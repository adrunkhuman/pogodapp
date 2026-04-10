from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.cities import GRID_DEGREES
from backend.climate_repository import (
    ClimateDataError,
    ClimateRepository,
    build_default_climate_repository,
)
from backend.config import DEFAULT_PREFERENCES, MAP_PROJECTION
from backend.logging_config import configure_backend_logging
from backend.runtime import resolve_climate_database_path
from backend.score_service import ScoreResponse, build_score_response
from backend.scoring import (
    PreferenceInputs,
    ProbeBreakdown,
    score_matrix_row_breakdown,
)

if TYPE_CHECKING:
    from starlette.middleware.base import RequestResponseEndpoint

    from backend.scoring import ClimateMatrix

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
logger = logging.getLogger(__name__)
CLIENT_ERROR_STATUS_MIN = 400
SERVER_ERROR_STATUS_MIN = 500
SCORE_CACHE_SIZE = 16


class _ScoreResponseCache:
    """Avoid recomputing identical score requests within one worker."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, int, int, int, int], ScoreResponse] = OrderedDict()

    def get(self, key: tuple[int, int, int, int, int]) -> ScoreResponse | None:
        response = self._entries.get(key)
        if response is None:
            return None
        self._entries.move_to_end(key)
        return response

    def set(self, key: tuple[int, int, int, int, int], response: ScoreResponse) -> None:
        self._entries[key] = response
        self._entries.move_to_end(key)
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class _SupportsProbeRepository(Protocol):
    def probe_nearest_cell(self, lat: float, lon: float) -> int | None: ...

    def get_climate_matrix(self) -> ClimateMatrix: ...


class ProbeMetricResponse(BaseModel):
    """One rendered metric row in the `/probe` payload."""

    key: str
    label: str
    value: float
    display_value: str
    score: float


class ProbeResponse(BaseModel):
    """Per-attribute climate breakdown for one hovered map point."""

    found: bool = False
    overall_score: float = 0.0
    metrics: list[ProbeMetricResponse] = Field(default_factory=list)


def build_probe_response(breakdown: ProbeBreakdown) -> ProbeResponse:
    """Map scoring breakdown data into the stable `/probe` response model."""
    return ProbeResponse(
        found=True,
        overall_score=breakdown.overall_score,
        metrics=[ProbeMetricResponse(**asdict(metric)) for metric in breakdown.metrics],
    )


def build_index_context() -> dict[str, object]:
    """Return template context for the initial page render."""
    return {
        "preferences": DEFAULT_PREFERENCES,
        "map_projection": MAP_PROJECTION,
        "probe_grid_degrees": GRID_DEGREES,
    }


def preload_repository(repository: ClimateRepository) -> None:
    """Warm startup caches without turning recoverable data issues into boot failures."""
    if not hasattr(repository, "get_climate_matrix") or not hasattr(repository, "get_indexed_cities"):
        return

    try:
        # Test doubles and fallback repositories may only implement the slower request path.
        repository.get_climate_matrix()
        repository.get_indexed_cities()
        if hasattr(repository, "get_heatmap_projection"):
            repository.get_heatmap_projection()
        logger.info(
            "repository preload finished",
            extra={"event": "startup_preload", "outcome": "ok", "repository": type(repository).__name__},
        )
    except ClimateDataError as error:
        logger.warning(
            "repository preload skipped",
            extra={"event": "startup_preload", "outcome": "skipped", "detail": str(error)},
        )


def _score_cache_key(preferences: PreferenceInputs) -> tuple[int, int, int, int, int]:
    return (
        preferences.preferred_day_temperature,
        preferences.summer_heat_limit,
        preferences.winter_cold_limit,
        preferences.dryness_preference,
        preferences.sunshine_preference,
    )


def _score_response_from_cache_or_repository(
    score_cache: _ScoreResponseCache,
    repository: ClimateRepository,
    preferences: PreferenceInputs,
) -> ScoreResponse:
    cache_key = _score_cache_key(preferences)
    cached_response = score_cache.get(cache_key)
    if cached_response is not None:
        return cached_response

    response = build_score_response(repository, preferences)
    score_cache.set(cache_key, response)
    return response


def probe_preferences_dependency(
    preferred_day_temperature: Annotated[int, Query(ge=5, le=35)],
    summer_heat_limit: Annotated[int, Query(ge=18, le=42)],
    winter_cold_limit: Annotated[int, Query(ge=-15, le=20)],
    dryness_preference: Annotated[int, Query(ge=0, le=100)],
    sunshine_preference: Annotated[int, Query(ge=0, le=100)],
) -> PreferenceInputs:
    """Validate `/probe` query preferences through the same model as `/score`."""
    try:
        return PreferenceInputs(
            preferred_day_temperature=preferred_day_temperature,
            summer_heat_limit=summer_heat_limit,
            winter_cold_limit=winter_cold_limit,
            dryness_preference=dryness_preference,
            sunshine_preference=sunshine_preference,
        )
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error


async def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": "Too many requests"}, status_code=429)


def _request_log_context(request: Request, response: Response | None = None) -> tuple[str, str, str, str, int | str]:
    client = request.client.host if request.client else "unknown"
    query = request.url.query or "-"
    content_length = response.headers.get("content-length", "-") if response is not None else "-"
    http_version = request.scope.get("http_version", "unknown")
    return client, query, request.url.scheme, http_version, content_length


def _request_outcome(status_code: int) -> str:
    if status_code >= SERVER_ERROR_STATUS_MIN:
        return "error"
    if status_code >= CLIENT_ERROR_STATUS_MIN:
        return "client_error"
    return "ok"


def _attach_http_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_http_requests(request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            client, query, scheme, http_version, content_length = _request_log_context(request)
            logger.exception(
                "http request failed",
                extra={
                    "event": "http_request",
                    "outcome": "error",
                    "method": request.method,
                    "path": request.url.path,
                    "query": query,
                    "httpStatus": 500,
                    "srcIp": client,
                    "scheme": scheme,
                    "httpVersion": http_version,
                    "txBytes": content_length,
                    "responseTime": round((perf_counter() - started) * 1000, 2),
                    "host": request.headers.get("host", "-"),
                },
            )
            raise

        client, query, scheme, http_version, content_length = _request_log_context(request, response)
        logger.info(
            "http request finished",
            extra={
                "event": "http_request",
                "outcome": _request_outcome(response.status_code),
                "method": request.method,
                "path": request.url.path,
                "query": query,
                "httpStatus": response.status_code,
                "srcIp": client,
                "scheme": scheme,
                "httpVersion": http_version,
                "txBytes": content_length,
                "responseTime": round((perf_counter() - started) * 1000, 2),
                "host": request.headers.get("host", "-"),
            },
        )
        return response


def create_app(
    climate_repository: ClimateRepository | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    The app serves the initial shell, static assets, `/score`, `/probe`, and
    `/health`, and wires standard response compression plus request logging.
    `climate_repository` is injectable for tests; production wiring falls back
    to the local DuckDB artifact when present and otherwise uses the in-repo
    stub dataset.
    """
    configure_backend_logging()
    app = FastAPI(title="Pogodapp")
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    _attach_http_request_logging(app)
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    repository = climate_repository or build_default_climate_repository(resolve_climate_database_path())
    score_cache = _ScoreResponseCache(SCORE_CACHE_SIZE)
    preload_repository(repository)

    # Pre-warm cache with default preferences so the load-triggered POST on first page
    # visit hits a cache entry rather than doing a full 5s computation cold.
    try:
        default_prefs = PreferenceInputs(**{f.name: f.value for f in DEFAULT_PREFERENCES})
        score_cache.set(_score_cache_key(default_prefs), build_score_response(repository, default_prefs))
        logger.info("startup default score cached", extra={"event": "startup_default_score", "outcome": "ok"})
    except ClimateDataError:
        logger.warning("startup default score skipped", extra={"event": "startup_default_score", "outcome": "skipped"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=build_index_context(),
        )

    @app.post("/score")
    @limiter.limit("30/minute")
    async def score(request: Request, preferences: Annotated[PreferenceInputs, Form()]) -> ScoreResponse:
        try:
            return _score_response_from_cache_or_repository(score_cache, repository, preferences)
        except ClimateDataError as error:
            logger.exception(
                "score request failed",
                extra={
                    "event": "score_request",
                    "outcome": "error",
                    "preferred_day_temperature": preferences.preferred_day_temperature,
                    "summer_heat_limit": preferences.summer_heat_limit,
                    "winter_cold_limit": preferences.winter_cold_limit,
                    "dryness_preference": preferences.dryness_preference,
                    "sunshine_preference": preferences.sunshine_preference,
                },
            )
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/probe")
    @limiter.limit("120/minute")
    async def probe(
        request: Request,
        lat: Annotated[float, Query(ge=-90, le=90)],
        lon: Annotated[float, Query(ge=-180, le=180)],
        preferences: Annotated[PreferenceInputs, Depends(probe_preferences_dependency)],
    ) -> ProbeResponse:
        if not hasattr(repository, "probe_nearest_cell"):
            return ProbeResponse()
        probe_repository = cast("_SupportsProbeRepository", repository)
        try:
            row_index = probe_repository.probe_nearest_cell(lat, lon)
            if row_index is None:
                return ProbeResponse()
            climate_matrix = probe_repository.get_climate_matrix()
        except ClimateDataError as error:
            logger.exception(
                "probe request failed",
                extra={"event": "probe_request", "outcome": "error", "lat": lat, "lon": lon},
            )
            raise HTTPException(status_code=503, detail=str(error)) from error
        breakdown = score_matrix_row_breakdown(climate_matrix, row_index, preferences)
        return build_probe_response(breakdown)

    return app


app = create_app()
