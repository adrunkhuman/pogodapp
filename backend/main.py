from __future__ import annotations

import asyncio
import logging
import mmap
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, NamedTuple, Protocol, cast
from urllib.parse import urlencode

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
from starlette.concurrency import run_in_threadpool

from backend.cities import GRID_DEGREES
from backend.climate_repository import (
    ClimateDataError,
    ClimateRepository,
    build_default_climate_repository,
)
from backend.config import DEFAULT_PREFERENCES, MAP_PROJECTION
from backend.logging_config import configure_backend_logging
from backend.runtime import resolve_climate_database_path
from backend.score_service import HeatmapField, ScoreResponse, build_heatmap_response, build_score_response
from backend.scoring import (
    ClimateCell,
    PreferenceInputs,
    ProbeBreakdown,
    score_climate_cell_breakdown,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.middleware.base import RequestResponseEndpoint

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
logger = logging.getLogger(__name__)
CLIENT_ERROR_STATUS_MIN = 400
SERVER_ERROR_STATUS_MIN = 500
SCORE_CACHE_SIZE = 16
HEATMAP_FIELD_CACHE_SIZE = 2
HEATMAP_FIELD_CACHE_TTL_SECONDS = 20.0
SCORE_REQUEST_CONCURRENCY = 2


class _ScoreResponseCache:
    """Avoid recomputing identical score requests within one worker."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, int, int, int, int], ScoreResponse] = OrderedDict()
        self._inflight: dict[tuple[int, int, int, int, int], threading.Event] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple[int, int, int, int, int]) -> ScoreResponse | None:
        with self._lock:
            response = self._entries.get(key)
            if response is None:
                return None
            self._entries.move_to_end(key)
            return response

    def set(self, key: tuple[int, int, int, int, int], response: ScoreResponse) -> None:
        with self._lock:
            self._entries[key] = response
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def get_or_set(
        self,
        key: tuple[int, int, int, int, int],
        build: Callable[[], ScoreResponse],
    ) -> ScoreResponse:
        return self.get_with_status_or_set(key, build).response

    def get_with_status_or_set(
        self,
        key: tuple[int, int, int, int, int],
        build: Callable[[], ScoreResponse],
    ) -> _ScoreCacheResult:
        waited_on_inflight = False
        while True:
            with self._lock:
                response = self._entries.get(key)
                if response is not None:
                    self._entries.move_to_end(key)
                    cache_status = "miss_waited" if waited_on_inflight else "hit"
                    return _ScoreCacheResult(
                        response=response, cache_hit=not waited_on_inflight, cache_status=cache_status
                    )

                inflight = self._inflight.get(key)
                if inflight is None:
                    inflight = threading.Event()
                    self._inflight[key] = inflight
                    break

            waited_on_inflight = True
            inflight.wait()

        try:
            response = build()
        except Exception:
            with self._lock:
                self._inflight.pop(key).set()
            raise

        with self._lock:
            self._entries[key] = response
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._inflight.pop(key).set()
            return _ScoreCacheResult(response=response, cache_hit=False, cache_status="miss_built")


class _ScoreCacheResult(NamedTuple):
    response: ScoreResponse
    cache_hit: bool
    cache_status: str


class _HeatmapFieldCache:
    """Keep a tiny, short-lived score-field cache for `/heatmap` reuse."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[tuple[int, int, int, int, int], tuple[float, HeatmapField]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[int, int, int, int, int]) -> HeatmapField | None:
        now = perf_counter()
        with self._lock:
            self._purge_expired(now)
            cached = self._entries.get(key)
            if cached is None:
                return None
            self._entries.move_to_end(key)
            return cached[1]

    def set(self, key: tuple[int, int, int, int, int], heatmap_field: HeatmapField) -> None:
        now = perf_counter()
        with self._lock:
            self._purge_expired(now)
            self._entries[key] = (now, heatmap_field)
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def _purge_expired(self, now: float) -> None:
        expired_keys = [key for key, (stored_at, _field) in self._entries.items() if now - stored_at > self.ttl_seconds]
        for key in expired_keys:
            self._entries.pop(key, None)


class _RequestConcurrencyTracker:
    """Track waiting and active request counts for one route class."""

    def __init__(self) -> None:
        self._waiting = 0
        self._active = 0
        self._lock = threading.Lock()

    def mark_waiting(self) -> int:
        with self._lock:
            self._waiting += 1
            return self._waiting

    def mark_started(self) -> tuple[int, int]:
        with self._lock:
            self._waiting -= 1
            self._active += 1
            return self._waiting, self._active

    def mark_finished(self) -> None:
        with self._lock:
            self._active -= 1


class _SupportsProbeRepository(Protocol):
    def probe_nearest_cell(self, lat: float, lon: float) -> int | None: ...

    def get_probe_cell(self, row_index: int) -> ClimateCell: ...


def _current_rss_mb() -> float | None:
    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return None

    try:
        rss_pages = int(statm_path.read_text(encoding="utf-8").split()[1])
        return round((rss_pages * mmap.PAGESIZE) / (1024 * 1024), 1)
    except (OSError, ValueError, IndexError):
        return None


def _log_runtime_memory(stage: str, repository: ClimateRepository) -> None:
    extra: dict[str, object] = {
        "event": "runtime_memory",
        "stage": stage,
        "repository": type(repository).__name__,
    }
    rss_mb = _current_rss_mb()
    if rss_mb is not None:
        extra["rss_mb"] = rss_mb
    get_runtime_cache_stats = getattr(repository, "get_runtime_cache_stats", None)
    if callable(get_runtime_cache_stats):
        extra.update(get_runtime_cache_stats())
    logger.info("runtime memory snapshot", extra=extra)


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
        _log_runtime_memory("before_preload", repository)
        # Test doubles and fallback repositories may only implement the slower request path.
        repository.get_climate_matrix()
        _log_runtime_memory("after_climate_matrix", repository)
        repository.get_indexed_cities()
        _log_runtime_memory("after_indexed_cities", repository)
        if hasattr(repository, "get_heatmap_projection"):
            repository.get_heatmap_projection()
            _log_runtime_memory("after_heatmap_projection", repository)
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


def _score_log_fields(preferences: PreferenceInputs) -> dict[str, int]:
    return {
        "preferred_day_temperature": preferences.preferred_day_temperature,
        "summer_heat_limit": preferences.summer_heat_limit,
        "winter_cold_limit": preferences.winter_cold_limit,
        "dryness_preference": preferences.dryness_preference,
        "sunshine_preference": preferences.sunshine_preference,
    }


def _score_response_from_cache_or_repository(
    score_cache: _ScoreResponseCache,
    heatmap_field_cache: _HeatmapFieldCache,
    repository: ClimateRepository,
    preferences: PreferenceInputs,
) -> _ScoreCacheResult:
    cache_key = _score_cache_key(preferences)
    return score_cache.get_with_status_or_set(
        cache_key,
        lambda: build_score_response(
            repository,
            preferences,
            store_heatmap_field=lambda heatmap_field: heatmap_field_cache.set(cache_key, heatmap_field),
        ),
    )


def _coerce_score_cache_result(result: _ScoreCacheResult | ScoreResponse) -> _ScoreCacheResult:
    if isinstance(result, _ScoreCacheResult):
        return result
    return _ScoreCacheResult(response=result, cache_hit=False, cache_status="miss_built")


def _heatmap_url(preferences: PreferenceInputs) -> str:
    return "/heatmap?" + urlencode(
        {
            "preferred_day_temperature": preferences.preferred_day_temperature,
            "summer_heat_limit": preferences.summer_heat_limit,
            "winter_cold_limit": preferences.winter_cold_limit,
            "dryness_preference": preferences.dryness_preference,
            "sunshine_preference": preferences.sunshine_preference,
        }
    )


def _build_probe_response_from_repository(
    repository: _SupportsProbeRepository,
    lat: float,
    lon: float,
    preferences: PreferenceInputs,
) -> ProbeResponse:
    row_index = repository.probe_nearest_cell(lat, lon)
    if row_index is None:
        return ProbeResponse()

    climate_cell = repository.get_probe_cell(row_index)
    breakdown = score_climate_cell_breakdown(climate_cell, preferences)
    return build_probe_response(breakdown)


def probe_preferences_dependency(
    preferred_day_temperature: Annotated[int, Query(ge=-5, le=35)],
    summer_heat_limit: Annotated[int, Query(ge=-5, le=42)],
    winter_cold_limit: Annotated[int, Query(ge=-15, le=35)],
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


def create_app(  # noqa: C901, PLR0915
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
    heatmap_field_cache = _HeatmapFieldCache(HEATMAP_FIELD_CACHE_SIZE, HEATMAP_FIELD_CACHE_TTL_SECONDS)
    # Let a small number of /score builds overlap so slider bursts do not serialize behind one miss.
    score_request_semaphore = asyncio.Semaphore(SCORE_REQUEST_CONCURRENCY)
    score_request_tracker = _RequestConcurrencyTracker()
    heatmap_request_semaphore = asyncio.Semaphore(1)
    preload_repository(repository)

    # Pre-warm default scores so the page-load HTMX POST hits cache instead of paying the cold path.
    try:
        default_prefs = PreferenceInputs(**{f.name: f.value for f in DEFAULT_PREFERENCES})
        default_cache_key = _score_cache_key(default_prefs)
        score_cache.set(
            default_cache_key,
            build_score_response(
                repository,
                default_prefs,
                store_heatmap_field=lambda heatmap_field: heatmap_field_cache.set(default_cache_key, heatmap_field),
            ),
        )
        _log_runtime_memory("after_default_score_cache", repository)
        logger.info("startup default score cached", extra={"event": "startup_default_score", "outcome": "ok"})
    except ClimateDataError:
        logger.warning("startup default score skipped", extra={"event": "startup_default_score", "outcome": "skipped"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    async def _serve_score_request(preferences: PreferenceInputs) -> _ScoreCacheResult:
        queue_started = perf_counter()
        score_request_tracker.mark_waiting()
        async with score_request_semaphore:
            score_queue_depth, score_inflight = score_request_tracker.mark_started()
            queue_wait_ms = round((perf_counter() - queue_started) * 1000, 2)
            try:
                raw_cache_result = await run_in_threadpool(
                    _score_response_from_cache_or_repository,
                    score_cache,
                    heatmap_field_cache,
                    repository,
                    preferences,
                )
            finally:
                score_request_tracker.mark_finished()
        cache_result = _coerce_score_cache_result(raw_cache_result)
        logger.info(
            "score request served",
            extra={
                "event": "score_request_route",
                "outcome": "ok",
                "cache_hit": cache_result.cache_hit,
                "cache_status": cache_result.cache_status,
                "queue_wait_ms": queue_wait_ms,
                "score_queue_depth": score_queue_depth,
                "score_inflight": score_inflight,
                **_score_log_fields(preferences),
            },
        )
        return cache_result

    async def _serve_heatmap_request(preferences: PreferenceInputs) -> bytes:
        cache_key = _score_cache_key(preferences)
        cached_heatmap_field = heatmap_field_cache.get(cache_key)
        queue_started = perf_counter()
        async with heatmap_request_semaphore:
            queue_wait_ms = round((perf_counter() - queue_started) * 1000, 2)
            heatmap_png = await run_in_threadpool(
                build_heatmap_response,
                repository,
                preferences,
                cached_heatmap_field=cached_heatmap_field,
            )
        outcome = "ok" if heatmap_png else "empty"
        logger.info(
            "heatmap request served",
            extra={
                "event": "heatmap_request_route",
                "outcome": outcome,
                "score_field_cache_hit": cached_heatmap_field is not None,
                "queue_wait_ms": queue_wait_ms,
                **_score_log_fields(preferences),
            },
        )
        return heatmap_png

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
    async def score(request: Request, preferences: Annotated[PreferenceInputs, Form()]) -> dict[str, object]:
        try:
            cache_result = await _serve_score_request(preferences)
        except ClimateDataError as error:
            logger.exception(
                "score request failed",
                extra={
                    "event": "score_request",
                    "outcome": "error",
                    **_score_log_fields(preferences),
                },
            )
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "scores": cache_result.response["scores"],
            "heatmap_url": _heatmap_url(preferences) if cache_result.response["scores"] else "",
        }

    @app.get("/heatmap")
    @limiter.limit("30/minute")
    async def heatmap(
        request: Request,
        preferences: Annotated[PreferenceInputs, Depends(probe_preferences_dependency)],
    ) -> Response:
        try:
            heatmap_png = await _serve_heatmap_request(preferences)
        except ClimateDataError as error:
            logger.exception(
                "heatmap request failed",
                extra={
                    "event": "heatmap_request",
                    "outcome": "error",
                    "preferred_day_temperature": preferences.preferred_day_temperature,
                    "summer_heat_limit": preferences.summer_heat_limit,
                    "winter_cold_limit": preferences.winter_cold_limit,
                    "dryness_preference": preferences.dryness_preference,
                    "sunshine_preference": preferences.sunshine_preference,
                },
            )
            raise HTTPException(status_code=503, detail=str(error)) from error
        if not heatmap_png:
            return Response(status_code=204)
        return Response(content=heatmap_png, media_type="image/png")

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
            return await run_in_threadpool(
                _build_probe_response_from_repository, probe_repository, lat, lon, preferences
            )
        except ClimateDataError as error:
            logger.exception(
                "probe request failed",
                extra={"event": "probe_request", "outcome": "error", "lat": lat, "lon": lon},
            )
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()
