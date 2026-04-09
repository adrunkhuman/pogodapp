from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
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
    from backend.scoring import ClimateMatrix

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
logger = logging.getLogger(__name__)


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
        logger.info("startup_preload outcome=ok repository=%s", type(repository).__name__)
    except ClimateDataError as error:
        logger.warning("startup_preload outcome=skipped detail=%s", error)


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


def create_app(
    climate_repository: ClimateRepository | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    The app serves the initial shell, static frontend assets, and the `/score`
    JSON API. `climate_repository` is injectable for tests; production wiring
    falls back to the local DuckDB artifact when present and otherwise uses the
    in-repo stub dataset.
    """
    configure_backend_logging()
    app = FastAPI(title="Pogodapp")
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    repository = climate_repository or build_default_climate_repository(resolve_climate_database_path())
    preload_repository(repository)

    initial_scores: ScoreResponse | None = None
    try:
        default_prefs = PreferenceInputs(**{f.name: f.value for f in DEFAULT_PREFERENCES})
        initial_scores = build_score_response(repository, default_prefs)
        logger.info("startup_default_scores outcome=ok cities=%d", len(initial_scores.get("scores", [])))
    except ClimateDataError:
        logger.warning("startup_default_scores outcome=skipped")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={**build_index_context(), "initial_scores": initial_scores},
        )

    @app.post("/score")
    @limiter.limit("30/minute")
    async def score(request: Request, preferences: Annotated[PreferenceInputs, Form()]) -> ScoreResponse:
        try:
            return build_score_response(repository, preferences)
        except ClimateDataError as error:
            logger.exception("score_request outcome=error")
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
            logger.exception("probe_request outcome=error")
            raise HTTPException(status_code=503, detail=str(error)) from error
        breakdown = score_matrix_row_breakdown(climate_matrix, row_index, preferences)
        return build_probe_response(breakdown)

    return app


app = create_app()
