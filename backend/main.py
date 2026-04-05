from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, cast

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.climate_repository import (
    ClimateDataError,
    ClimateRepository,
    build_default_climate_repository,
)
from backend.config import DEFAULT_PREFERENCES, MAP_PROJECTION
from backend.logging_config import configure_backend_logging
from backend.score_service import ScoreResponse, build_score_response
from backend.scoring import (
    PreferenceInputs,
    score_matrix_row_breakdown,
)

if TYPE_CHECKING:
    from backend.scoring import ClimateMatrix

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
CLIMATE_DATABASE_PATH = ROOT_DIR / "data" / "climate.duckdb"
CLIMATE_DATABASE_ENV_VAR = "POGODAPP_CLIMATE_DB"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
logger = logging.getLogger(__name__)


class _SupportsProbeRepository(Protocol):
    def probe_nearest_cell(self, lat: float, lon: float) -> int | None: ...

    def get_climate_matrix(self) -> ClimateMatrix: ...


class ProbeResponse(BaseModel):
    """Per-attribute climate breakdown for one hovered map point."""

    found: bool = False
    avg_temp_c: float = 0.0
    avg_precip_mm: float = 0.0
    avg_cloud_pct: float = 0.0
    temp_score: float = 0.0
    rain_score: float = 0.0
    cloud_score: float = 0.0
    overall_score: float = 0.0


def build_index_context() -> dict[str, object]:
    """Return template context for the initial page render."""
    return {"preferences": DEFAULT_PREFERENCES, "map_projection": MAP_PROJECTION}


def resolve_climate_database_path() -> Path:
    """Resolve the runtime climate database path from env or the default location."""
    configured_path = os.getenv(CLIMATE_DATABASE_ENV_VAR)
    return Path(configured_path) if configured_path else CLIMATE_DATABASE_PATH


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
    except ClimateDataError as error:
        logger.warning("startup_preload outcome=skipped detail=%s", error)


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
    repository = climate_repository or build_default_climate_repository(resolve_climate_database_path())
    preload_repository(repository)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=build_index_context(),
        )

    @app.post("/score")
    async def score(preferences: Annotated[PreferenceInputs, Form()]) -> ScoreResponse:
        try:
            return build_score_response(repository, preferences)
        except ClimateDataError as error:
            logger.exception("score_request outcome=error")
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/probe")
    async def probe(  # noqa: PLR0913
        lat: Annotated[float, Query(ge=-90, le=90)],
        lon: Annotated[float, Query(ge=-180, le=180)],
        ideal_temperature: Annotated[int, Query(ge=-10, le=35)],
        cold_tolerance: Annotated[int, Query(ge=0, le=15)],
        heat_tolerance: Annotated[int, Query(ge=0, le=15)],
        rain_sensitivity: Annotated[int, Query(ge=0, le=100)],
        sun_preference: Annotated[int, Query(ge=0, le=100)],
    ) -> ProbeResponse:
        if not hasattr(repository, "probe_nearest_cell"):
            return ProbeResponse()
        probe_repository = cast("_SupportsProbeRepository", repository)
        row_index = probe_repository.probe_nearest_cell(lat, lon)
        if row_index is None:
            return ProbeResponse()
        try:
            climate_matrix = probe_repository.get_climate_matrix()
        except ClimateDataError as error:
            logger.exception("probe_request outcome=error")
            raise HTTPException(status_code=503, detail=str(error)) from error
        preferences = PreferenceInputs(
            ideal_temperature=ideal_temperature,
            cold_tolerance=cold_tolerance,
            heat_tolerance=heat_tolerance,
            rain_sensitivity=rain_sensitivity,
            sun_preference=sun_preference,
        )
        breakdown = score_matrix_row_breakdown(climate_matrix, row_index, preferences)
        return ProbeResponse(found=True, **breakdown)

    return app


app = create_app()
