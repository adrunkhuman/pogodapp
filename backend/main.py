from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.climate_repository import (
    ClimateDataError,
    ClimateRepository,
    build_default_climate_repository,
)
from backend.config import DEFAULT_PREFERENCES, MAP_PROJECTION
from backend.logging_config import configure_backend_logging
from backend.score_service import ScoreResponse, build_score_response
from backend.scoring import PreferenceInputs  # noqa: TC001 - FastAPI needs the runtime symbol for Form model parsing

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
CLIMATE_DATABASE_PATH = ROOT_DIR / "data" / "climate.duckdb"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def build_index_context() -> dict[str, object]:
    """Return template context for the initial page render."""
    return {"preferences": DEFAULT_PREFERENCES, "map_projection": MAP_PROJECTION}


def preload_repository(repository: ClimateRepository) -> None:
    """Warm the optimized repository path during app startup."""
    if not hasattr(repository, "get_climate_matrix") or not hasattr(repository, "get_indexed_cities"):
        return

    repository.get_climate_matrix()
    repository.get_indexed_cities()


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
    repository = climate_repository or build_default_climate_repository(CLIMATE_DATABASE_PATH)
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
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()
