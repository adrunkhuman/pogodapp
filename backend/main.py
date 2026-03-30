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
from backend.config import DEFAULT_PREFERENCES
from backend.scoring import PreferenceInputs, ScorePoint, score_climate_cells

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
CLIMATE_DATABASE_PATH = ROOT_DIR / "data" / "climate.duckdb"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def build_index_context() -> dict[str, object]:
    """Return template context for the initial page render."""
    return {"preferences": DEFAULT_PREFERENCES}


def create_app(climate_repository: ClimateRepository | None = None) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Pogodapp")
    repository = climate_repository or build_default_climate_repository(CLIMATE_DATABASE_PATH)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=build_index_context(),
        )

    @app.post("/score")
    async def score(preferences: Annotated[PreferenceInputs, Form()]) -> list[ScorePoint]:
        try:
            climate_cells = repository.list_cells()
        except ClimateDataError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        return score_climate_cells(climate_cells, preferences)

    return app


app = create_app()
