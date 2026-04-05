from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated, TypedDict

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.cities import CityScorePoint, rank_city_scores
from backend.climate_repository import (
    ClimateDataError,
    ClimateRepository,
    build_default_climate_repository,
)
from backend.config import DEFAULT_PREFERENCES
from backend.heatmap import render_heatmap_png
from backend.scoring import CellScorePoint, PreferenceInputs, score_climate_cells

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
CLIMATE_DATABASE_PATH = ROOT_DIR / "data" / "climate.duckdb"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ScoreResponse(TypedDict):
    """Combined score + heatmap response so cells are only scored once per request."""

    scores: list[CityScorePoint]
    heatmap: str  # data:image/png;base64,... covering the full world extent


def build_index_context() -> dict[str, object]:
    """Return template context for the initial page render."""
    return {"preferences": DEFAULT_PREFERENCES}


def create_app(
    climate_repository: ClimateRepository | None = None,
) -> FastAPI:
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
    async def score(preferences: Annotated[PreferenceInputs, Form()]) -> ScoreResponse:
        try:
            climate_cells = repository.list_cells()
            cities = repository.list_cities()
        except ClimateDataError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

        raw = score_climate_cells(climate_cells, preferences)
        if not raw:
            return {"scores": [], "heatmap": ""}

        max_score = max(p["score"] for p in raw)
        if max_score == 0:
            return {"scores": [], "heatmap": ""}

        normalized: list[CellScorePoint] = [
            {"lat": p["lat"], "lon": p["lon"], "score": round(p["score"] / max_score, 4)} for p in raw
        ]

        top20 = rank_city_scores(cities, normalized, limit=20)
        png = render_heatmap_png(normalized)
        heatmap_data_url = "data:image/png;base64," + base64.b64encode(png).decode()

        return {"scores": top20, "heatmap": heatmap_data_url}

    return app


app = create_app()
