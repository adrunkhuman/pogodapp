from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.config import DEFAULT_PREFERENCES
from backend.scoring import PreferenceInputs, ScorePoint, score_preferences

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Keep FastAPI validation aligned with the slider contract in backend config.
PREFERENCE_BOUNDS = {field.name: (field.minimum, field.maximum) for field in DEFAULT_PREFERENCES}


def build_index_context() -> dict[str, object]:
    """Return template context for the initial page render."""
    return {"preferences": DEFAULT_PREFERENCES}


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="Pogodapp")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=build_index_context(),
        )

    @app.post("/score")
    async def score(
        ideal_temperature: Annotated[
            int,
            Form(
                ge=PREFERENCE_BOUNDS["ideal_temperature"][0],
                le=PREFERENCE_BOUNDS["ideal_temperature"][1],
            ),
        ],
        cold_tolerance: Annotated[
            int,
            Form(
                ge=PREFERENCE_BOUNDS["cold_tolerance"][0],
                le=PREFERENCE_BOUNDS["cold_tolerance"][1],
            ),
        ],
        heat_tolerance: Annotated[
            int,
            Form(
                ge=PREFERENCE_BOUNDS["heat_tolerance"][0],
                le=PREFERENCE_BOUNDS["heat_tolerance"][1],
            ),
        ],
        rain_sensitivity: Annotated[
            int,
            Form(
                ge=PREFERENCE_BOUNDS["rain_sensitivity"][0],
                le=PREFERENCE_BOUNDS["rain_sensitivity"][1],
            ),
        ],
        sun_preference: Annotated[
            int,
            Form(
                ge=PREFERENCE_BOUNDS["sun_preference"][0],
                le=PREFERENCE_BOUNDS["sun_preference"][1],
            ),
        ],
    ) -> list[ScorePoint]:
        preferences = PreferenceInputs(
            ideal_temperature=ideal_temperature,
            cold_tolerance=cold_tolerance,
            heat_tolerance=heat_tolerance,
            rain_sensitivity=rain_sensitivity,
            sun_preference=sun_preference,
        )
        return score_preferences(preferences)

    return app


app = create_app()
