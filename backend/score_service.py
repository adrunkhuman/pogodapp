from __future__ import annotations

import base64
from typing import TYPE_CHECKING, TypedDict

from backend.cities import CityScorePoint, rank_city_scores
from backend.heatmap import render_heatmap_png
from backend.scoring import CellScorePoint, PreferenceInputs, score_climate_cells

if TYPE_CHECKING:
    from backend.climate_repository import ClimateRepository

TOP_CITY_RESULTS = 20


class ScoreResponse(TypedDict):
    """Backend response contract for one scored user preference request."""

    scores: list[CityScorePoint]
    heatmap: str


def build_score_response(repository: ClimateRepository, preferences: PreferenceInputs) -> ScoreResponse:
    """Score all available climate cells and shape the `/score` API payload.

    The route keeps FastAPI concerns only; this function owns the application-level
    scoring flow so ranking, normalization, and heatmap rendering stay testable
    outside HTTP wiring.
    """
    climate_cells = repository.list_cells()
    cities = repository.list_cities()
    raw_scores = score_climate_cells(climate_cells, preferences)

    if not raw_scores:
        return {"scores": [], "heatmap": ""}

    max_score = max(point["score"] for point in raw_scores)
    if max_score == 0:
        # An all-zero result carries no useful ranking or map signal for the UI.
        return {"scores": [], "heatmap": ""}

    # Re-normalize each response so the best match in the current result set lands
    # at 1.0, which keeps the heatmap visually informative across very different queries.
    normalized_scores: list[CellScorePoint] = [
        {"lat": point["lat"], "lon": point["lon"], "score": round(point["score"] / max_score, 4)}
        for point in raw_scores
    ]
    top_cities = rank_city_scores(cities, normalized_scores, limit=TOP_CITY_RESULTS)
    heatmap_png = render_heatmap_png(normalized_scores)

    return {
        "scores": top_cities,
        "heatmap": "data:image/png;base64," + base64.b64encode(heatmap_png).decode(),
    }
