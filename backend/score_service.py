from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

from backend.cities import CityScorePoint, rank_city_scores
from backend.heatmap import render_heatmap_png
from backend.scoring import CellScorePoint, PreferenceInputs, score_climate_cells

if TYPE_CHECKING:
    from backend.climate_repository import ClimateRepository

TOP_CITY_RESULTS = 20
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoreTimings:
    """Per-step timing snapshot for one `/score` calculation."""

    cells_ms: float = 0.0
    cities_ms: float = 0.0
    scoring_ms: float = 0.0
    normalize_ms: float = 0.0
    ranking_ms: float = 0.0
    heatmap_ms: float = 0.0
    total_ms: float = 0.0


class ScoreResponse(TypedDict):
    """Backend response contract for one scored user preference request."""

    scores: list[CityScorePoint]
    heatmap: str


def _elapsed_ms(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000, 2)


def _log_score_timings(
    timings: ScoreTimings,
    *,
    climate_cell_count: int,
    city_count: int,
    ranked_city_count: int,
    outcome: str,
) -> None:
    logger.info(
        "score_request outcome=%s total_ms=%.2f cells_ms=%.2f cities_ms=%.2f scoring_ms=%.2f normalize_ms=%.2f ranking_ms=%.2f heatmap_ms=%.2f climate_cells=%d cities=%d ranked_cities=%d",
        outcome,
        timings.total_ms,
        timings.cells_ms,
        timings.cities_ms,
        timings.scoring_ms,
        timings.normalize_ms,
        timings.ranking_ms,
        timings.heatmap_ms,
        climate_cell_count,
        city_count,
        ranked_city_count,
    )


def build_score_response(repository: ClimateRepository, preferences: PreferenceInputs) -> ScoreResponse:
    """Score all available climate cells and shape the `/score` API payload.

    The route keeps FastAPI concerns only; this function owns the application-level
    scoring flow so ranking, normalization, and heatmap rendering stay testable
    outside HTTP wiring.
    """
    request_started = perf_counter()
    timings = ScoreTimings()

    cells_started = perf_counter()
    climate_cells = repository.list_cells()
    timings.cells_ms = _elapsed_ms(cells_started)

    cities_started = perf_counter()
    cities = repository.list_cities()
    timings.cities_ms = _elapsed_ms(cities_started)

    scoring_started = perf_counter()
    raw_scores = score_climate_cells(climate_cells, preferences)
    timings.scoring_ms = _elapsed_ms(scoring_started)

    if not raw_scores:
        timings.total_ms = _elapsed_ms(request_started)
        _log_score_timings(
            timings,
            climate_cell_count=len(climate_cells),
            city_count=len(cities),
            ranked_city_count=0,
            outcome="empty",
        )
        return {"scores": [], "heatmap": ""}

    max_score = max(point["score"] for point in raw_scores)
    if max_score == 0:
        # An all-zero result carries no useful ranking or map signal for the UI.
        timings.total_ms = _elapsed_ms(request_started)
        _log_score_timings(
            timings,
            climate_cell_count=len(climate_cells),
            city_count=len(cities),
            ranked_city_count=0,
            outcome="all_zero",
        )
        return {"scores": [], "heatmap": ""}

    # Re-normalize each response so the best match in the current result set lands
    # at 1.0, which keeps the heatmap visually informative across very different queries.
    normalize_started = perf_counter()
    normalized_scores: list[CellScorePoint] = [
        {"lat": point["lat"], "lon": point["lon"], "score": round(point["score"] / max_score, 4)}
        for point in raw_scores
    ]
    timings.normalize_ms = _elapsed_ms(normalize_started)

    ranking_started = perf_counter()
    top_cities = rank_city_scores(cities, normalized_scores, limit=TOP_CITY_RESULTS)
    timings.ranking_ms = _elapsed_ms(ranking_started)

    heatmap_started = perf_counter()
    heatmap_png = render_heatmap_png(normalized_scores)
    timings.heatmap_ms = _elapsed_ms(heatmap_started)
    timings.total_ms = _elapsed_ms(request_started)

    _log_score_timings(
        timings,
        climate_cell_count=len(climate_cells),
        city_count=len(cities),
        ranked_city_count=len(top_cities),
        outcome="ok",
    )

    return {
        "scores": top_cities,
        "heatmap": "data:image/png;base64," + base64.b64encode(heatmap_png).decode(),
    }
