from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from backend.cities import CityScorePoint, continent_of, rank_city_scores, rank_indexed_city_scores
from backend.heatmap import render_heatmap_png, render_heatmap_png_from_arrays, render_heatmap_png_from_projection
from backend.scoring import (
    CellScorePoint,
    PreferenceInputs,
    normalize_score_array,
    score_climate_cells,
    score_climate_matrix,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.cities import CityRankingCache
    from backend.climate_repository import ClimateRepository

TOP_CITY_RESULTS = 30
# Larger unfiltered pool for map markers: top N globally + per-continent fill.
MARKER_GLOBAL_LIMIT = 80
MARKER_PER_CONTINENT_MIN = 3
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
    markers: list[CityScorePoint]
    heatmap: str


def _elapsed_ms(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000, 2)


def _select_markers(
    city_catalog: CityRankingCache,
    normalized_scores: NDArray[np.float32],
) -> list[CityScorePoint]:
    """Return map marker cities: top MARKER_GLOBAL_LIMIT globally plus per-continent fill.

    Uses raw score order (no diversity suppression) so hot regions show cluster density
    on the map. Continent fill ensures every inhabited region has at least
    MARKER_PER_CONTINENT_MIN dots even when its climate scores are low.
    """
    if not city_catalog.cities:
        return []

    city_scores = normalized_scores[city_catalog.climate_indexes]
    # Descending sort order by score.
    order = np.argsort(city_scores)[::-1]

    markers: list[CityScorePoint] = []
    continent_counts: dict[str, int] = {}
    seen: set[int] = set()

    for idx in order:
        city = city_catalog.cities[idx]
        continent = continent_of(city.country_code)
        in_global_top = len(markers) < MARKER_GLOBAL_LIMIT

        if not in_global_top and continent_counts.get(continent, 0) >= MARKER_PER_CONTINENT_MIN:
            continue

        markers.append(
            {
                "name": city.name,
                "country_code": city.country_code,
                "flag": city_catalog.flags[idx],
                "score": round(float(city_scores[idx]), 4),
                "lat": city.lat,
                "lon": city.lon,
            }
        )
        continent_counts[continent] = continent_counts.get(continent, 0) + 1
        seen.add(int(idx))

        if len(markers) >= MARKER_GLOBAL_LIMIT + len(continent_counts) * MARKER_PER_CONTINENT_MIN:
            break

    return markers


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

    if hasattr(repository, "get_climate_matrix") and hasattr(repository, "get_indexed_cities"):
        return _build_score_response_from_matrix(repository, preferences, request_started, timings)

    return _build_score_response_from_cells(repository, preferences, request_started, timings)


def _build_score_response_from_matrix(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    request_started: float,
    timings: ScoreTimings,
) -> ScoreResponse:
    cells_started = perf_counter()
    climate_matrix = repository.get_climate_matrix()
    timings.cells_ms = _elapsed_ms(cells_started)

    cities_started = perf_counter()
    indexed_cities = repository.get_indexed_cities()
    timings.cities_ms = _elapsed_ms(cities_started)

    scoring_started = perf_counter()
    raw_scores = score_climate_matrix(climate_matrix, preferences)
    timings.scoring_ms = _elapsed_ms(scoring_started)

    if raw_scores.size == 0:
        timings.total_ms = _elapsed_ms(request_started)
        _log_score_timings(
            timings,
            climate_cell_count=len(climate_matrix.latitudes),
            city_count=len(indexed_cities.cities),
            ranked_city_count=0,
            outcome="empty",
        )
        return {"scores": [], "markers": [], "heatmap": ""}

    max_score = float(raw_scores.max())
    if max_score == 0.0:
        timings.total_ms = _elapsed_ms(request_started)
        _log_score_timings(
            timings,
            climate_cell_count=len(climate_matrix.latitudes),
            city_count=len(indexed_cities.cities),
            ranked_city_count=0,
            outcome="all_zero",
        )
        return {"scores": [], "markers": [], "heatmap": ""}

    normalize_started = perf_counter()
    normalized_scores = normalize_score_array(raw_scores)
    timings.normalize_ms = _elapsed_ms(normalize_started)

    ranking_started = perf_counter()
    top_cities = rank_indexed_city_scores(indexed_cities, normalized_scores, limit=TOP_CITY_RESULTS)
    markers = _select_markers(indexed_cities, normalized_scores)
    timings.ranking_ms = _elapsed_ms(ranking_started)

    heatmap_started = perf_counter()
    # Some injected repositories only support the cached matrix/ranking fast path.
    if hasattr(repository, "get_heatmap_projection"):
        heatmap_png = render_heatmap_png_from_projection(repository.get_heatmap_projection(), normalized_scores)
    else:
        heatmap_png = render_heatmap_png_from_arrays(
            climate_matrix.latitudes,
            climate_matrix.longitudes,
            normalized_scores,
        )
    timings.heatmap_ms = _elapsed_ms(heatmap_started)
    timings.total_ms = _elapsed_ms(request_started)

    _log_score_timings(
        timings,
        climate_cell_count=len(climate_matrix.latitudes),
        city_count=len(indexed_cities.cities),
        ranked_city_count=len(top_cities),
        outcome="ok",
    )

    return {
        "scores": top_cities,
        "markers": markers,
        "heatmap": "data:image/png;base64," + base64.b64encode(heatmap_png).decode(),
    }


def _build_score_response_from_cells(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    request_started: float,
    timings: ScoreTimings,
) -> ScoreResponse:

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
        return {"scores": [], "markers": [], "heatmap": ""}

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
        return {"scores": [], "markers": [], "heatmap": ""}

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
    # Cells path has no CityRankingCache; use top_cities itself as markers (already has lat/lon).
    markers = top_cities
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
        "markers": markers,
        "heatmap": "data:image/png;base64," + base64.b64encode(heatmap_png).decode(),
    }
