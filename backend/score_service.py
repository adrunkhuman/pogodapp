from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from backend.cities import (
    CityCandidate,
    CityRankingCache,
    CityScorePoint,
    IndexedRankingTimings,
    rank_city_scores,
    rank_indexed_city_scores,
)
from backend.config import RANKING_MIN_POPULATION
from backend.heatmap import render_heatmap_png, render_heatmap_png_from_arrays, render_heatmap_png_from_projection
from backend.scoring import (
    CellScorePoint,
    MatrixScoreTimings,
    PreferenceInputs,
    normalize_score_array,
    score_climate_cells,
    score_climate_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from backend.climate_repository import ClimateRepository

INITIAL_CITY_RESULTS = 30
SIDEBAR_CONTINENT_RESERVE = 30
CONTINENT_COUNT = 6
# Keep a larger diversified pool so continent backfill doesn't fall back to clustered raw-score picks.
RANKING_POOL_SIZE = SIDEBAR_CONTINENT_RESERVE * CONTINENT_COUNT * 3
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoreTimings:
    """Per-step timing snapshot for one `/score` calculation."""

    cells_ms: float = 0.0
    cities_ms: float = 0.0
    scoring_ms: float = 0.0
    normalize_ms: float = 0.0
    ranking_ms: float = 0.0
    ranking_filter_ms: float = 0.0
    ranking_candidates_ms: float = 0.0
    ranking_candidates_setup_ms: float = 0.0
    ranking_candidates_winner_select_ms: float = 0.0
    ranking_candidates_distance_ms: float = 0.0
    ranking_candidates_penalty_ms: float = 0.0
    ranking_sidebar_ms: float = 0.0
    ranking_rescore_ms: float = 0.0
    scoring_setup_ms: float = 0.0
    scoring_temperature_ms: float = 0.0
    scoring_rain_ms: float = 0.0
    scoring_sun_ms: float = 0.0
    scoring_combine_ms: float = 0.0
    heatmap_ms: float = 0.0
    response_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class HeatmapTimings:
    """Per-step timing snapshot for one `/heatmap` calculation."""

    scoring_ms: float = 0.0
    normalize_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True, frozen=True)
class ScoreContext:
    """Shared score-request counts reused by both ranking paths."""

    climate_cell_count: int
    city_count: int


@dataclass(slots=True)
class HeatmapField:
    """Reusable normalized score field for one preference tuple."""

    normalized_scores: NDArray[np.float32] | list[CellScorePoint]
    latitudes: NDArray[np.float32] | None = None
    longitudes: NDArray[np.float32] | None = None


class ScoreResponse(TypedDict):
    """Backend response contract for one scored user preference request."""

    scores: list[CityScorePoint]


EMPTY_SCORE_RESPONSE: ScoreResponse = {"scores": []}


def _elapsed_ms(start_time: float) -> float:
    return round((perf_counter() - start_time) * 1000, 2)


def _empty_score_response(
    timings: ScoreTimings,
    *,
    preferences: PreferenceInputs,
    request_started: float,
    context: ScoreContext,
    outcome: str,
) -> ScoreResponse:
    timings.total_ms = _elapsed_ms(request_started)
    _log_score_timings(
        timings,
        preferences=preferences,
        climate_cell_count=context.climate_cell_count,
        city_count=context.city_count,
        ranked_city_count=0,
        outcome=outcome,
    )
    return EMPTY_SCORE_RESPONSE


def _finalize_score_response(
    timings: ScoreTimings,
    *,
    preferences: PreferenceInputs,
    request_started: float,
    context: ScoreContext,
    ranked_cities: list[CityScorePoint],
) -> ScoreResponse:
    timings.total_ms = _elapsed_ms(request_started)
    _log_score_timings(
        timings,
        preferences=preferences,
        climate_cell_count=context.climate_cell_count,
        city_count=context.city_count,
        ranked_city_count=len(ranked_cities),
        outcome="ok",
    )
    return {"scores": ranked_cities}


def _log_heatmap_timings(  # noqa: PLR0913
    timings: HeatmapTimings,
    *,
    preferences: PreferenceInputs,
    climate_cell_count: int,
    png_bytes: int,
    field_cache_hit: bool,
    outcome: str,
) -> None:
    logger.info(
        "heatmap request finished",
        extra={
            "event": "heatmap_request",
            "outcome": outcome,
            "score_field_cache_hit": field_cache_hit,
            "total_ms": timings.total_ms,
            "scoring_ms": timings.scoring_ms,
            "normalize_ms": timings.normalize_ms,
            "render_ms": timings.render_ms,
            "climate_cells": climate_cell_count,
            "png_bytes": png_bytes,
            "preferred_day_temperature": preferences.preferred_day_temperature,
            "summer_heat_limit": preferences.summer_heat_limit,
            "winter_cold_limit": preferences.winter_cold_limit,
            "dryness_preference": preferences.dryness_preference,
            "sunshine_preference": preferences.sunshine_preference,
        },
    )


def _filter_ranking_catalog(city_catalog: CityRankingCache) -> CityRankingCache:
    """Return a catalog limited to cities with meaningful population for the sidebar list.

    Population 0 means the DB predates the population column — let those cities through
    so old databases still produce a ranked list.
    """
    if RANKING_MIN_POPULATION == 0 or not city_catalog.cities:
        return city_catalog

    mask = [city.population >= RANKING_MIN_POPULATION or city.population == 0 for city in city_catalog.cities]
    if all(mask):
        return city_catalog

    filtered_cities = tuple(city for city, keep in zip(city_catalog.cities, mask, strict=True) if keep)
    filtered_indexes = city_catalog.climate_indexes[np.array(mask)]
    return CityRankingCache.from_cities(filtered_cities, filtered_indexes)


def _filter_city_candidates(city_catalog: tuple[CityCandidate, ...]) -> tuple[CityCandidate, ...]:
    """Return sidebar-eligible cities for the fallback ranking path."""
    if RANKING_MIN_POPULATION == 0 or not city_catalog:
        return city_catalog

    return tuple(city for city in city_catalog if city.population >= RANKING_MIN_POPULATION or city.population == 0)


def _ensure_continent_coverage(
    ranked: list[CityScorePoint],
    fill_pool: list[CityScorePoint],
    min_per_continent: int = 3,
) -> list[CityScorePoint]:
    """Guarantee at least min_per_continent entries per continent in the sidebar list.

    fill_pool must already be diversity-suppressed and score-ordered so that
    fill cities are geographically spread — not raw-score clusters.
    """
    continent_counts: dict[str, int] = {}
    present: set[tuple[str, str, float, float]] = set()
    for entry in ranked:
        present.add(_city_identity(entry))
        cont = entry["continent"]
        if cont != "Other":
            continent_counts[cont] = continent_counts.get(cont, 0) + 1

    all_continents = {city["continent"] for city in fill_pool} - {"Other"}
    needs_fill = {c for c in all_continents if continent_counts.get(c, 0) < min_per_continent}

    if not needs_fill:
        return ranked

    for city in fill_pool:
        cont = city["continent"]
        if cont not in needs_fill or _city_identity(city) in present:
            continue
        ranked.append(city)
        present.add(_city_identity(city))
        continent_counts[cont] = continent_counts.get(cont, 0) + 1
        if continent_counts[cont] >= min_per_continent:
            needs_fill.discard(cont)
        if not needs_fill:
            break

    return ranked


def _build_sidebar_scores(ranked_pool: list[CityScorePoint]) -> list[CityScorePoint]:
    """Keep a deeper reserve per continent so the UI can progressively reveal cities."""
    continent_counts: dict[str, int] = {}
    sidebar_scores: list[CityScorePoint] = []

    for city in ranked_pool:
        continent = city["continent"]
        if continent == "Other":
            continue
        if continent_counts.get(continent, 0) >= SIDEBAR_CONTINENT_RESERVE:
            continue
        sidebar_scores.append(city)
        continent_counts[continent] = continent_counts.get(continent, 0) + 1

    return sidebar_scores


def _build_ranked_sidebar_scores(ranked_pool: list[CityScorePoint]) -> list[CityScorePoint]:
    """Apply continent backfill and reserve trimming to one ranked candidate pool."""
    initial_cities = list(ranked_pool[:INITIAL_CITY_RESULTS])
    initial_cities = _ensure_continent_coverage(initial_cities, ranked_pool[INITIAL_CITY_RESULTS:])
    return _build_sidebar_scores(initial_cities + ranked_pool[INITIAL_CITY_RESULTS:])


def _city_identity(city: CityScorePoint) -> tuple[str, str, float, float]:
    return (city["name"], city["country_code"], city["lat"], city["lon"])


def _deduplicate_city_points(cities: list[CityScorePoint]) -> list[CityScorePoint]:
    seen: set[tuple[str, str, float, float]] = set()
    deduplicated: list[CityScorePoint] = []

    for city in cities:
        identity = _city_identity(city)
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(city)

    return deduplicated


def _with_city_score(city: CityScorePoint, score: float) -> CityScorePoint:
    return {
        "name": city["name"],
        "continent": city["continent"],
        "country_code": city["country_code"],
        "flag": city["flag"],
        "score": score,
        "lat": city["lat"],
        "lon": city["lon"],
        "probe_lat": city["probe_lat"],
        "probe_lon": city["probe_lon"],
    }


def _rescore_city_points_from_cache(
    city_catalog: CityRankingCache,
    ranked_cities: list[CityScorePoint],
    raw_scores: NDArray[np.float32],
) -> list[CityScorePoint]:
    """Replace shortlist scores with each city's own nearest-cell score and re-sort by it."""
    score_by_city = {
        (city.name, city.country_code, city.lat, city.lon): round(float(raw_scores[climate_index]), 4)
        for city, climate_index in zip(city_catalog.cities, city_catalog.climate_indexes, strict=True)
    }
    rescored = [
        _with_city_score(city, score_by_city.get(_city_identity(city), city["score"])) for city in ranked_cities
    ]
    return _deduplicate_city_points(sorted(rescored, key=lambda city: city["score"], reverse=True))


def _rescore_city_points_from_cells(
    city_catalog: tuple[CityCandidate, ...],
    ranked_cities: list[CityScorePoint],
    raw_scores: list[CellScorePoint],
) -> list[CityScorePoint]:
    """Replace shortlist scores with each city's snapped-cell score and re-sort by it."""
    score_by_cell = {
        (round(score_point["lat"], 4), round(score_point["lon"], 4)): round(score_point["score"], 4)
        for score_point in raw_scores
    }
    score_by_city = {
        (city.name, city.country_code, city.lat, city.lon): score_by_cell.get(
            (round(city.cell_lat, 4), round(city.cell_lon, 4)), 0.0
        )
        for city in city_catalog
    }
    rescored = [
        _with_city_score(city, score_by_city.get(_city_identity(city), city["score"])) for city in ranked_cities
    ]
    return _deduplicate_city_points(sorted(rescored, key=lambda city: city["score"], reverse=True))


def _log_score_timings(  # noqa: PLR0913
    timings: ScoreTimings,
    *,
    preferences: PreferenceInputs,
    climate_cell_count: int,
    city_count: int,
    ranked_city_count: int,
    outcome: str,
) -> None:
    logger.info(
        "score request finished",
        extra={
            "event": "score_request",
            "outcome": outcome,
            "total_ms": timings.total_ms,
            "cells_ms": timings.cells_ms,
            "cities_ms": timings.cities_ms,
            "scoring_ms": timings.scoring_ms,
            "scoring_setup_ms": timings.scoring_setup_ms,
            "scoring_temperature_ms": timings.scoring_temperature_ms,
            "scoring_rain_ms": timings.scoring_rain_ms,
            "scoring_sun_ms": timings.scoring_sun_ms,
            "scoring_combine_ms": timings.scoring_combine_ms,
            "normalize_ms": timings.normalize_ms,
            "ranking_ms": timings.ranking_ms,
            "ranking_filter_ms": timings.ranking_filter_ms,
            "ranking_candidates_ms": timings.ranking_candidates_ms,
            "ranking_candidates_setup_ms": timings.ranking_candidates_setup_ms,
            "ranking_candidates_winner_select_ms": timings.ranking_candidates_winner_select_ms,
            "ranking_candidates_distance_ms": timings.ranking_candidates_distance_ms,
            "ranking_candidates_penalty_ms": timings.ranking_candidates_penalty_ms,
            "ranking_sidebar_ms": timings.ranking_sidebar_ms,
            "ranking_rescore_ms": timings.ranking_rescore_ms,
            "heatmap_ms": timings.heatmap_ms,
            "response_ms": timings.response_ms,
            "climate_cells": climate_cell_count,
            "cities": city_count,
            "ranked_cities": ranked_city_count,
            "preferred_day_temperature": preferences.preferred_day_temperature,
            "summer_heat_limit": preferences.summer_heat_limit,
            "winter_cold_limit": preferences.winter_cold_limit,
            "dryness_preference": preferences.dryness_preference,
            "sunshine_preference": preferences.sunshine_preference,
        },
    )


def build_score_response(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    *,
    store_heatmap_field: Callable[[HeatmapField], None] | None = None,
) -> ScoreResponse:
    """Score all available climate cells and shape the `/score` API payload.

    The route keeps FastAPI concerns only; this function owns the application-level
    scoring flow so ranking, normalization, and heatmap rendering stay testable
    outside HTTP wiring.
    """
    request_started = perf_counter()
    timings = ScoreTimings()

    if hasattr(repository, "get_climate_matrix") and hasattr(repository, "get_indexed_cities"):
        return _build_score_response_from_matrix(
            repository,
            preferences,
            request_started,
            timings,
            store_heatmap_field=store_heatmap_field,
        )

    return _build_score_response_from_cells(
        repository,
        preferences,
        request_started,
        timings,
        store_heatmap_field=store_heatmap_field,
    )


def _build_score_response_from_matrix(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    request_started: float,
    timings: ScoreTimings,
    *,
    store_heatmap_field: Callable[[HeatmapField], None] | None,
) -> ScoreResponse:
    cells_started = perf_counter()
    climate_matrix = repository.get_climate_matrix()
    timings.cells_ms = _elapsed_ms(cells_started)

    cities_started = perf_counter()
    indexed_cities = repository.get_indexed_cities()
    timings.cities_ms = _elapsed_ms(cities_started)
    context = ScoreContext(climate_cell_count=len(climate_matrix.latitudes), city_count=len(indexed_cities.cities))

    scoring_started = perf_counter()
    scoring_breakdown = MatrixScoreTimings()
    raw_scores = score_climate_matrix(climate_matrix, preferences, timings=scoring_breakdown)
    timings.scoring_ms = _elapsed_ms(scoring_started)
    timings.scoring_setup_ms = scoring_breakdown.setup_ms
    timings.scoring_temperature_ms = scoring_breakdown.temperature_ms
    timings.scoring_rain_ms = scoring_breakdown.rain_ms
    timings.scoring_sun_ms = scoring_breakdown.sun_ms
    timings.scoring_combine_ms = scoring_breakdown.combine_ms

    if raw_scores.size == 0:
        return _empty_score_response(
            timings,
            preferences=preferences,
            request_started=request_started,
            context=context,
            outcome="empty",
        )

    max_score = float(raw_scores.max())
    if max_score == 0.0:
        return _empty_score_response(
            timings,
            preferences=preferences,
            request_started=request_started,
            context=context,
            outcome="all_zero",
        )

    normalize_started = perf_counter()
    normalized_scores = normalize_score_array(raw_scores)
    timings.normalize_ms = _elapsed_ms(normalize_started)
    if store_heatmap_field is not None:
        store_heatmap_field(
            HeatmapField(
                normalized_scores=normalized_scores,
                latitudes=climate_matrix.latitudes,
                longitudes=climate_matrix.longitudes,
            )
        )

    ranking_started = perf_counter()
    ranking_filter_started = perf_counter()
    ranking_catalog = _filter_ranking_catalog(indexed_cities)
    timings.ranking_filter_ms = _elapsed_ms(ranking_filter_started)
    # Build a large diversity-suppressed pool so the continent fill draws from
    # already-spread candidates rather than raw score clusters.
    ranking_candidates_started = perf_counter()
    ranking_candidates_breakdown = IndexedRankingTimings()
    diverse_pool = rank_indexed_city_scores(
        ranking_catalog,
        normalized_scores,
        limit=RANKING_POOL_SIZE,
        timings=ranking_candidates_breakdown,
    )
    timings.ranking_candidates_ms = _elapsed_ms(ranking_candidates_started)
    timings.ranking_candidates_setup_ms = ranking_candidates_breakdown.setup_ms
    timings.ranking_candidates_winner_select_ms = ranking_candidates_breakdown.winner_select_ms
    timings.ranking_candidates_distance_ms = ranking_candidates_breakdown.distance_ms
    timings.ranking_candidates_penalty_ms = ranking_candidates_breakdown.penalty_ms
    ranking_sidebar_started = perf_counter()
    top_cities = _build_ranked_sidebar_scores(diverse_pool)
    timings.ranking_sidebar_ms = _elapsed_ms(ranking_sidebar_started)
    ranking_rescore_started = perf_counter()
    top_cities = _rescore_city_points_from_cache(ranking_catalog, top_cities, raw_scores)
    timings.ranking_rescore_ms = _elapsed_ms(ranking_rescore_started)
    timings.ranking_ms = _elapsed_ms(ranking_started)

    return _finalize_score_response(
        timings,
        preferences=preferences,
        request_started=request_started,
        context=context,
        ranked_cities=top_cities,
    )


def _build_score_response_from_cells(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    request_started: float,
    timings: ScoreTimings,
    *,
    store_heatmap_field: Callable[[HeatmapField], None] | None,
) -> ScoreResponse:

    cells_started = perf_counter()
    climate_cells = repository.list_cells()
    timings.cells_ms = _elapsed_ms(cells_started)

    cities_started = perf_counter()
    cities = repository.list_cities()
    timings.cities_ms = _elapsed_ms(cities_started)
    context = ScoreContext(climate_cell_count=len(climate_cells), city_count=len(cities))

    scoring_started = perf_counter()
    raw_scores = score_climate_cells(climate_cells, preferences)
    timings.scoring_ms = _elapsed_ms(scoring_started)

    if not raw_scores:
        return _empty_score_response(
            timings,
            preferences=preferences,
            request_started=request_started,
            context=context,
            outcome="empty",
        )

    max_score = max(point["score"] for point in raw_scores)
    if max_score == 0:
        # An all-zero result carries no useful ranking or map signal for the UI.
        return _empty_score_response(
            timings,
            preferences=preferences,
            request_started=request_started,
            context=context,
            outcome="all_zero",
        )

    # Re-normalize each response so the best match in the current result set lands
    # at 1.0, which keeps the heatmap visually informative across very different queries.
    normalize_started = perf_counter()
    normalized_scores: list[CellScorePoint] = [
        {"lat": point["lat"], "lon": point["lon"], "score": round(point["score"] / max_score, 4)}
        for point in raw_scores
    ]
    timings.normalize_ms = _elapsed_ms(normalize_started)
    if store_heatmap_field is not None:
        store_heatmap_field(HeatmapField(normalized_scores=normalized_scores))

    ranking_started = perf_counter()
    ranking_filter_started = perf_counter()
    ranking_catalog = _filter_city_candidates(cities)
    timings.ranking_filter_ms = _elapsed_ms(ranking_filter_started)
    ranking_candidates_started = perf_counter()
    diverse_pool = rank_city_scores(ranking_catalog, normalized_scores, limit=RANKING_POOL_SIZE)
    timings.ranking_candidates_ms = _elapsed_ms(ranking_candidates_started)
    ranking_sidebar_started = perf_counter()
    top_cities = _build_ranked_sidebar_scores(diverse_pool)
    timings.ranking_sidebar_ms = _elapsed_ms(ranking_sidebar_started)
    ranking_rescore_started = perf_counter()
    top_cities = _rescore_city_points_from_cells(ranking_catalog, top_cities, raw_scores)
    timings.ranking_rescore_ms = _elapsed_ms(ranking_rescore_started)
    timings.ranking_ms = _elapsed_ms(ranking_started)

    return _finalize_score_response(
        timings,
        preferences=preferences,
        request_started=request_started,
        context=context,
        ranked_cities=top_cities,
    )


def build_heatmap_response(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    *,
    cached_heatmap_field: HeatmapField | None = None,
) -> bytes:
    """Build the `/heatmap` PNG payload for one user preference request."""
    request_started = perf_counter()
    timings = HeatmapTimings()
    if cached_heatmap_field is not None:
        render_started = perf_counter()
        heatmap_png = _render_heatmap_from_field(repository, cached_heatmap_field)
        timings.render_ms = _elapsed_ms(render_started)
        timings.total_ms = _elapsed_ms(request_started)
        _log_heatmap_timings(
            timings,
            preferences=preferences,
            climate_cell_count=_heatmap_field_cell_count(cached_heatmap_field),
            png_bytes=len(heatmap_png),
            field_cache_hit=True,
            outcome="ok" if heatmap_png else "empty",
        )
        return heatmap_png

    if hasattr(repository, "get_climate_matrix"):
        return _build_heatmap_response_from_matrix(repository, preferences, request_started, timings)

    return _build_heatmap_response_from_cells(repository, preferences, request_started, timings)


def _build_heatmap_response_from_matrix(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    request_started: float,
    timings: HeatmapTimings,
) -> bytes:
    climate_matrix = repository.get_climate_matrix()
    scoring_started = perf_counter()
    raw_scores = score_climate_matrix(climate_matrix, preferences)
    timings.scoring_ms = _elapsed_ms(scoring_started)

    if raw_scores.size == 0:
        timings.total_ms = _elapsed_ms(request_started)
        _log_heatmap_timings(
            timings,
            preferences=preferences,
            climate_cell_count=len(climate_matrix.latitudes),
            png_bytes=0,
            field_cache_hit=False,
            outcome="empty",
        )
        return b""

    max_score = float(raw_scores.max())
    if max_score == 0.0:
        timings.total_ms = _elapsed_ms(request_started)
        _log_heatmap_timings(
            timings,
            preferences=preferences,
            climate_cell_count=len(climate_matrix.latitudes),
            png_bytes=0,
            field_cache_hit=False,
            outcome="all_zero",
        )
        return b""

    normalize_started = perf_counter()
    normalized_scores = normalize_score_array(raw_scores)
    timings.normalize_ms = _elapsed_ms(normalize_started)

    # Some injected repositories only support the cached matrix/ranking fast path.
    render_started = perf_counter()
    if hasattr(repository, "get_heatmap_projection"):
        heatmap_png = render_heatmap_png_from_projection(repository.get_heatmap_projection(), normalized_scores)
    else:
        heatmap_png = render_heatmap_png_from_arrays(
            climate_matrix.latitudes,
            climate_matrix.longitudes,
            normalized_scores,
        )
    timings.render_ms = _elapsed_ms(render_started)
    timings.total_ms = _elapsed_ms(request_started)
    _log_heatmap_timings(
        timings,
        preferences=preferences,
        climate_cell_count=len(climate_matrix.latitudes),
        png_bytes=len(heatmap_png),
        field_cache_hit=False,
        outcome="ok",
    )
    return heatmap_png


def _build_heatmap_response_from_cells(
    repository: ClimateRepository,
    preferences: PreferenceInputs,
    request_started: float,
    timings: HeatmapTimings,
) -> bytes:
    scoring_started = perf_counter()
    raw_scores = score_climate_cells(repository.list_cells(), preferences)
    timings.scoring_ms = _elapsed_ms(scoring_started)

    if not raw_scores:
        timings.total_ms = _elapsed_ms(request_started)
        _log_heatmap_timings(
            timings,
            preferences=preferences,
            climate_cell_count=0,
            png_bytes=0,
            field_cache_hit=False,
            outcome="empty",
        )
        return b""

    max_score = max(point["score"] for point in raw_scores)
    if max_score == 0:
        timings.total_ms = _elapsed_ms(request_started)
        _log_heatmap_timings(
            timings,
            preferences=preferences,
            climate_cell_count=len(raw_scores),
            png_bytes=0,
            field_cache_hit=False,
            outcome="all_zero",
        )
        return b""

    normalize_started = perf_counter()
    normalized_scores: list[CellScorePoint] = [
        {"lat": point["lat"], "lon": point["lon"], "score": round(point["score"] / max_score, 4)}
        for point in raw_scores
    ]
    timings.normalize_ms = _elapsed_ms(normalize_started)
    render_started = perf_counter()
    heatmap_png = render_heatmap_png(normalized_scores)
    timings.render_ms = _elapsed_ms(render_started)
    timings.total_ms = _elapsed_ms(request_started)
    _log_heatmap_timings(
        timings,
        preferences=preferences,
        climate_cell_count=len(raw_scores),
        png_bytes=len(heatmap_png),
        field_cache_hit=False,
        outcome="ok",
    )
    return heatmap_png


def _render_heatmap_from_field(repository: ClimateRepository, heatmap_field: HeatmapField) -> bytes:
    if heatmap_field.latitudes is not None and heatmap_field.longitudes is not None:
        if hasattr(repository, "get_heatmap_projection"):
            return render_heatmap_png_from_projection(
                repository.get_heatmap_projection(),
                cast("NDArray[np.float32]", heatmap_field.normalized_scores),
            )
        return render_heatmap_png_from_arrays(
            heatmap_field.latitudes,
            heatmap_field.longitudes,
            cast("NDArray[np.float32]", heatmap_field.normalized_scores),
        )

    return render_heatmap_png(cast("list[CellScorePoint]", heatmap_field.normalized_scores))


def _heatmap_field_cell_count(heatmap_field: HeatmapField) -> int:
    if heatmap_field.latitudes is not None:
        return len(heatmap_field.latitudes)
    return len(cast("list[CellScorePoint]", heatmap_field.normalized_scores))
