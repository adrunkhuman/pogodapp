from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from backend.cities import (
    STUB_CITY_CANDIDATES,
    CityCandidate,
    CityRankingCache,
    apply_regional_penalty,
    continent_of,
    country_flag,
    haversine_distance_km,
    rank_city_scores,
    rank_indexed_city_scores,
    snap_city_to_cell_key,
)
from backend.config import CITY_DIVERSITY_DECAY_KM

if TYPE_CHECKING:
    from backend.scoring import CellScorePoint


def test_stub_city_candidates_match_stub_climate_fixture_order() -> None:
    assert [city.name for city in STUB_CITY_CANDIDATES] == [
        "San Francisco",
        "Rome",
        "Cape Town",
        "Tokyo",
        "Rio de Janeiro",
    ]


def test_snap_city_to_cell_key_matches_native_worldclim_centers() -> None:
    snapped = snap_city_to_cell_key(
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=0.0, cell_lon=0.0)
    )

    assert snapped == (4.7083, -74.0417)


def test_snap_city_to_cell_key_supports_finer_worldclim_grids() -> None:
    snapped = snap_city_to_cell_key(
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=0.0, cell_lon=0.0),
        grid_degrees=2.5 / 60,
    )

    assert snapped == (4.7292, -74.0625)


def test_rank_city_scores_prefers_the_highest_scoring_nearby_city_cells() -> None:
    cities = (
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=4.75, cell_lon=-74.0833),
        CityCandidate(name="Medellin", country_code="CO", lat=6.2442, lon=-75.5812, cell_lat=6.25, cell_lon=-75.5833),
        CityCandidate(name="Quito", country_code="EC", lat=-0.1807, lon=-78.4678, cell_lat=-0.25, cell_lon=-78.4167),
    )
    scores: list[CellScorePoint] = [
        {"lat": 4.75, "lon": -74.0833, "score": 1.0},
        {"lat": 6.25, "lon": -75.5833, "score": 0.82},
        {"lat": -0.25, "lon": -78.4167, "score": 0.91},
    ]

    ranked = rank_city_scores(cities, scores, limit=2, diversity_decay_km=200.0)

    assert [city["name"] for city in ranked] == ["Bogota", "Quito"]
    assert ranked[0]["score"] == 1.0
    assert 0.82 < ranked[1]["score"] < 0.91


def test_rank_city_scores_penalizes_nearby_duplicates_after_strong_regional_center() -> None:
    cities = (
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=4.75, cell_lon=-74.0833),
        CityCandidate(name="Medellin", country_code="CO", lat=6.2442, lon=-75.5812, cell_lat=6.25, cell_lon=-75.5833),
        CityCandidate(name="Lima", country_code="PE", lat=-12.0464, lon=-77.0428, cell_lat=-12.0833, cell_lon=-77.0833),
    )
    scores: list[CellScorePoint] = [
        {"lat": 4.75, "lon": -74.0833, "score": 1.0},
        {"lat": 6.25, "lon": -75.5833, "score": 0.97},
        {"lat": -12.0833, "lon": -77.0833, "score": 0.88},
    ]

    ranked = rank_city_scores(cities, scores, limit=2, diversity_decay_km=200.0)

    assert [city["name"] for city in ranked] == ["Bogota", "Lima"]
    assert ranked[0]["score"] == 1.0
    assert 0.87 < ranked[1]["score"] < 0.88


def test_apply_regional_penalty_drops_same_place_to_zero() -> None:
    bogota = CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=4.75, cell_lon=-74.0833)

    assert apply_regional_penalty(0.9, bogota, bogota, 1.0) == 0.0


def test_rank_city_scores_uses_configured_decay_radius() -> None:
    cities = (
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=4.75, cell_lon=-74.0833),
        CityCandidate(name="Medellin", country_code="CO", lat=6.2442, lon=-75.5812, cell_lat=6.25, cell_lon=-75.5833),
    )
    scores: list[CellScorePoint] = [
        {"lat": 4.75, "lon": -74.0833, "score": 1.0},
        {"lat": 6.25, "lon": -75.5833, "score": 0.97},
    ]

    ranked = rank_city_scores(cities, scores, limit=2)

    assert ranked[0]["name"] == "Bogota"
    assert ranked[1]["name"] == "Medellin"
    assert ranked[1]["score"] == round(
        apply_regional_penalty(0.97, cities[0], cities[1], 1.0, decay_km=CITY_DIVERSITY_DECAY_KM),
        4,
    )


def test_rank_indexed_city_scores_matches_classic_ranking_behavior() -> None:
    cities = (
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=4.75, cell_lon=-74.0833),
        CityCandidate(name="Medellin", country_code="CO", lat=6.2442, lon=-75.5812, cell_lat=6.25, cell_lon=-75.5833),
        CityCandidate(name="Quito", country_code="EC", lat=-0.1807, lon=-78.4678, cell_lat=-0.25, cell_lon=-78.4167),
    )
    classic_scores: list[CellScorePoint] = [
        {"lat": 4.75, "lon": -74.0833, "score": 1.0},
        {"lat": 6.25, "lon": -75.5833, "score": 0.82},
        {"lat": -0.25, "lon": -78.4167, "score": 0.91},
    ]
    indexed_catalog = CityRankingCache(
        cities=cities,
        climate_indexes=np.array([0, 1, 2], dtype=np.int32),
        latitude_radians=np.radians(np.array([city.lat for city in cities], dtype=np.float32)),
        longitude_radians=np.radians(np.array([city.lon for city in cities], dtype=np.float32)),
        cosine_latitudes=np.cos(np.radians(np.array([city.lat for city in cities], dtype=np.float32))).astype(
            np.float32,
            copy=False,
        ),
        flags=tuple(country_flag(city.country_code) for city in cities),
    )
    indexed_scores = np.array([1.0, 0.82, 0.91], dtype=np.float32)

    classic_ranked = rank_city_scores(cities, classic_scores, limit=2)
    indexed_ranked = rank_indexed_city_scores(indexed_catalog, indexed_scores, limit=2)

    assert indexed_ranked == classic_ranked


def test_rank_city_scores_prefers_larger_population_center_when_scores_are_nearly_tied() -> None:
    cities = (
        CityCandidate(
            name="Tiny Village",
            country_code="CO",
            lat=4.711,
            lon=-74.0721,
            cell_lat=4.75,
            cell_lon=-74.0833,
            population=2_000,
        ),
        CityCandidate(
            name="Bogota",
            country_code="CO",
            lat=4.8,
            lon=-74.1,
            cell_lat=4.8333,
            cell_lon=-74.125,
            population=8_000_000,
        ),
    )
    scores: list[CellScorePoint] = [
        {"lat": 4.75, "lon": -74.0833, "score": 0.9},
        {"lat": 4.8333, "lon": -74.125, "score": 0.892},
    ]

    ranked = rank_city_scores(cities, scores, limit=1, diversity_decay_km=200.0)

    assert ranked[0]["name"] == "Bogota"


def test_rank_indexed_city_scores_prefers_larger_population_center_when_scores_are_nearly_tied() -> None:
    cities = (
        CityCandidate(
            name="Tiny Village",
            country_code="CO",
            lat=4.711,
            lon=-74.0721,
            cell_lat=4.75,
            cell_lon=-74.0833,
            population=2_000,
        ),
        CityCandidate(
            name="Bogota",
            country_code="CO",
            lat=4.8,
            lon=-74.1,
            cell_lat=4.8333,
            cell_lon=-74.125,
            population=8_000_000,
        ),
    )
    indexed_catalog = CityRankingCache(
        cities=cities,
        climate_indexes=np.array([0, 1], dtype=np.int32),
        latitude_radians=np.radians(np.array([city.lat for city in cities], dtype=np.float32)),
        longitude_radians=np.radians(np.array([city.lon for city in cities], dtype=np.float32)),
        cosine_latitudes=np.cos(np.radians(np.array([city.lat for city in cities], dtype=np.float32))).astype(
            np.float32,
            copy=False,
        ),
        flags=tuple(country_flag(city.country_code) for city in cities),
    )

    ranked = rank_indexed_city_scores(indexed_catalog, np.array([0.9, 0.892], dtype=np.float32), limit=1)

    assert ranked[0]["name"] == "Bogota"


def test_continent_of_treats_western_russia_as_europe() -> None:
    assert continent_of("RU", 37.6173) == "Europe"


def test_continent_of_treats_siberia_as_asia() -> None:
    assert continent_of("RU", 82.9204) == "Asia"


def test_haversine_distance_km_is_small_for_nearby_cities() -> None:
    distance = haversine_distance_km(4.711, -74.0721, 6.2442, -75.5812)

    assert 200 < distance < 260


def test_country_flag_uses_iso_country_code() -> None:
    assert country_flag("jp") == "🇯🇵"
