from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from backend.cities import CityCandidate, CityRankingCache, CityScorePoint
from backend.climate_repository import StubClimateRepository
from backend.logging_config import configure_backend_logging
from backend.score_service import _deduplicate_city_points, build_score_response
from backend.scoring import ClimateCell, ClimateMatrix, PreferenceInputs

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from backend.climate_repository import ClimateRepository


def test_build_score_response_logs_step_timings(caplog: LogCaptureFixture) -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=7,
        heat_tolerance=5,
        rain_sensitivity=55,
        sun_preference=60,
    )

    configure_backend_logging()
    backend_logger = logging.getLogger("backend")
    original_handlers = backend_logger.handlers[:]
    original_propagate = backend_logger.propagate

    backend_logger.handlers = []
    backend_logger.propagate = True

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = build_score_response(StubClimateRepository(), preferences)
    finally:
        backend_logger.handlers = original_handlers
        backend_logger.propagate = original_propagate

    assert response["scores"]
    assert "markers" not in response
    assert response["heatmap"].startswith("data:image/png;base64,")
    assert "score_request outcome=ok" in caplog.text
    assert "total_ms=" in caplog.text
    assert "cells_ms=" in caplog.text
    assert "cities_ms=" in caplog.text
    assert "scoring_ms=" in caplog.text
    assert "normalize_ms=" in caplog.text
    assert "ranking_ms=" in caplog.text
    assert "heatmap_ms=" in caplog.text


def test_build_score_response_returns_empty_payload_for_empty_matrix() -> None:
    class EmptyMatrixRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix(
                latitudes=np.array([], dtype=np.float32),
                longitudes=np.array([], dtype=np.float32),
                temperature_c=np.empty((0, 12), dtype=np.float32),
                precipitation_mm=np.empty((0, 12), dtype=np.float32),
                cloud_cover_pct=np.empty((0, 12), dtype=np.uint8),
            )

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities((), np.array([], dtype=np.int32))

    response = build_score_response(
        cast("ClimateRepository", EmptyMatrixRepository()),
        PreferenceInputs(
            ideal_temperature=22, cold_tolerance=7, heat_tolerance=5, rain_sensitivity=55, sun_preference=60
        ),
    )

    assert response == {"scores": [], "heatmap": ""}


def test_build_score_response_returns_empty_payload_for_all_zero_matrix_scores(monkeypatch: MonkeyPatch) -> None:
    class SingleCellRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells(
                (
                    ClimateCell(
                        lat=1.0,
                        lon=2.0,
                        temperature_c=(22.0,) * 12,
                        precipitation_mm=(0.0,) * 12,
                        cloud_cover_pct=(15,) * 12,
                    ),
                )
            )

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities((), np.array([], dtype=np.int32))

    monkeypatch.setattr("backend.score_service.score_climate_matrix", lambda *_args: np.array([0.0], dtype=np.float32))

    response = build_score_response(
        cast("ClimateRepository", SingleCellRepository()),
        PreferenceInputs(
            ideal_temperature=22, cold_tolerance=7, heat_tolerance=5, rain_sensitivity=55, sun_preference=60
        ),
    )

    assert response == {"scores": [], "heatmap": ""}


def test_build_score_response_falls_back_to_array_heatmap_path_when_projection_cache_is_absent() -> None:
    class MatrixOnlyRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells(
                (
                    ClimateCell(
                        lat=1.0,
                        lon=2.0,
                        temperature_c=(22.0,) * 12,
                        precipitation_mm=(0.0,) * 12,
                        cloud_cover_pct=(15,) * 12,
                    ),
                )
            )

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities((), np.array([], dtype=np.int32))

    response = build_score_response(
        cast("ClimateRepository", MatrixOnlyRepository()),
        PreferenceInputs(
            ideal_temperature=22, cold_tolerance=7, heat_tolerance=5, rain_sensitivity=55, sun_preference=60
        ),
    )

    assert response["scores"] == []
    assert response["heatmap"].startswith("data:image/png;base64,")


def test_deduplicate_city_points_removes_duplicate_substituted_cities() -> None:
    cities: list[CityScorePoint] = [
        {
            "name": "Bogota",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 0.91,
            "lat": 4.711,
            "lon": -74.0721,
            "probe_lat": 4.7083,
            "probe_lon": -74.0417,
        },
        {
            "name": "Bogota",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 0.9,
            "lat": 4.711,
            "lon": -74.0721,
            "probe_lat": 4.7083,
            "probe_lon": -74.0417,
        },
        {
            "name": "Medellin",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 0.89,
            "lat": 6.2442,
            "lon": -75.5812,
            "probe_lat": 6.25,
            "probe_lon": -75.5833,
        },
    ]
    deduplicated = _deduplicate_city_points(cities)

    assert deduplicated == [
        cities[0],
        cities[2],
    ]


def test_build_score_response_includes_backend_continent_labels() -> None:
    response = build_score_response(
        StubClimateRepository(),
        PreferenceInputs(
            ideal_temperature=22,
            cold_tolerance=7,
            heat_tolerance=5,
            rain_sensitivity=55,
            sun_preference=60,
        ),
    )

    assert response["scores"]
    assert all(city["continent"] != "Other" for city in response["scores"])


def test_build_score_response_keeps_fallback_ranking_behavior_aligned_with_matrix_path() -> None:
    climate_cells = (
        ClimateCell(
            lat=1.0,
            lon=2.0,
            temperature_c=(22.0,) * 12,
            precipitation_mm=(0.0,) * 12,
            cloud_cover_pct=(15,) * 12,
        ),
        ClimateCell(
            lat=10.0,
            lon=11.0,
            temperature_c=(22.0,) * 12,
            precipitation_mm=(0.0,) * 12,
            cloud_cover_pct=(15,) * 12,
        ),
    )
    cities = (
        CityCandidate(
            name="Capital", country_code="CO", lat=1.0, lon=2.0, cell_lat=1.0, cell_lon=2.0, population=100_000
        ),
        CityCandidate(
            name="Small Town", country_code="CO", lat=10.0, lon=11.0, cell_lat=10.0, cell_lon=11.0, population=5_000
        ),
    )
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=7,
        heat_tolerance=5,
        rain_sensitivity=55,
        sun_preference=60,
    )

    class CellsRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return climate_cells

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return cities

    class MatrixRepository(CellsRepository):
        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells(climate_cells)

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities(cities, np.array([0, 1], dtype=np.int32))

    cells_response = build_score_response(cast("ClimateRepository", CellsRepository()), preferences)
    matrix_response = build_score_response(cast("ClimateRepository", MatrixRepository()), preferences)

    assert cells_response["scores"] == matrix_response["scores"]


def test_build_score_response_filters_low_population_but_keeps_legacy_zero_population(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("backend.score_service.RANKING_MIN_POPULATION", 30_000)
    climate_cells = (
        ClimateCell(
            lat=1.0, lon=2.0, temperature_c=(22.0,) * 12, precipitation_mm=(0.0,) * 12, cloud_cover_pct=(15,) * 12
        ),
        ClimateCell(
            lat=3.0, lon=4.0, temperature_c=(22.0,) * 12, precipitation_mm=(0.0,) * 12, cloud_cover_pct=(15,) * 12
        ),
        ClimateCell(
            lat=5.0, lon=6.0, temperature_c=(22.0,) * 12, precipitation_mm=(0.0,) * 12, cloud_cover_pct=(15,) * 12
        ),
    )
    cities = (
        CityCandidate(
            name="Legacy City", country_code="CO", lat=1.0, lon=2.0, cell_lat=1.0, cell_lon=2.0, population=0
        ),
        CityCandidate(
            name="Small Town", country_code="CO", lat=3.0, lon=4.0, cell_lat=3.0, cell_lon=4.0, population=5_000
        ),
        CityCandidate(
            name="Capital", country_code="CO", lat=5.0, lon=6.0, cell_lat=5.0, cell_lon=6.0, population=100_000
        ),
    )
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=7,
        heat_tolerance=5,
        rain_sensitivity=55,
        sun_preference=60,
    )

    class CellsRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return climate_cells

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return cities

    class MatrixRepository(CellsRepository):
        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells(climate_cells)

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities(cities, np.array([0, 1, 2], dtype=np.int32))

    matrix_names = [
        city["name"]
        for city in build_score_response(cast("ClimateRepository", MatrixRepository()), preferences)["scores"]
    ]
    fallback_names = [
        city["name"]
        for city in build_score_response(cast("ClimateRepository", CellsRepository()), preferences)["scores"]
    ]

    assert set(matrix_names) == {"Legacy City", "Capital"}
    assert set(fallback_names) == {"Legacy City", "Capital"}
