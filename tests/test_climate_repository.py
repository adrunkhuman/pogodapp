from __future__ import annotations

from typing import TYPE_CHECKING, cast

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

    from backend.climate_repository import ClimateRepository

from backend.cities import CityCandidate, CityRankingCache, snap_city_to_cell_key
from backend.climate_pipeline import (
    DEFAULT_WORLDCLIM_RESOLUTION,
    MonthlyClimateRasters,
    build_insert_rows,
    copy_rows_into_cities_table,
    copy_rows_into_climate_table,
    create_cities_table,
    create_climate_cells_table,
)
from backend.climate_repository import (
    ClimateDataError,
    DuckDbClimateRepository,
    StubClimateRepository,
)
from backend.heatmap import HeatmapProjection
from backend.main import create_app
from backend.scoring import ClimateCell, ClimateMatrix


def default_form_data() -> dict[str, str]:
    return {
        "preferred_day_temperature": "22",
        "summer_heat_limit": "30",
        "winter_cold_limit": "5",
        "dryness_preference": "60",
        "sunshine_preference": "60",
    }


def test_stub_climate_repository_returns_climate_cells() -> None:
    repository = StubClimateRepository()

    cells = repository.list_cells()

    assert cells
    assert all(isinstance(cell, ClimateCell) for cell in cells)


def test_stub_climate_repository_returns_city_candidates() -> None:
    repository = StubClimateRepository()

    cities = repository.list_cities()

    assert cities
    assert all(isinstance(city, CityCandidate) for city in cities)


def test_duckdb_climate_repository_loads_monthly_climate_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                10.5 AS lat,
                20.5 AS lon,
                1.0 AS t_jan, 2.0 AS t_feb, 3.0 AS t_mar, 4.0 AS t_apr, 5.0 AS t_may, 6.0 AS t_jun,
                7.0 AS t_jul, 8.0 AS t_aug, 9.0 AS t_sep, 10.0 AS t_oct, 11.0 AS t_nov, 12.0 AS t_dec,
                -1.0 AS tmin_jan, 0.0 AS tmin_feb, 1.0 AS tmin_mar, 2.0 AS tmin_apr, 3.0 AS tmin_may, 4.0 AS tmin_jun,
                5.0 AS tmin_jul, 6.0 AS tmin_aug, 7.0 AS tmin_sep, 8.0 AS tmin_oct, 9.0 AS tmin_nov, 10.0 AS tmin_dec,
                3.0 AS tmax_jan, 4.0 AS tmax_feb, 5.0 AS tmax_mar, 6.0 AS tmax_apr, 7.0 AS tmax_may, 8.0 AS tmax_jun,
                9.0 AS tmax_jul, 10.0 AS tmax_aug, 11.0 AS tmax_sep, 12.0 AS tmax_oct, 13.0 AS tmax_nov, 14.0 AS tmax_dec,
                13.0 AS prec_jan, 14.0 AS prec_feb, 15.0 AS prec_mar, 16.0 AS prec_apr, 17.0 AS prec_may, 18.0 AS prec_jun,
                19.0 AS prec_jul, 20.0 AS prec_aug, 21.0 AS prec_sep, 22.0 AS prec_oct, 23.0 AS prec_nov, 24.0 AS prec_dec,
                25 AS cloud_jan, 26 AS cloud_feb, 27 AS cloud_mar, 28 AS cloud_apr, 29 AS cloud_may, 30 AS cloud_jun,
                31 AS cloud_jul, 32 AS cloud_aug, 33 AS cloud_sep, 34 AS cloud_oct, 35 AS cloud_nov, 36 AS cloud_dec
            """
        )

    repository = DuckDbClimateRepository(database_path)

    cells = repository.list_cells()

    assert cells == (
        ClimateCell(
            lat=10.5,
            lon=20.5,
            temperature_c=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
            temperature_min_c=(-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
            temperature_max_c=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0),
            precipitation_mm=(13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0),
            cloud_cover_pct=(25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36),
        ),
    )


def test_duckdb_climate_repository_loads_city_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT
                'Bogota' AS name,
                'CO' AS country_code,
                4.711 AS lat,
                -74.0721 AS lon,
                4.75 AS cell_lat,
                -74.0833 AS cell_lon,
                100000 AS population
            """
        )

    repository = DuckDbClimateRepository(database_path)

    cities = repository.list_cities()

    assert cities == (
        CityCandidate(
            name="Bogota",
            country_code="CO",
            lat=4.711,
            lon=-74.0721,
            cell_lat=4.75,
            cell_lon=-74.0833,
            population=100000,
        ),
    )


def test_duckdb_climate_repository_defaults_missing_population_column_to_zero(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT
                'Bogota' AS name,
                'CO' AS country_code,
                4.711 AS lat,
                -74.0721 AS lon,
                4.75 AS cell_lat,
                -74.0833 AS cell_lon
            """
        )

    cities = DuckDbClimateRepository(database_path).list_cities()

    assert cities == (
        CityCandidate(
            name="Bogota",
            country_code="CO",
            lat=4.711,
            lon=-74.0721,
            cell_lat=4.75,
            cell_lon=-74.0833,
            population=0,
        ),
    )


def test_duckdb_climate_repository_raises_clear_error_for_missing_database(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.duckdb"

    with pytest.raises(ClimateDataError, match="Climate database file not found"):
        DuckDbClimateRepository(missing_path).list_cells()


def test_duckdb_climate_repository_raises_clear_error_for_schema_mismatch(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("CREATE TABLE climate_cells (lat DOUBLE, lon DOUBLE)")

    with pytest.raises(ClimateDataError, match="Failed to read climate data"):
        DuckDbClimateRepository(database_path).list_cells()


def test_duckdb_climate_repository_raises_clear_error_for_missing_cities_table(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("CREATE TABLE climate_cells (lat DOUBLE, lon DOUBLE)")

    with pytest.raises(ClimateDataError, match="missing the cities table"):
        DuckDbClimateRepository(database_path).list_cities()


def test_duckdb_climate_repository_raises_clear_error_for_bad_city_row_values(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT
                'Bogota' AS name,
                'CO' AS country_code,
                NULL AS lat,
                -74.0721 AS lon,
                4.75 AS cell_lat,
                -74.0833 AS cell_lon,
                0 AS population
            """
        )

    with pytest.raises(ClimateDataError, match="Failed to map city data"):
        DuckDbClimateRepository(database_path).list_cities()


def test_duckdb_climate_repository_probe_nearest_cell_returns_index_for_known_land_cell(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    probe_city = CityCandidate(name="", country_code="", lat=10.5, lon=20.5, cell_lat=0.0, cell_lon=0.0)
    snapped_lat, snapped_lon = snap_city_to_cell_key(probe_city)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                ? AS lat,
                ? AS lon,
                1.0 AS t_jan, 2.0 AS t_feb, 3.0 AS t_mar, 4.0 AS t_apr, 5.0 AS t_may, 6.0 AS t_jun,
                7.0 AS t_jul, 8.0 AS t_aug, 9.0 AS t_sep, 10.0 AS t_oct, 11.0 AS t_nov, 12.0 AS t_dec,
                -1.0 AS tmin_jan, 0.0 AS tmin_feb, 1.0 AS tmin_mar, 2.0 AS tmin_apr, 3.0 AS tmin_may, 4.0 AS tmin_jun,
                5.0 AS tmin_jul, 6.0 AS tmin_aug, 7.0 AS tmin_sep, 8.0 AS tmin_oct, 9.0 AS tmin_nov, 10.0 AS tmin_dec,
                3.0 AS tmax_jan, 4.0 AS tmax_feb, 5.0 AS tmax_mar, 6.0 AS tmax_apr, 7.0 AS tmax_may, 8.0 AS tmax_jun,
                9.0 AS tmax_jul, 10.0 AS tmax_aug, 11.0 AS tmax_sep, 12.0 AS tmax_oct, 13.0 AS tmax_nov, 14.0 AS tmax_dec,
                13.0 AS prec_jan, 14.0 AS prec_feb, 15.0 AS prec_mar, 16.0 AS prec_apr, 17.0 AS prec_may, 18.0 AS prec_jun,
                19.0 AS prec_jul, 20.0 AS prec_aug, 21.0 AS prec_sep, 22.0 AS prec_oct, 23.0 AS prec_nov, 24.0 AS prec_dec,
                25 AS cloud_jan, 26 AS cloud_feb, 27 AS cloud_mar, 28 AS cloud_apr, 29 AS cloud_may, 30 AS cloud_jun,
                31 AS cloud_jul, 32 AS cloud_aug, 33 AS cloud_sep, 34 AS cloud_oct, 35 AS cloud_nov, 36 AS cloud_dec
            """,
            [snapped_lat, snapped_lon],
        )

    repository = DuckDbClimateRepository(database_path)

    assert repository.probe_nearest_cell(probe_city.lat, probe_city.lon) == 0


def test_duckdb_climate_repository_probe_nearest_cell_returns_none_for_unmapped_coords(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                10.5 AS lat,
                20.5 AS lon,
                1.0 AS t_jan, 2.0 AS t_feb, 3.0 AS t_mar, 4.0 AS t_apr, 5.0 AS t_may, 6.0 AS t_jun,
                7.0 AS t_jul, 8.0 AS t_aug, 9.0 AS t_sep, 10.0 AS t_oct, 11.0 AS t_nov, 12.0 AS t_dec,
                -1.0 AS tmin_jan, 0.0 AS tmin_feb, 1.0 AS tmin_mar, 2.0 AS tmin_apr, 3.0 AS tmin_may, 4.0 AS tmin_jun,
                5.0 AS tmin_jul, 6.0 AS tmin_aug, 7.0 AS tmin_sep, 8.0 AS tmin_oct, 9.0 AS tmin_nov, 10.0 AS tmin_dec,
                3.0 AS tmax_jan, 4.0 AS tmax_feb, 5.0 AS tmax_mar, 6.0 AS tmax_apr, 7.0 AS tmax_may, 8.0 AS tmax_jun,
                9.0 AS tmax_jul, 10.0 AS tmax_aug, 11.0 AS tmax_sep, 12.0 AS tmax_oct, 13.0 AS tmax_nov, 14.0 AS tmax_dec,
                13.0 AS prec_jan, 14.0 AS prec_feb, 15.0 AS prec_mar, 16.0 AS prec_apr, 17.0 AS prec_may, 18.0 AS prec_jun,
                19.0 AS prec_jul, 20.0 AS prec_aug, 21.0 AS prec_sep, 22.0 AS prec_oct, 23.0 AS prec_nov, 24.0 AS prec_dec,
                25 AS cloud_jan, 26 AS cloud_feb, 27 AS cloud_mar, 28 AS cloud_apr, 29 AS cloud_may, 30 AS cloud_jun,
                31 AS cloud_jul, 32 AS cloud_aug, 33 AS cloud_sep, 34 AS cloud_oct, 35 AS cloud_nov, 36 AS cloud_dec
            """
        )

    repository = DuckDbClimateRepository(database_path)

    assert repository.probe_nearest_cell(0.0, 0.0) is None


def test_app_can_use_an_injected_climate_repository() -> None:
    class SingleCellRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return (
                ClimateCell(
                    lat=1.0,
                    lon=2.0,
                    temperature_c=(22.0,) * 12,
                    temperature_min_c=(22.0,) * 12,
                    temperature_max_c=(22.0,) * 12,
                    precipitation_mm=(0.0,) * 12,
                    cloud_cover_pct=(15,) * 12,
                ),
            )

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells(self.list_cells())

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities((), np.array([], dtype=np.int32))

        def get_heatmap_projection(self) -> HeatmapProjection:
            climate_matrix = self.get_climate_matrix()
            return HeatmapProjection.from_coordinates(climate_matrix.latitudes, climate_matrix.longitudes)

    response = TestClient(create_app(climate_repository=cast("ClimateRepository", SingleCellRepository()))).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    assert response.json()["scores"] == []


def test_create_app_preloads_optimized_repository() -> None:
    class PreloadedRepository:
        def __init__(self) -> None:
            self.preload_calls: list[str] = []

        def get_climate_matrix(self) -> ClimateMatrix:
            self.preload_calls.append("matrix")
            return ClimateMatrix(
                latitudes=np.array([], dtype=np.float32),
                longitudes=np.array([], dtype=np.float32),
                temperature_c=np.empty((0, 12), dtype=np.float32),
                temperature_min_c=np.empty((0, 12), dtype=np.float32),
                temperature_max_c=np.empty((0, 12), dtype=np.float32),
                precipitation_mm=np.empty((0, 12), dtype=np.float32),
                cloud_cover_pct=np.empty((0, 12), dtype=np.uint8),
            )

        def get_indexed_cities(self) -> CityRankingCache:
            self.preload_calls.append("cities")
            return CityRankingCache.from_cities((), np.array([], dtype=np.int32))

        def get_heatmap_projection(self) -> HeatmapProjection:
            self.preload_calls.append("heatmap")
            return HeatmapProjection.from_coordinates(np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

    repository = PreloadedRepository()

    create_app(climate_repository=cast("ClimateRepository", repository))

    # Preload warms all three caches first; default score precompute may add further calls after.
    assert repository.preload_calls[:3] == ["matrix", "cities", "heatmap"]


def test_create_app_preload_is_best_effort_for_data_errors() -> None:
    class BrokenPreloadRepository:
        def get_climate_matrix(self) -> ClimateMatrix:
            msg = "Climate database file not found: data/climate.duckdb"
            raise ClimateDataError(msg)

        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

    app = create_app(climate_repository=cast("ClimateRepository", BrokenPreloadRepository()))

    assert app.title == "Pogodapp"


def test_create_app_preload_warms_caches_without_rebuilding_on_first_request() -> None:
    class CountingRepository:
        def __init__(self) -> None:
            self.matrix_builds = 0
            self.city_cache_builds = 0
            self.heatmap_builds = 0
            self._matrix: ClimateMatrix | None = None
            self._cities: CityRankingCache | None = None
            self._projection: HeatmapProjection | None = None

        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

        def get_climate_matrix(self) -> ClimateMatrix:
            if self._matrix is None:
                self.matrix_builds += 1
                self._matrix = ClimateMatrix.from_cells(
                    (
                        ClimateCell(
                            lat=1.0,
                            lon=2.0,
                            temperature_c=(22.0,) * 12,
                            temperature_min_c=(22.0,) * 12,
                            temperature_max_c=(22.0,) * 12,
                            precipitation_mm=(0.0,) * 12,
                            cloud_cover_pct=(15,) * 12,
                        ),
                    )
                )
            return self._matrix

        def get_indexed_cities(self) -> CityRankingCache:
            if self._cities is None:
                self.city_cache_builds += 1
                self._cities = CityRankingCache.from_cities((), np.array([], dtype=np.int32))
            return self._cities

        def get_heatmap_projection(self) -> HeatmapProjection:
            if self._projection is None:
                self.heatmap_builds += 1
                climate_matrix = self.get_climate_matrix()
                self._projection = HeatmapProjection.from_coordinates(
                    climate_matrix.latitudes, climate_matrix.longitudes
                )
            return self._projection

    repository = CountingRepository()
    client = TestClient(create_app(climate_repository=cast("ClimateRepository", repository)))

    response = client.post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    assert repository.matrix_builds == 1
    assert repository.city_cache_builds == 1
    assert repository.heatmap_builds == 1


def test_app_can_use_an_injected_climate_repository_with_city_catalog() -> None:
    class SingleCellRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return (
                ClimateCell(
                    lat=1.0,
                    lon=2.0,
                    temperature_c=(22.0,) * 12,
                    temperature_min_c=(22.0,) * 12,
                    temperature_max_c=(22.0,) * 12,
                    precipitation_mm=(0.0,) * 12,
                    cloud_cover_pct=(15,) * 12,
                ),
            )

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return (CityCandidate(name="Test City", country_code="CO", lat=1.0, lon=2.0, cell_lat=1.0, cell_lon=2.0),)

        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells(self.list_cells())

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities(self.list_cities(), np.array([0], dtype=np.int32))

        def get_heatmap_projection(self) -> HeatmapProjection:
            climate_matrix = self.get_climate_matrix()
            return HeatmapProjection.from_coordinates(climate_matrix.latitudes, climate_matrix.longitudes)

    response = TestClient(create_app(climate_repository=cast("ClimateRepository", SingleCellRepository()))).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [
        {
            "name": "Test City",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 1.0,
            "lat": 1.0,
            "lon": 2.0,
            "probe_lat": 1.0,
            "probe_lon": 2.0,
        }
    ]


def test_app_returns_clear_503_when_climate_repository_fails() -> None:
    class BrokenRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            msg = "Climate database file not found: data/climate.duckdb"
            raise ClimateDataError(msg)

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

    response = TestClient(create_app(climate_repository=cast("ClimateRepository", BrokenRepository()))).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Climate database file not found: data/climate.duckdb"}


def test_app_returns_clear_503_when_city_lookup_fails() -> None:
    class BrokenCityRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return (
                ClimateCell(
                    lat=1.0,
                    lon=2.0,
                    temperature_c=(22.0,) * 12,
                    temperature_min_c=(22.0,) * 12,
                    temperature_max_c=(22.0,) * 12,
                    precipitation_mm=(0.0,) * 12,
                    cloud_cover_pct=(15,) * 12,
                ),
            )

        def list_cities(self) -> tuple[CityCandidate, ...]:
            msg = "Climate database file is missing the cities table: data/climate.duckdb"
            raise ClimateDataError(msg)

    response = TestClient(create_app(climate_repository=cast("ClimateRepository", BrokenCityRepository()))).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Climate database file is missing the cities table: data/climate.duckdb"}


def test_app_scores_from_duckdb(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                1.0 AS lat,
                2.0 AS lon,
                22.0 AS t_jan, 22.0 AS t_feb, 22.0 AS t_mar, 22.0 AS t_apr, 22.0 AS t_may, 22.0 AS t_jun,
                22.0 AS t_jul, 22.0 AS t_aug, 22.0 AS t_sep, 22.0 AS t_oct, 22.0 AS t_nov, 22.0 AS t_dec,
                22.0 AS tmin_jan, 22.0 AS tmin_feb, 22.0 AS tmin_mar, 22.0 AS tmin_apr, 22.0 AS tmin_may, 22.0 AS tmin_jun,
                22.0 AS tmin_jul, 22.0 AS tmin_aug, 22.0 AS tmin_sep, 22.0 AS tmin_oct, 22.0 AS tmin_nov, 22.0 AS tmin_dec,
                22.0 AS tmax_jan, 22.0 AS tmax_feb, 22.0 AS tmax_mar, 22.0 AS tmax_apr, 22.0 AS tmax_may, 22.0 AS tmax_jun,
                22.0 AS tmax_jul, 22.0 AS tmax_aug, 22.0 AS tmax_sep, 22.0 AS tmax_oct, 22.0 AS tmax_nov, 22.0 AS tmax_dec,
                0.0 AS prec_jan, 0.0 AS prec_feb, 0.0 AS prec_mar, 0.0 AS prec_apr, 0.0 AS prec_may, 0.0 AS prec_jun,
                0.0 AS prec_jul, 0.0 AS prec_aug, 0.0 AS prec_sep, 0.0 AS prec_oct, 0.0 AS prec_nov, 0.0 AS prec_dec,
                15 AS cloud_jan, 15 AS cloud_feb, 15 AS cloud_mar, 15 AS cloud_apr, 15 AS cloud_may, 15 AS cloud_jun,
                15 AS cloud_jul, 15 AS cloud_aug, 15 AS cloud_sep, 15 AS cloud_oct, 15 AS cloud_nov, 15 AS cloud_dec
            """
        )
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT
                'Test City' AS name,
                'CO' AS country_code,
                1.0 AS lat,
                2.0 AS lon,
                1.0 AS cell_lat,
                2.0 AS cell_lon,
                0 AS population
            """
        )

    response = TestClient(create_app(climate_repository=DuckDbClimateRepository(database_path))).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [
        {
            "name": "Test City",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 1.0,
            "lat": 1.0,
            "lon": 2.0,
            "probe_lat": 1.0,
            "probe_lon": 2.0,
        }
    ]


def test_duckdb_city_cache_aligns_indexes_with_shuffled_climate_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT * FROM (
                VALUES
                    (5.0, 6.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                     22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                     22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15),
                    (1.0, 2.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                     22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                     22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
            ) AS t(
                lat, lon,
                t_jan, t_feb, t_mar, t_apr, t_may, t_jun, t_jul, t_aug, t_sep, t_oct, t_nov, t_dec,
                tmin_jan, tmin_feb, tmin_mar, tmin_apr, tmin_may, tmin_jun, tmin_jul, tmin_aug, tmin_sep, tmin_oct, tmin_nov, tmin_dec,
                tmax_jan, tmax_feb, tmax_mar, tmax_apr, tmax_may, tmax_jun, tmax_jul, tmax_aug, tmax_sep, tmax_oct, tmax_nov, tmax_dec,
                prec_jan, prec_feb, prec_mar, prec_apr, prec_may, prec_jun, prec_jul, prec_aug, prec_sep, prec_oct, prec_nov, prec_dec,
                cloud_jan, cloud_feb, cloud_mar, cloud_apr, cloud_may, cloud_jun, cloud_jul, cloud_aug, cloud_sep, cloud_oct, cloud_nov, cloud_dec
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT * FROM (
                VALUES
                    ('First Match', 'CO', 1.0, 2.0, 1.0, 2.0, 0),
                    ('Missing Match', 'CO', 7.0, 8.0, 7.0, 8.0, 0)
            ) AS t(name, country_code, lat, lon, cell_lat, cell_lon, population)
            """
        )

    repository = DuckDbClimateRepository(database_path)
    climate_matrix = repository.get_climate_matrix()
    city_cache = repository.get_indexed_cities()

    assert [city.name for city in city_cache.cities] == ["First Match"]
    mapped_index = int(city_cache.climate_indexes[0])
    assert float(climate_matrix.latitudes[mapped_index]) == 1.0
    assert float(climate_matrix.longitudes[mapped_index]) == 2.0


def test_app_uses_duckdb_automatically_when_default_database_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                1.0 AS lat,
                2.0 AS lon,
                22.0 AS t_jan, 22.0 AS t_feb, 22.0 AS t_mar, 22.0 AS t_apr, 22.0 AS t_may, 22.0 AS t_jun,
                22.0 AS t_jul, 22.0 AS t_aug, 22.0 AS t_sep, 22.0 AS t_oct, 22.0 AS t_nov, 22.0 AS t_dec,
                22.0 AS tmin_jan, 22.0 AS tmin_feb, 22.0 AS tmin_mar, 22.0 AS tmin_apr, 22.0 AS tmin_may, 22.0 AS tmin_jun,
                22.0 AS tmin_jul, 22.0 AS tmin_aug, 22.0 AS tmin_sep, 22.0 AS tmin_oct, 22.0 AS tmin_nov, 22.0 AS tmin_dec,
                22.0 AS tmax_jan, 22.0 AS tmax_feb, 22.0 AS tmax_mar, 22.0 AS tmax_apr, 22.0 AS tmax_may, 22.0 AS tmax_jun,
                22.0 AS tmax_jul, 22.0 AS tmax_aug, 22.0 AS tmax_sep, 22.0 AS tmax_oct, 22.0 AS tmax_nov, 22.0 AS tmax_dec,
                0.0 AS prec_jan, 0.0 AS prec_feb, 0.0 AS prec_mar, 0.0 AS prec_apr, 0.0 AS prec_may, 0.0 AS prec_jun,
                0.0 AS prec_jul, 0.0 AS prec_aug, 0.0 AS prec_sep, 0.0 AS prec_oct, 0.0 AS prec_nov, 0.0 AS prec_dec,
                15 AS cloud_jan, 15 AS cloud_feb, 15 AS cloud_mar, 15 AS cloud_apr, 15 AS cloud_may, 15 AS cloud_jun,
                15 AS cloud_jul, 15 AS cloud_aug, 15 AS cloud_sep, 15 AS cloud_oct, 15 AS cloud_nov, 15 AS cloud_dec
            """
        )
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT
                'Test City' AS name,
                'CO' AS country_code,
                1.0 AS lat,
                2.0 AS lon,
                1.0 AS cell_lat,
                2.0 AS cell_lon,
                0 AS population
            """
        )

    monkeypatch.setenv("POGODAPP_CLIMATE_DB", str(database_path))

    response = TestClient(create_app()).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [
        {
            "name": "Test City",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 1.0,
            "lat": 1.0,
            "lon": 2.0,
            "probe_lat": 1.0,
            "probe_lon": 2.0,
        }
    ]


def test_duckdb_climate_repository_raises_clear_error_for_bad_row_values(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                1.0 AS lat,
                2.0 AS lon,
                NULL AS t_jan, 22.0 AS t_feb, 22.0 AS t_mar, 22.0 AS t_apr, 22.0 AS t_may, 22.0 AS t_jun,
                22.0 AS t_jul, 22.0 AS t_aug, 22.0 AS t_sep, 22.0 AS t_oct, 22.0 AS t_nov, 22.0 AS t_dec,
                22.0 AS tmin_jan, 22.0 AS tmin_feb, 22.0 AS tmin_mar, 22.0 AS tmin_apr, 22.0 AS tmin_may, 22.0 AS tmin_jun,
                22.0 AS tmin_jul, 22.0 AS tmin_aug, 22.0 AS tmin_sep, 22.0 AS tmin_oct, 22.0 AS tmin_nov, 22.0 AS tmin_dec,
                22.0 AS tmax_jan, 22.0 AS tmax_feb, 22.0 AS tmax_mar, 22.0 AS tmax_apr, 22.0 AS tmax_may, 22.0 AS tmax_jun,
                22.0 AS tmax_jul, 22.0 AS tmax_aug, 22.0 AS tmax_sep, 22.0 AS tmax_oct, 22.0 AS tmax_nov, 22.0 AS tmax_dec,
                0.0 AS prec_jan, 0.0 AS prec_feb, 0.0 AS prec_mar, 0.0 AS prec_apr, 0.0 AS prec_may, 0.0 AS prec_jun,
                0.0 AS prec_jul, 0.0 AS prec_aug, 0.0 AS prec_sep, 0.0 AS prec_oct, 0.0 AS prec_nov, 0.0 AS prec_dec,
                15 AS cloud_jan, 15 AS cloud_feb, 15 AS cloud_mar, 15 AS cloud_apr, 15 AS cloud_may, 15 AS cloud_jun,
                15 AS cloud_jul, 15 AS cloud_aug, 15 AS cloud_sep, 15 AS cloud_oct, 15 AS cloud_nov, 15 AS cloud_dec
            """
        )

    with pytest.raises(ClimateDataError, match="Failed to map climate data"):
        DuckDbClimateRepository(database_path).list_cells()


def test_duckdb_climate_repository_loads_rows_built_in_pipeline_shape(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    raster_height, raster_width = DEFAULT_WORLDCLIM_RESOLUTION.raster_shape

    monthly_temperature = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_temperature_min = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_temperature_max = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_precipitation = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_solar_radiation = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))

    for month_index, month in enumerate(monthly_temperature, start=1):
        month[0, 0] = float(month_index)

    for month_index, month in enumerate(monthly_temperature_min, start=1):
        month[0, 0] = float(month_index - 2)

    for month_index, month in enumerate(monthly_temperature_max, start=1):
        month[0, 0] = float(month_index + 2)

    for month_index, month in enumerate(monthly_precipitation, start=1):
        month[0, 0] = float(month_index * 10)

    for month_index, month in enumerate(monthly_solar_radiation, start=1):
        month[0, 0] = float(month_index * 100)

    rows = build_insert_rows(
        MonthlyClimateRasters(
            temperature_mean=monthly_temperature,
            temperature_min=monthly_temperature_min,
            temperature_max=monthly_temperature_max,
            precipitation=monthly_precipitation,
            solar_radiation=monthly_solar_radiation,
        ),
        DEFAULT_WORLDCLIM_RESOLUTION,
    )

    with duckdb.connect(str(database_path)) as connection:
        create_climate_cells_table(connection)
        copy_rows_into_climate_table(connection, rows)
        create_cities_table(connection)
        copy_rows_into_cities_table(
            connection,
            [("North Pole Test City", "NO", 89.9, -179.9, 89.9583, -179.9583, 0)],
        )

    cells = DuckDbClimateRepository(database_path).list_cells()
    cities = DuckDbClimateRepository(database_path).list_cities()

    assert cells == (
        ClimateCell(
            lat=89.9583,
            lon=-179.9583,
            temperature_c=tuple(float(month_index) for month_index in range(1, 13)),
            temperature_min_c=tuple(float(month_index - 2) for month_index in range(1, 13)),
            temperature_max_c=tuple(float(month_index + 2) for month_index in range(1, 13)),
            precipitation_mm=tuple(float(month_index * 10) for month_index in range(1, 13)),
            cloud_cover_pct=(50,) * 12,
        ),
    )
    assert cities == (
        CityCandidate(
            name="North Pole Test City",
            country_code="NO",
            lat=89.9,
            lon=-179.9,
            cell_lat=89.9583,
            cell_lon=-179.9583,
        ),
    )


def test_build_insert_rows_drops_cells_with_nonfinite_tmin_or_tmax() -> None:
    raster_height, raster_width = DEFAULT_WORLDCLIM_RESOLUTION.raster_shape

    monthly_temperature = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_temperature_min = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_temperature_max = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_precipitation = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))
    monthly_solar_radiation = tuple(np.full((raster_height, raster_width), np.nan, dtype=np.float64) for _ in range(12))

    for month in monthly_temperature:
        month[0, 0] = 20.0
        month[0, 1] = 20.0
    for month in monthly_temperature_min:
        month[0, 0] = 10.0
        month[0, 1] = 10.0
    for month in monthly_temperature_max:
        month[0, 0] = 25.0
        month[0, 1] = 25.0
    for month in monthly_precipitation:
        month[0, 0] = 50.0
        month[0, 1] = 50.0
    for month in monthly_solar_radiation:
        month[0, 0] = 100.0
        month[0, 1] = 100.0

    monthly_temperature_min[0][0, 1] = np.nan
    monthly_temperature_max[1][0, 1] = np.nan

    rows = build_insert_rows(
        MonthlyClimateRasters(
            temperature_mean=monthly_temperature,
            temperature_min=monthly_temperature_min,
            temperature_max=monthly_temperature_max,
            precipitation=monthly_precipitation,
            solar_radiation=monthly_solar_radiation,
        ),
        DEFAULT_WORLDCLIM_RESOLUTION,
    )

    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(89.9583, abs=1e-4)
    assert rows[0][1] == pytest.approx(-179.9583, abs=1e-4)


def test_create_app_reads_climate_database_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "climate-5m.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE climate_cells AS
            SELECT
                1.0 AS lat,
                2.0 AS lon,
                22.0 AS t_jan, 22.0 AS t_feb, 22.0 AS t_mar, 22.0 AS t_apr, 22.0 AS t_may, 22.0 AS t_jun,
                22.0 AS t_jul, 22.0 AS t_aug, 22.0 AS t_sep, 22.0 AS t_oct, 22.0 AS t_nov, 22.0 AS t_dec,
                22.0 AS tmin_jan, 22.0 AS tmin_feb, 22.0 AS tmin_mar, 22.0 AS tmin_apr, 22.0 AS tmin_may, 22.0 AS tmin_jun,
                22.0 AS tmin_jul, 22.0 AS tmin_aug, 22.0 AS tmin_sep, 22.0 AS tmin_oct, 22.0 AS tmin_nov, 22.0 AS tmin_dec,
                22.0 AS tmax_jan, 22.0 AS tmax_feb, 22.0 AS tmax_mar, 22.0 AS tmax_apr, 22.0 AS tmax_may, 22.0 AS tmax_jun,
                22.0 AS tmax_jul, 22.0 AS tmax_aug, 22.0 AS tmax_sep, 22.0 AS tmax_oct, 22.0 AS tmax_nov, 22.0 AS tmax_dec,
                0.0 AS prec_jan, 0.0 AS prec_feb, 0.0 AS prec_mar, 0.0 AS prec_apr, 0.0 AS prec_may, 0.0 AS prec_jun,
                0.0 AS prec_jul, 0.0 AS prec_aug, 0.0 AS prec_sep, 0.0 AS prec_oct, 0.0 AS prec_nov, 0.0 AS prec_dec,
                15 AS cloud_jan, 15 AS cloud_feb, 15 AS cloud_mar, 15 AS cloud_apr, 15 AS cloud_may, 15 AS cloud_jun,
                15 AS cloud_jul, 15 AS cloud_aug, 15 AS cloud_sep, 15 AS cloud_oct, 15 AS cloud_nov, 15 AS cloud_dec
            """
        )
        connection.execute(
            """
            CREATE TABLE cities AS
            SELECT
                'Env Test City' AS name,
                'CO' AS country_code,
                1.0 AS lat,
                2.0 AS lon,
                1.0 AS cell_lat,
                2.0 AS cell_lon,
                0 AS population
            """
        )

    monkeypatch.setenv("POGODAPP_CLIMATE_DB", str(database_path))

    response = TestClient(create_app()).post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [
        {
            "name": "Env Test City",
            "continent": "South America",
            "country_code": "CO",
            "flag": "🇨🇴",
            "score": 1.0,
            "lat": 1.0,
            "lon": 2.0,
            "probe_lat": 1.0,
            "probe_lon": 2.0,
        }
    ]
