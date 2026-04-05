from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path

from backend.cities import CityCandidate
from backend.climate_pipeline import (
    build_insert_rows,
    copy_rows_into_cities_table,
    copy_rows_into_climate_table,
    create_cities_table,
    create_climate_cells_table,
)
from backend.climate_repository import ClimateDataError, DuckDbClimateRepository, StubClimateRepository
from backend.main import create_app
from backend.scoring import ClimateCell


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
                -74.0833 AS cell_lon
            """
        )

    repository = DuckDbClimateRepository(database_path)

    cities = repository.list_cities()

    assert cities == (
        CityCandidate(name="Bogota", country_code="CO", lat=4.711, lon=-74.0721, cell_lat=4.75, cell_lon=-74.0833),
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
                -74.0833 AS cell_lon
            """
        )

    with pytest.raises(ClimateDataError, match="Failed to map city data"):
        DuckDbClimateRepository(database_path).list_cities()


def test_app_can_use_an_injected_climate_repository() -> None:
    class SingleCellRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return (
                ClimateCell(
                    lat=1.0,
                    lon=2.0,
                    temperature_c=(22.0,) * 12,
                    precipitation_mm=(0.0,) * 12,
                    cloud_cover_pct=(15,) * 12,
                ),
            )

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

    response = TestClient(create_app(climate_repository=SingleCellRepository())).post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 200
    assert response.json()["scores"] == []


def test_app_can_use_an_injected_climate_repository_with_city_catalog() -> None:
    class SingleCellRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            return (
                ClimateCell(
                    lat=1.0,
                    lon=2.0,
                    temperature_c=(22.0,) * 12,
                    precipitation_mm=(0.0,) * 12,
                    cloud_cover_pct=(15,) * 12,
                ),
            )

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return (CityCandidate(name="Test City", country_code="CO", lat=1.0, lon=2.0, cell_lat=1.0, cell_lon=2.0),)

    response = TestClient(create_app(climate_repository=SingleCellRepository())).post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [{"name": "Test City", "country_code": "CO", "flag": "🇨🇴", "score": 1.0}]


def test_app_returns_clear_503_when_climate_repository_fails() -> None:
    class BrokenRepository:
        def list_cells(self) -> tuple[ClimateCell, ...]:
            msg = "Climate database file not found: data/climate.duckdb"
            raise ClimateDataError(msg)

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

    response = TestClient(create_app(climate_repository=BrokenRepository())).post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
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
                    precipitation_mm=(0.0,) * 12,
                    cloud_cover_pct=(15,) * 12,
                ),
            )

        def list_cities(self) -> tuple[CityCandidate, ...]:
            msg = "Climate database file is missing the cities table: data/climate.duckdb"
            raise ClimateDataError(msg)

    response = TestClient(create_app(climate_repository=BrokenCityRepository())).post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
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
                2.0 AS cell_lon
            """
        )

    response = TestClient(create_app(climate_repository=DuckDbClimateRepository(database_path))).post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [{"name": "Test City", "country_code": "CO", "flag": "🇨🇴", "score": 1.0}]


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
                2.0 AS cell_lon
            """
        )

    monkeypatch.setattr("backend.main.CLIMATE_DATABASE_PATH", database_path)

    response = TestClient(create_app()).post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 200
    assert response.json()["scores"] == [{"name": "Test City", "country_code": "CO", "flag": "🇨🇴", "score": 1.0}]


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

    monthly_temperature = tuple(np.full((1080, 2160), np.nan, dtype=np.float64) for _ in range(12))
    monthly_precipitation = tuple(np.full((1080, 2160), np.nan, dtype=np.float64) for _ in range(12))
    monthly_solar_radiation = tuple(np.full((1080, 2160), np.nan, dtype=np.float64) for _ in range(12))

    for month_index, month in enumerate(monthly_temperature, start=1):
        month[0, 0] = float(month_index)

    for month_index, month in enumerate(monthly_precipitation, start=1):
        month[0, 0] = float(month_index * 10)

    for month_index, month in enumerate(monthly_solar_radiation, start=1):
        month[0, 0] = float(month_index * 100)

    rows = build_insert_rows(monthly_temperature, monthly_precipitation, monthly_solar_radiation)

    with duckdb.connect(str(database_path)) as connection:
        create_climate_cells_table(connection)
        copy_rows_into_climate_table(connection, rows)
        create_cities_table(connection)
        copy_rows_into_cities_table(
            connection,
            [("North Pole Test City", "NO", 89.9, -179.9, 89.9167, -179.9167)],
        )

    cells = DuckDbClimateRepository(database_path).list_cells()
    cities = DuckDbClimateRepository(database_path).list_cities()

    assert cells == (
        ClimateCell(
            lat=89.9167,
            lon=-179.9167,
            temperature_c=tuple(float(month_index) for month_index in range(1, 13)),
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
            cell_lat=89.9167,
            cell_lon=-179.9167,
        ),
    )
