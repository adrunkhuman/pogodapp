from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest
import tifffile

if TYPE_CHECKING:
    from pathlib import Path

from backend.climate_pipeline import (
    EXPECTED_CITY_COLUMNS,
    EXPECTED_CLIMATE_COLUMNS,
    INSERT_CITY_QUERY,
    INSERT_CLIMATE_CELL_QUERY,
    NODATA_CUTOFF,
    build_city_rows,
    create_cities_table,
    ensure_geonames_cities,
    load_raster,
    solar_radiation_to_cloud_proxy,
    validate_climate_database,
    validate_climate_database_with_row_range,
)


def test_load_raster_converts_nodata_sentinel_to_nan(tmp_path: Path) -> None:
    sentinel = np.finfo(np.float32).min  # ≈ -3.4e38, WorldClim ocean nodata value
    raw = np.full((1080, 2160), sentinel, dtype=np.float32)
    raw[0, 0] = 18.5  # one land pixel

    tif_path = tmp_path / "test.tif"
    tifffile.imwrite(str(tif_path), raw)

    result = load_raster(tif_path)

    assert result.dtype == np.float64
    assert result[0, 0] == pytest.approx(18.5)
    # All other pixels were nodata — they must be NaN, not the raw sentinel
    assert np.all(np.isnan(result[1:, :]))
    assert np.all(np.isnan(result[0, 1:]))
    assert sentinel <= NODATA_CUTOFF  # confirm the sentinel is caught by the cutoff


def test_solar_radiation_to_cloud_proxy_inverts_relative_brightness() -> None:
    monthly_solar_radiation = np.array(
        [
            [100.0, 200.0],
            [300.0, 400.0],
        ]
    )

    cloud_proxy = solar_radiation_to_cloud_proxy(monthly_solar_radiation)

    assert cloud_proxy[0, 0] == 100
    assert cloud_proxy[1, 1] == 0
    assert cloud_proxy[0, 1] > cloud_proxy[1, 0]


def test_validate_climate_database_accepts_expected_schema_and_row_count(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        columns = ", ".join(f"{column} DOUBLE" for column in EXPECTED_CLIMATE_COLUMNS[:26])
        cloud_columns = ", ".join(f"{column} INTEGER" for column in EXPECTED_CLIMATE_COLUMNS[26:])
        connection.execute(f"CREATE TABLE climate_cells ({columns}, {cloud_columns})")
        connection.executemany(INSERT_CLIMATE_CELL_QUERY, [tuple(0 for _ in EXPECTED_CLIMATE_COLUMNS)] * 2)
        create_cities_table(connection)
        connection.executemany(INSERT_CITY_QUERY, [("Bogota", "CO", 4.711, -74.0721, 4.75, -74.0833)])

    summary = validate_climate_database_with_row_range(database_path, (1, 10))

    assert summary.row_count == 2
    assert summary.columns == EXPECTED_CLIMATE_COLUMNS
    assert summary.city_count == 1
    assert summary.city_columns == EXPECTED_CITY_COLUMNS


def test_validate_climate_database_rejects_unexpected_row_count(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        columns = ", ".join(f"{column} DOUBLE" for column in EXPECTED_CLIMATE_COLUMNS[:26])
        cloud_columns = ", ".join(f"{column} INTEGER" for column in EXPECTED_CLIMATE_COLUMNS[26:])
        connection.execute(f"CREATE TABLE climate_cells ({columns}, {cloud_columns})")
        connection.executemany(INSERT_CLIMATE_CELL_QUERY, [tuple(0 for _ in EXPECTED_CLIMATE_COLUMNS)])
        create_cities_table(connection)
        connection.executemany(INSERT_CITY_QUERY, [("Bogota", "CO", 4.711, -74.0721, 4.75, -74.0833)])

    with pytest.raises(ValueError, match="outside expected rough range"):
        validate_climate_database(database_path)


def test_validate_climate_database_rejects_zero_city_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        columns = ", ".join(f"{column} DOUBLE" for column in EXPECTED_CLIMATE_COLUMNS[:26])
        cloud_columns = ", ".join(f"{column} INTEGER" for column in EXPECTED_CLIMATE_COLUMNS[26:])
        connection.execute(f"CREATE TABLE climate_cells ({columns}, {cloud_columns})")
        connection.executemany(INSERT_CLIMATE_CELL_QUERY, [tuple(0 for _ in EXPECTED_CLIMATE_COLUMNS)] * 2)
        create_cities_table(connection)

    with pytest.raises(ValueError, match="City count is zero"):
        validate_climate_database_with_row_range(database_path, (1, 10))


def test_build_city_rows_filters_out_cities_that_snap_to_ocean_cells(tmp_path: Path) -> None:
    city_catalog_path = tmp_path / "cities15000.txt"
    city_catalog_path.write_text(
        "1\tBogota\tBogota\t\t4.711\t-74.0721\tP\tPPLA\tCO\t\t\t\t\t\t0\t0\t0\tAmerica/Bogota\t2024-01-01\n"
        "2\tOcean Test\tOcean Test\t\t0.0\t0.0\tP\tPPL\tZZ\t\t\t\t\t\t0\t0\t0\tEtc/UTC\t2024-01-01\n",
        encoding="utf-8",
    )

    rows = build_city_rows(city_catalog_path, [(4.75, -74.0833, *([0.0] * 36))])

    assert rows == [("Bogota", "CO", 4.711, -74.0721, 4.75, -74.0833)]


def test_ensure_geonames_cities_uses_cached_extract(tmp_path: Path) -> None:
    extracted_path = tmp_path / "cities15000.txt"
    extracted_path.write_text("cached", encoding="utf-8")

    assert ensure_geonames_cities(tmp_path) == extracted_path
