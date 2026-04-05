from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest
import tifffile

if TYPE_CHECKING:
    from pathlib import Path

from backend.climate_pipeline import (
    EXPECTED_CLIMATE_COLUMNS,
    INSERT_CLIMATE_CELL_QUERY,
    NODATA_CUTOFF,
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

    summary = validate_climate_database_with_row_range(database_path, (1, 10))

    assert summary.row_count == 2
    assert summary.columns == EXPECTED_CLIMATE_COLUMNS


def test_validate_climate_database_rejects_unexpected_row_count(tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        columns = ", ".join(f"{column} DOUBLE" for column in EXPECTED_CLIMATE_COLUMNS[:26])
        cloud_columns = ", ".join(f"{column} INTEGER" for column in EXPECTED_CLIMATE_COLUMNS[26:])
        connection.execute(f"CREATE TABLE climate_cells ({columns}, {cloud_columns})")
        connection.executemany(INSERT_CLIMATE_CELL_QUERY, [tuple(0 for _ in EXPECTED_CLIMATE_COLUMNS)])

    with pytest.raises(ValueError, match="outside expected rough range"):
        validate_climate_database(database_path)
