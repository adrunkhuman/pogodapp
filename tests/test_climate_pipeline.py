from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from backend.climate_pipeline import (
    EXPECTED_CLIMATE_COLUMNS,
    INSERT_CLIMATE_CELL_QUERY,
    aggregate_raster_to_half_degree,
    solar_radiation_to_cloud_proxy,
    validate_climate_database,
    validate_climate_database_with_row_range,
)


def test_aggregate_raster_to_half_degree_averages_valid_source_cells() -> None:
    raster = np.full((1080, 2160), -3.4e38, dtype=np.float32)
    raster[:3, :3] = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )

    aggregated = aggregate_raster_to_half_degree(raster)

    assert aggregated.shape == (360, 720)
    assert aggregated[0, 0] == pytest.approx(5.0)
    assert np.isnan(aggregated[0, 1])


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
