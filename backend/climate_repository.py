from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import duckdb

from backend.cities import STUB_CITY_CANDIDATES, CityCandidate
from backend.scoring import MONTHS_PER_YEAR, STUB_CLIMATE_CELLS, ClimateCell

if TYPE_CHECKING:
    from pathlib import Path

SELECT_CLIMATE_CELLS_QUERY = """
SELECT
    lat,
    lon,
    t_jan, t_feb, t_mar, t_apr, t_may, t_jun, t_jul, t_aug, t_sep, t_oct, t_nov, t_dec,
    prec_jan, prec_feb, prec_mar, prec_apr, prec_may, prec_jun, prec_jul, prec_aug, prec_sep, prec_oct, prec_nov, prec_dec,
    cloud_jan, cloud_feb, cloud_mar, cloud_apr, cloud_may, cloud_jun, cloud_jul, cloud_aug, cloud_sep, cloud_oct, cloud_nov, cloud_dec
FROM climate_cells
"""

SELECT_CITIES_QUERY = """
SELECT
    name,
    country_code,
    lat,
    lon,
    cell_lat,
    cell_lon
FROM cities
"""


class ClimateDataError(RuntimeError):
    """Raised when climate data cannot be loaded into the scoring model."""


class ClimateRepository(Protocol):
    """Small boundary between scoring and climate data storage."""

    def list_cells(self) -> tuple[ClimateCell, ...]:
        """Return climate rows ready for scoring."""

    def list_cities(self) -> tuple[CityCandidate, ...]:
        """Return user-facing cities already mapped onto the dataset."""


class StubClimateRepository:
    """Fallback repository used when the local DuckDB artifact is absent."""

    def list_cells(self) -> tuple[ClimateCell, ...]:
        """Return deterministic in-repo climate fixtures."""
        return STUB_CLIMATE_CELLS

    def list_cities(self) -> tuple[CityCandidate, ...]:
        """Return deterministic in-repo city fixtures."""
        return STUB_CITY_CANDIDATES


def build_default_climate_repository(database_path: Path) -> ClimateRepository:
    """Return the local DuckDB repository when available, otherwise use stubs."""
    if database_path.exists():
        return DuckDbClimateRepository(database_path)

    return StubClimateRepository()


class DuckDbClimateRepository:
    """Load monthly climate rows from DuckDB without leaking SQL into routing code."""

    def __init__(self, database_path: Path) -> None:
        """Store the database file path used for subsequent reads."""
        self.database_path = database_path

    def list_cells(self) -> tuple[ClimateCell, ...]:
        """Read climate rows and map them onto the scoring domain model.

        Raises:
            ClimateDataError: The database file is missing, unreadable, or does
                not match the runtime climate row contract.
        """
        rows = self._fetch_rows(SELECT_CLIMATE_CELLS_QUERY)

        try:
            return tuple(self._row_to_climate_cell(row) for row in rows)
        except (TypeError, ValueError) as error:
            msg = f"Failed to map climate data from {self.database_path} into climate rows: {error}"
            raise ClimateDataError(msg) from error

    def list_cities(self) -> tuple[CityCandidate, ...]:
        """Read build-time city rows already mapped onto climate cells.

        Raises:
            ClimateDataError: The database file is missing, unreadable, missing
                the `cities` table, or contains invalid city rows.
        """
        rows = self._fetch_rows(SELECT_CITIES_QUERY, table_name="cities")

        try:
            return tuple(self._row_to_city(row) for row in rows)
        except (TypeError, ValueError) as error:
            msg = f"Failed to map city data from {self.database_path} into city rows: {error}"
            raise ClimateDataError(msg) from error

    def _fetch_rows(self, query: str, *, table_name: str = "climate_cells") -> list[tuple[object, ...]]:
        if not self.database_path.exists():
            msg = f"Climate database file not found: {self.database_path}"
            raise ClimateDataError(msg)

        try:
            with duckdb.connect(str(self.database_path), read_only=True) as connection:
                return connection.execute(query).fetchall()
        except duckdb.Error as error:
            if table_name == "cities" and "Table with name cities does not exist" in str(error):
                msg = (
                    f"Climate database file is missing the cities table: {self.database_path}. "
                    "Rebuild it with `uv run python scripts/build_climate_db.py`."
                )
                raise ClimateDataError(msg) from error
            msg = f"Failed to read climate data from {self.database_path}: {error}"
            raise ClimateDataError(msg) from error

    def _row_to_climate_cell(self, row: tuple[object, ...]) -> ClimateCell:
        """Convert one database row into the in-memory scoring shape."""
        latitude, longitude, *monthly_values = row
        return ClimateCell(
            lat=float(cast("int | float", latitude)),
            lon=float(cast("int | float", longitude)),
            temperature_c=tuple(float(cast("int | float", value)) for value in monthly_values[:MONTHS_PER_YEAR]),
            precipitation_mm=tuple(
                float(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR : MONTHS_PER_YEAR * 2]
            ),
            cloud_cover_pct=tuple(
                int(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR * 2 : MONTHS_PER_YEAR * 3]
            ),
        )

    def _row_to_city(self, row: tuple[object, ...]) -> CityCandidate:
        """Convert one city row into the in-memory ranking shape."""
        name, country_code, latitude, longitude, cell_latitude, cell_longitude = row
        return CityCandidate(
            name=str(cast("str", name)),
            country_code=str(cast("str", country_code)),
            lat=float(cast("int | float", latitude)),
            lon=float(cast("int | float", longitude)),
            cell_lat=float(cast("int | float", cell_latitude)),
            cell_lon=float(cast("int | float", cell_longitude)),
        )
