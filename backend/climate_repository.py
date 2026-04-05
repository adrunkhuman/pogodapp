from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import duckdb
import numpy as np

from backend.cities import (
    STUB_CITY_CANDIDATES,
    CityCandidate,
    CityRankingCache,
    coordinate_key,
    snap_city_to_cell_key,
)
from backend.heatmap import HeatmapProjection
from backend.scoring import MONTHS_PER_YEAR, STUB_CLIMATE_CELLS, ClimateCell, ClimateMatrix

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

CITY_BASE_COLUMNS = ("name", "country_code", "lat", "lon", "cell_lat", "cell_lon")


class ClimateDataError(RuntimeError):
    """Raised when climate data cannot be loaded into the scoring model."""


class ClimateRepository(Protocol):
    """Small boundary between scoring and climate data storage."""

    def list_cells(self) -> tuple[ClimateCell, ...]:
        """Return climate rows ready for scoring."""

    def list_cities(self) -> tuple[CityCandidate, ...]:
        """Return user-facing cities already mapped onto the dataset."""

    def get_climate_matrix(self) -> ClimateMatrix:
        """Return compact climate arrays ready for vectorized scoring."""

    def get_indexed_cities(self) -> CityRankingCache:
        """Return ranking-ready cities aligned with the current climate-matrix row order."""

    def get_heatmap_projection(self) -> HeatmapProjection:
        """Return cached heatmap pixel coordinates aligned with the current climate matrix."""


class StubClimateRepository:
    """Fallback repository used when the local DuckDB artifact is absent."""

    def list_cells(self) -> tuple[ClimateCell, ...]:
        """Return deterministic in-repo climate fixtures."""
        return STUB_CLIMATE_CELLS

    def list_cities(self) -> tuple[CityCandidate, ...]:
        """Return deterministic in-repo city fixtures."""
        return STUB_CITY_CANDIDATES

    def get_climate_matrix(self) -> ClimateMatrix:
        """Return the stub dataset in compact array form."""
        return ClimateMatrix.from_cells(STUB_CLIMATE_CELLS)

    def get_indexed_cities(self) -> CityRankingCache:
        """Return stub cities aligned to the stub matrix rows."""
        return CityRankingCache.from_cities(STUB_CITY_CANDIDATES, np.arange(len(STUB_CITY_CANDIDATES), dtype=np.int32))

    def get_heatmap_projection(self) -> HeatmapProjection:
        """Return cached heatmap projection for the stub dataset."""
        climate_matrix = self.get_climate_matrix()
        return HeatmapProjection.from_coordinates(climate_matrix.latitudes, climate_matrix.longitudes)


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
        self._climate_matrix: ClimateMatrix | None = None
        self._cities: tuple[CityCandidate, ...] | None = None
        self._indexed_cities: CityRankingCache | None = None
        self._heatmap_projection: HeatmapProjection | None = None
        self._sorted_climate_keys: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None
        self._sorted_climate_indexes: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None

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
        if self._cities is not None:
            return self._cities

        rows = self._fetch_rows(self._select_cities_query(), table_name="cities")

        try:
            self._cities = tuple(self._row_to_city(row) for row in rows)
        except (TypeError, ValueError) as error:
            msg = f"Failed to map city data from {self.database_path} into city rows: {error}"
            raise ClimateDataError(msg) from error

        return self._cities

    def get_climate_matrix(self) -> ClimateMatrix:
        """Load the climate table once into compact arrays for repeated scoring."""
        if self._climate_matrix is not None:
            return self._climate_matrix

        rows = self._fetch_rows(SELECT_CLIMATE_CELLS_QUERY)
        row_count = len(rows)
        latitudes = np.empty(row_count, dtype=np.float32)
        longitudes = np.empty(row_count, dtype=np.float32)
        temperature_c = np.empty((row_count, MONTHS_PER_YEAR), dtype=np.float32)
        precipitation_mm = np.empty((row_count, MONTHS_PER_YEAR), dtype=np.float32)
        cloud_cover_pct = np.empty((row_count, MONTHS_PER_YEAR), dtype=np.uint8)

        try:
            for index, row in enumerate(rows):
                latitude, longitude, *monthly_values = row
                latitudes[index] = float(cast("int | float", latitude))
                longitudes[index] = float(cast("int | float", longitude))
                temperature_c[index] = tuple(
                    float(cast("int | float", value)) for value in monthly_values[:MONTHS_PER_YEAR]
                )
                precipitation_mm[index] = tuple(
                    float(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR : MONTHS_PER_YEAR * 2]
                )
                cloud_cover_pct[index] = tuple(
                    int(cast("int | float", value))
                    for value in monthly_values[MONTHS_PER_YEAR * 2 : MONTHS_PER_YEAR * 3]
                )
        except (TypeError, ValueError) as error:
            msg = f"Failed to map climate data from {self.database_path} into climate rows: {error}"
            raise ClimateDataError(msg) from error

        self._climate_matrix = ClimateMatrix(
            latitudes=latitudes,
            longitudes=longitudes,
            temperature_c=temperature_c,
            precipitation_mm=precipitation_mm,
            cloud_cover_pct=cloud_cover_pct,
        )
        return self._climate_matrix

    def get_indexed_cities(self) -> CityRankingCache:
        """Resolve every city to its climate-matrix row once and cache the result."""
        if self._indexed_cities is not None:
            return self._indexed_cities

        cities = self.list_cities()
        sorted_climate_keys, sorted_climate_indexes = self._get_sorted_climate_keys()
        resolved_cities: list[CityCandidate] = []
        climate_indexes: list[int] = []

        for city in cities:
            climate_key = coordinate_key(city.cell_lat, city.cell_lon)
            position = int(np.searchsorted(sorted_climate_keys, climate_key))
            if position >= len(sorted_climate_keys) or int(sorted_climate_keys[position]) != climate_key:
                continue
            resolved_cities.append(city)
            climate_indexes.append(int(sorted_climate_indexes[position]))

        self._indexed_cities = CityRankingCache.from_cities(
            tuple(resolved_cities),
            np.array(climate_indexes, dtype=np.int32),
        )
        return self._indexed_cities

    def get_heatmap_projection(self) -> HeatmapProjection:
        """Project the fixed climate grid into heatmap pixels once."""
        if self._heatmap_projection is not None:
            return self._heatmap_projection

        climate_matrix = self.get_climate_matrix()
        self._heatmap_projection = HeatmapProjection.from_coordinates(
            climate_matrix.latitudes, climate_matrix.longitudes
        )
        return self._heatmap_projection

    def probe_nearest_cell(self, lat: float, lon: float) -> int | None:
        """Return the climate-matrix index of the land cell nearest to (lat, lon).

        Returns None when the coordinates fall on ocean or outside the grid.
        """
        probe_city = CityCandidate(name="", country_code="", lat=lat, lon=lon, cell_lat=0.0, cell_lon=0.0)
        snapped_lat, snapped_lon = snap_city_to_cell_key(probe_city)
        key = coordinate_key(snapped_lat, snapped_lon)
        sorted_keys, sorted_indexes = self._get_sorted_climate_keys()
        pos = int(np.searchsorted(sorted_keys, key))
        if pos >= len(sorted_keys) or int(sorted_keys[pos]) != key:
            return None
        return int(sorted_indexes[pos])

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
        population_column_index = 6
        name, country_code, latitude, longitude, cell_latitude, cell_longitude = row[:6]
        population = int(cast("int | float", row[population_column_index])) if len(row) > population_column_index else 0
        return CityCandidate(
            name=str(cast("str", name)),
            country_code=str(cast("str", country_code)),
            lat=float(cast("int | float", latitude)),
            lon=float(cast("int | float", longitude)),
            cell_lat=float(cast("int | float", cell_latitude)),
            cell_lon=float(cast("int | float", cell_longitude)),
            population=population,
        )

    def _select_cities_query(self) -> str:
        columns = set(self._fetch_table_columns("cities"))
        selected_columns = list(CITY_BASE_COLUMNS)
        if "population" in columns:
            selected_columns.append("population")
        return "SELECT\n    " + ",\n    ".join(selected_columns) + "\nFROM cities"

    def _fetch_table_columns(self, table_name: str) -> tuple[str, ...]:
        if not self.database_path.exists():
            msg = f"Climate database file not found: {self.database_path}"
            raise ClimateDataError(msg)

        try:
            with duckdb.connect(str(self.database_path), read_only=True) as connection:
                rows = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        except duckdb.Error as error:
            if table_name == "cities" and "Table with name cities does not exist" in str(error):
                msg = (
                    f"Climate database file is missing the cities table: {self.database_path}. "
                    "Rebuild it with `uv run python scripts/build_climate_db.py`."
                )
                raise ClimateDataError(msg) from error
            msg = f"Failed to read climate data from {self.database_path}: {error}"
            raise ClimateDataError(msg) from error

        return tuple(str(cast("str", row[1])) for row in rows)

    def _get_sorted_climate_keys(
        self,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int], np.dtype[np.int32]]]:
        if self._sorted_climate_keys is not None and self._sorted_climate_indexes is not None:
            return self._sorted_climate_keys, self._sorted_climate_indexes

        climate_matrix = self.get_climate_matrix()
        climate_keys = np.fromiter(
            (
                coordinate_key(float(lat), float(lon))
                for lat, lon in zip(climate_matrix.latitudes, climate_matrix.longitudes, strict=True)
            ),
            dtype=np.int64,
            count=len(climate_matrix.latitudes),
        )
        self._sorted_climate_indexes = np.argsort(climate_keys).astype(np.int32, copy=False)
        self._sorted_climate_keys = climate_keys[self._sorted_climate_indexes]
        return self._sorted_climate_keys, self._sorted_climate_indexes
