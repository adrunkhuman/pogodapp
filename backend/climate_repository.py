from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, overload

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

    from numpy.typing import NDArray

MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
FLOAT32_DTYPE = np.dtype(np.float32)
UINT8_DTYPE = np.dtype(np.uint8)

SELECT_CLIMATE_CELLS_QUERY = """
SELECT
    lat,
    lon,
    t_jan, t_feb, t_mar, t_apr, t_may, t_jun, t_jul, t_aug, t_sep, t_oct, t_nov, t_dec,
    tmin_jan, tmin_feb, tmin_mar, tmin_apr, tmin_may, tmin_jun, tmin_jul, tmin_aug, tmin_sep, tmin_oct, tmin_nov, tmin_dec,
    tmax_jan, tmax_feb, tmax_mar, tmax_apr, tmax_may, tmax_jun, tmax_jul, tmax_aug, tmax_sep, tmax_oct, tmax_nov, tmax_dec,
    prec_jan, prec_feb, prec_mar, prec_apr, prec_may, prec_jun, prec_jul, prec_aug, prec_sep, prec_oct, prec_nov, prec_dec,
    cloud_jan, cloud_feb, cloud_mar, cloud_apr, cloud_may, cloud_jun, cloud_jul, cloud_aug, cloud_sep, cloud_oct, cloud_nov, cloud_dec
FROM climate_cells
"""

TEMPERATURE_COLUMNS = tuple(f"t_{month}" for month in MONTH_NAMES)
TEMPERATURE_MIN_COLUMNS = tuple(f"tmin_{month}" for month in MONTH_NAMES)
TEMPERATURE_MAX_COLUMNS = tuple(f"tmax_{month}" for month in MONTH_NAMES)
PRECIPITATION_COLUMNS = tuple(f"prec_{month}" for month in MONTH_NAMES)
CLOUD_COLUMNS = tuple(f"cloud_{month}" for month in MONTH_NAMES)
TEMPERATURE_MAX_LIST = f"list_value({', '.join(TEMPERATURE_MAX_COLUMNS)})"
PRECIPITATION_LIST = f"list_value({', '.join(PRECIPITATION_COLUMNS)})"
CLOUD_LIST = f"list_value({', '.join(CLOUD_COLUMNS)})"
AVERAGE_CLOUD_EXPR = f"list_avg({CLOUD_LIST})"
AVERAGE_CLOUD_BANKERS_ROUND_EXPR = (
    f"CAST(floor({AVERAGE_CLOUD_EXPR}) + CASE "
    f"WHEN {AVERAGE_CLOUD_EXPR} - floor({AVERAGE_CLOUD_EXPR}) > 0.5 THEN 1 "
    f"WHEN {AVERAGE_CLOUD_EXPR} - floor({AVERAGE_CLOUD_EXPR}) < 0.5 THEN 0 "
    f"WHEN mod(CAST(floor({AVERAGE_CLOUD_EXPR}) AS BIGINT), 2) = 0 THEN 0 "
    f"ELSE 1 END AS FLOAT)"
)
SELECT_CLIMATE_MATRIX_QUERY = (
    f"SELECT\n"
    f"    lat,\n"
    f"    lon,\n"
    f"    list_median({TEMPERATURE_MAX_LIST}) AS typical_highs_c,\n"
    f"    greatest({', '.join(TEMPERATURE_MAX_COLUMNS)}) AS hottest_month_highs_c,\n"
    f"    least({', '.join(TEMPERATURE_MIN_COLUMNS)}) AS coldest_month_lows_c,\n"
    f"    list_median({PRECIPITATION_LIST}) AS median_precipitation_mm,\n"
    f"    greatest({', '.join(PRECIPITATION_COLUMNS)}) AS wettest_precipitation_mm,\n"
    f"    {AVERAGE_CLOUD_BANKERS_ROUND_EXPR} AS average_cloud_cover_pct,\n"
    f"    CAST(greatest({', '.join(CLOUD_COLUMNS)}) AS FLOAT) AS gloomiest_cloud_cover_pct\n"
    f"FROM climate_cells"
)
SELECT_PROBE_CLIMATE_CELL_QUERY = SELECT_CLIMATE_CELLS_QUERY + "\nWHERE round(lat, 4) = ? AND round(lon, 4) = ?"

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
        """Return compact score-time arrays ready for vectorized scoring.

        DuckDB-backed repositories may omit all monthly arrays and keep only
        yearly scoring aggregates resident; callers that need monthly probe
        data must use `get_probe_cell()`.
        """

    def get_indexed_cities(self) -> CityRankingCache:
        """Return ranking-ready cities aligned with the current climate-matrix row order."""

    def get_heatmap_projection(self) -> HeatmapProjection:
        """Return cached heatmap pixel coordinates aligned with the current climate matrix."""

    def get_probe_cell(self, row_index: int) -> ClimateCell:
        """Return one full climate row for probe rendering."""


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

    def get_probe_cell(self, row_index: int) -> ClimateCell:
        """Return one stub climate row by row index."""
        return STUB_CLIMATE_CELLS[row_index]


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
        except (AssertionError, TypeError, ValueError) as error:
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
        except (AssertionError, TypeError, ValueError) as error:
            msg = f"Failed to map city data from {self.database_path} into city rows: {error}"
            raise ClimateDataError(msg) from error

        return self._cities

    def get_climate_matrix(self) -> ClimateMatrix:
        """Load the score-time yearly aggregates once into compact arrays.

        Raises:
            ClimateDataError: The database file is missing, unreadable, or does
                not match the runtime matrix contract.
        """
        if self._climate_matrix is not None:
            return self._climate_matrix

        try:
            columns = self._fetch_numpy_columns(SELECT_CLIMATE_MATRIX_QUERY)
            latitudes = np.asarray(columns["lat"], dtype=FLOAT32_DTYPE)
            longitudes = np.asarray(columns["lon"], dtype=FLOAT32_DTYPE)
            typical_highs_c = np.asarray(columns["typical_highs_c"], dtype=FLOAT32_DTYPE)
            hottest_month_highs_c = np.asarray(columns["hottest_month_highs_c"], dtype=FLOAT32_DTYPE)
            coldest_month_lows_c = np.asarray(columns["coldest_month_lows_c"], dtype=FLOAT32_DTYPE)
            median_precipitation_mm = np.asarray(columns["median_precipitation_mm"], dtype=FLOAT32_DTYPE)
            wettest_precipitation_mm = np.asarray(columns["wettest_precipitation_mm"], dtype=FLOAT32_DTYPE)
            average_cloud_cover_pct = np.asarray(columns["average_cloud_cover_pct"], dtype=UINT8_DTYPE)
            gloomiest_cloud_cover_pct = np.asarray(columns["gloomiest_cloud_cover_pct"], dtype=UINT8_DTYPE)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            msg = f"Failed to map climate data from {self.database_path} into climate rows: {error}"
            raise ClimateDataError(msg) from error

        self._climate_matrix = ClimateMatrix(
            latitudes=latitudes,
            longitudes=longitudes,
            temperature_c=None,
            typical_highs_c=typical_highs_c,
            hottest_month_highs_c=hottest_month_highs_c,
            coldest_month_lows_c=coldest_month_lows_c,
            median_precipitation_mm=median_precipitation_mm,
            wettest_precipitation_mm=wettest_precipitation_mm,
            average_cloud_cover_pct=average_cloud_cover_pct,
            gloomiest_cloud_cover_pct=gloomiest_cloud_cover_pct,
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

    def get_probe_cell(self, row_index: int) -> ClimateCell:
        """Fetch one full climate row on demand for `/probe`."""
        climate_matrix = self.get_climate_matrix()
        probe_lat = round(float(climate_matrix.latitudes[row_index]), 4)
        probe_lon = round(float(climate_matrix.longitudes[row_index]), 4)
        rows = self._fetch_rows(SELECT_PROBE_CLIMATE_CELL_QUERY, parameters=(probe_lat, probe_lon))
        if not rows:
            msg = f"Failed to read climate data from {self.database_path}: probe row not found for {probe_lat}, {probe_lon}"
            raise ClimateDataError(msg)

        try:
            return self._row_to_climate_cell(rows[0])
        except (AssertionError, TypeError, ValueError) as error:
            msg = f"Failed to map climate data from {self.database_path} into climate rows: {error}"
            raise ClimateDataError(msg) from error

    def get_runtime_cache_stats(self) -> dict[str, float]:
        """Return approximate resident cache sizes for preload logging."""
        climate_matrix_mb = 0.0
        if self._climate_matrix is not None:
            climate_matrix_mb = round(self._climate_matrix_nbytes(self._climate_matrix) / (1024 * 1024), 1)

        city_cache_mb = 0.0
        if self._indexed_cities is not None:
            city_cache_mb = round(self._city_cache_nbytes(self._indexed_cities) / (1024 * 1024), 1)

        probe_lookup_mb = 0.0
        if self._sorted_climate_keys is not None and self._sorted_climate_indexes is not None:
            probe_lookup_mb = round(self._probe_lookup_nbytes() / (1024 * 1024), 1)

        heatmap_projection_mb = 0.0
        if self._heatmap_projection is not None:
            heatmap_projection_mb = round(self._heatmap_projection_nbytes(self._heatmap_projection) / (1024 * 1024), 1)

        return {
            "climate_matrix_mb": climate_matrix_mb,
            "city_cache_mb": city_cache_mb,
            "probe_lookup_mb": probe_lookup_mb,
            "heatmap_projection_mb": heatmap_projection_mb,
            "runtime_cache_mb": round(climate_matrix_mb + city_cache_mb + probe_lookup_mb + heatmap_projection_mb, 1),
        }

    def _fetch_rows(
        self,
        query: str,
        *,
        table_name: str = "climate_cells",
        parameters: tuple[object, ...] = (),
    ) -> list[tuple[object, ...]]:
        if not self.database_path.exists():
            msg = f"Climate database file not found: {self.database_path}"
            raise ClimateDataError(msg)

        try:
            with duckdb.connect(str(self.database_path), read_only=True) as connection:
                return connection.execute(query, parameters).fetchall()
        except duckdb.Error as error:
            if table_name == "cities" and "Table with name cities does not exist" in str(error):
                msg = (
                    f"Climate database file is missing the cities table: {self.database_path}. "
                    "Rebuild it with `uv run python scripts/build_climate_db.py`."
                )
                raise ClimateDataError(msg) from error
            msg = f"Failed to read climate data from {self.database_path}: {error}"
            raise ClimateDataError(msg) from error

    def _fetch_numpy_columns(self, query: str) -> dict[str, NDArray[np.generic]]:
        if not self.database_path.exists():
            msg = f"Climate database file not found: {self.database_path}"
            raise ClimateDataError(msg)

        try:
            with duckdb.connect(str(self.database_path), read_only=True) as connection:
                return cast("dict[str, NDArray[np.generic]]", connection.execute(query).fetchnumpy())
        except duckdb.Error as error:
            msg = f"Failed to read climate data from {self.database_path}: {error}"
            raise ClimateDataError(msg) from error

    @overload
    def _build_monthly_matrix(
        self,
        columns: dict[str, NDArray[np.generic]],
        month_columns: tuple[str, ...],
        *,
        dtype: np.dtype[np.float32],
    ) -> NDArray[np.float32]: ...

    @overload
    def _build_monthly_matrix(
        self,
        columns: dict[str, NDArray[np.generic]],
        month_columns: tuple[str, ...],
        *,
        dtype: np.dtype[np.uint8],
    ) -> NDArray[np.uint8]: ...

    def _build_monthly_matrix(
        self,
        columns: dict[str, NDArray[np.generic]],
        month_columns: tuple[str, ...],
        *,
        dtype: np.dtype[np.float32] | np.dtype[np.uint8],
    ) -> NDArray[np.float32] | NDArray[np.uint8]:
        row_count = len(columns[month_columns[0]])
        matrix = np.empty((row_count, MONTHS_PER_YEAR), dtype=dtype)
        for month_index, column_name in enumerate(month_columns):
            matrix[:, month_index] = np.asarray(columns[column_name], dtype=dtype)
        return cast("NDArray[np.float32] | NDArray[np.uint8]", matrix)

    def _row_to_climate_cell(self, row: tuple[object, ...]) -> ClimateCell:
        """Convert one database row into the in-memory scoring shape."""
        latitude, longitude, *monthly_values = row
        return ClimateCell(
            lat=float(cast("int | float", latitude)),
            lon=float(cast("int | float", longitude)),
            temperature_c=tuple(float(cast("int | float", value)) for value in monthly_values[:MONTHS_PER_YEAR]),
            temperature_min_c=tuple(
                float(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR : MONTHS_PER_YEAR * 2]
            ),
            temperature_max_c=tuple(
                float(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR * 2 : MONTHS_PER_YEAR * 3]
            ),
            precipitation_mm=tuple(
                float(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR * 3 : MONTHS_PER_YEAR * 4]
            ),
            cloud_cover_pct=tuple(
                int(cast("int | float", value)) for value in monthly_values[MONTHS_PER_YEAR * 4 : MONTHS_PER_YEAR * 5]
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

    def _climate_matrix_nbytes(self, climate_matrix: ClimateMatrix) -> int:
        total = (
            climate_matrix.latitudes.nbytes
            + climate_matrix.longitudes.nbytes
            + climate_matrix.typical_highs_c.nbytes
            + climate_matrix.hottest_month_highs_c.nbytes
            + climate_matrix.coldest_month_lows_c.nbytes
            + climate_matrix.median_precipitation_mm.nbytes
            + climate_matrix.wettest_precipitation_mm.nbytes
            + climate_matrix.average_cloud_cover_pct.nbytes
            + climate_matrix.gloomiest_cloud_cover_pct.nbytes
        )
        if climate_matrix.temperature_c is not None:
            total += climate_matrix.temperature_c.nbytes
        if climate_matrix.temperature_min_c is not None:
            total += climate_matrix.temperature_min_c.nbytes
        if climate_matrix.temperature_max_c is not None:
            total += climate_matrix.temperature_max_c.nbytes
        if climate_matrix.precipitation_mm is not None:
            total += climate_matrix.precipitation_mm.nbytes
        if climate_matrix.cloud_cover_pct is not None:
            total += climate_matrix.cloud_cover_pct.nbytes
        return total

    def _city_cache_nbytes(self, city_cache: CityRankingCache) -> int:
        return (
            city_cache.climate_indexes.nbytes
            + city_cache.latitude_radians.nbytes
            + city_cache.longitude_radians.nbytes
            + city_cache.cosine_latitudes.nbytes
        )

    def _probe_lookup_nbytes(self) -> int:
        if self._sorted_climate_keys is None or self._sorted_climate_indexes is None:
            return 0
        return self._sorted_climate_keys.nbytes + self._sorted_climate_indexes.nbytes

    def _heatmap_projection_nbytes(self, heatmap_projection: HeatmapProjection) -> int:
        return (
            heatmap_projection.score_indexes.nbytes
            + heatmap_projection.xs.nbytes
            + heatmap_projection.ys.nbytes
            + heatmap_projection.work_indexes.nbytes
            + heatmap_projection.land_mask.nbytes
        )
