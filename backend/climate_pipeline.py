from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.request import urlopen

import duckdb
import numpy as np
import tifffile

if TYPE_CHECKING:
    from numpy.typing import NDArray

from backend.cities import CityCandidate, snap_city_to_cell_key

MONTH_NAMES: Final[tuple[str, ...]] = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)
GEONAMES_CITIES_URL: Final[str] = "https://download.geonames.org/export/dump/cities15000.zip"
# WorldClim stores ocean pixels as ≈ -3.4e38 (np.finfo(np.float32).min).
# Any value below this cutoff is nodata; no real climate measurement comes close to -1e20.
NODATA_CUTOFF: Final[float] = -1e20
MONTHS_PER_YEAR: Final[int] = 12
TEMPERATURE_COLUMNS: Final[tuple[str, ...]] = tuple(f"t_{month_name}" for month_name in MONTH_NAMES)
PRECIPITATION_COLUMNS: Final[tuple[str, ...]] = tuple(f"prec_{month_name}" for month_name in MONTH_NAMES)
CLOUD_COLUMNS: Final[tuple[str, ...]] = tuple(f"cloud_{month_name}" for month_name in MONTH_NAMES)
EXPECTED_CLIMATE_COLUMNS: Final[tuple[str, ...]] = (
    "lat",
    "lon",
    *TEMPERATURE_COLUMNS,
    *PRECIPITATION_COLUMNS,
    *CLOUD_COLUMNS,
)
EXPECTED_CITY_COLUMNS: Final[tuple[str, ...]] = (
    "name",
    "country_code",
    "lat",
    "lon",
    "cell_lat",
    "cell_lon",
    "population",
)
INSERT_CLIMATE_CELL_QUERY: Final[str] = (
    "INSERT INTO climate_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
INSERT_CITY_QUERY: Final[str] = "INSERT INTO cities VALUES (?, ?, ?, ?, ?, ?, ?)"


@dataclass(frozen=True, slots=True)
class WorldClimResolution:
    """One supported WorldClim grid resolution and its build-time assumptions."""

    name: str
    grid_degrees: float
    raster_shape: tuple[int, int]
    rough_row_count_range: tuple[int, int]

    @property
    def archive_urls(self) -> dict[str, str]:
        """Return the three source archives needed for this WorldClim resolution."""
        return {
            variable_name: f"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_{self.name}_{variable_name}.zip"
            for variable_name in ("tavg", "prec", "srad")
        }


WORLDCLIM_RESOLUTIONS: Final[dict[str, WorldClimResolution]] = {
    "10m": WorldClimResolution(
        name="10m",
        grid_degrees=10 / 60,
        raster_shape=(1080, 2160),
        rough_row_count_range=(780_000, 840_000),
    ),
    "5m": WorldClimResolution(
        name="5m",
        grid_degrees=5 / 60,
        raster_shape=(2160, 4320),
        rough_row_count_range=(3_100_000, 3_350_000),
    ),
    "2.5m": WorldClimResolution(
        name="2.5m",
        grid_degrees=2.5 / 60,
        raster_shape=(4320, 8640),
        rough_row_count_range=(12_500_000, 13_400_000),
    ),
    "30s": WorldClimResolution(
        name="30s",
        grid_degrees=30 / 3600,
        raster_shape=(21600, 43200),
        rough_row_count_range=(315_000_000, 332_000_000),
    ),
}
DEFAULT_WORLDCLIM_RESOLUTION: Final[WorldClimResolution] = WORLDCLIM_RESOLUTIONS["5m"]


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Report the output of one pipeline run."""

    output_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Report schema and row-count checks for a built database."""

    row_count: int
    columns: tuple[str, ...]
    city_count: int
    city_columns: tuple[str, ...]


def ensure_geonames_cities(cache_dir: Path) -> Path:
    """Download and extract the GeoNames cities15000 catalog once into the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / Path(GEONAMES_CITIES_URL).name
    extracted_path = cache_dir / "cities15000.txt"

    if not archive_path.exists():
        with urlopen(GEONAMES_CITIES_URL) as response, archive_path.open("wb") as destination:  # noqa: S310
            shutil.copyfileobj(response, destination)

    if not extracted_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract("cities15000.txt", path=cache_dir)

    return extracted_path


def load_city_catalog(cities_txt_path: Path) -> tuple[CityCandidate, ...]:
    """Read the GeoNames cities15000 dump that feeds the build-time city mapping."""
    population_column_index = 14
    with cities_txt_path.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(source, delimiter="\t")
        return tuple(
            CityCandidate(
                name=row[1],
                country_code=row[8],
                lat=float(row[4]),
                lon=float(row[5]),
                cell_lat=0.0,
                cell_lon=0.0,
                population=int(row[population_column_index])
                if len(row) > population_column_index and row[population_column_index]
                else 0,
            )
            for row in rows
        )


def solar_radiation_to_cloud_proxy(monthly_solar_radiation: NDArray[np.float64]) -> NDArray[np.int16]:
    """Invert solar radiation into a temporary 0..100 cloud proxy until a direct cloud source is chosen."""
    valid_values = monthly_solar_radiation[np.isfinite(monthly_solar_radiation)]
    lower_bound, upper_bound = np.percentile(valid_values, [5, 95])
    if lower_bound == upper_bound:
        return np.full(monthly_solar_radiation.shape, 50, dtype=np.int16)

    scaled = np.clip((monthly_solar_radiation - lower_bound) / (upper_bound - lower_bound), 0.0, 1.0)
    return np.rint((1.0 - scaled) * 100).astype(np.int16)


def build_coordinate_grids(resolution: WorldClimResolution) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return lat/lon center coordinates for one native WorldClim grid."""
    half = resolution.grid_degrees / 2
    latitudes = np.linspace(90 - half, -90 + half, num=resolution.raster_shape[0], dtype=np.float64)
    longitudes = np.linspace(-180 + half, 180 - half, num=resolution.raster_shape[1], dtype=np.float64)
    return np.meshgrid(latitudes, longitudes, indexing="ij")


def ensure_worldclim_archives(cache_dir: Path, resolution: WorldClimResolution) -> dict[str, Path]:
    """Download the three source archives once into the local cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(resolution.archive_urls)) as executor:
        archive_paths = executor.map(
            lambda item: ensure_worldclim_archive(cache_dir, *item), resolution.archive_urls.items()
        )

    return dict(archive_paths)


def ensure_worldclim_archive(cache_dir: Path, variable_name: str, url: str) -> tuple[str, Path]:
    """Download one source archive when it is missing from the local cache."""
    archive_path = cache_dir / Path(url).name
    if not archive_path.exists():
        with urlopen(url) as response, archive_path.open("wb") as destination:  # noqa: S310
            shutil.copyfileobj(response, destination)
    return variable_name, archive_path


def extract_worldclim_archives(archive_paths: dict[str, Path], extracted_dir: Path) -> dict[str, tuple[Path, ...]]:
    """Extract source GeoTIFFs and return them ordered by month."""
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: dict[str, tuple[Path, ...]] = {}
    for variable_name, archive_path in archive_paths.items():
        variable_dir = extracted_dir / variable_name
        variable_dir.mkdir(parents=True, exist_ok=True)
        tif_paths = tuple(sorted(variable_dir.glob("*.tif")))
        if len(tif_paths) != MONTHS_PER_YEAR:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(variable_dir)
            tif_paths = tuple(sorted(variable_dir.glob("*.tif")))
        if len(tif_paths) != MONTHS_PER_YEAR:
            msg = f"Expected {MONTHS_PER_YEAR} monthly rasters for {variable_name}, found {len(tif_paths)}"
            raise ValueError(msg)
        extracted_paths[variable_name] = tif_paths
    return extracted_paths


def load_raster(path: Path) -> NDArray[np.float64]:
    """Read one WorldClim GeoTIFF, converting ocean nodata pixels to NaN.

    tifffile does not honour TIFF nodata metadata, so ocean pixels arrive as
    ≈ -3.4e38 (np.finfo(np.float32).min). Masking them here lets the rest of
    the pipeline use np.isfinite as a clean land filter.
    """
    logging.getLogger("tifffile").setLevel(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = tifffile.imread(path).astype(np.float64)
    raw[raw <= NODATA_CUTOFF] = np.nan
    return raw


def load_monthly_rasters(
    extracted_paths: dict[str, tuple[Path, ...]],
) -> dict[str, tuple[NDArray[np.float64], ...]]:
    """Read source GeoTIFFs at the selected native WorldClim resolution."""
    monthly: dict[str, tuple[NDArray[np.float64], ...]] = {}
    for variable_name, tif_paths in extracted_paths.items():
        with ThreadPoolExecutor(max_workers=4) as executor:
            monthly[variable_name] = tuple(executor.map(load_raster, tif_paths))
    return monthly


def build_insert_rows(
    monthly_temperature: tuple[NDArray[np.float64], ...],
    monthly_precipitation: tuple[NDArray[np.float64], ...],
    monthly_solar_radiation: tuple[NDArray[np.float64], ...],
    resolution: WorldClimResolution,
) -> list[tuple[float | int, ...]]:
    """Flatten monthly rasters into the DuckDB row shape.

    Only cells with finite data for every month across all three variables are
    kept. That makes the runtime table a pure land-cell dataset with no partial
    yearly rows to special-case later.
    """
    finite_mask = np.logical_and.reduce(
        [
            *[np.isfinite(month) for month in monthly_temperature],
            *[np.isfinite(month) for month in monthly_precipitation],
            *[np.isfinite(month) for month in monthly_solar_radiation],
        ]
    )
    latitudes, longitudes = build_coordinate_grids(resolution)
    flat_mask = finite_mask.ravel()
    column_vectors: list[list[float | int]] = [
        latitudes.ravel()[flat_mask].round(4).tolist(),
        longitudes.ravel()[flat_mask].round(4).tolist(),
    ]

    column_vectors.extend(month.ravel()[flat_mask].round(4).tolist() for month in monthly_temperature)
    column_vectors.extend(month.ravel()[flat_mask].round(4).tolist() for month in monthly_precipitation)
    column_vectors.extend(
        solar_radiation_to_cloud_proxy(month).ravel()[flat_mask].astype(int).tolist()
        for month in monthly_solar_radiation
    )

    return list(zip(*column_vectors, strict=True))


def create_climate_cells_table(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the normalized monthly climate table expected by runtime scoring."""
    connection.execute("DROP TABLE IF EXISTS climate_cells")
    connection.execute(
        """
        CREATE TABLE climate_cells (
            lat DOUBLE NOT NULL,
            lon DOUBLE NOT NULL,
            t_jan DOUBLE NOT NULL,
            t_feb DOUBLE NOT NULL,
            t_mar DOUBLE NOT NULL,
            t_apr DOUBLE NOT NULL,
            t_may DOUBLE NOT NULL,
            t_jun DOUBLE NOT NULL,
            t_jul DOUBLE NOT NULL,
            t_aug DOUBLE NOT NULL,
            t_sep DOUBLE NOT NULL,
            t_oct DOUBLE NOT NULL,
            t_nov DOUBLE NOT NULL,
            t_dec DOUBLE NOT NULL,
            prec_jan DOUBLE NOT NULL,
            prec_feb DOUBLE NOT NULL,
            prec_mar DOUBLE NOT NULL,
            prec_apr DOUBLE NOT NULL,
            prec_may DOUBLE NOT NULL,
            prec_jun DOUBLE NOT NULL,
            prec_jul DOUBLE NOT NULL,
            prec_aug DOUBLE NOT NULL,
            prec_sep DOUBLE NOT NULL,
            prec_oct DOUBLE NOT NULL,
            prec_nov DOUBLE NOT NULL,
            prec_dec DOUBLE NOT NULL,
            cloud_jan INTEGER NOT NULL,
            cloud_feb INTEGER NOT NULL,
            cloud_mar INTEGER NOT NULL,
            cloud_apr INTEGER NOT NULL,
            cloud_may INTEGER NOT NULL,
            cloud_jun INTEGER NOT NULL,
            cloud_jul INTEGER NOT NULL,
            cloud_aug INTEGER NOT NULL,
            cloud_sep INTEGER NOT NULL,
            cloud_oct INTEGER NOT NULL,
            cloud_nov INTEGER NOT NULL,
            cloud_dec INTEGER NOT NULL
        )
        """
    )


def copy_rows_into_climate_table(connection: duckdb.DuckDBPyConnection, rows: list[tuple[float | int, ...]]) -> None:
    """Bulk-load climate rows through a temporary CSV to avoid slow row-by-row inserts."""
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, suffix=".csv") as temporary_file:
        csv_path = Path(temporary_file.name)
        csv.writer(temporary_file).writerows(rows)

    try:
        copy_query = f"COPY climate_cells FROM '{csv_path.as_posix()}'"
        connection.execute(copy_query)
    finally:
        csv_path.unlink(missing_ok=True)


def build_city_rows(
    cities_txt_path: Path, climate_rows: list[tuple[float | int, ...]], resolution: WorldClimResolution
) -> list[tuple[str | float, ...]]:
    """Keep only cities that snap onto valid land climate cells."""
    valid_cells = {(float(row[0]), float(row[1])) for row in climate_rows}
    city_rows: list[tuple[str | float, ...]] = []

    for city in load_city_catalog(cities_txt_path):
        cell_lat, cell_lon = snap_city_to_cell_key(city, grid_degrees=resolution.grid_degrees)
        if (cell_lat, cell_lon) not in valid_cells:
            continue

        city_rows.append((city.name, city.country_code, city.lat, city.lon, cell_lat, cell_lon, city.population))

    return city_rows


def create_cities_table(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the build-time city lookup table used by runtime ranking."""
    connection.execute("DROP TABLE IF EXISTS cities")
    connection.execute(
        """
        CREATE TABLE cities (
            name VARCHAR NOT NULL,
            country_code VARCHAR NOT NULL,
            lat DOUBLE NOT NULL,
            lon DOUBLE NOT NULL,
            cell_lat DOUBLE NOT NULL,
            cell_lon DOUBLE NOT NULL,
            population INTEGER NOT NULL
        )
        """
    )


def copy_rows_into_cities_table(connection: duckdb.DuckDBPyConnection, rows: list[tuple[str | float, ...]]) -> None:
    """Bulk-load city rows through a temporary CSV like the climate table path."""
    with tempfile.NamedTemporaryFile("w", newline="", delete=False, suffix=".csv", encoding="utf-8") as temporary_file:
        csv_path = Path(temporary_file.name)
        csv.writer(temporary_file).writerows(rows)

    try:
        copy_query = f"COPY cities FROM '{csv_path.as_posix()}'"
        connection.execute(copy_query)
    finally:
        csv_path.unlink(missing_ok=True)


def build_worldclim_database(
    output_path: Path, cache_dir: Path, *, resolution: WorldClimResolution = DEFAULT_WORLDCLIM_RESOLUTION
) -> BuildSummary:
    """Download, build, and overwrite the runtime DuckDB climate artifact.

    This performs network I/O, writes cache files under `cache_dir`, and
    replaces `output_path` if it already exists.
    """
    resolution_cache_dir = cache_dir / resolution.name
    archive_paths = ensure_worldclim_archives(resolution_cache_dir, resolution)
    cities_txt_path = ensure_geonames_cities(cache_dir)
    extracted_paths = extract_worldclim_archives(archive_paths, resolution_cache_dir / "extracted")
    monthly = load_monthly_rasters(extracted_paths)
    rows = build_insert_rows(monthly["tavg"], monthly["prec"], monthly["srad"], resolution)
    city_rows = build_city_rows(cities_txt_path, rows, resolution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with duckdb.connect(str(output_path)) as connection:
        create_climate_cells_table(connection)
        copy_rows_into_climate_table(connection, rows)
        create_cities_table(connection)
        copy_rows_into_cities_table(connection, city_rows)

    return BuildSummary(output_path=output_path, row_count=len(rows))


def validate_climate_database(
    database_path: Path, *, resolution: WorldClimResolution = DEFAULT_WORLDCLIM_RESOLUTION
) -> ValidationSummary:
    """Check that the built artifact matches the expected runtime contract."""
    return validate_climate_database_with_row_range(database_path, resolution.rough_row_count_range)


def validate_climate_database_with_row_range(
    database_path: Path,
    expected_row_count_range: tuple[int, int],
) -> ValidationSummary:
    """Validate runtime schema plus rough climate/city row-count expectations.

    Raises:
        ValueError: The schema, row counts, or city import output drifted from
            the expected build contract.
        TypeError: DuckDB returned unexpected count value types.
    """
    with duckdb.connect(str(database_path), read_only=True) as connection:
        columns = tuple(
            column_name for _, column_name, *_ in connection.execute("PRAGMA table_info('climate_cells')").fetchall()
        )
        if columns != EXPECTED_CLIMATE_COLUMNS:
            msg = f"Unexpected climate_cells schema: {columns}"
            raise ValueError(msg)
        city_columns = tuple(
            column_name for _, column_name, *_ in connection.execute("PRAGMA table_info('cities')").fetchall()
        )
        if city_columns != EXPECTED_CITY_COLUMNS:
            msg = f"Unexpected cities schema: {city_columns}"
            raise ValueError(msg)
        row_count_result = connection.execute("SELECT COUNT(*) FROM climate_cells").fetchone()
        city_count_result = connection.execute("SELECT COUNT(*) FROM cities").fetchone()

    if row_count_result is None:
        msg = "Failed to fetch climate row count"
        raise ValueError(msg)
    if city_count_result is None:
        msg = "Failed to fetch city row count"
        raise ValueError(msg)

    row_count = row_count_result[0]
    city_count = city_count_result[0]

    if not isinstance(row_count, int):
        msg = f"Unexpected row count type: {type(row_count)!r}"
        raise TypeError(msg)
    if not isinstance(city_count, int):
        msg = f"Unexpected city count type: {type(city_count)!r}"
        raise TypeError(msg)
    if city_count == 0:
        msg = "City count is zero; GeoNames import likely failed"
        raise ValueError(msg)

    minimum_rows, maximum_rows = expected_row_count_range
    if not minimum_rows <= row_count <= maximum_rows:
        msg = f"Row count {row_count} outside expected rough range {minimum_rows}..{maximum_rows}"
        raise ValueError(msg)

    return ValidationSummary(row_count=row_count, columns=columns, city_count=city_count, city_columns=city_columns)


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments for the one-shot pipeline."""
    parser = argparse.ArgumentParser(description="Build the prototype WorldClim climate.duckdb dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "climate.duckdb",
        help="Path for the generated DuckDB file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data") / "worldclim",
        help="Cache directory for downloaded WorldClim source archives.",
    )
    parser.add_argument(
        "--resolution",
        choices=tuple(WORLDCLIM_RESOLUTIONS),
        default=DEFAULT_WORLDCLIM_RESOLUTION.name,
        help="Native WorldClim grid resolution to build.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full build and validation flow from the command line."""
    args = parse_args()
    resolution = WORLDCLIM_RESOLUTIONS[args.resolution]
    summary = build_worldclim_database(output_path=args.output, cache_dir=args.cache_dir, resolution=resolution)
    validation = validate_climate_database(summary.output_path, resolution=resolution)
    print(f"Built {summary.output_path} with {validation.row_count} rows.")


if __name__ == "__main__":
    main()
