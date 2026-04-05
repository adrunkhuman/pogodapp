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
WORLDCLIM_ARCHIVES: Final[dict[str, str]] = {
    "tavg": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_tavg.zip",
    "prec": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_prec.zip",
    "srad": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_srad.zip",
}
GRID_DEGREES: Final[float] = 10 / 60
# WorldClim stores ocean pixels as ≈ -3.4e38 (np.finfo(np.float32).min).
# Any value below this cutoff is nodata; no real climate measurement comes close to -1e20.
NODATA_CUTOFF: Final[float] = -1e20
RASTER_SHAPE: Final[tuple[int, int]] = (1080, 2160)
MONTHS_PER_YEAR: Final[int] = 12
# WorldClim 10' land coverage (including coastal pixels) is ~808k cells — more than the
# ~29% dry-land-area fraction because small islands and coastal strips are included.
ROUGH_ROW_COUNT_RANGE: Final[tuple[int, int]] = (780_000, 840_000)
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
INSERT_CLIMATE_CELL_QUERY: Final[str] = (
    "INSERT INTO climate_cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


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


def solar_radiation_to_cloud_proxy(monthly_solar_radiation: NDArray[np.float64]) -> NDArray[np.int16]:
    """Invert solar radiation into a temporary 0..100 cloud proxy until a direct cloud source is chosen."""
    valid_values = monthly_solar_radiation[np.isfinite(monthly_solar_radiation)]
    lower_bound, upper_bound = np.percentile(valid_values, [5, 95])
    if lower_bound == upper_bound:
        return np.full(monthly_solar_radiation.shape, 50, dtype=np.int16)

    scaled = np.clip((monthly_solar_radiation - lower_bound) / (upper_bound - lower_bound), 0.0, 1.0)
    return np.rint((1.0 - scaled) * 100).astype(np.int16)


def build_coordinate_grids() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return lat/lon center coordinates for the native 10-arcminute grid."""
    half = GRID_DEGREES / 2
    latitudes = np.linspace(90 - half, -90 + half, num=RASTER_SHAPE[0], dtype=np.float64)
    longitudes = np.linspace(-180 + half, 180 - half, num=RASTER_SHAPE[1], dtype=np.float64)
    return np.meshgrid(latitudes, longitudes, indexing="ij")


def ensure_worldclim_archives(cache_dir: Path) -> dict[str, Path]:
    """Download the three source archives once into the local cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(WORLDCLIM_ARCHIVES)) as executor:
        archive_paths = executor.map(
            lambda item: ensure_worldclim_archive(cache_dir, *item), WORLDCLIM_ARCHIVES.items()
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
    """Read source GeoTIFFs at native 10-arcminute resolution."""
    monthly: dict[str, tuple[NDArray[np.float64], ...]] = {}
    for variable_name, tif_paths in extracted_paths.items():
        with ThreadPoolExecutor(max_workers=4) as executor:
            monthly[variable_name] = tuple(executor.map(load_raster, tif_paths))
    return monthly


def build_insert_rows(
    monthly_temperature: tuple[NDArray[np.float64], ...],
    monthly_precipitation: tuple[NDArray[np.float64], ...],
    monthly_solar_radiation: tuple[NDArray[np.float64], ...],
) -> list[tuple[float | int, ...]]:
    """Flatten aggregated monthly grids into the DuckDB row shape."""
    finite_mask = np.logical_and.reduce(
        [
            *[np.isfinite(month) for month in monthly_temperature],
            *[np.isfinite(month) for month in monthly_precipitation],
            *[np.isfinite(month) for month in monthly_solar_radiation],
        ]
    )
    latitudes, longitudes = build_coordinate_grids()
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


def build_worldclim_database(output_path: Path, cache_dir: Path) -> BuildSummary:
    """Download and persist the 10-arcminute climate dataset."""
    archive_paths = ensure_worldclim_archives(cache_dir)
    extracted_paths = extract_worldclim_archives(archive_paths, cache_dir / "extracted")
    monthly = load_monthly_rasters(extracted_paths)
    rows = build_insert_rows(monthly["tavg"], monthly["prec"], monthly["srad"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with duckdb.connect(str(output_path)) as connection:
        create_climate_cells_table(connection)
        copy_rows_into_climate_table(connection, rows)

    return BuildSummary(output_path=output_path, row_count=len(rows))


def validate_climate_database(database_path: Path) -> ValidationSummary:
    """Check that the built artifact matches the expected prototype contract."""
    return validate_climate_database_with_row_range(database_path, ROUGH_ROW_COUNT_RANGE)


def validate_climate_database_with_row_range(
    database_path: Path,
    expected_row_count_range: tuple[int, int],
) -> ValidationSummary:
    """Check that the built artifact matches the expected schema and a supplied row-count range."""
    with duckdb.connect(str(database_path), read_only=True) as connection:
        columns = tuple(
            column_name for _, column_name, *_ in connection.execute("PRAGMA table_info('climate_cells')").fetchall()
        )
        if columns != EXPECTED_CLIMATE_COLUMNS:
            msg = f"Unexpected climate_cells schema: {columns}"
            raise ValueError(msg)
        row_count_result = connection.execute("SELECT COUNT(*) FROM climate_cells").fetchone()

    if row_count_result is None:
        msg = "Failed to fetch climate row count"
        raise ValueError(msg)

    row_count = row_count_result[0]

    if not isinstance(row_count, int):
        msg = f"Unexpected row count type: {type(row_count)!r}"
        raise TypeError(msg)

    minimum_rows, maximum_rows = expected_row_count_range
    if not minimum_rows <= row_count <= maximum_rows:
        msg = f"Row count {row_count} outside expected rough range {minimum_rows}..{maximum_rows}"
        raise ValueError(msg)

    return ValidationSummary(row_count=row_count, columns=columns)


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
    return parser.parse_args()


def main() -> None:
    """Run the full build and validation flow from the command line."""
    args = parse_args()
    summary = build_worldclim_database(output_path=args.output, cache_dir=args.cache_dir)
    validation = validate_climate_database(summary.output_path)
    print(f"Built {summary.output_path} with {validation.row_count} rows.")


if __name__ == "__main__":
    main()
