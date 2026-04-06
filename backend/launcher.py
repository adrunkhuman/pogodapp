from __future__ import annotations

import argparse
import logging

import uvicorn

from backend.climate_pipeline import WORLDCLIM_RESOLUTIONS, build_worldclim_database, validate_climate_database
from backend.runtime import (
    resolve_build_climate_db_if_missing,
    resolve_climate_cache_dir,
    resolve_climate_database_path,
    resolve_climate_resolution,
    resolve_host,
    resolve_port,
    resolve_reload,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse launcher options for local or deployed environments."""
    parser = argparse.ArgumentParser(description="Run Pogodapp.")
    parser.add_argument("--host", default=resolve_host(), help="Host interface for the server.")
    parser.add_argument("--port", type=int, default=resolve_port(), help="Port for the server.")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable code reload.",
    )
    return parser.parse_args()


def ensure_climate_database() -> None:
    """Optionally build the climate database before the app imports its runtime repository."""
    database_path = resolve_climate_database_path()
    if database_path.exists() or not resolve_build_climate_db_if_missing():
        return

    resolution_name = resolve_climate_resolution()
    try:
        resolution = WORLDCLIM_RESOLUTIONS[resolution_name]
    except KeyError as error:
        supported_resolutions = ", ".join(sorted(WORLDCLIM_RESOLUTIONS))
        msg = f"Unsupported climate resolution {resolution_name!r}. Use one of: {supported_resolutions}."
        raise ValueError(msg) from error

    cache_dir = resolve_climate_cache_dir()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "startup_bootstrap outcome=building_climate_db path=%s resolution=%s cache_dir=%s",
        database_path,
        resolution.name,
        cache_dir,
    )
    build_worldclim_database(output_path=database_path, cache_dir=cache_dir, resolution=resolution)
    validation = validate_climate_database(database_path, resolution=resolution)
    logger.info(
        "startup_bootstrap outcome=ready_climate_db path=%s rows=%s cities=%s",
        database_path,
        validation.row_count,
        validation.city_count,
    )


def main() -> None:
    """Launch the app server after any requested climate bootstrap work."""
    args = parse_args()
    ensure_climate_database()
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=resolve_reload() and not args.no_reload)


if __name__ == "__main__":
    main()
