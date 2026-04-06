from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_CLIMATE_DATABASE_NAME = "climate.duckdb"
DEFAULT_CLIMATE_CACHE_DIR_NAME = "worldclim"
DEFAULT_HOST = "127.0.0.1"
DEPLOY_HOST = "0.0.0.0"  # noqa: S104 - deployment platforms require an externally reachable bind address.
DEFAULT_PORT = 8000

CLIMATE_DATABASE_ENV_VAR = "POGODAPP_CLIMATE_DB"
DATA_DIR_ENV_VAR = "POGODAPP_DATA_DIR"
CLIMATE_CACHE_DIR_ENV_VAR = "POGODAPP_CLIMATE_CACHE_DIR"
BUILD_IF_MISSING_ENV_VAR = "POGODAPP_BUILD_CLIMATE_DB_IF_MISSING"
CLIMATE_RESOLUTION_ENV_VAR = "POGODAPP_CLIMATE_RESOLUTION"
HOST_ENV_VAR = "POGODAPP_HOST"
PORT_ENV_VAR = "PORT"
RELOAD_ENV_VAR = "POGODAPP_RELOAD"

TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_ENV_VALUES = frozenset({"0", "false", "no", "off"})


def parse_bool_env(name: str, *, default: bool) -> bool:
    """Parse one boolean environment variable using a small stable vocabulary."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()
    if normalized_value in TRUE_ENV_VALUES:
        return True
    if normalized_value in FALSE_ENV_VALUES:
        return False

    msg = f"Invalid boolean value for {name}: {raw_value!r}. Use one of {sorted(TRUE_ENV_VALUES | FALSE_ENV_VALUES)}."
    raise ValueError(msg)


def resolve_data_dir() -> Path:
    """Return the base directory for generated runtime artifacts."""
    configured_path = os.getenv(DATA_DIR_ENV_VAR)
    return Path(configured_path) if configured_path else DEFAULT_DATA_DIR


def resolve_climate_database_path() -> Path:
    """Return the climate database path from env or the standard data dir."""
    configured_path = os.getenv(CLIMATE_DATABASE_ENV_VAR)
    if configured_path:
        return Path(configured_path)
    return resolve_data_dir() / DEFAULT_CLIMATE_DATABASE_NAME


def resolve_climate_cache_dir() -> Path:
    """Return the download cache path used when bootstrapping climate data."""
    configured_path = os.getenv(CLIMATE_CACHE_DIR_ENV_VAR)
    if configured_path:
        return Path(configured_path)
    return resolve_data_dir() / DEFAULT_CLIMATE_CACHE_DIR_NAME


def resolve_build_climate_db_if_missing() -> bool:
    """Return whether startup should generate climate data when the DB is absent."""
    return parse_bool_env(BUILD_IF_MISSING_ENV_VAR, default=False)


def resolve_climate_resolution() -> str:
    """Return the requested WorldClim resolution for runtime bootstrap."""
    return os.getenv(CLIMATE_RESOLUTION_ENV_VAR, "5m")


def resolve_host() -> str:
    """Return the bind host, preferring provider-friendly defaults when a port is injected."""
    configured_host = os.getenv(HOST_ENV_VAR)
    if configured_host:
        return configured_host
    return DEPLOY_HOST if os.getenv(PORT_ENV_VAR) else DEFAULT_HOST


def resolve_port() -> int:
    """Return the bind port from the platform or the local default."""
    raw_port = os.getenv(PORT_ENV_VAR)
    if raw_port is None:
        return DEFAULT_PORT
    try:
        return int(raw_port)
    except ValueError as error:
        msg = f"Invalid port value for {PORT_ENV_VAR}: {raw_port!r}. Expected an integer."
        raise ValueError(msg) from error


def resolve_reload() -> bool:
    """Return whether Uvicorn reload should be enabled."""
    return parse_bool_env(RELOAD_ENV_VAR, default=os.getenv(PORT_ENV_VAR) is None)
