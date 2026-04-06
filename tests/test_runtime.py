from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from backend import runtime

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_climate_database_path_uses_data_dir_when_database_not_overridden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "runtime-data"
    monkeypatch.setenv(runtime.DATA_DIR_ENV_VAR, str(data_dir))
    monkeypatch.delenv(runtime.CLIMATE_DATABASE_ENV_VAR, raising=False)

    assert runtime.resolve_climate_database_path() == data_dir / "climate.duckdb"


def test_resolve_climate_database_path_prefers_explicit_database_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "runtime-data"
    database_path = tmp_path / "custom" / "climate.duckdb"
    monkeypatch.setenv(runtime.DATA_DIR_ENV_VAR, str(data_dir))
    monkeypatch.setenv(runtime.CLIMATE_DATABASE_ENV_VAR, str(database_path))

    assert runtime.resolve_climate_database_path() == database_path


def test_resolve_host_defaults_to_deploy_host_when_port_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.HOST_ENV_VAR, raising=False)
    monkeypatch.setenv(runtime.PORT_ENV_VAR, "9000")

    assert runtime.resolve_host() == runtime.DEPLOY_HOST


def test_parse_bool_env_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.BUILD_IF_MISSING_ENV_VAR, "maybe")

    with pytest.raises(ValueError, match="Invalid boolean value"):
        runtime.resolve_build_climate_db_if_missing()


def test_resolve_reload_defaults_to_false_when_port_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.RELOAD_ENV_VAR, raising=False)
    monkeypatch.setenv(runtime.PORT_ENV_VAR, "9000")

    assert runtime.resolve_reload() is False


def test_resolve_reload_honors_explicit_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.PORT_ENV_VAR, "9000")
    monkeypatch.setenv(runtime.RELOAD_ENV_VAR, "true")

    assert runtime.resolve_reload() is True
