from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from backend import launcher

if TYPE_CHECKING:
    from pathlib import Path


def test_ensure_climate_database_skips_when_file_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    database_path = tmp_path / "climate.duckdb"
    database_path.write_text("ready", encoding="utf-8")
    build_calls: list[tuple[Path, Path, str]] = []

    monkeypatch.setattr(launcher, "resolve_climate_database_path", lambda: database_path)
    monkeypatch.setattr(launcher, "resolve_build_climate_db_if_missing", lambda: True)
    monkeypatch.setattr(
        launcher,
        "build_worldclim_database",
        lambda output_path, cache_dir, resolution: build_calls.append((output_path, cache_dir, resolution.name)),
    )

    launcher.ensure_climate_database()

    assert build_calls == []


def test_ensure_climate_database_skips_when_bootstrap_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "climate.duckdb"
    build_calls: list[tuple[Path, Path, str]] = []

    monkeypatch.setattr(launcher, "resolve_climate_database_path", lambda: database_path)
    monkeypatch.setattr(launcher, "resolve_build_climate_db_if_missing", lambda: False)
    monkeypatch.setattr(
        launcher,
        "build_worldclim_database",
        lambda output_path, cache_dir, resolution: build_calls.append((output_path, cache_dir, resolution.name)),
    )

    launcher.ensure_climate_database()

    assert build_calls == []


def test_ensure_climate_database_builds_and_validates_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "nested" / "climate.duckdb"
    cache_dir = tmp_path / "cache"
    build_calls: list[tuple[Path, Path, str]] = []
    validate_calls: list[tuple[Path, str]] = []

    monkeypatch.setattr(launcher, "resolve_climate_database_path", lambda: database_path)
    monkeypatch.setattr(launcher, "resolve_build_climate_db_if_missing", lambda: True)
    monkeypatch.setattr(launcher, "resolve_climate_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(launcher, "resolve_climate_resolution", lambda: "5m")
    monkeypatch.setattr(
        launcher,
        "build_worldclim_database",
        lambda output_path, cache_dir, resolution: build_calls.append((output_path, cache_dir, resolution.name)),
    )
    monkeypatch.setattr(
        launcher,
        "validate_climate_database",
        lambda database_path, resolution: (
            validate_calls.append((database_path, resolution.name))
            or type("ValidationSummary", (), {"row_count": 1, "city_count": 1})()
        ),
    )

    launcher.ensure_climate_database()

    assert database_path.parent.exists()
    assert cache_dir.exists()
    assert build_calls == [(database_path, cache_dir, "5m")]
    assert validate_calls == [(database_path, "5m")]


def test_ensure_climate_database_rejects_unknown_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "resolve_climate_database_path", lambda: tmp_path / "climate.duckdb")
    monkeypatch.setattr(launcher, "resolve_build_climate_db_if_missing", lambda: True)
    monkeypatch.setattr(launcher, "resolve_climate_resolution", lambda: "bad")

    with pytest.raises(ValueError, match="Unsupported climate resolution"):
        launcher.ensure_climate_database()
