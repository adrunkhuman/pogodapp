from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, cast

import numpy as np
from fastapi.testclient import TestClient

from backend.cities import GRID_DEGREES, CityCandidate, CityRankingCache, continent_of
from backend.climate_repository import ClimateDataError, StubClimateRepository
from backend.config import DEFAULT_PREFERENCES, MAP_PROJECTION, PREFERENCE_FIELD_NAMES
from backend.heatmap import HeatmapProjection
from backend.logging_config import configure_backend_logging
from backend.main import build_probe_response, create_app
from backend.scoring import ClimateCell, ClimateMatrix, PreferenceInputs, score_matrix_row_breakdown, score_preferences

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from backend.climate_repository import ClimateRepository
    from backend.score_service import ScoreResponse


from backend import main as backend_main


def _capture_backend_logs() -> tuple[logging.Logger, list[logging.Handler], bool]:
    configure_backend_logging()
    backend_logger = logging.getLogger("backend")
    original_handlers = backend_logger.handlers[:]
    original_propagate = backend_logger.propagate
    backend_logger.handlers = []
    backend_logger.propagate = True
    return backend_logger, original_handlers, original_propagate


def _restore_backend_logs(backend_logger: logging.Logger, handlers: list[logging.Handler], propagate: bool) -> None:
    backend_logger.handlers = handlers
    backend_logger.propagate = propagate


client = TestClient(create_app(climate_repository=StubClimateRepository()))


def default_form_data() -> dict[str, str]:
    return {
        "preferred_day_temperature": "22",
        "summer_heat_limit": "30",
        "winter_cold_limit": "5",
        "dryness_preference": "60",
        "sunshine_preference": "60",
    }


def default_query_params() -> dict[str, int | float]:
    return {
        "lat": 37.5,
        "lon": -122.0,
        "preferred_day_temperature": 22,
        "summer_heat_limit": 30,
        "winter_cold_limit": 5,
        "dryness_preference": 60,
        "sunshine_preference": 60,
    }


def rendered_default_form_data() -> dict[str, str]:
    return {preference.name: str(preference.value) for preference in DEFAULT_PREFERENCES}


class ManyCitiesRepository:
    def __init__(self) -> None:
        self._cells = tuple(
            ClimateCell(
                lat=float(index),
                lon=float(index),
                temperature_c=(22.0,) * 12,
                temperature_min_c=(22.0,) * 12,
                temperature_max_c=(22.0,) * 12,
                precipitation_mm=(0.0,) * 12,
                cloud_cover_pct=(15,) * 12,
            )
            for index in range(25)
        )
        self._cities = tuple(
            CityCandidate(
                name=f"City {index:02d}",
                country_code="US",
                lat=self._cells[index].lat,
                lon=self._cells[index].lon,
                cell_lat=self._cells[index].lat,
                cell_lon=self._cells[index].lon,
            )
            for index in range(25)
        )
        self._matrix = ClimateMatrix.from_cells(self._cells)
        self._ranking_cache = CityRankingCache.from_cities(self._cities, np.arange(len(self._cities), dtype=np.int32))
        self._projection = HeatmapProjection.from_coordinates(self._matrix.latitudes, self._matrix.longitudes)

    def list_cells(self) -> tuple[ClimateCell, ...]:
        return self._cells

    def list_cities(self) -> tuple[CityCandidate, ...]:
        return self._cities

    def get_climate_matrix(self) -> ClimateMatrix:
        return self._matrix

    def get_indexed_cities(self) -> CityRankingCache:
        return self._ranking_cache

    def get_heatmap_projection(self) -> HeatmapProjection:
        return self._projection


def test_home_page_renders() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "POGODAPP" in response.text
    assert "Pick the climate you like and see where it shows up." in response.text
    assert 'hx-post="/score"' in response.text
    assert 'hx-trigger="load, change delay:500ms"' in response.text
    assert 'hx-sync="this:replace"' in response.text
    assert 'hx-swap="none"' in response.text
    assert 'id="score-loading-indicator"' in response.text
    assert 'id="map-description"' in response.text
    assert 'id="map-status"' in response.text
    assert 'id="map-legend"' in response.text
    assert "Climate compatibility" in response.text
    assert ">Map</h2>" in response.text
    assert (
        'id="map" role="region" aria-label="Interactive climate score map" aria-describedby="map-description map-legend map-status"'
        in response.text
    )
    assert 'id="score-results-list"' in response.text
    assert "/static/vendor/maplibre-gl.css" in response.text
    assert "/static/vendor/maplibre-gl.js" in response.text
    assert "/static/map-core.js" in response.text
    assert "/static/map-sidebar.js" in response.text
    assert "/static/map-probe.js" in response.text
    assert "/static/map-layers.js" in response.text
    assert "/static/map.js" in response.text
    assert "/static/app.js" in response.text
    assert "window.POGODAPP_MAP_CONFIG" in response.text
    assert MAP_PROJECTION.name in response.text
    assert "probeGridDegrees" in response.text
    assert str(GRID_DEGREES) in response.text
    assert "POGODAPP_INITIAL_SCORES" not in response.text


def test_home_page_uses_backend_default_preferences() -> None:
    response = client.get("/")

    assert response.status_code == 200

    for preference in DEFAULT_PREFERENCES:
        assert f'id="{preference.name}"' in response.text
        assert f'name="{preference.name}"' in response.text
        assert f'data-field="{preference.name}"' in response.text
        assert f'min="{preference.minimum}"' in response.text
        assert f'max="{preference.maximum}"' in response.text
        assert f'step="{preference.step}"' in response.text
        assert f'value="{preference.value}"' in response.text


def test_app_bootstrap_relies_on_htmx_load_trigger_instead_of_manual_submit() -> None:
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert 'window.htmx.trigger(form, "submit")' not in response.text


def test_preference_contract_matches_issue_scope() -> None:
    expected_names = (
        "preferred_day_temperature",
        "summer_heat_limit",
        "winter_cold_limit",
        "dryness_preference",
        "sunshine_preference",
    )

    assert expected_names == PREFERENCE_FIELD_NAMES
    assert tuple(preference.name for preference in DEFAULT_PREFERENCES) == expected_names


def test_preference_input_bounds_match_backend_config() -> None:
    schema = PreferenceInputs.model_json_schema()

    for preference in DEFAULT_PREFERENCES:
        field_schema = schema["properties"][preference.name]

        assert field_schema["minimum"] == preference.minimum
        assert field_schema["maximum"] == preference.maximum


def test_static_files_are_served() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "font-family" in response.text


def test_styles_lock_desktop_shell_to_viewport_height() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "html {" in response.text
    assert "height: 100vh;" in response.text
    assert "overflow: hidden;" in response.text
    assert "#preferences {" in response.text
    assert "align-content: start;" in response.text
    assert "overflow: auto;" in response.text
    assert "width: 100%;" in response.text
    assert "justify-self: stretch;" in response.text


def test_local_map_assets_are_served() -> None:
    css_response = client.get("/static/vendor/maplibre-gl.css")
    js_response = client.get("/static/vendor/maplibre-gl.js")
    geojson_response = client.get("/static/data/world.geojson")

    assert css_response.status_code == 200
    assert ".maplibregl-map" in css_response.text
    assert js_response.status_code == 200
    assert "maplibregl" in js_response.text
    assert geojson_response.status_code == 200
    assert geojson_response.json()["type"] == "FeatureCollection"


def test_map_script_initializes_maplibre_score_layer() -> None:
    response = client.get("/static/map.js")
    core_response = client.get("/static/map-core.js")
    layers_response = client.get("/static/map-layers.js")

    assert response.status_code == 200
    assert core_response.status_code == 200
    assert layers_response.status_code == 200
    assert "new window.maplibregl.Map" in response.text
    assert "WORLD_BACKDROP_URL" in response.text
    assert "HEATMAP_SOURCE_ID" in layers_response.text
    assert "data: WORLD_BACKDROP_URL" in response.text
    assert "id: LAND_LAYER_ID" in response.text
    assert "id: BORDER_LAYER_ID" in response.text
    assert 'type: "image"' in layers_response.text
    assert 'type: "raster"' in layers_response.text
    assert "projection: { type: MAP_CONFIG.projection }" in response.text
    assert "window.POGODAPP_MAP_CONFIG" in core_response.text
    assert "WORLD_CORNERS" in core_response.text
    assert "updateImage" in layers_response.text
    assert 'setMapStatus("Map backdrop ready.");' in response.text
    assert 'setMapStatus("Map library failed to load.");' in response.text


def test_map_contract_does_not_depend_on_remote_basemap_assets() -> None:
    home_response = client.get("/")
    script_response = client.get("/static/map.js")
    core_response = client.get("/static/map-core.js")
    layers_response = client.get("/static/map-layers.js")
    probe_response = client.get("/static/map-probe.js")

    assert home_response.status_code == 200
    assert script_response.status_code == 200
    assert core_response.status_code == 200
    assert layers_response.status_code == 200
    assert probe_response.status_code == 200
    assert "pmtiles" not in home_response.text
    assert "protomaps" not in home_response.text
    assert "unpkg.com/maplibre-gl" not in home_response.text
    assert "pmtiles" not in script_response.text
    assert "protomaps" not in script_response.text
    assert "https://" not in script_response.text
    assert "https://" not in core_response.text
    assert "https://" not in layers_response.text
    assert "https://" not in probe_response.text


def test_probe_script_snaps_cache_keys_and_query_params_to_backend_grid() -> None:
    response = client.get("/static/map-probe.js")
    core_response = client.get("/static/map-core.js")

    assert response.status_code == 200
    assert core_response.status_code == 200
    assert "function snapProbeCoordinate" in response.text
    assert "probeGridDegrees" in response.text
    assert "const snapped = snapProbeCoordinate(lat, lon);" in response.text
    assert "const cacheKey = `${snapped.lat},${snapped.lon},${new URLSearchParams(prefs)}`;" in response.text
    assert "new URLSearchParams({ lat: snapped.lat, lon: snapped.lon, ...prefs });" in response.text
    assert "const PROBE_HOVER_COOLDOWN_MS = 250;" in core_response.text
    assert "let probeRequestToken = 0;" in core_response.text
    assert "probeRequestToken += 1;" in response.text
    assert "cancelProbeCooldown();" in response.text
    assert "abortActiveProbe();" in response.text
    assert "requestToken !== probeRequestToken" in response.text
    assert "requestToken = ++probeRequestToken" in response.text


def test_home_page_uses_gzip_when_requested() -> None:
    response = client.get("/", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    assert "Accept-Encoding" in response.headers.get("vary", "")


def test_small_health_response_stays_uncompressed() -> None:
    response = client.get("/health", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None


def test_http_requests_are_logged(caplog: LogCaptureFixture) -> None:
    backend_logger, original_handlers, original_propagate = _capture_backend_logs()

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = client.get("/health")
    finally:
        _restore_backend_logs(backend_logger, original_handlers, original_propagate)

    assert response.status_code == 200
    assert caplog.records
    record = caplog.records[-1]
    assert record.message == "http request finished"
    assert record.__dict__["event"] == "http_request"
    assert record.__dict__["outcome"] == "ok"
    assert record.__dict__["method"] == "GET"
    assert record.__dict__["path"] == "/health"
    assert record.__dict__["httpStatus"] == 200
    assert record.__dict__["srcIp"] == "testclient"
    assert record.__dict__["scheme"] == "http"
    assert record.__dict__["httpVersion"] == "1.1"


def test_http_request_logs_client_errors(caplog: LogCaptureFixture) -> None:
    backend_logger, original_handlers, original_propagate = _capture_backend_logs()

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = client.get("/missing-route")
    finally:
        _restore_backend_logs(backend_logger, original_handlers, original_propagate)

    assert response.status_code == 404
    assert caplog.records
    record = caplog.records[-1]
    assert record.message == "http request finished"
    assert record.__dict__["event"] == "http_request"
    assert record.__dict__["outcome"] == "client_error"
    assert record.__dict__["path"] == "/missing-route"
    assert record.__dict__["httpStatus"] == 404


def test_http_request_logs_server_errors(caplog: LogCaptureFixture) -> None:
    class FailingRepository(StubClimateRepository):
        def get_climate_matrix(self) -> ClimateMatrix:
            msg = "db unavailable"
            raise ClimateDataError(msg)

    failing_client = TestClient(create_app(climate_repository=FailingRepository()))

    backend_logger, original_handlers, original_propagate = _capture_backend_logs()

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = failing_client.post("/score", data=default_form_data())
    finally:
        _restore_backend_logs(backend_logger, original_handlers, original_propagate)

    assert response.status_code == 503
    assert caplog.records
    record = caplog.records[-1]
    assert record.message == "http request finished"
    assert record.__dict__["event"] == "http_request"
    assert record.__dict__["outcome"] == "error"
    assert record.__dict__["path"] == "/score"
    assert record.__dict__["httpStatus"] == 503


def test_http_request_logs_uncaught_exceptions(caplog: LogCaptureFixture) -> None:
    app = create_app(climate_repository=StubClimateRepository())

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        msg = "boom"
        raise RuntimeError(msg)

    failing_client = TestClient(app, raise_server_exceptions=False)
    backend_logger, original_handlers, original_propagate = _capture_backend_logs()

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = failing_client.get("/boom?x=1")
    finally:
        _restore_backend_logs(backend_logger, original_handlers, original_propagate)

    assert response.status_code == 500
    assert caplog.records
    record = caplog.records[-1]
    assert record.message == "http request failed"
    assert record.__dict__["event"] == "http_request"
    assert record.__dict__["outcome"] == "error"
    assert record.__dict__["path"] == "/boom"
    assert record.__dict__["query"] == "x=1"


def test_http_requests_are_logged_with_structured_fields_by_default(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    logging_client = TestClient(create_app(climate_repository=StubClimateRepository()))
    backend_logger, original_handlers, original_propagate = _capture_backend_logs()

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = logging_client.get("/health")
    finally:
        _restore_backend_logs(backend_logger, original_handlers, original_propagate)

    assert response.status_code == 200
    assert caplog.records
    record = caplog.records[-1]
    assert record.__dict__["event"] == "http_request"
    assert record.__dict__["path"] == "/health"
    assert record.__dict__["httpStatus"] == 200


def test_score_endpoint_accepts_form_encoded_preferences() -> None:
    response = client.post(
        "/score",
        data=default_form_data(),
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()

    assert isinstance(payload, dict)
    assert set(payload) == {"scores", "heatmap"}

    scores = payload["scores"]
    assert isinstance(scores, list)
    assert scores

    score_values = [item["score"] for item in scores]
    for item in scores:
        assert set(item) == {
            "name",
            "continent",
            "country_code",
            "flag",
            "score",
            "lat",
            "lon",
            "probe_lat",
            "probe_lon",
        }
        assert isinstance(item["name"], str)
        assert item["name"]
        assert isinstance(item["continent"], str)
        assert item["continent"]
        assert isinstance(item["country_code"], str)
        assert len(item["country_code"]) == 2
        assert isinstance(item["flag"], str)
        assert item["flag"]
        assert 0 <= item["score"] <= 1
    # City scores inherit normalized cell scores, but the best cell may have no nearby city.
    assert max(score_values) <= 1.0
    assert max(score_values) > 0
    continent_counts: dict[str, int] = {}
    for item in scores:
        continent = continent_of(item["country_code"], item["lon"])
        if continent == "Other":
            continue
        continent_counts[continent] = continent_counts.get(continent, 0) + 1
    assert continent_counts
    assert all(count <= 30 for count in continent_counts.values())
    # Heatmap is a PNG data URL
    assert payload["heatmap"].startswith("data:image/png;base64,")


def test_score_endpoint_offloads_scoring_to_threadpool(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []
    app = create_app(climate_repository=StubClimateRepository())
    threadpool_client = TestClient(app)

    async def fake_run_in_threadpool(func: Callable[..., object], *args: object) -> object:
        calls.append((func, args))
        return func(*args)

    monkeypatch.setattr(backend_main, "run_in_threadpool", fake_run_in_threadpool)

    response = threadpool_client.post("/score", data=default_form_data())

    assert response.status_code == 200
    assert len(calls) == 1
    assert getattr(calls[0][0], "__name__", "") == "_score_response_from_cache_or_repository"


def test_score_endpoint_uses_gzip_when_requested() -> None:
    response = client.post(
        "/score",
        data=default_form_data(),
        headers={"Accept-Encoding": "gzip"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers.get("content-encoding") == "gzip"
    assert "Accept-Encoding" in response.headers.get("vary", "")


def test_score_endpoint_is_deterministic_for_the_same_preferences() -> None:
    form_data = default_form_data()

    first_response = client.post("/score", data=form_data)
    second_response = client.post("/score", data=form_data)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["scores"] == second_response.json()["scores"]
    assert first_response.json()["heatmap"] == second_response.json()["heatmap"]


def test_score_endpoint_reuses_cached_response_for_identical_preferences() -> None:
    call_count = 0
    original_builder = backend_main.build_score_response

    def counted_builder(repository: StubClimateRepository, preferences: PreferenceInputs) -> ScoreResponse:
        nonlocal call_count
        call_count += 1
        return original_builder(repository, preferences)

    backend_main.__dict__["build_score_response"] = counted_builder
    cached_client = TestClient(create_app(climate_repository=StubClimateRepository()))

    try:
        first_response = cached_client.post("/score", data=default_form_data())
        second_response = cached_client.post("/score", data=default_form_data())
    finally:
        backend_main.__dict__["build_score_response"] = original_builder

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert call_count == 2  # 1 pre-warm + 1 miss for non-default form data


def test_score_endpoint_uses_prewarmed_default_preferences_cache() -> None:
    call_count = 0
    original_builder = backend_main.build_score_response

    def counted_builder(repository: StubClimateRepository, preferences: PreferenceInputs) -> ScoreResponse:
        nonlocal call_count
        call_count += 1
        return original_builder(repository, preferences)

    backend_main.__dict__["build_score_response"] = counted_builder

    try:
        cached_client = TestClient(create_app(climate_repository=StubClimateRepository()))
        response = cached_client.post("/score", data=rendered_default_form_data())
    finally:
        backend_main.__dict__["build_score_response"] = original_builder

    assert response.status_code == 200
    assert call_count == 1  # pre-warm only; the first default request should hit cache


def test_score_endpoint_evicts_oldest_cached_preferences_after_cache_limit() -> None:
    call_count = 0
    original_builder = backend_main.build_score_response

    def counted_builder(repository: StubClimateRepository, preferences: PreferenceInputs) -> ScoreResponse:
        nonlocal call_count
        call_count += 1
        return original_builder(repository, preferences)

    backend_main.__dict__["build_score_response"] = counted_builder
    cached_client = TestClient(create_app(climate_repository=StubClimateRepository()))
    base_form = default_form_data()

    try:
        for offset in range(17):
            response = cached_client.post(
                "/score",
                data={**base_form, "dryness_preference": str(offset)},
            )
            assert response.status_code == 200

        repeated_first = cached_client.post("/score", data={**base_form, "dryness_preference": "0"})
        newest_repeat = cached_client.post("/score", data={**base_form, "dryness_preference": "16"})
    finally:
        backend_main.__dict__["build_score_response"] = original_builder

    assert repeated_first.status_code == 200
    assert newest_repeat.status_code == 200
    assert call_count == 19  # 1 pre-warm + 17 unique keys + 1 recompute after LRU eviction


def test_score_response_cache_deduplicates_concurrent_identical_misses() -> None:
    score_cache_class = backend_main.__dict__["_ScoreResponseCache"]
    cache = score_cache_class(4)
    key = (18, 30, 0, 30, 50)
    build_started = threading.Event()
    build_count = 0
    results: list[dict[str, object]] = []

    def build() -> dict[str, object]:
        nonlocal build_count
        build_count += 1
        build_started.set()
        time.sleep(0.05)
        return {"scores": [], "heatmap": "cached"}

    def worker() -> None:
        results.append(cache.get_or_set(key, build))

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    threads[0].start()
    assert build_started.wait(timeout=1)
    threads[1].start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert build_count == 1
    assert results == [{"scores": [], "heatmap": "cached"}, {"scores": [], "heatmap": "cached"}]


def test_score_response_cache_recovers_after_failing_inflight_build() -> None:
    score_cache_class = backend_main.__dict__["_ScoreResponseCache"]
    cache = score_cache_class(4)
    key = (18, 30, 0, 30, 50)
    build_started = threading.Event()
    call_count = 0
    failures: list[str] = []
    successes: list[dict[str, object]] = []

    def flaky_build() -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            build_started.set()
            time.sleep(0.05)
            msg = "boom"
            raise RuntimeError(msg)
        return {"scores": [], "heatmap": "recovered"}

    def worker() -> None:
        try:
            successes.append(cache.get_or_set(key, flaky_build))
        except RuntimeError as error:
            failures.append(str(error))

    threads = [threading.Thread(target=worker), threading.Thread(target=worker)]
    threads[0].start()
    assert build_started.wait(timeout=1)
    threads[1].start()
    for thread in threads:
        thread.join(timeout=1)
        assert not thread.is_alive()

    assert failures == ["boom"]
    assert successes == [{"scores": [], "heatmap": "recovered"}]
    assert call_count == 2
    assert cache.get_or_set(key, flaky_build) == {"scores": [], "heatmap": "recovered"}
    assert call_count == 2


def test_dryness_preference_penalizes_rainier_cells() -> None:
    dry_tolerant_scores = score_preferences(
        PreferenceInputs(
            preferred_day_temperature=22,
            summer_heat_limit=30,
            winter_cold_limit=5,
            dryness_preference=0,
            sunshine_preference=60,
        )
    )
    rain_sensitive_scores = score_preferences(
        PreferenceInputs(
            preferred_day_temperature=22,
            summer_heat_limit=30,
            winter_cold_limit=5,
            dryness_preference=100,
            sunshine_preference=60,
        )
    )

    # Hard-code stub indices so the test compares driest vs. rainiest fixtures.
    driest_index = 2
    rainiest_index = 4

    assert rain_sensitive_scores[driest_index]["score"] > rain_sensitive_scores[rainiest_index]["score"]
    assert rain_sensitive_scores[rainiest_index]["score"] < dry_tolerant_scores[rainiest_index]["score"]


def test_score_endpoint_rejects_out_of_range_preferences() -> None:
    response = client.post(
        "/score",
        data={
            **default_form_data(),
            "preferred_day_temperature": "99",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any(item["loc"][-1] == "preferred_day_temperature" for item in detail)


def test_score_endpoint_rejects_missing_preferences() -> None:
    response = client.post(
        "/score",
        data={
            "preferred_day_temperature": "22",
            "summer_heat_limit": "30",
            "winter_cold_limit": "5",
            "dryness_preference": "60",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any(item["loc"][-1] == "sunshine_preference" for item in detail)


def test_score_endpoint_rejects_non_numeric_preferences() -> None:
    response = client.post(
        "/score",
        data={
            **default_form_data(),
            "preferred_day_temperature": "warm",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any(item["loc"][-1] == "preferred_day_temperature" for item in detail)


def test_score_endpoint_rejects_typical_day_above_summer_limit() -> None:
    response = client.post(
        "/score",
        data={
            **default_form_data(),
            "preferred_day_temperature": "31",
            "summer_heat_limit": "30",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any("summer_heat_limit" in item["msg"] for item in detail)


def test_score_endpoint_rejects_typical_day_below_winter_limit() -> None:
    response = client.post(
        "/score",
        data={
            **default_form_data(),
            "preferred_day_temperature": "6",
            "winter_cold_limit": "7",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any("winter_cold_limit" in item["msg"] for item in detail)


def test_probe_endpoint_rejects_typical_day_above_summer_limit() -> None:
    response = client.get(
        "/probe",
        params={
            **default_query_params(),
            "preferred_day_temperature": 31,
            "summer_heat_limit": 30,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any("summer_heat_limit" in item["msg"] for item in detail)


def test_probe_endpoint_rejects_typical_day_below_winter_limit() -> None:
    response = client.get(
        "/probe",
        params={
            **default_query_params(),
            "preferred_day_temperature": 6,
            "winter_cold_limit": 7,
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any("winter_cold_limit" in item["msg"] for item in detail)


def test_score_endpoint_returns_all_available_cities_when_under_continent_reserve() -> None:
    many_cities_client = TestClient(create_app(climate_repository=cast("ClimateRepository", ManyCitiesRepository())))

    response = many_cities_client.post(
        "/score",
        data=default_form_data(),
    )

    assert response.status_code == 200
    scores = response.json()["scores"]
    returned_names = [item["name"] for item in scores]
    all_names = {f"City {index:02d}" for index in range(25)}

    # 25 cities available in total — all returned since no continent reserve is hit.
    assert len(scores) == 25
    assert set(returned_names) == all_names


def test_probe_endpoint_returns_scored_breakdown_for_a_valid_cell() -> None:
    class ProbeRepository(StubClimateRepository):
        def probe_nearest_cell(self, lat: float, lon: float) -> int | None:
            return 0

        def get_probe_cell(self, row_index: int) -> ClimateCell:
            return self.list_cells()[row_index]

    probe_client = TestClient(create_app(climate_repository=ProbeRepository()))

    response = probe_client.get(
        "/probe",
        params=default_query_params(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is True
    assert 0 <= payload["overall_score"] <= 1
    assert set(payload) == {"found", "overall_score", "metrics"}
    assert len(payload["metrics"]) == 5
    assert [metric["key"] for metric in payload["metrics"]] == ["temp", "high", "low", "rain", "sun"]
    assert all(set(metric) == {"key", "label", "value", "display_value", "score"} for metric in payload["metrics"])


def test_probe_endpoint_offloads_breakdown_to_threadpool(monkeypatch: MonkeyPatch) -> None:
    class ProbeRepository(StubClimateRepository):
        def probe_nearest_cell(self, lat: float, lon: float) -> int | None:
            return 0

        def get_probe_cell(self, row_index: int) -> ClimateCell:
            return self.list_cells()[row_index]

    calls: list[tuple[object, tuple[object, ...]]] = []
    app = create_app(climate_repository=ProbeRepository())
    threadpool_client = TestClient(app)

    async def fake_run_in_threadpool(func: Callable[..., object], *args: object) -> object:
        calls.append((func, args))
        return func(*args)

    monkeypatch.setattr(backend_main, "run_in_threadpool", fake_run_in_threadpool)

    response = threadpool_client.get("/probe", params=default_query_params())

    assert response.status_code == 200
    assert response.json()["found"] is True
    assert len(calls) == 1
    assert getattr(calls[0][0], "__name__", "") == "_build_probe_response_from_repository"


def test_probe_endpoint_uses_probe_cell_without_loading_resident_matrix() -> None:
    class ProbeOnlyRepository(StubClimateRepository):
        def __init__(self) -> None:
            self.matrix_calls = 0

        def probe_nearest_cell(self, lat: float, lon: float) -> int | None:
            return 0

        def get_climate_matrix(self) -> ClimateMatrix:
            self.matrix_calls += 1
            return super().get_climate_matrix()

        def get_probe_cell(self, row_index: int) -> ClimateCell:
            return self.list_cells()[row_index]

    repository = ProbeOnlyRepository()
    probe_client = TestClient(create_app(climate_repository=cast("ClimateRepository", repository)))
    matrix_calls_after_startup = repository.matrix_calls

    response = probe_client.get("/probe", params=default_query_params())

    assert response.status_code == 200
    assert response.json()["found"] is True
    assert repository.matrix_calls == matrix_calls_after_startup


def test_probe_endpoint_returns_empty_payload_when_repository_has_no_probe_support() -> None:
    probe_client = TestClient(create_app(climate_repository=StubClimateRepository()))

    response = probe_client.get(
        "/probe",
        params=default_query_params(),
    )

    assert response.status_code == 200
    assert response.json() == {"found": False, "overall_score": 0.0, "metrics": []}


def test_probe_endpoint_returns_empty_payload_when_no_cell_is_found() -> None:
    class MissingProbeRepository(StubClimateRepository):
        def probe_nearest_cell(self, lat: float, lon: float) -> int | None:
            return None

    probe_client = TestClient(create_app(climate_repository=MissingProbeRepository()))

    response = probe_client.get(
        "/probe",
        params=default_query_params(),
    )

    assert response.status_code == 200
    assert response.json() == {"found": False, "overall_score": 0.0, "metrics": []}


def test_probe_endpoint_returns_503_for_repository_failures() -> None:
    class BrokenProbeRepository(StubClimateRepository):
        def probe_nearest_cell(self, lat: float, lon: float) -> int | None:
            return 0

        def get_probe_cell(self, row_index: int) -> ClimateCell:
            msg = "Climate database file not found: data/climate.duckdb"
            raise ClimateDataError(msg)

    probe_client = TestClient(create_app(climate_repository=BrokenProbeRepository()))

    response = probe_client.get(
        "/probe",
        params=default_query_params(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Climate database file not found: data/climate.duckdb"}


def test_home_page_registers_htmx_handoff_script() -> None:
    response = client.get("/")
    app_script = client.get("/static/app.js")

    assert response.status_code == 200
    assert app_script.status_code == 200
    assert "/static/app.js" in response.text
    assert "htmx:afterRequest" in app_script.text
    assert "htmx:beforeRequest" in app_script.text
    assert "window.renderScores(JSON.parse(event.detail.xhr.responseText));" in app_script.text
    assert "loadingIndicator.hidden = !isLoading;" in app_script.text
    assert "summerHeatInput.min = preferredDayInput.value" in app_script.text
    assert "winterColdInput.max = preferredDayInput.value" in app_script.text


def test_map_script_renders_city_labels_instead_of_coordinates() -> None:
    sidebar_response = client.get("/static/map-sidebar.js")
    probe_response = client.get("/static/map-probe.js")
    layers_response = client.get("/static/map-layers.js")

    assert sidebar_response.status_code == 200
    assert probe_response.status_code == 200
    assert layers_response.status_code == 200
    assert "point.country_code" in sidebar_response.text
    assert "point.name" in sidebar_response.text
    assert "point.flag" in sidebar_response.text
    assert "score-results__item" in sidebar_response.text
    assert "if (!response.ok) throw new Error" in probe_response.text
    assert "metric.display_value" in probe_response.text
    assert "metric.score" in probe_response.text
    assert "probe_lat" in layers_response.text


def test_build_probe_response_preserves_metric_order_and_fields() -> None:
    response = build_probe_response(
        score_matrix_row_breakdown(
            ClimateMatrix.from_cells((StubClimateRepository().list_cells()[0],)),
            0,
            PreferenceInputs(
                preferred_day_temperature=22,
                summer_heat_limit=30,
                winter_cold_limit=5,
                dryness_preference=60,
                sunshine_preference=60,
            ),
        )
    )

    assert response.found is True
    assert [metric.key for metric in response.metrics] == ["temp", "high", "low", "rain", "sun"]
    assert [metric.label for metric in response.metrics] == ["temp", "high", "low", "rain", "sun"]


def test_preload_logs_runtime_memory_snapshots(monkeypatch: MonkeyPatch) -> None:
    class InstrumentedRepository:
        def get_climate_matrix(self) -> ClimateMatrix:
            return ClimateMatrix.from_cells((StubClimateRepository().list_cells()[0],))

        def get_indexed_cities(self) -> CityRankingCache:
            return CityRankingCache.from_cities((), np.array([], dtype=np.int32))

        def get_heatmap_projection(self) -> HeatmapProjection:
            return HeatmapProjection.from_coordinates(np.array([], dtype=np.float32), np.array([], dtype=np.float32))

        def list_cells(self) -> tuple[ClimateCell, ...]:
            return ()

        def list_cities(self) -> tuple[CityCandidate, ...]:
            return ()

        def get_runtime_cache_stats(self) -> dict[str, float]:
            return {"runtime_cache_mb": 12.3}

    runtime_stages: list[str] = []

    def capture_runtime_memory(stage: str, repository: ClimateRepository) -> None:
        runtime_stages.append(stage)

    monkeypatch.setattr(backend_main, "_log_runtime_memory", capture_runtime_memory)

    create_app(climate_repository=cast("ClimateRepository", InstrumentedRepository()))

    assert runtime_stages == [
        "before_preload",
        "after_climate_matrix",
        "after_indexed_cities",
        "after_heatmap_projection",
        "after_default_score_cache",
    ]


def test_current_rss_mb_returns_none_when_proc_statm_is_unavailable(monkeypatch: MonkeyPatch) -> None:
    class MissingStatmPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def exists(self) -> bool:
            return False

    monkeypatch.setattr(backend_main, "Path", MissingStatmPath)

    assert backend_main._current_rss_mb() is None  # noqa: SLF001
