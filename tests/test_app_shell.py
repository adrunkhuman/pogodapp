import numpy as np
from fastapi.testclient import TestClient

from backend.cities import CityCandidate, CityRankingCache
from backend.climate_repository import StubClimateRepository
from backend.config import DEFAULT_PREFERENCES, MAP_PROJECTION, PREFERENCE_FIELD_NAMES
from backend.heatmap import HeatmapProjection
from backend.main import create_app
from backend.scoring import ClimateCell, ClimateMatrix, PreferenceInputs, score_preferences

client = TestClient(create_app(climate_repository=StubClimateRepository()))


class ManyCitiesRepository:
    def __init__(self) -> None:
        self._cells = tuple(
            ClimateCell(
                lat=float(index),
                lon=float(index),
                temperature_c=(22.0,) * 12,
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
    assert 'hx-trigger="input changed delay:300ms"' in response.text
    assert 'hx-swap="none"' in response.text
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
    assert "window.POGODAPP_MAP_CONFIG" in response.text
    assert MAP_PROJECTION.name in response.text


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


def test_preference_contract_matches_issue_scope() -> None:
    expected_names = (
        "ideal_temperature",
        "cold_tolerance",
        "heat_tolerance",
        "rain_sensitivity",
        "sun_preference",
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

    assert response.status_code == 200
    assert "new window.maplibregl.Map" in response.text
    assert "data: WORLD_BACKDROP_URL" in response.text
    assert "id: LAND_LAYER_ID" in response.text
    assert "id: BORDER_LAYER_ID" in response.text
    assert "HEATMAP_SOURCE_ID" in response.text
    assert 'type: "image"' in response.text
    assert 'type: "raster"' in response.text
    assert "projection: { type: MAP_CONFIG.projection }" in response.text
    assert "window.POGODAPP_MAP_CONFIG" in response.text
    assert "WORLD_CORNERS" in response.text
    assert "updateImage" in response.text
    assert 'setMapStatus("Map backdrop ready.");' in response.text
    assert 'setMapStatus("Map library failed to load.");' in response.text


def test_map_contract_does_not_depend_on_remote_basemap_assets() -> None:
    home_response = client.get("/")
    script_response = client.get("/static/map.js")

    assert home_response.status_code == 200
    assert script_response.status_code == 200
    assert "pmtiles" not in home_response.text
    assert "protomaps" not in home_response.text
    assert "unpkg.com/maplibre-gl" not in home_response.text
    assert "pmtiles" not in script_response.text
    assert "protomaps" not in script_response.text
    assert "https://" not in script_response.text


def test_score_endpoint_accepts_form_encoded_preferences() -> None:
    response = client.post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()

    assert isinstance(payload, dict)
    assert "scores" in payload
    assert "heatmap" in payload

    scores = payload["scores"]
    assert isinstance(scores, list)
    assert scores

    score_values = [item["score"] for item in scores]
    for item in scores:
        assert set(item) == {"name", "country_code", "flag", "score"}
        assert isinstance(item["name"], str)
        assert item["name"]
        assert isinstance(item["country_code"], str)
        assert len(item["country_code"]) == 2
        assert isinstance(item["flag"], str)
        assert item["flag"]
        assert 0 <= item["score"] <= 1
    # City scores inherit normalized cell scores, but the best cell may have no nearby city.
    assert max(score_values) <= 1.0
    assert max(score_values) > 0
    # List capped at top 20 for the text panel
    assert len(scores) <= 20
    # Heatmap is a PNG data URL
    assert payload["heatmap"].startswith("data:image/png;base64,")


def test_score_endpoint_is_deterministic_for_the_same_preferences() -> None:
    form_data = {
        "ideal_temperature": "22",
        "cold_tolerance": "7",
        "heat_tolerance": "5",
        "rain_sensitivity": "55",
        "sun_preference": "60",
    }

    first_response = client.post("/score", data=form_data)
    second_response = client.post("/score", data=form_data)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["scores"] == second_response.json()["scores"]
    assert first_response.json()["heatmap"] == second_response.json()["heatmap"]


def test_rain_sensitivity_penalizes_rainier_cells() -> None:
    dry_tolerant_scores = score_preferences(
        PreferenceInputs(
            ideal_temperature=22,
            cold_tolerance=7,
            heat_tolerance=5,
            rain_sensitivity=0,
            sun_preference=60,
        )
    )
    rain_sensitive_scores = score_preferences(
        PreferenceInputs(
            ideal_temperature=22,
            cold_tolerance=7,
            heat_tolerance=5,
            rain_sensitivity=100,
            sun_preference=60,
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
            "ideal_temperature": "99",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any(item["loc"][-1] == "ideal_temperature" for item in detail)


def test_score_endpoint_rejects_missing_preferences() -> None:
    response = client.post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any(item["loc"][-1] == "sun_preference" for item in detail)


def test_score_endpoint_rejects_non_numeric_preferences() -> None:
    response = client.post(
        "/score",
        data={
            "ideal_temperature": "warm",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]

    assert detail
    assert any(item["loc"][-1] == "ideal_temperature" for item in detail)


def test_score_endpoint_caps_city_results_to_top_twenty() -> None:
    many_cities_client = TestClient(create_app(climate_repository=ManyCitiesRepository()))

    response = many_cities_client.post(
        "/score",
        data={
            "ideal_temperature": "22",
            "cold_tolerance": "7",
            "heat_tolerance": "5",
            "rain_sensitivity": "55",
            "sun_preference": "60",
        },
    )

    assert response.status_code == 200
    scores = response.json()["scores"]
    returned_names = [item["name"] for item in scores]
    all_names = {f"City {index:02d}" for index in range(25)}

    assert len(scores) == 20
    assert len(set(returned_names)) == 20
    assert set(returned_names) <= all_names
    assert len(all_names - set(returned_names)) == 5


def test_home_page_registers_htmx_handoff_script() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "htmx:afterRequest" in response.text
    assert "window.renderScores(scores);" in response.text


def test_map_script_renders_city_labels_instead_of_coordinates() -> None:
    response = client.get("/static/map.js")

    assert response.status_code == 200
    assert "Intl.DisplayNames" in response.text
    assert "point.country_code" in response.text
    assert "point.name" in response.text
    assert "point.flag" in response.text
    assert "score-results__item" in response.text
