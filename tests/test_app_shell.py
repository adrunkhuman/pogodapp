from fastapi.testclient import TestClient

from backend.config import DEFAULT_PREFERENCES, PREFERENCE_FIELD_NAMES
from backend.main import app
from backend.scoring import PreferenceInputs, score_preferences

client = TestClient(app)


def test_home_page_renders() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Pogodapp" in response.text
    assert "Climate preference search" in response.text
    assert 'hx-post="/score"' in response.text
    assert 'hx-trigger="input changed delay:300ms"' in response.text
    assert 'hx-swap="none"' in response.text
    assert 'id="map-description"' in response.text
    assert 'id="map-status"' in response.text
    assert 'id="map-legend"' in response.text
    assert (
        'id="map" role="region" aria-label="Interactive climate score map" aria-describedby="map-description map-legend map-status"'
        in response.text
    )
    assert 'id="score-results-list"' in response.text
    assert "maplibre-gl.css" in response.text
    assert "maplibre-gl.js" in response.text
    assert "pmtiles.js" in response.text
    assert "basemaps.js" in response.text


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


def test_map_script_initializes_maplibre_score_layer() -> None:
    response = client.get("/static/map.js")

    assert response.status_code == 200
    assert "new window.maplibregl.Map" in response.text
    assert "map.addSource(SCORE_SOURCE_ID" in response.text
    assert "map.addLayer({" in response.text
    assert 'window.maplibregl.addProtocol("pmtiles"' in response.text
    assert 'window.basemaps.layers("protomaps"' in response.text
    assert "url: PROTOMAPS_PM_TILES_URL" in response.text
    assert "renderScoreList(collection);" in response.text
    assert 'setMapStatus("Map libraries failed to load.");' in response.text


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

    assert isinstance(payload, list)
    assert payload

    for item in payload:
        assert set(item) == {"lat", "lon", "score"}
        assert isinstance(item["lat"], float)
        assert isinstance(item["lon"], float)
        assert 0 <= item["score"] <= 1


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
    assert first_response.json() == second_response.json()


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


def test_home_page_registers_htmx_handoff_script() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "htmx:afterRequest" in response.text
    assert "window.renderScores(scores);" in response.text
