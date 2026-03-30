from fastapi.testclient import TestClient

from backend.config import DEFAULT_PREFERENCES, PREFERENCE_FIELD_NAMES
from backend.main import app

client = TestClient(app)


def test_home_page_renders() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Pogodapp" in response.text
    assert "Climate preference search" in response.text
    assert '<form id="preferences" hx-post="/score" hx-trigger="change delay:200ms">' in response.text
    assert '<div id="map" role="img" aria-label="Map results placeholder"></div>' in response.text


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


def test_static_files_are_served() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "font-family" in response.text
