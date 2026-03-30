from fastapi.testclient import TestClient

from backend.config import DEFAULT_PREFERENCES
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
        assert f'value="{preference.value}"' in response.text


def test_static_files_are_served() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "font-family" in response.text
