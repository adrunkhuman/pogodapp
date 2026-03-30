from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_home_page_renders() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Pogodapp" in response.text
    assert "Climate preference search" in response.text


def test_static_files_are_served() -> None:
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "font-family" in response.text
