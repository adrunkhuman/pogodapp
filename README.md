# Pogodapp

Pogodapp is a small app for finding climates you like.

It is not a weather app. It scores long-term climate normals against a few user preferences and returns a city list plus a map overlay.

## What It Does

- Takes a few climate preferences.
- Scores land cells around the world.
- Returns a ranked city list and a heatmap.
- Keeps everything on one page.

## Stack

| Layer | Choice |
| --- | --- |
| Backend | FastAPI, DuckDB, NumPy |
| Frontend render | Jinja2 |
| Frontend interaction | HTMX |
| Map | MapLibre GL with local static assets |
| Heatmap | Server-rendered PNG via Pillow |
| Tooling | uv, Ruff, ty, pytest, prek |

## How It Works

- `GET /` renders the page.
- `POST /score` accepts standard form fields and returns JSON.
- The response shape is `{"scores": [{"name", "continent", "country_code", "flag", "score", "lat", "lon", "probe_lat", "probe_lon"}, ...], "heatmap": "data:image/png;base64,..."}`.
- Empty or all-zero results return `{"scores": [], "heatmap": ""}`.
- `GET /probe` accepts the same preference fields plus `lat` and `lon`, then returns `{"found": bool, "overall_score": 0..1, "metrics": [{"key", "label", "value", "display_value", "score"}, ...]}`.
- `/probe` returns `{"found": false, "overall_score": 0.0, "metrics": []}` for ocean points, unmapped cells, or repositories without probe support.
- FastAPI handles HTTP and validation.
- Scoring, ranking, and heatmap rendering stay out of the route layer.
- `frontend/static/map.js` only renders. HTMX submits the form and hands the response to the map code.

## Data

- Source: WorldClim 2.1 monthly normals.
- Current baseline: native `5m` WorldClim grids, meaning 5 arc-minutes per cell.
- Runtime tables: `climate_cells(...)` and `cities(...)` inside `data/climate.duckdb`.
- If the database is missing, the app falls back to a small in-repo stub dataset.
- Cloud cover is currently approximated from solar radiation.

## Scoring

- Temperature uses one preferred temperature band plus separate summer and winter guardrails.
- The current prototype still applies those semantics to monthly mean temperature until the dataset grows dedicated high/low normals.
- Dryness is one-sided: wetter months only hurt the score.
- Sunshine preference becomes a cloud-cover tolerance threshold.
- Scores are normalized per request so the best available match lands at `1.0`.
- Ranked cities are grouped by continent, apply a regional diversity penalty so one cluster does not dominate the output, and keep a deeper reserve of up to 30 cities per continent for progressive reveal in the sidebar.

Coordinate notes:

- `lat` and `lon` are the display coordinates for the city marker and focused map state.
- `probe_lat` and `probe_lon` are the snapped climate-cell coordinates used for `/probe` lookups and tooltip scoring.

## Run Locally

```bash
uv sync
uv run pogodapp
```

Run against a non-default database file:

```bash
$env:POGODAPP_CLIMATE_DB = "data/climate-5m.duckdb"
uv run pogodapp
```

Optional flags:

```bash
uv run pogodapp --port 9000
uv run pogodapp --host 0.0.0.0
uv run pogodapp --no-reload
```

Notes:

- Default local URL: `http://127.0.0.1:8000`
- Live reload is on by default.
- If `data/climate.duckdb` exists, startup warms the climate matrix, city cache, and heatmap projection.
- If preload fails, startup logs the problem and requests fall back to the existing `503` path.

## Build Climate Data

```bash
uv run python scripts/build_climate_db.py
```

This builds `data/climate.duckdb`, downloads the required WorldClim rasters and GeoNames city source, keeps only valid land cells, and populates `climate_cells` plus `cities`.

Optional flag:

```bash
uv run python scripts/build_climate_db.py --resolution 10m
```

Supported resolutions are `10m`, `5m`, `2.5m`, and `30s`. The default is now `5m`.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```
