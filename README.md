# Pogodapp

Pogodapp helps people search for climates they like.

It is not a weather app. The user sets a few climate preferences, the backend scores long-term climate normals across the world, and the UI returns a ranked list of cities plus a map overlay.

## What It Does

- Accepts climate preferences through one form.
- Scores global land cells against those preferences.
- Returns a ranked city shortlist and a heatmap image for the map.
- Keeps the whole interaction on one page.

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

- `GET /` renders the app shell with slider defaults and shared map config.
- `POST /score` accepts standard form fields and returns JSON.
- The response shape is `{"scores": [{"name", "country_code", "flag", "score"}, ...], "heatmap": "data:image/png;base64,..."}`.
- Empty or all-zero results return `{"scores": [], "heatmap": ""}`.
- FastAPI handles HTTP and validation.
- Scoring, city ranking, and heatmap rendering stay in backend service modules instead of the route layer.
- `frontend/static/map.js` only renders. HTMX submits the form, and `htmx:afterRequest` hands the JSON payload to the map code.

## Data

- Source: WorldClim 2.1 monthly normals.
- Current baseline: native `10m` WorldClim grids, meaning 10 arc-minutes per cell.
- Runtime tables: `climate_cells(...)` and `cities(...)` inside `data/climate.duckdb`.
- If the database is missing, the app falls back to a small in-repo stub dataset.
- Cloud cover is currently approximated from solar radiation so the scoring schema can stay stable while the real source is still undecided.

## Scoring

- Temperature uses an ideal value plus separate cold and heat tolerance.
- Rain is one-sided: more rain only hurts the score.
- Sun preference becomes a cloud-cover tolerance threshold.
- Scores are normalized per request so the best available match lands at `1.0`.
- The city list is capped at 20 results and applies a regional diversity penalty so one cluster does not dominate the output.

## Run Locally

```bash
uv sync
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
- If preload fails, startup logs the problem and requests still use the existing `503` path.

## Build Climate Data

```bash
uv run python scripts/build_climate_db.py
```

This builds `data/climate.duckdb`, downloads the required WorldClim rasters and GeoNames city source, keeps only valid land cells, and populates both `climate_cells` and `cities`.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

## Current State

- Recent work moved the app onto the native `10m` grid, server-side heatmap rendering, ranked city results, and cached vectorized scoring.
- MapLibre and the world backdrop are served locally.
- HTMX is still loaded externally. That policy is not settled yet.

## Open Questions

- `#45`: is `10m` still the right baseline, and are we describing it clearly?
- `#44`: should frontend runtime assets be fully vendored or intentionally mixed?
- `#43`: how should the score surface handle local maxima better?
- `#41`: are the current temperature controls the right product model?
- `#11`: how should `climate.duckdb` be distributed?
