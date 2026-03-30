# Pogodapp

Pogodapp is a climate preference search tool, not a weather app. The user describes an ideal climate profile and the app scores places around the world against long-term climate normals, then shows the result as an interactive world map.

## Goal

- Let the user tune climate preferences with simple controls
- Score global grid cells against those preferences
- Render scored matches on a world map so stronger results stand out clearly

## Stack

| Layer | Choice |
| --- | --- |
| Backend | FastAPI + DuckDB |
| Frontend render | Jinja2 |
| Frontend interaction | HTMX |
| Map rendering | MapLibre GL + Protomaps PMTiles |
| Hosting | Railway |
| Tooling | uv, Ruff, ty, pytest |

## Architecture

- One FastAPI app serves the initial page, static assets, and the scoring API
- `GET /` renders the page through Jinja2 with default slider values from backend config
- `POST /score` accepts form-encoded inputs and returns JSON as `[{lat, lon, score}, ...]`; out-of-range values now fail with `422`
- `DEFAULT_PREFERENCES` drives the form control ranges, `PreferenceInputs` enforces the same `/score` bounds, and `tests/test_app_shell.py` guards drift between them
- HTMX submits form changes to `/score`
- `htmx:afterRequest` bridges the response into the map update path
- `frontend/static/map.js` stays focused on rendering, not networking
- The current map uses MapLibre GL with the PMTiles browser protocol and Protomaps basemap layers loaded from public CDN/asset hosts
- If those external map assets are blocked or unavailable, the map falls back to an in-page failure message while the textual score list still renders
- DuckDB is the only runtime data store

## Data Model Direction

- Source: WorldClim monthly climate normals
- Prototype resolution: `0.5deg` grid
- Likely later target: `10'` grid
- Planned row shape: one row per grid cell
- Planned schema: `lat`, `lon`, `t_jan..t_dec`, `prec_jan..prec_dec`, `cloud_jan..cloud_dec`
- Distribution policy for `climate.duckdb` is still open: direct git, Git LFS, or build-time download

## Scoring Model

Each grid cell is scored month-by-month, then combined into one annual score.

- Temperature penalty is asymmetric: cold deviations and heat deviations use different slopes, with a comfort band around the ideal temperature
- Rain is one-sided: higher `rain_sensitivity` increases penalties for wetter cells without rewarding rain
- `sun_preference` currently uses a symmetric preference-fit curve against the stub sun signal
- The user-facing form contract uses `sun_preference`; later scoring maps that preference onto the cloud-cover signal
- Final scores are normalized to the `0..1` range for map rendering
- The current prototype renders scores as colored circle markers with larger, warmer markers indicating better matches

## Repository Layout

```text
.
|-- backend/
|   |-- config.py
|   |-- main.py
|   `-- scoring.py
|-- data/
|-- frontend/
|   |-- static/
|   |   |-- map.js
|   |   `-- styles.css
|   `-- templates/
|       `-- index.html
|-- tests/
|-- .github/workflows/ci.yml
`-- pyproject.toml
```

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn backend.main:app --reload
```

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

## Status

- The scaffold and tooling are in place
- Implementation work is tracked in GitHub issues
- The current backlog lives in `https://github.com/adrunkhuman/pogodapp/issues`
