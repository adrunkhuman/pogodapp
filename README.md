# Pogodapp

Pogodapp is a climate preference search tool, not a weather app. The user describes an ideal climate profile and the app scores places around the world against long-term climate normals, then shows the result as an interactive map heatmap.

## Goal

- Let the user tune climate preferences with simple controls
- Score global grid cells against those preferences
- Render good matches brightly and bad matches dimly on a world map

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
- `POST /score` accepts form-encoded inputs and returns JSON as `[{lat, lon, score}, ...]`
- HTMX submits form changes to `/score`
- `htmx:afterRequest` bridges the response into the map update path
- `frontend/static/map.js` stays focused on rendering, not networking
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
- Rain penalty scales with precipitation and user sensitivity
- Cloud penalty uses the agreed misery-style curve, harsher at high cover
- Final scores are normalized to the `0..1` range for map rendering

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
