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
| Map rendering | MapLibre GL + local world backdrop |
| Hosting | Railway |
| Tooling | uv, Ruff, ty, pytest |

## Architecture

- One FastAPI app serves the initial page, static assets, and the scoring API
- `GET /` renders the page through Jinja2 with default slider values from backend config
- `POST /score` accepts form-encoded inputs and returns JSON as `[{name, country_code, flag, score}, ...]`; out-of-range values now fail with `422`
- Climate data access sits behind a small repository boundary so routing code does not know whether rows come from stubs or DuckDB; DuckDB-backed runtime now reads both climate cells and pre-mapped cities from the same artifact
- `DEFAULT_PREFERENCES` drives the form control ranges, `PreferenceInputs` enforces the same `/score` bounds, and `tests/test_app_shell.py` guards drift between them
- HTMX submits form changes to `/score`
- `htmx:afterRequest` bridges the response into the map update path
- `frontend/static/map.js` stays focused on rendering, not networking
- The current map uses MapLibre GL with app-served static assets and a lightweight local GeoJSON world backdrop
- The score overlay now renders as a colored grid-cell surface on top of that backdrop, and the textual score list now ranks nearby cities instead of raw coordinates
- DuckDB is the only runtime data store
- The app uses `data/climate.duckdb` automatically when that file exists; otherwise it falls back to stub climate rows and stub cities until the dataset lands

## Data Model Direction

- Source: WorldClim monthly climate normals
- Resolution: `10'` (~18km) grid — native WorldClim resolution, no aggregation
- Land mask: ocean pixels are identified by WorldClim's nodata sentinel (~-3.4e38) and excluded; only land cells enter the DB
- Planned row shape: one row per grid cell (~675k land cells)
- Runtime tables: `climate_cells(lat, lon, t_jan..t_dec, prec_jan..prec_dec, cloud_jan..cloud_dec)` and `cities(name, country_code, lat, lon, cell_lat, cell_lon)`
- Distribution policy for `climate.duckdb` is still open: direct git, Git LFS, or build-time download

## Scoring Model

Each grid cell is scored month-by-month, then combined into one annual score.

- Temperature penalty is asymmetric: cold deviations and heat deviations use different slopes, with a comfort band around the ideal temperature
- Rain is one-sided: higher `rain_sensitivity` increases penalties for wetter months without rewarding rain
- `sun_preference` maps onto a cloud-cover tolerance threshold and then applies a one-sided misery-style cloud penalty
- The stub scorer now uses one grid-cell record with 12 monthly values per climate signal and averages monthly composite scores into one annual score; DuckDB-backed data still lands in later issues
- Final scores are normalized to the `0..1` range for map rendering
- The current prototype renders scores as colored climate cells so stronger matching regions remain readable at world scale
- Cities are downloaded from GeoNames `cities15000.zip`, filtered onto valid land climate cells at build time, then ranked at request time from the already-scored cells

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
|   |   |-- data/
|   |   |   `-- world.geojson
|   |   |-- vendor/
|   |   |   |-- maplibre-gl.css
|   |   |   `-- maplibre-gl.js
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
uv run pogodapp
```

- The app serves its own map assets from `/static`, so local map rendering no longer depends on remote PMTiles, sprite, or glyph hosts

## Local Testing

```bash
uv run pogodapp
```

- Starts the FastAPI app on `http://127.0.0.1:8000`
- Uses live reload by default for local iteration
- Still runs with stub climate rows when `data/climate.duckdb` is absent

Optional flags:

```bash
uv run pogodapp --port 9000
uv run pogodapp --no-reload
```

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

## Data Build

```bash
uv run python scripts/build_climate_db.py
```

- Builds `data/climate.duckdb` with both `climate_cells` and the pre-mapped `cities` table
- Downloads and caches GeoNames `cities15000.zip` under `data/worldclim/` by default
- If your local `data/climate.duckdb` predates the `cities` table, rebuild it with this command

## Build Climate Data

```bash
uv run python scripts/build_climate_db.py
```

- The pipeline downloads WorldClim 2.1 `10m` monthly `tavg`, `prec`, and `srad` rasters into `data/worldclim/`
- It aggregates each `3x3` block into the planned `0.5deg` prototype grid and writes `data/climate.duckdb`
- `cloud_jan..cloud_dec` currently come from an inverted month-wise solar-radiation proxy so the scoring schema stays usable until a direct cloud-cover source is chosen
- The build validates the final schema and checks that the row count stays in the expected rough prototype range

## Status

- The scaffold and tooling are in place
- Implementation work is tracked in GitHub issues
- The current backlog lives in `https://github.com/adrunkhuman/pogodapp/issues`
