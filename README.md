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
- `POST /score` accepts `preferred_day_temperature`, `summer_heat_limit`, `winter_cold_limit`, `dryness_preference`, and `sunshine_preference` as form fields and returns JSON.
- The response shape is `{"scores": [{"name", "continent", "country_code", "flag", "score", "lat", "lon", "probe_lat", "probe_lon"}, ...], "heatmap": "data:image/png;base64,..."}`.
- Empty or all-zero results return `{"scores": [], "heatmap": ""}`.
- `GET /probe` accepts the same preference fields plus `lat` and `lon`, then returns `{"found": bool, "overall_score": 0..1, "metrics": [{"key", "label", "value", "display_value", "score"}, ...]}`.
- Both `/score` and `/probe` return `422` when `preferred_day_temperature` falls above `summer_heat_limit` or below `winter_cold_limit`.
- `/probe` metric keys are `temp`, `high`, `low`, `rain`, and `sun`.
- `/probe` temperature metrics mean: `temp` = typical day from median monthly high, `high` = hottest-month high, `low` = coldest-month low.
- `/probe` returns `{"found": false, "overall_score": 0.0, "metrics": []}` for ocean points, unmapped cells, or repositories without probe support.
- FastAPI handles HTTP and validation.
- Scoring, ranking, and heatmap rendering stay out of the route layer.
- `frontend/static/map.js` only renders. HTMX submits the form and hands the response to the map code.

## Data

- Source: WorldClim 2.1 monthly normals.
- Current baseline: native `5m` WorldClim grids, meaning 5 arc-minutes per cell.
- Runtime tables: `climate_cells(...)` and `cities(...)` inside `data/climate.duckdb`.
- If the database is missing, the app falls back to a small in-repo stub dataset.
- Temperature inputs now store monthly mean, monthly average daily low, and monthly average daily high normals.
- Older `data/climate.duckdb` files from before issue `#41` are incompatible because the runtime now requires `tmin_*` and `tmax_*` columns. Rebuild the database instead of trying to migrate it in place.
- Cloud cover is currently approximated from solar radiation.

## Scoring

- Temperature uses average daily highs for the preferred day and summer limit, plus average daily lows for the winter limit.
- Temperature is the dominant block in the final score; rain and sun only gain more influence when the user gives stronger non-neutral answers.
- Dryness uses a profile score built from typical monthly rain plus the wettest month, so a bad wet season still matters.
- Sunshine uses average cloud plus the gloomiest month, so one dark season still hurts.
- The final composite is multiplicative rather than a plain weighted average, so a place cannot fully buy back a bad summer or winter with good rain/sun traits.
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

## Deployment

The app can bootstrap `climate.duckdb` on startup, but deployment config stays provider-agnostic and env-driven.

Relevant environment variables:

- `POGODAPP_DATA_DIR`: base directory for runtime artifacts. Default: `data`.
- `POGODAPP_CLIMATE_DB`: explicit DuckDB path. Default: `{POGODAPP_DATA_DIR}/climate.duckdb`.
- `POGODAPP_CLIMATE_CACHE_DIR`: cache directory for downloaded source archives. Default: `{POGODAPP_DATA_DIR}/worldclim`.
- `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING`: when `true`, startup builds the climate database before launching the app.
- `POGODAPP_CLIMATE_RESOLUTION`: WorldClim resolution used by startup bootstrap. Default: `5m`.
- `POGODAPP_HOST`: bind host override. Default: `127.0.0.1` locally, `0.0.0.0` when a platform injects `PORT`.
- `PORT`: bind port. Default: `8000`.
- `POGODAPP_RELOAD`: toggles Uvicorn reload. Default: on for local runs, off when `PORT` is injected.

Generic deployment shape:

1. Mount persistent storage for the data directory.
2. Set `POGODAPP_DATA_DIR` and `POGODAPP_CLIMATE_DB` to that persistent path.
3. Set `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING=true` for first-boot bootstrap.
4. Start the app with `uv run pogodapp`.

Railway example:

1. Attach a volume mounted at `/app/data`.
2. Set service variables:
   - `POGODAPP_DATA_DIR=/app/data`
   - `POGODAPP_CLIMATE_DB=/app/data/climate.duckdb`
   - `POGODAPP_CLIMATE_CACHE_DIR=/app/data/worldclim`
   - `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING=true`
   - `POGODAPP_CLIMATE_RESOLUTION=5m`
   - `POGODAPP_RELOAD=false`
3. Use the checked-in `railway.toml`, which starts the service with `uv run pogodapp`.

Railway-specific notes from the platform docs:

- Volumes are mounted only at runtime, not during build or pre-deploy.
- Variables configured in Railway are exposed to the app as normal environment variables.
- The first deploy on an empty volume can take a while because the app builds the climate database before Uvicorn starts.

## Build Climate Data

```bash
uv run python scripts/build_climate_db.py
```

This builds `data/climate.duckdb`, downloads the required WorldClim rasters and GeoNames city source, keeps only land cells whose monthly mean temperature, min temperature, max temperature, precipitation, and solar-radiation inputs are all finite for all 12 months, and populates `climate_cells` plus `cities`.

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
