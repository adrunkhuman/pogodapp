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
- `GET /` renders the shell only; the first results arrive through an automatic HTMX `load` `POST /score` using the backend default preferences.
- `POST /score` accepts `preferred_day_temperature`, `summer_heat_limit`, `winter_cold_limit`, `dryness_preference`, and `sunshine_preference` as form fields and returns JSON.
- After the initial page-load score, HTMX re-submits the form on settled control `change` events with a `500ms` delay, so quick slider edits collapse into one `/score` request instead of fan-out bursts.
- The response shape is `{"scores": [{"name", "continent", "country_code", "flag", "score", "lat", "lon", "probe_lat", "probe_lon"}, ...], "heatmap": "data:image/png;base64,..."}`.
- Empty or all-zero results return `{"scores": [], "heatmap": ""}`.
- `/score` is rate-limited to `30/minute` per client and returns `429` when that limit is exceeded.
- `GET /probe` accepts the same preference fields plus `lat` and `lon`, then returns `{"found": bool, "overall_score": 0..1, "metrics": [{"key", "label", "value", "display_value", "score"}, ...]}`.
- `/probe` is rate-limited to `120/minute` per client and returns `429` when that limit is exceeded.
- `preferred_day_temperature` accepts `-5..35`.
- `summer_heat_limit` accepts `-5..42` and must stay greater than or equal to `preferred_day_temperature`.
- `winter_cold_limit` accepts `-15..35` and must stay less than or equal to `preferred_day_temperature`.
- Both `/score` and `/probe` return `422` when those temperature fields violate the ordering rule.
- `/probe` metric keys are `temp`, `high`, `low`, `rain`, and `sun`.
- `/probe` temperature metrics mean: `temp` = typical day from median monthly high, `high` = hottest-month high, `low` = coldest-month low.
- `/probe` returns `{"found": false, "overall_score": 0.0, "metrics": []}` for ocean points, unmapped cells, or repositories without probe support.
- FastAPI handles HTTP and validation.
- Scoring, ranking, and heatmap rendering stay out of the route layer.
- `frontend/static/map.js` only renders. HTMX submits the form, shows a brief `Calculating...` status in the controls panel, and hands the JSON response to the map code. Any non-`200` `/score` response shows a generic inline error in the controls panel.
- Tooltip probes snap hover points to the climate grid and cache results by snapped cell plus current preferences.

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
POGODAPP_CLIMATE_DB=data/climate-5m.duckdb uv run pogodapp
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
- Large dynamic responses are gzip-compressed when the client sends `Accept-Encoding: gzip`; small responses like `/health` may stay uncompressed.

## Deployment

The app can bootstrap `climate.duckdb` on startup, and deployment remains env-driven.

For container deployments, mount persistent storage and point the data env vars at that mount. For example, if your runtime volume is mounted at `/app/data`, set `POGODAPP_DATA_DIR=/app/data`, `POGODAPP_CLIMATE_DB=/app/data/climate.duckdb`, and `POGODAPP_CLIMATE_CACHE_DIR=/app/data/worldclim`.

Relevant environment variables:

- `POGODAPP_DATA_DIR`: base directory for runtime artifacts. Default: `data`.
- `POGODAPP_CLIMATE_DB`: explicit DuckDB path. Default: `{POGODAPP_DATA_DIR}/climate.duckdb`.
- `POGODAPP_CLIMATE_CACHE_DIR`: cache directory for downloaded source archives. Default: `{POGODAPP_DATA_DIR}/worldclim`.
- `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING`: when `true`, startup builds the climate database before launching the app.
- `POGODAPP_CLIMATE_RESOLUTION`: WorldClim resolution used by startup bootstrap. Default: `5m`.
- `POGODAPP_HOST`: bind host override. Default: `127.0.0.1` locally, `0.0.0.0` when a platform injects `PORT`.
- `PORT`: bind port. Default: `8000`.
- `POGODAPP_RELOAD`: toggles Uvicorn reload. Default: on for local runs, off when `PORT` is injected.
- `LOG_LEVEL`: log level override for app and server logs. Default: `INFO`.
- `LOG_FORMAT`: stdout formatter for `backend`, `uvicorn`, and `uvicorn.error`. Supported values: `json` and `plain`. Default: `json`. Unset or unrecognized values fall back to `json`.

## Observability

- Pogodapp emits its own request logs instead of Uvicorn access logs.
- Logs are JSON by default. Set `LOG_FORMAT=plain` for traditional local stdout logs.
- Log format is no longer inferred from Railway or any other platform environment variables.
- JSON logs always include `level`, `message`, `timestamp`, `logger`, and any event-specific fields attached to the record.
- Main event families are `http_request`, `startup`, `startup_db`, `startup_bootstrap`, `startup_preload`, `startup_default_score`, `score_request`, `probe_request`, and `runtime_memory`.

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
