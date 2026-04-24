# Pogodapp

Find climates you like.

Pogodapp is a small FastAPI app that scores long-term climate normals against a few preferences, then shows matching cities and a world heatmap.

## What it does
- Single-page app: controls, map, and ranked results stay on one screen.
- Scores land cells from WorldClim climate normals.
- Ranks cities near good-scoring cells.
- Lets you hover the map to inspect one cell with `/probe`.

## Stack
- FastAPI
- DuckDB
- NumPy
- Jinja2 for first render
- HTMX for form submission
- MapLibre GL for the map
- Pillow for PNG heatmaps

## Routes
- `GET /` renders the app shell.
- `POST /score` accepts standard form fields and returns JSON with ranked cities plus a `heatmap_url`.
- `GET /heatmap` returns the rendered PNG for the current preferences, or `204` when nothing matches.
- `GET /probe` returns a score breakdown for one map point.
- `GET /health` is a basic health check.

Input fields for scoring:
- `preferred_day_temperature`
- `summer_heat_limit`
- `winter_cold_limit`
- `dryness_preference`
- `sunshine_preference`

Temperature rules:
- `preferred_day_temperature`: `-5..35`
- `summer_heat_limit`: `-5..42`
- `winter_cold_limit`: `-15..35`
- `preferred_day_temperature <= summer_heat_limit`
- `preferred_day_temperature >= winter_cold_limit`

Rate limits:
- `/score`: `30/minute`
- `/heatmap`: `30/minute`
- `/probe`: `120/minute`

## Scoring
- Temperature is the main signal.
- Dryness and sunshine matter more when the user moves those sliders away from neutral.
- Scores are normalized per request, so the best available match is `1.0`.
- City results are spread out so one region does not flood the list.

## Data
- Source: WorldClim 2.1 monthly normals.
- Default runtime dataset: native `5m` resolution.
- Main database: `data/climate.duckdb`.
- `climate.duckdb` is generated, not committed to the repo.
- If the database is missing, the app falls back to a small in-repo stub dataset by default.
- Set `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING=true` to generate and validate `climate.duckdb` during app launch.
- Production deployments should use persistent `data/` storage so startup generation is a one-time bootstrap.
- Older databases from before the `tmin_*` and `tmax_*` schema change are not compatible. Rebuild them.

## Run locally
```bash
uv sync
uv run pogodapp
```

Default local URL: `http://127.0.0.1:8000`

Useful variants:

```bash
POGODAPP_CLIMATE_DB=data/climate-5m.duckdb uv run pogodapp
uv run pogodapp --port 9000
uv run pogodapp --host 0.0.0.0
uv run pogodapp --no-reload
```

## Build climate data
```bash
uv run python scripts/build_climate_db.py
```

Optional resolution override:

```bash
uv run python scripts/build_climate_db.py --resolution 10m
```

Supported resolutions: `10m`, `5m`, `2.5m`, `30s`.

## Config
- `POGODAPP_DATA_DIR`: base directory for runtime data. Default: `data`
- `POGODAPP_CLIMATE_DB`: DuckDB path. Default: `{POGODAPP_DATA_DIR}/climate.duckdb`
- `POGODAPP_CLIMATE_CACHE_DIR`: download cache directory. Default: `{POGODAPP_DATA_DIR}/worldclim`
- `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING`: build the database on startup when missing. Default: disabled, using stub data instead
- `POGODAPP_CLIMATE_RESOLUTION`: bootstrap resolution. Default: `5m`
- `POGODAPP_HOST`: bind host override
- `PORT`: bind port. Default: `8000`
- `POGODAPP_RELOAD`: toggles reload mode
- `LOG_LEVEL`: log level override. Default: `INFO`
- `LOG_FORMAT`: `json` or `plain`. Default: `json`

## Development
```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```
