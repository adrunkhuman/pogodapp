# Development

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

## Scoring inputs

`POST /score` accepts these form fields:

| Field | Range |
| --- | --- |
| `preferred_day_temperature` | `-5..35` |
| `summer_heat_limit` | `-5..42` |
| `winter_cold_limit` | `-15..35` |

Constraints: `preferred_day_temperature <= summer_heat_limit` and `preferred_day_temperature >= winter_cold_limit`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `POGODAPP_DATA_DIR` | `data` | Base directory for runtime data. |
| `POGODAPP_CLIMATE_DB` | `{POGODAPP_DATA_DIR}/climate.duckdb` | DuckDB path. |
| `POGODAPP_CLIMATE_CACHE_DIR` | `{POGODAPP_DATA_DIR}/worldclim` | Download cache directory. |
| `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING` | disabled | Build the database on startup when missing; otherwise use stub data. |
| `POGODAPP_CLIMATE_RESOLUTION` | `5m` | Bootstrap resolution. |
| `POGODAPP_HOST` | — | Bind host override. |
| `PORT` | `8000` | Bind port. |
| `POGODAPP_RELOAD` | — | Toggles reload mode. |
| `LOG_LEVEL` | `INFO` | Log level override. |
| `LOG_FORMAT` | `json` | `json` or `plain`. |

## Docker

```bash
docker build -t pogodapp .
docker run -p 8000:8000 pogodapp
```

The image excludes generated DuckDB data. Mount persistent data when running with a real climate database:

```bash
docker run -p 8000:8000 -v pogodapp-data:/app/data pogodapp
```

Set `POGODAPP_BUILD_CLIMATE_DB_IF_MISSING=true` to bootstrap `/app/data/climate.duckdb` on startup.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```