from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

_JSON_LEVEL_NAMES = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}
_LOG_RECORD_DEFAULTS = frozenset(logging.makeLogRecord({}).__dict__)
_LOG_FORMAT_ENV_VAR = "LOG_FORMAT"


def _use_plain_logging() -> bool:
    return os.getenv(_LOG_FORMAT_ENV_VAR, "json").strip().lower() == "plain"


def _serialize_log_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple):
        return [_serialize_log_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_log_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_log_value(item) for key, item in value.items()}
    return str(value)


class _JSONFormatter(logging.Formatter):
    """JSON formatter for stdout logs."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "level": _JSON_LEVEL_NAMES.get(record.levelno, "info"),
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
        }
        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_DEFAULTS or key.startswith("_"):
                continue
            entry[key] = _serialize_log_value(value)
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))


_PLAIN_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_PLAIN_DATEFMT = "%H:%M:%S"


def _build_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if _use_plain_logging():
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_PLAIN_DATEFMT))
    else:
        handler.setFormatter(_JSONFormatter())

    return handler


def _configure_named_logger(name: str, level: int) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(_build_handler(level))


def configure_backend_logging() -> None:
    """Configure backend and Uvicorn stdout logging.

    Logs are JSON by default. Set LOG_FORMAT=plain for human-oriented local output.
    Uses LOG_LEVEL env var to override the default INFO level.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    for logger_name in ("backend", "uvicorn", "uvicorn.error"):
        _configure_named_logger(logger_name, level)
