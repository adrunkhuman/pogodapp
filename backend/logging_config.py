from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime


def _is_railway() -> bool:
    return os.getenv("RAILWAY_ENVIRONMENT") is not None or os.getenv("RAILWAY_SERVICE_NAME") is not None


class _JSONFormatter(logging.Formatter):
    """Single-line JSON for Railway log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
        }
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))


_PLAIN_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_PLAIN_DATEFMT = "%H:%M:%S"


def configure_backend_logging() -> None:
    """Configure the backend logger for Railway (JSON/stdout) or local (plain/stdout).

    Uses LOG_LEVEL env var to override the default INFO level.
    Railway is detected via RAILWAY_ENVIRONMENT or RAILWAY_SERVICE_NAME.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    backend_logger = logging.getLogger("backend")
    backend_logger.setLevel(level)
    backend_logger.propagate = False

    if backend_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if _is_railway():
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_PLAIN_DATEFMT))

    backend_logger.addHandler(handler)
