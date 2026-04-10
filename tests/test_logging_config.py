from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from backend.logging_config import _JSONFormatter, configure_backend_logging

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_json_formatter_emits_railway_friendly_fields() -> None:
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="backend.main",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="http request finished",
        args=(),
        exc_info=None,
    )
    record.event = "http_request"
    record.httpStatus = 200

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "warn"
    assert payload["logger"] == "backend.main"
    assert payload["message"] == "http request finished"
    assert payload["event"] == "http_request"
    assert payload["httpStatus"] == 200
    assert "timestamp" in payload


def test_json_formatter_serializes_nested_extra_values() -> None:
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="backend.main",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="score request finished",
        args=(),
        exc_info=None,
    )
    record.event = "score_request"
    record.metrics = {"durations": [1.2, 3], "cache": ("hit", True)}

    payload = json.loads(formatter.format(record))

    assert payload["metrics"] == {"durations": [1.2, 3], "cache": ["hit", True]}


def test_configure_backend_logging_uses_plain_formatter_locally(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_NAME", raising=False)

    configure_backend_logging()

    for logger_name in ("backend", "uvicorn", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        assert logger.propagate is False
        assert len(logger.handlers) == 1
        assert not isinstance(logger.handlers[0].formatter, _JSONFormatter)


def test_configure_backend_logging_applies_log_level_and_replaces_handlers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_NAME", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    configure_backend_logging()
    original_handlers = {name: logging.getLogger(name).handlers[0] for name in ("backend", "uvicorn", "uvicorn.error")}

    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    configure_backend_logging()

    for logger_name in ("backend", "uvicorn", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1
        assert logger.handlers[0] is not original_handlers[logger_name]
        assert isinstance(logger.handlers[0].formatter, _JSONFormatter)


def test_configure_backend_logging_uses_json_formatter_on_railway(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")

    configure_backend_logging()

    for logger_name in ("backend", "uvicorn", "uvicorn.error"):
        logger = logging.getLogger(logger_name)
        assert logger.propagate is False
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, _JSONFormatter)
