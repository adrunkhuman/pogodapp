from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.climate_repository import StubClimateRepository
from backend.logging_config import configure_backend_logging
from backend.score_service import build_score_response
from backend.scoring import PreferenceInputs

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


def test_build_score_response_logs_step_timings(caplog: LogCaptureFixture) -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=7,
        heat_tolerance=5,
        rain_sensitivity=55,
        sun_preference=60,
    )

    configure_backend_logging()
    backend_logger = logging.getLogger("backend")
    original_handlers = backend_logger.handlers[:]
    original_propagate = backend_logger.propagate

    backend_logger.handlers = []
    backend_logger.propagate = True

    try:
        with caplog.at_level(logging.INFO, logger="backend"):
            response = build_score_response(StubClimateRepository(), preferences)
    finally:
        backend_logger.handlers = original_handlers
        backend_logger.propagate = original_propagate

    assert response["scores"]
    assert response["heatmap"].startswith("data:image/png;base64,")
    assert "score_request outcome=ok" in caplog.text
    assert "total_ms=" in caplog.text
    assert "cells_ms=" in caplog.text
    assert "cities_ms=" in caplog.text
    assert "scoring_ms=" in caplog.text
    assert "normalize_ms=" in caplog.text
    assert "ranking_ms=" in caplog.text
    assert "heatmap_ms=" in caplog.text
