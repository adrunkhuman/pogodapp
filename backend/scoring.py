from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class PreferenceInputs:
    """Normalized form inputs passed into the scoring layer."""

    ideal_temperature: int
    cold_tolerance: int
    heat_tolerance: int
    rain_sensitivity: int
    sun_preference: int


class ScorePoint(TypedDict):
    """JSON score payload returned to the frontend."""

    lat: float
    lon: float
    score: float


def score_preferences(preferences: PreferenceInputs) -> list[ScorePoint]:
    """Return a placeholder score payload until climate scoring lands."""
    _ = preferences
    return []
