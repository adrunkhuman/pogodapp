from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field


class PreferenceInputs(BaseModel):
    """Normalized form inputs passed into the scoring layer."""

    ideal_temperature: int = Field(ge=-10, le=35)
    cold_tolerance: int = Field(ge=0, le=15)
    heat_tolerance: int = Field(ge=0, le=15)
    rain_sensitivity: int = Field(ge=0, le=100)
    sun_preference: int = Field(ge=0, le=100)


@dataclass(frozen=True, slots=True)
class StubClimateCell:
    """Deterministic placeholder climate data for one grid cell."""

    lat: float
    lon: float
    temperature: float
    rain_index: int
    sun_index: int


class ScorePoint(TypedDict):
    """JSON score payload returned to the frontend."""

    lat: float
    lon: float
    score: float


STUB_CLIMATE_CELLS: tuple[StubClimateCell, ...] = (
    StubClimateCell(lat=37.5, lon=-122.0, temperature=17.0, rain_index=30, sun_index=72),
    StubClimateCell(lat=41.9, lon=12.5, temperature=19.0, rain_index=42, sun_index=68),
    StubClimateCell(lat=-33.9, lon=18.4, temperature=21.0, rain_index=28, sun_index=80),
    StubClimateCell(lat=35.7, lon=139.7, temperature=16.0, rain_index=58, sun_index=54),
    StubClimateCell(lat=-22.9, lon=-43.2, temperature=25.0, rain_index=63, sun_index=77),
)


def clamp_score(value: float) -> float:
    """Keep scores in the API's normalized range."""
    return max(0.0, min(1.0, value))


def temperature_score(cell: StubClimateCell, preferences: PreferenceInputs) -> float:
    """Use cold vs. heat tolerance based on which side of the ideal a cell falls."""
    delta = cell.temperature - preferences.ideal_temperature
    tolerance = preferences.heat_tolerance if delta >= 0 else preferences.cold_tolerance
    scale = max(tolerance, 1)
    return clamp_score(1 - abs(delta) / scale)


def rain_score(observed: int, sensitivity: int) -> float:
    """Rain is one-sided: higher sensitivity should only penalize wetter cells more."""
    return clamp_score(1 - (observed / 100) * (sensitivity / 100))


def preference_score(observed: int, preferred: int) -> float:
    """Approximate fit for 0..100 preference-style controls."""
    return clamp_score(1 - abs(observed - preferred) / 100)


def score_preferences(preferences: PreferenceInputs) -> list[ScorePoint]:
    """Return deterministic placeholder scores until real climate scoring lands."""
    scores: list[ScorePoint] = []

    for cell in STUB_CLIMATE_CELLS:
        score = clamp_score(
            (
                temperature_score(cell, preferences)
                + rain_score(cell.rain_index, preferences.rain_sensitivity)
                + preference_score(cell.sun_index, preferences.sun_preference)
            )
            / 3
        )
        scores.append({"lat": cell.lat, "lon": cell.lon, "score": score})

    return scores
