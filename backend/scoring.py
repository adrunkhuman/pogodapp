from dataclasses import dataclass
from typing import TypedDict

from pydantic import BaseModel, Field

MONTHS_PER_YEAR = 12
TEMPERATURE_COMFORT_BAND_C = 1.5
TEMPERATURE_SLOPE_BASE_C = 6.0
SATURATING_MONTHLY_RAIN_MM = 300.0
MAX_TOLERATED_CLOUD_COVER = 85.0
MIN_TOLERATED_CLOUD_COVER = 15.0


class PreferenceInputs(BaseModel):
    """Validated `/score` form inputs before FastAPI hands them to scoring."""

    ideal_temperature: int = Field(ge=-10, le=35)
    cold_tolerance: int = Field(ge=0, le=15)
    heat_tolerance: int = Field(ge=0, le=15)
    rain_sensitivity: int = Field(ge=0, le=100)
    sun_preference: int = Field(ge=0, le=100)


@dataclass(frozen=True, slots=True)
class ClimateCell:
    """Monthly climate normals for one scored grid cell."""

    lat: float
    lon: float
    temperature_c: tuple[float, ...]
    precipitation_mm: tuple[float, ...]
    cloud_cover_pct: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject incomplete monthly rows before they reach scoring code."""
        if len(self.temperature_c) != MONTHS_PER_YEAR:
            msg = "temperature_c must contain 12 monthly values"
            raise ValueError(msg)

        if len(self.precipitation_mm) != MONTHS_PER_YEAR:
            msg = "precipitation_mm must contain 12 monthly values"
            raise ValueError(msg)

        if len(self.cloud_cover_pct) != MONTHS_PER_YEAR:
            msg = "cloud_cover_pct must contain 12 monthly values"
            raise ValueError(msg)


class ScorePoint(TypedDict):
    """JSON score payload returned to the frontend."""

    lat: float
    lon: float
    score: float


STUB_CLIMATE_CELLS: tuple[ClimateCell, ...] = (
    ClimateCell(
        lat=37.5,
        lon=-122.0,
        temperature_c=(11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 19.0, 20.0, 20.0, 18.0, 15.0, 12.0),
        precipitation_mm=(110.0, 90.0, 70.0, 35.0, 15.0, 5.0, 1.0, 2.0, 8.0, 30.0, 70.0, 100.0),
        cloud_cover_pct=(60, 55, 50, 40, 30, 20, 15, 15, 20, 30, 45, 55),
    ),
    ClimateCell(
        lat=41.9,
        lon=12.5,
        temperature_c=(8.0, 9.0, 12.0, 15.0, 19.0, 23.0, 26.0, 26.0, 23.0, 18.0, 13.0, 9.0),
        precipitation_mm=(80.0, 70.0, 60.0, 65.0, 45.0, 30.0, 20.0, 25.0, 70.0, 105.0, 115.0, 95.0),
        cloud_cover_pct=(55, 50, 45, 40, 30, 20, 15, 15, 25, 35, 50, 55),
    ),
    ClimateCell(
        lat=-33.9,
        lon=18.4,
        temperature_c=(21.0, 21.0, 20.0, 18.0, 16.0, 14.0, 13.0, 14.0, 15.0, 17.0, 18.0, 20.0),
        precipitation_mm=(15.0, 18.0, 20.0, 35.0, 75.0, 95.0, 110.0, 90.0, 45.0, 30.0, 20.0, 15.0),
        cloud_cover_pct=(20, 22, 28, 35, 45, 55, 60, 58, 45, 35, 28, 22),
    ),
    ClimateCell(
        lat=35.7,
        lon=139.7,
        temperature_c=(6.0, 7.0, 10.0, 15.0, 19.0, 22.0, 26.0, 28.0, 24.0, 18.0, 13.0, 8.0),
        precipitation_mm=(55.0, 60.0, 115.0, 125.0, 135.0, 160.0, 155.0, 170.0, 210.0, 195.0, 95.0, 55.0),
        cloud_cover_pct=(45, 45, 50, 55, 60, 70, 75, 70, 65, 55, 45, 40),
    ),
    ClimateCell(
        lat=-22.9,
        lon=-43.2,
        temperature_c=(27.0, 28.0, 27.0, 26.0, 24.0, 22.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0),
        precipitation_mm=(175.0, 160.0, 180.0, 140.0, 110.0, 90.0, 75.0, 70.0, 110.0, 130.0, 145.0, 165.0),
        cloud_cover_pct=(40, 38, 42, 45, 50, 55, 52, 48, 45, 42, 40, 38),
    ),
)


def clamp_score(value: float) -> float:
    """Keep scores in the API's normalized range."""
    return max(0.0, min(1.0, value))


def temperature_score(observed_temperature_c: float, preferences: PreferenceInputs) -> float:
    """Apply a comfort band around the ideal and separate cold vs. heat slopes outside it."""
    delta = observed_temperature_c - preferences.ideal_temperature
    distance_from_band = max(abs(delta) - TEMPERATURE_COMFORT_BAND_C, 0.0)

    if distance_from_band == 0:
        return 1.0

    tolerance = preferences.heat_tolerance if delta >= 0 else preferences.cold_tolerance
    slope_span = TEMPERATURE_SLOPE_BASE_C + tolerance
    return clamp_score(1 - distance_from_band / slope_span)


def rain_score(monthly_precipitation_mm: float, sensitivity: int) -> float:
    """Penalize wetter months without ever rewarding rain."""
    precipitation_ratio = monthly_precipitation_mm / SATURATING_MONTHLY_RAIN_MM
    return clamp_score(1 - precipitation_ratio * (sensitivity / 100))


def cloud_score(monthly_cloud_cover_pct: int, sun_preference: int) -> float:
    """Treat cloud as a misery-style penalty with a preference-dependent tolerance threshold."""
    tolerated_cloud_cover = MAX_TOLERATED_CLOUD_COVER - (
        (MAX_TOLERATED_CLOUD_COVER - MIN_TOLERATED_CLOUD_COVER) * (sun_preference / 100)
    )

    if monthly_cloud_cover_pct <= tolerated_cloud_cover:
        return 1.0

    excess_ratio = (monthly_cloud_cover_pct - tolerated_cloud_cover) / (100 - tolerated_cloud_cover)
    return clamp_score(1 - excess_ratio**2)


def monthly_score(cell: ClimateCell, month_index: int, preferences: PreferenceInputs) -> float:
    """Combine the three monthly climate signals into one monthly fit score."""
    return (
        temperature_score(cell.temperature_c[month_index], preferences)
        + rain_score(cell.precipitation_mm[month_index], preferences.rain_sensitivity)
        + cloud_score(cell.cloud_cover_pct[month_index], preferences.sun_preference)
    ) / 3


def annual_score(cell: ClimateCell, preferences: PreferenceInputs) -> float:
    """Average monthly scores into one normalized annual score."""
    score_total = sum(monthly_score(cell, month_index, preferences) for month_index in range(MONTHS_PER_YEAR))
    return clamp_score(score_total / MONTHS_PER_YEAR)


def score_preferences(preferences: PreferenceInputs) -> list[ScorePoint]:
    """Score every stub climate row until the DuckDB adapter lands."""
    return [
        {
            "lat": cell.lat,
            "lon": cell.lon,
            "score": annual_score(cell, preferences),
        }
        for cell in STUB_CLIMATE_CELLS
    ]
