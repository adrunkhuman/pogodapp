from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from numpy.typing import NDArray

MONTHS_PER_YEAR = 12
TEMPERATURE_COMFORT_BAND_C = 2.0
TEMPERATURE_IDEAL_SLOPE_C = 8.0
TEMPERATURE_LIMIT_SLOPE_C = 10.0
TEMPERATURE_IDEAL_WEIGHT = 0.5
TEMPERATURE_HEAT_WEIGHT = 0.25
TEMPERATURE_COLD_WEIGHT = 0.25
SATURATING_MONTHLY_RAIN_MM = 300.0
MAX_TOLERATED_CLOUD_COVER = 85.0
MIN_TOLERATED_CLOUD_COVER = 15.0


class PreferenceInputs(BaseModel):
    """Validated scoring inputs for the `/score` workflow."""

    preferred_day_temperature: int = Field(ge=5, le=35)
    summer_heat_limit: int = Field(ge=18, le=42)
    winter_cold_limit: int = Field(ge=-15, le=20)
    dryness_preference: int = Field(ge=0, le=100)
    sunshine_preference: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_temperature_order(self) -> PreferenceInputs:
        """Reject self-contradictory temperature ranges before scoring starts."""
        if self.preferred_day_temperature > self.summer_heat_limit:
            msg = "preferred_day_temperature must be less than or equal to summer_heat_limit"
            raise ValueError(msg)

        if self.preferred_day_temperature < self.winter_cold_limit:
            msg = "preferred_day_temperature must be greater than or equal to winter_cold_limit"
            raise ValueError(msg)

        return self


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


@dataclass(frozen=True, slots=True)
class ClimateMatrix:
    """Compact climate features for vectorized scoring."""

    latitudes: NDArray[np.float32]
    longitudes: NDArray[np.float32]
    temperature_c: NDArray[np.float32]
    precipitation_mm: NDArray[np.float32]
    cloud_cover_pct: NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Reject malformed matrix shapes before they reach the scorer."""
        cell_count = self.latitudes.shape[0]

        if self.longitudes.shape != (cell_count,):
            msg = "longitudes must align with latitudes"
            raise ValueError(msg)

        if self.temperature_c.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "temperature_c must be shaped (cells, 12)"
            raise ValueError(msg)

        if self.precipitation_mm.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "precipitation_mm must be shaped (cells, 12)"
            raise ValueError(msg)

        if self.cloud_cover_pct.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "cloud_cover_pct must be shaped (cells, 12)"
            raise ValueError(msg)

    @classmethod
    def from_cells(cls, climate_cells: tuple[ClimateCell, ...]) -> ClimateMatrix:
        """Build the compact scoring matrix from cell objects."""
        return cls(
            latitudes=np.array([cell.lat for cell in climate_cells], dtype=np.float32),
            longitudes=np.array([cell.lon for cell in climate_cells], dtype=np.float32),
            temperature_c=np.array([cell.temperature_c for cell in climate_cells], dtype=np.float32),
            precipitation_mm=np.array([cell.precipitation_mm for cell in climate_cells], dtype=np.float32),
            cloud_cover_pct=np.array([cell.cloud_cover_pct for cell in climate_cells], dtype=np.uint8),
        )


@dataclass(frozen=True, slots=True)
class ProbeMetricBreakdown:
    """One user-facing metric row for the map probe tooltip."""

    key: str
    label: str
    value: float
    display_value: str
    score: float


@dataclass(frozen=True, slots=True)
class ProbeBreakdown:
    """Structured tooltip payload for one scored climate-matrix row."""

    overall_score: float
    metrics: tuple[ProbeMetricBreakdown, ...]


class CellScorePoint(TypedDict):
    """Normalized score payload for one climate cell."""

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
    """Score temperature with one preferred band plus separate hot and cold guardrails."""
    ideal_delta = abs(observed_temperature_c - preferences.preferred_day_temperature)
    ideal_distance = max(ideal_delta - TEMPERATURE_COMFORT_BAND_C, 0.0)
    ideal_score = clamp_score(1 - ideal_distance / TEMPERATURE_IDEAL_SLOPE_C)

    heat_excess = max(observed_temperature_c - preferences.summer_heat_limit, 0.0)
    heat_score = clamp_score(1 - heat_excess / TEMPERATURE_LIMIT_SLOPE_C)

    cold_excess = max(preferences.winter_cold_limit - observed_temperature_c, 0.0)
    cold_score = clamp_score(1 - cold_excess / TEMPERATURE_LIMIT_SLOPE_C)

    return (
        ideal_score * TEMPERATURE_IDEAL_WEIGHT
        + heat_score * TEMPERATURE_HEAT_WEIGHT
        + cold_score * TEMPERATURE_COLD_WEIGHT
    )


def rain_score(monthly_precipitation_mm: float, dryness_preference: int) -> float:
    """Penalize wetter months without ever rewarding rain."""
    precipitation_ratio = monthly_precipitation_mm / SATURATING_MONTHLY_RAIN_MM
    return clamp_score(1 - precipitation_ratio * (dryness_preference / 100))


def cloud_score(monthly_cloud_cover_pct: int, sunshine_preference: int) -> float:
    """Treat cloud as a misery-style penalty with a preference-dependent tolerance threshold."""
    tolerated_cloud_cover = MAX_TOLERATED_CLOUD_COVER - (
        (MAX_TOLERATED_CLOUD_COVER - MIN_TOLERATED_CLOUD_COVER) * (sunshine_preference / 100)
    )

    if monthly_cloud_cover_pct <= tolerated_cloud_cover:
        return 1.0

    excess_ratio = (monthly_cloud_cover_pct - tolerated_cloud_cover) / (100 - tolerated_cloud_cover)
    return clamp_score(1 - excess_ratio**2)


def monthly_score(cell: ClimateCell, month_index: int, preferences: PreferenceInputs) -> float:
    """Combine the three monthly climate signals into one monthly fit score."""
    return (
        temperature_score(cell.temperature_c[month_index], preferences)
        + rain_score(cell.precipitation_mm[month_index], preferences.dryness_preference)
        + cloud_score(cell.cloud_cover_pct[month_index], preferences.sunshine_preference)
    ) / 3


def annual_score(cell: ClimateCell, preferences: PreferenceInputs) -> float:
    """Average monthly scores into one normalized annual score."""
    score_total = sum(monthly_score(cell, month_index, preferences) for month_index in range(MONTHS_PER_YEAR))
    return clamp_score(score_total / MONTHS_PER_YEAR)


def score_climate_cells(climate_cells: tuple[ClimateCell, ...], preferences: PreferenceInputs) -> list[CellScorePoint]:
    """Score the supplied climate rows without caring where they came from."""
    return [
        {
            "lat": cell.lat,
            "lon": cell.lon,
            "score": annual_score(cell, preferences),
        }
        for cell in climate_cells
    ]


def score_climate_matrix(climate_matrix: ClimateMatrix, preferences: PreferenceInputs) -> NDArray[np.float32]:
    """Score the compact climate matrix with vectorized NumPy operations."""
    tolerated_cloud_cover = MAX_TOLERATED_CLOUD_COVER - (
        (MAX_TOLERATED_CLOUD_COVER - MIN_TOLERATED_CLOUD_COVER) * (preferences.sunshine_preference / 100.0)
    )
    preferred_day_temperature = np.float32(preferences.preferred_day_temperature)
    summer_heat_limit = np.float32(preferences.summer_heat_limit)
    winter_cold_limit = np.float32(preferences.winter_cold_limit)
    dryness_ratio = np.float32(preferences.dryness_preference / 100.0)
    cloud_denominator = np.float32(100.0 - tolerated_cloud_cover)
    annual_scores = np.zeros(climate_matrix.latitudes.shape[0], dtype=np.float32)

    for month_index in range(MONTHS_PER_YEAR):
        monthly_temperature = climate_matrix.temperature_c[:, month_index]
        monthly_precipitation = climate_matrix.precipitation_mm[:, month_index]
        monthly_cloud_cover = climate_matrix.cloud_cover_pct[:, month_index].astype(np.float32, copy=False)

        ideal_distance = np.maximum(
            np.abs(monthly_temperature - preferred_day_temperature) - TEMPERATURE_COMFORT_BAND_C,
            0.0,
            dtype=np.float32,
        )
        ideal_scores = np.clip(1.0 - (ideal_distance / TEMPERATURE_IDEAL_SLOPE_C), 0.0, 1.0)
        heat_excess = np.maximum(monthly_temperature - summer_heat_limit, 0.0, dtype=np.float32)
        heat_scores = np.clip(1.0 - (heat_excess / TEMPERATURE_LIMIT_SLOPE_C), 0.0, 1.0)
        cold_excess = np.maximum(winter_cold_limit - monthly_temperature, 0.0, dtype=np.float32)
        cold_scores = np.clip(1.0 - (cold_excess / TEMPERATURE_LIMIT_SLOPE_C), 0.0, 1.0)
        temperature_scores = (
            ideal_scores * TEMPERATURE_IDEAL_WEIGHT
            + heat_scores * TEMPERATURE_HEAT_WEIGHT
            + cold_scores * TEMPERATURE_COLD_WEIGHT
        )

        rain_scores = np.clip(
            1.0 - (monthly_precipitation / SATURATING_MONTHLY_RAIN_MM) * dryness_ratio,
            0.0,
            1.0,
        )

        excess_ratio = (monthly_cloud_cover - tolerated_cloud_cover) / cloud_denominator
        cloud_scores = np.where(
            monthly_cloud_cover <= tolerated_cloud_cover,
            1.0,
            np.clip(1.0 - excess_ratio**2, 0.0, 1.0),
        )

        annual_scores += (temperature_scores + rain_scores + cloud_scores) / 3.0

    annual_scores /= MONTHS_PER_YEAR
    return np.clip(annual_scores, 0.0, 1.0).astype(np.float32, copy=False)


def normalize_score_array(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale one score vector so the best match lands at 1.0."""
    if scores.size == 0:
        return scores

    max_score = float(scores.max())
    if max_score == 0.0:
        return scores

    return np.round(scores / max_score, 4).astype(np.float32, copy=False)


def score_matrix_row_breakdown(
    climate_matrix: ClimateMatrix,
    row_index: int,
    preferences: PreferenceInputs,
) -> ProbeBreakdown:
    """Break one climate-matrix row into probe-ready averages and per-metric scores."""
    temp_scores = []
    rain_scores = []
    cloud_scores = []

    for month in range(MONTHS_PER_YEAR):
        temp_scores.append(temperature_score(float(climate_matrix.temperature_c[row_index, month]), preferences))
        rain_scores.append(
            rain_score(float(climate_matrix.precipitation_mm[row_index, month]), preferences.dryness_preference)
        )
        cloud_scores.append(
            cloud_score(int(climate_matrix.cloud_cover_pct[row_index, month]), preferences.sunshine_preference)
        )

    avg_temp_c = round(float(np.mean(climate_matrix.temperature_c[row_index])), 1)
    avg_precip_mm = round(float(np.mean(climate_matrix.precipitation_mm[row_index])), 1)
    avg_cloud_pct = round(float(np.mean(climate_matrix.cloud_cover_pct[row_index].astype(np.float32))), 1)
    avg_sun_pct = round(100.0 - avg_cloud_pct, 1)
    metric_scores = (
        ProbeMetricBreakdown(
            key="temp",
            label="temperature",
            value=avg_temp_c,
            display_value=f"{'+' if avg_temp_c > 0 else ''}{avg_temp_c:.1f}°C",
            score=round(sum(temp_scores) / MONTHS_PER_YEAR, 3),
        ),
        ProbeMetricBreakdown(
            key="rain",
            label="dryness",
            value=avg_precip_mm,
            display_value=f"{round(avg_precip_mm)}mm/mo",
            score=round(sum(rain_scores) / MONTHS_PER_YEAR, 3),
        ),
        ProbeMetricBreakdown(
            key="sun",
            label="sunshine",
            value=avg_sun_pct,
            display_value=f"{round(avg_sun_pct)}% sun",
            score=round(sum(cloud_scores) / MONTHS_PER_YEAR, 3),
        ),
    )
    return ProbeBreakdown(
        overall_score=round((sum(temp_scores) + sum(rain_scores) + sum(cloud_scores)) / (3 * MONTHS_PER_YEAR), 4),
        metrics=metric_scores,
    )


def score_preferences(preferences: PreferenceInputs) -> list[CellScorePoint]:
    """Score against the in-repo stub dataset.

    This stays as a small test helper while the main runtime path reads climate
    rows through a repository.
    """
    return score_climate_cells(STUB_CLIMATE_CELLS, preferences)
