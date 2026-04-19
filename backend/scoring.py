from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from numpy.typing import NDArray

MONTHS_PER_YEAR = 12
# These are product-tuning knobs, not source-data constants.
# Temperature stays dominant because users usually mean temp limits as near-hard
# constraints, while rain and sun act as refinements whose weight grows only
# when the user gives more extreme answers.
TEMPERATURE_COMFORT_BAND_C = 2.0
TEMPERATURE_IDEAL_SLOPE_C = 8.0
TEMPERATURE_LIMIT_SLOPE_C = 10.0
TEMPERATURE_IDEAL_WEIGHT = 0.5
TEMPERATURE_HEAT_WEIGHT = 0.30
TEMPERATURE_COLD_WEIGHT = 0.20
SATURATING_MONTHLY_RAIN_MM = 300.0
SATURATING_WETTEST_MONTH_RAIN_MM = 400.0
MAX_TOLERATED_CLOUD_COVER = 85.0
MIN_TOLERATED_CLOUD_COVER = 15.0
TEMPERATURE_BLOCK_BASE_WEIGHT = 0.88
TEMPERATURE_BLOCK_MIN_WEIGHT = 0.72
PRECIPITATION_PROFILE_MEDIAN_WEIGHT = 0.7
PRECIPITATION_PROFILE_PEAK_WEIGHT = 0.3
SUN_PROFILE_AVERAGE_WEIGHT = 0.7
SUN_PROFILE_GLOOM_WEIGHT = 0.3
MULTIPLICATIVE_SCORE_FLOOR = 0.02
TEMPERATURE_COMFORT_BAND_C32 = np.float32(TEMPERATURE_COMFORT_BAND_C)
TEMPERATURE_IDEAL_SLOPE_C32 = np.float32(TEMPERATURE_IDEAL_SLOPE_C)
TEMPERATURE_LIMIT_SLOPE_C32 = np.float32(TEMPERATURE_LIMIT_SLOPE_C)
TEMPERATURE_IDEAL_WEIGHT32 = np.float32(TEMPERATURE_IDEAL_WEIGHT)
TEMPERATURE_HEAT_WEIGHT32 = np.float32(TEMPERATURE_HEAT_WEIGHT)
TEMPERATURE_COLD_WEIGHT32 = np.float32(TEMPERATURE_COLD_WEIGHT)
SATURATING_MONTHLY_RAIN_MM32 = np.float32(SATURATING_MONTHLY_RAIN_MM)
SATURATING_WETTEST_MONTH_RAIN_MM32 = np.float32(SATURATING_WETTEST_MONTH_RAIN_MM)
PRECIPITATION_PROFILE_MEDIAN_WEIGHT32 = np.float32(PRECIPITATION_PROFILE_MEDIAN_WEIGHT)
PRECIPITATION_PROFILE_PEAK_WEIGHT32 = np.float32(PRECIPITATION_PROFILE_PEAK_WEIGHT)
SUN_PROFILE_AVERAGE_WEIGHT32 = np.float32(SUN_PROFILE_AVERAGE_WEIGHT)
SUN_PROFILE_GLOOM_WEIGHT32 = np.float32(SUN_PROFILE_GLOOM_WEIGHT)
MULTIPLICATIVE_SCORE_FLOOR32 = np.float32(MULTIPLICATIVE_SCORE_FLOOR)


class PreferenceInputs(BaseModel):
    """Validated scoring inputs for the `/score` workflow."""

    preferred_day_temperature: int = Field(ge=-5, le=35)
    summer_heat_limit: int = Field(ge=-5, le=42)
    winter_cold_limit: int = Field(ge=-15, le=35)
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
    temperature_min_c: tuple[float, ...]
    temperature_max_c: tuple[float, ...]
    precipitation_mm: tuple[float, ...]
    cloud_cover_pct: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject incomplete monthly rows before they reach scoring code."""
        if len(self.temperature_c) != MONTHS_PER_YEAR:
            msg = "temperature_c must contain 12 monthly values"
            raise ValueError(msg)

        if len(self.temperature_min_c) != MONTHS_PER_YEAR:
            msg = "temperature_min_c must contain 12 monthly values"
            raise ValueError(msg)

        if len(self.temperature_max_c) != MONTHS_PER_YEAR:
            msg = "temperature_max_c must contain 12 monthly values"
            raise ValueError(msg)

        if len(self.precipitation_mm) != MONTHS_PER_YEAR:
            msg = "precipitation_mm must contain 12 monthly values"
            raise ValueError(msg)

        if len(self.cloud_cover_pct) != MONTHS_PER_YEAR:
            msg = "cloud_cover_pct must contain 12 monthly values"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ClimateMatrix:
    """Compact score-time features for vectorized scoring.

    Monthly arrays are optional because the runtime can keep yearly aggregates
    resident for `/score` while `/probe` works from a full `ClimateCell`.
    """

    latitudes: NDArray[np.float32]
    longitudes: NDArray[np.float32]
    temperature_c: NDArray[np.float32] | None
    temperature_min_c: NDArray[np.float32] | None = None
    temperature_max_c: NDArray[np.float32] | None = None
    precipitation_mm: NDArray[np.float32] | None = None
    cloud_cover_pct: NDArray[np.uint8] | None = None
    typical_highs_c: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    hottest_month_highs_c: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    coldest_month_lows_c: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    median_precipitation_mm: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    wettest_precipitation_mm: NDArray[np.float32] = field(default_factory=lambda: np.array([], dtype=np.float32))
    average_cloud_cover_pct: NDArray[np.uint8] = field(default_factory=lambda: np.array([], dtype=np.uint8))
    gloomiest_cloud_cover_pct: NDArray[np.uint8] = field(default_factory=lambda: np.array([], dtype=np.uint8))

    def __post_init__(self) -> None:
        """Reject malformed matrix shapes before they reach the scorer."""
        cell_count = self.latitudes.shape[0]

        self._validate_coordinate_shapes(cell_count)

        if self._has_monthly_inputs():
            self._validate_base_shapes(cell_count)
            temperature_min_c = cast("NDArray[np.float32]", self.temperature_min_c)
            temperature_max_c = cast("NDArray[np.float32]", self.temperature_max_c)
            precipitation_mm = cast("NDArray[np.float32]", self.precipitation_mm)
            cloud_cover_pct = cast("NDArray[np.uint8]", self.cloud_cover_pct)
            self._ensure_derived_vector(
                "typical_highs_c",
                np.median(temperature_max_c, axis=1).astype(np.float32, copy=False),
                cell_count,
            )
            self._ensure_derived_vector(
                "hottest_month_highs_c",
                np.max(temperature_max_c, axis=1).astype(np.float32, copy=False),
                cell_count,
            )
            self._ensure_derived_vector(
                "coldest_month_lows_c",
                np.min(temperature_min_c, axis=1).astype(np.float32, copy=False),
                cell_count,
            )
            self._ensure_derived_vector(
                "median_precipitation_mm",
                np.median(precipitation_mm, axis=1).astype(np.float32, copy=False),
                cell_count,
            )
            self._ensure_derived_vector(
                "wettest_precipitation_mm",
                np.max(precipitation_mm, axis=1).astype(np.float32, copy=False),
                cell_count,
            )
            self._ensure_derived_vector(
                "average_cloud_cover_pct",
                np.rint(np.mean(cloud_cover_pct.astype(np.float32), axis=1)).astype(np.uint8, copy=False),
                cell_count,
            )
            self._ensure_derived_vector(
                "gloomiest_cloud_cover_pct",
                np.max(cloud_cover_pct, axis=1).astype(np.uint8, copy=False),
                cell_count,
            )
            return

        self._validate_derived_only_shapes(cell_count)

    def _validate_coordinate_shapes(self, cell_count: int) -> None:
        if self.longitudes.shape != (cell_count,):
            msg = "longitudes must align with latitudes"
            raise ValueError(msg)

    def _has_monthly_inputs(self) -> bool:
        monthly_arrays = (
            self.temperature_min_c,
            self.temperature_max_c,
            self.precipitation_mm,
            self.cloud_cover_pct,
        )
        has_any = any(array is not None for array in monthly_arrays)
        has_all = all(array is not None for array in monthly_arrays)
        if has_any and not has_all:
            msg = (
                "temperature_min_c, temperature_max_c, precipitation_mm, and cloud_cover_pct must be provided together"
            )
            raise ValueError(msg)
        return has_all

    def _validate_base_shapes(self, cell_count: int) -> None:
        temperature_min_c = cast("NDArray[np.float32]", self.temperature_min_c)
        temperature_max_c = cast("NDArray[np.float32]", self.temperature_max_c)
        precipitation_mm = cast("NDArray[np.float32]", self.precipitation_mm)
        cloud_cover_pct = cast("NDArray[np.uint8]", self.cloud_cover_pct)

        if self.temperature_c is not None and self.temperature_c.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "temperature_c must be shaped (cells, 12)"
            raise ValueError(msg)

        if temperature_min_c.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "temperature_min_c must be shaped (cells, 12)"
            raise ValueError(msg)

        if temperature_max_c.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "temperature_max_c must be shaped (cells, 12)"
            raise ValueError(msg)

        if precipitation_mm.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "precipitation_mm must be shaped (cells, 12)"
            raise ValueError(msg)

        if cloud_cover_pct.shape != (cell_count, MONTHS_PER_YEAR):
            msg = "cloud_cover_pct must be shaped (cells, 12)"
            raise ValueError(msg)

    def _validate_derived_only_shapes(self, cell_count: int) -> None:
        for attribute in (
            "typical_highs_c",
            "hottest_month_highs_c",
            "coldest_month_lows_c",
            "median_precipitation_mm",
            "wettest_precipitation_mm",
            "average_cloud_cover_pct",
            "gloomiest_cloud_cover_pct",
        ):
            current = getattr(self, attribute)
            if current.shape != (cell_count,):
                msg = f"{attribute} must align with latitudes"
                raise ValueError(msg)

    def _ensure_derived_vector(self, attribute: str, derived: NDArray[np.generic], cell_count: int) -> None:
        current = getattr(self, attribute)
        if current.shape == (0,):
            object.__setattr__(self, attribute, derived)
            return

        if current.shape != (cell_count,):
            msg = f"{attribute} must align with latitudes"
            raise ValueError(msg)

    @classmethod
    def from_cells(cls, climate_cells: tuple[ClimateCell, ...]) -> ClimateMatrix:
        """Build the compact scoring matrix from cell objects."""
        return cls(
            latitudes=np.array([cell.lat for cell in climate_cells], dtype=np.float32),
            longitudes=np.array([cell.lon for cell in climate_cells], dtype=np.float32),
            temperature_c=np.array([cell.temperature_c for cell in climate_cells], dtype=np.float32),
            temperature_min_c=np.array([cell.temperature_min_c for cell in climate_cells], dtype=np.float32),
            temperature_max_c=np.array([cell.temperature_max_c for cell in climate_cells], dtype=np.float32),
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


@dataclass(slots=True)
class MatrixScoreTimings:
    """Optional cold-path timings for vectorized matrix scoring."""

    setup_ms: float = 0.0
    temperature_ms: float = 0.0
    rain_ms: float = 0.0
    sun_ms: float = 0.0
    combine_ms: float = 0.0


def _combine_score_blocks(
    temperature_scores: NDArray[np.float32],
    rain_scores: NDArray[np.float32],
    sun_scores: NDArray[np.float32],
    weights: tuple[float, float, float],
) -> NDArray[np.float32]:
    """Combine precomputed score blocks while reusing temporary buffers."""
    temperature_weight, rain_weight, sun_weight = weights
    preference_scores = np.empty_like(rain_scores)
    np.power(rain_scores, rain_weight, out=preference_scores)
    scores = np.empty_like(sun_scores)
    np.power(sun_scores, sun_weight, out=scores)
    np.multiply(preference_scores, scores, out=preference_scores)

    np.power(temperature_scores, temperature_weight, out=scores)
    np.power(preference_scores, np.float32(1.0) - temperature_weight, out=preference_scores)
    np.multiply(scores, preference_scores, out=scores)
    np.clip(scores, 0.0, 1.0, out=scores)
    return scores


def _temperature_score_block(
    climate_matrix: ClimateMatrix,
    preferred_day_temperature: np.float32,
    summer_heat_limit: np.float32,
    winter_cold_limit: np.float32,
) -> NDArray[np.float32]:
    """Score the temperature constraints while reusing temporary buffers."""
    ideal_scores = np.empty_like(climate_matrix.typical_highs_c)
    np.subtract(climate_matrix.typical_highs_c, preferred_day_temperature, out=ideal_scores)
    np.absolute(ideal_scores, out=ideal_scores)
    np.subtract(ideal_scores, TEMPERATURE_COMFORT_BAND_C32, out=ideal_scores)
    np.clip(ideal_scores, 0.0, None, out=ideal_scores)
    np.divide(ideal_scores, TEMPERATURE_IDEAL_SLOPE_C32, out=ideal_scores)
    np.subtract(1.0, ideal_scores, out=ideal_scores)
    np.clip(ideal_scores, 0.0, 1.0, out=ideal_scores)

    heat_scores = np.empty_like(climate_matrix.hottest_month_highs_c)
    np.subtract(climate_matrix.hottest_month_highs_c, summer_heat_limit, out=heat_scores)
    np.clip(heat_scores, 0.0, None, out=heat_scores)
    np.divide(heat_scores, TEMPERATURE_LIMIT_SLOPE_C32, out=heat_scores)
    np.subtract(1.0, heat_scores, out=heat_scores)
    np.clip(heat_scores, 0.0, 1.0, out=heat_scores)

    cold_scores = np.empty_like(climate_matrix.coldest_month_lows_c)
    np.subtract(winter_cold_limit, climate_matrix.coldest_month_lows_c, out=cold_scores)
    np.clip(cold_scores, 0.0, None, out=cold_scores)
    np.divide(cold_scores, TEMPERATURE_LIMIT_SLOPE_C32, out=cold_scores)
    np.subtract(1.0, cold_scores, out=cold_scores)
    np.clip(cold_scores, 0.0, 1.0, out=cold_scores)

    np.power(ideal_scores, TEMPERATURE_IDEAL_WEIGHT32, out=ideal_scores)
    np.power(heat_scores, TEMPERATURE_HEAT_WEIGHT32, out=heat_scores)
    np.multiply(ideal_scores, heat_scores, out=ideal_scores)
    np.power(cold_scores, TEMPERATURE_COLD_WEIGHT32, out=cold_scores)
    np.multiply(ideal_scores, cold_scores, out=ideal_scores)
    return ideal_scores


def _rain_score_block(climate_matrix: ClimateMatrix, dryness_ratio: np.float32) -> NDArray[np.float32]:
    """Score the precipitation constraints while reusing temporary buffers."""
    typical_rain_scores = np.empty_like(climate_matrix.median_precipitation_mm)
    np.divide(climate_matrix.median_precipitation_mm, SATURATING_MONTHLY_RAIN_MM32, out=typical_rain_scores)
    np.multiply(typical_rain_scores, dryness_ratio, out=typical_rain_scores)
    np.subtract(1.0, typical_rain_scores, out=typical_rain_scores)
    np.clip(typical_rain_scores, MULTIPLICATIVE_SCORE_FLOOR32, 1.0, out=typical_rain_scores)

    wettest_month_scores = np.empty_like(climate_matrix.wettest_precipitation_mm)
    np.divide(climate_matrix.wettest_precipitation_mm, SATURATING_WETTEST_MONTH_RAIN_MM32, out=wettest_month_scores)
    np.multiply(wettest_month_scores, dryness_ratio, out=wettest_month_scores)
    np.subtract(1.0, wettest_month_scores, out=wettest_month_scores)
    np.clip(wettest_month_scores, MULTIPLICATIVE_SCORE_FLOOR32, 1.0, out=wettest_month_scores)

    np.power(typical_rain_scores, PRECIPITATION_PROFILE_MEDIAN_WEIGHT32, out=typical_rain_scores)
    np.power(wettest_month_scores, PRECIPITATION_PROFILE_PEAK_WEIGHT32, out=wettest_month_scores)
    np.multiply(typical_rain_scores, wettest_month_scores, out=typical_rain_scores)
    return typical_rain_scores


STUB_CLIMATE_CELLS: tuple[ClimateCell, ...] = (
    ClimateCell(
        lat=37.5,
        lon=-122.0,
        temperature_c=(11.0, 12.0, 13.0, 14.0, 16.0, 18.0, 19.0, 20.0, 20.0, 18.0, 15.0, 12.0),
        temperature_min_c=(6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 14.0, 12.0, 9.0, 7.0),
        temperature_max_c=(16.0, 17.0, 18.0, 20.0, 22.0, 24.0, 25.0, 26.0, 26.0, 24.0, 20.0, 17.0),
        precipitation_mm=(110.0, 90.0, 70.0, 35.0, 15.0, 5.0, 1.0, 2.0, 8.0, 30.0, 70.0, 100.0),
        cloud_cover_pct=(60, 55, 50, 40, 30, 20, 15, 15, 20, 30, 45, 55),
    ),
    ClimateCell(
        lat=41.9,
        lon=12.5,
        temperature_c=(8.0, 9.0, 12.0, 15.0, 19.0, 23.0, 26.0, 26.0, 23.0, 18.0, 13.0, 9.0),
        temperature_min_c=(3.0, 4.0, 6.0, 9.0, 13.0, 17.0, 20.0, 20.0, 17.0, 13.0, 8.0, 4.0),
        temperature_max_c=(13.0, 14.0, 18.0, 21.0, 25.0, 30.0, 33.0, 33.0, 29.0, 24.0, 18.0, 14.0),
        precipitation_mm=(80.0, 70.0, 60.0, 65.0, 45.0, 30.0, 20.0, 25.0, 70.0, 105.0, 115.0, 95.0),
        cloud_cover_pct=(55, 50, 45, 40, 30, 20, 15, 15, 25, 35, 50, 55),
    ),
    ClimateCell(
        lat=-33.9,
        lon=18.4,
        temperature_c=(21.0, 21.0, 20.0, 18.0, 16.0, 14.0, 13.0, 14.0, 15.0, 17.0, 18.0, 20.0),
        temperature_min_c=(16.0, 16.0, 15.0, 13.0, 11.0, 8.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0),
        temperature_max_c=(27.0, 27.0, 26.0, 24.0, 21.0, 18.0, 17.0, 18.0, 19.0, 22.0, 24.0, 26.0),
        precipitation_mm=(15.0, 18.0, 20.0, 35.0, 75.0, 95.0, 110.0, 90.0, 45.0, 30.0, 20.0, 15.0),
        cloud_cover_pct=(20, 22, 28, 35, 45, 55, 60, 58, 45, 35, 28, 22),
    ),
    ClimateCell(
        lat=35.7,
        lon=139.7,
        temperature_c=(6.0, 7.0, 10.0, 15.0, 19.0, 22.0, 26.0, 28.0, 24.0, 18.0, 13.0, 8.0),
        temperature_min_c=(1.0, 2.0, 5.0, 10.0, 14.0, 18.0, 22.0, 24.0, 20.0, 14.0, 8.0, 3.0),
        temperature_max_c=(11.0, 12.0, 15.0, 20.0, 24.0, 27.0, 31.0, 33.0, 29.0, 23.0, 18.0, 13.0),
        precipitation_mm=(55.0, 60.0, 115.0, 125.0, 135.0, 160.0, 155.0, 170.0, 210.0, 195.0, 95.0, 55.0),
        cloud_cover_pct=(45, 45, 50, 55, 60, 70, 75, 70, 65, 55, 45, 40),
    ),
    ClimateCell(
        lat=-22.9,
        lon=-43.2,
        temperature_c=(27.0, 28.0, 27.0, 26.0, 24.0, 22.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0),
        temperature_min_c=(23.0, 24.0, 23.0, 21.0, 19.0, 17.0, 17.0, 18.0, 19.0, 20.0, 21.0, 23.0),
        temperature_max_c=(31.0, 32.0, 31.0, 30.0, 28.0, 26.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0),
        precipitation_mm=(175.0, 160.0, 180.0, 140.0, 110.0, 90.0, 75.0, 70.0, 110.0, 130.0, 145.0, 165.0),
        cloud_cover_pct=(40, 38, 42, 45, 50, 55, 52, 48, 45, 42, 40, 38),
    ),
)


def clamp_score(value: float) -> float:
    """Keep scores in the API's normalized range."""
    return max(0.0, min(1.0, value))


def preference_extremity(preference: int) -> float:
    """Map a centered 0..100 preference onto a 0..1 extremity scale."""
    return abs(preference - 50) / 50


def weighted_product_score(*components: tuple[float, float], floor: float = 0.0) -> float:
    """Combine 0..1 criteria multiplicatively without letting one miss force exact zero."""
    score = 1.0
    for value, weight in components:
        score *= max(value, floor) ** weight
    return clamp_score(score)


def temperature_score(
    observed_average_c: float,
    observed_min_c: float,
    observed_max_c: float,
    preferences: PreferenceInputs,
) -> float:
    """Score temperature using daytime highs and nighttime lows, not just monthly means."""
    ideal_score, heat_score, cold_score = temperature_component_scores(
        observed_average_c,
        observed_min_c,
        observed_max_c,
        preferences,
    )

    return weighted_product_score(
        (ideal_score, TEMPERATURE_IDEAL_WEIGHT),
        (heat_score, TEMPERATURE_HEAT_WEIGHT),
        (cold_score, TEMPERATURE_COLD_WEIGHT),
    )


def temperature_component_scores(
    observed_average_c: float,
    observed_min_c: float,
    observed_max_c: float,
    preferences: PreferenceInputs,
) -> tuple[float, float, float]:
    """Return separate scores for typical temperature, yearly high, and yearly low."""
    ideal_delta = abs(observed_average_c - preferences.preferred_day_temperature)
    ideal_distance = max(ideal_delta - TEMPERATURE_COMFORT_BAND_C, 0.0)
    ideal_score = clamp_score(1 - ideal_distance / TEMPERATURE_IDEAL_SLOPE_C)

    heat_excess = max(observed_max_c - preferences.summer_heat_limit, 0.0)
    heat_score = clamp_score(1 - heat_excess / TEMPERATURE_LIMIT_SLOPE_C)

    cold_excess = max(preferences.winter_cold_limit - observed_min_c, 0.0)
    cold_score = clamp_score(1 - cold_excess / TEMPERATURE_LIMIT_SLOPE_C)
    return ideal_score, heat_score, cold_score


def temperature_profile_score(cell: ClimateCell, preferences: PreferenceInputs) -> float:
    """Score yearly temperature by typical high, hottest month, and coldest month."""
    typical_high_c = float(np.median(np.array(cell.temperature_max_c, dtype=np.float32)))
    hottest_month_high_c = max(cell.temperature_max_c)
    coldest_month_low_c = min(cell.temperature_min_c)
    return temperature_score(typical_high_c, coldest_month_low_c, hottest_month_high_c, preferences)


def rain_score(monthly_precipitation_mm: float, dryness_preference: int) -> float:
    """Penalize wetter months without ever rewarding rain."""
    precipitation_ratio = monthly_precipitation_mm / SATURATING_MONTHLY_RAIN_MM
    return clamp_score(1 - precipitation_ratio * (dryness_preference / 100))


def rain_profile_score(monthly_precipitation_mm: tuple[float, ...], dryness_preference: int) -> float:
    """Blend the usual rain level with the worst wet-season spike."""
    median_precipitation_mm = float(np.median(np.array(monthly_precipitation_mm, dtype=np.float32)))
    wettest_month_precipitation_mm = max(monthly_precipitation_mm)
    typical_rain_score = rain_score(median_precipitation_mm, dryness_preference)
    wettest_month_score = clamp_score(
        1 - (wettest_month_precipitation_mm / SATURATING_WETTEST_MONTH_RAIN_MM) * (dryness_preference / 100)
    )
    return weighted_product_score(
        (typical_rain_score, PRECIPITATION_PROFILE_MEDIAN_WEIGHT),
        (wettest_month_score, PRECIPITATION_PROFILE_PEAK_WEIGHT),
        floor=MULTIPLICATIVE_SCORE_FLOOR,
    )


def cloud_score(monthly_cloud_cover_pct: int, sunshine_preference: int) -> float:
    """Treat cloud as a misery-style penalty with a preference-dependent tolerance threshold."""
    tolerated_cloud_cover = MAX_TOLERATED_CLOUD_COVER - (
        (MAX_TOLERATED_CLOUD_COVER - MIN_TOLERATED_CLOUD_COVER) * (sunshine_preference / 100)
    )

    if monthly_cloud_cover_pct <= tolerated_cloud_cover:
        return 1.0

    excess_ratio = (monthly_cloud_cover_pct - tolerated_cloud_cover) / (100 - tolerated_cloud_cover)
    return clamp_score(1 - excess_ratio**2)


def sunshine_profile_score(monthly_cloud_cover_pct: tuple[int, ...], sunshine_preference: int) -> float:
    """Blend average sky conditions with the gloomiest month."""
    average_cloud_cover_pct = round(float(np.mean(np.array(monthly_cloud_cover_pct, dtype=np.float32))))
    gloomiest_month_cloud_cover_pct = max(monthly_cloud_cover_pct)
    average_sun_score = cloud_score(average_cloud_cover_pct, sunshine_preference)
    gloomiest_month_score = cloud_score(gloomiest_month_cloud_cover_pct, sunshine_preference)
    return weighted_product_score(
        (average_sun_score, SUN_PROFILE_AVERAGE_WEIGHT),
        (gloomiest_month_score, SUN_PROFILE_GLOOM_WEIGHT),
        floor=MULTIPLICATIVE_SCORE_FLOOR,
    )


def preference_block_weights(dryness_preference: int, sunshine_preference: int) -> tuple[float, float, float]:
    """Rebalance rain and sun only when the user signals a strong non-temperature preference."""
    dryness_extremity = preference_extremity(dryness_preference)
    sunshine_extremity = preference_extremity(sunshine_preference)
    preference_block_weight = (1 - TEMPERATURE_BLOCK_BASE_WEIGHT) + 0.16 * max(dryness_extremity, sunshine_extremity)
    temperature_block_weight = max(TEMPERATURE_BLOCK_MIN_WEIGHT, 1 - preference_block_weight)
    combined_preference_importance = (1 + dryness_extremity) + (1 + sunshine_extremity)
    rain_weight = (1 + dryness_extremity) / combined_preference_importance
    sun_weight = (1 + sunshine_extremity) / combined_preference_importance
    return temperature_block_weight, rain_weight, sun_weight


def annual_score(cell: ClimateCell, preferences: PreferenceInputs) -> float:
    """Keep temperature limit adherence dominant while letting strong rain/sun opinions matter more."""
    temperature_component = temperature_profile_score(cell, preferences)
    rain_component = rain_profile_score(cell.precipitation_mm, preferences.dryness_preference)
    sun_component = sunshine_profile_score(cell.cloud_cover_pct, preferences.sunshine_preference)
    temperature_weight, rain_weight, sun_weight = preference_block_weights(
        preferences.dryness_preference,
        preferences.sunshine_preference,
    )
    preference_component = weighted_product_score((rain_component, rain_weight), (sun_component, sun_weight))
    return weighted_product_score(
        (temperature_component, temperature_weight),
        (preference_component, 1 - temperature_weight),
    )


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


def score_climate_matrix(
    climate_matrix: ClimateMatrix,
    preferences: PreferenceInputs,
    *,
    timings: MatrixScoreTimings | None = None,
) -> NDArray[np.float32]:
    """Score the compact climate matrix with vectorized NumPy operations.

    If provided, ``timings`` is populated in place with per-phase cold-path timings.
    """
    setup_started = perf_counter()
    tolerated_cloud_cover = MAX_TOLERATED_CLOUD_COVER - (
        (MAX_TOLERATED_CLOUD_COVER - MIN_TOLERATED_CLOUD_COVER) * (preferences.sunshine_preference / 100.0)
    )
    preferred_day_temperature = np.float32(preferences.preferred_day_temperature)
    summer_heat_limit = np.float32(preferences.summer_heat_limit)
    winter_cold_limit = np.float32(preferences.winter_cold_limit)
    dryness_ratio = np.float32(preferences.dryness_preference / 100.0)
    cloud_denominator = np.float32(100.0 - tolerated_cloud_cover)
    temperature_weight, rain_weight, sun_weight = preference_block_weights(
        preferences.dryness_preference,
        preferences.sunshine_preference,
    )
    if timings is not None:
        timings.setup_ms = (perf_counter() - setup_started) * 1000

    temperature_started = perf_counter()
    temperature_scores = _temperature_score_block(
        climate_matrix,
        preferred_day_temperature,
        summer_heat_limit,
        winter_cold_limit,
    )
    if timings is not None:
        timings.temperature_ms = (perf_counter() - temperature_started) * 1000

    rain_started = perf_counter()
    rain_scores = _rain_score_block(climate_matrix, dryness_ratio)
    if timings is not None:
        timings.rain_ms = (perf_counter() - rain_started) * 1000

    sun_started = perf_counter()
    average_excess_ratio = np.subtract(
        climate_matrix.average_cloud_cover_pct.astype(np.float32, copy=False),
        tolerated_cloud_cover,
        dtype=np.float32,
    )
    np.divide(average_excess_ratio, cloud_denominator, out=average_excess_ratio)
    np.clip(average_excess_ratio, 0.0, None, out=average_excess_ratio)
    average_sun_scores = np.subtract(1.0, average_excess_ratio * average_excess_ratio, dtype=np.float32)
    np.clip(average_sun_scores, MULTIPLICATIVE_SCORE_FLOOR32, 1.0, out=average_sun_scores)

    gloom_excess_ratio = np.subtract(
        climate_matrix.gloomiest_cloud_cover_pct.astype(np.float32, copy=False),
        tolerated_cloud_cover,
        dtype=np.float32,
    )
    np.divide(gloom_excess_ratio, cloud_denominator, out=gloom_excess_ratio)
    np.clip(gloom_excess_ratio, 0.0, None, out=gloom_excess_ratio)
    gloomiest_sun_scores = np.subtract(1.0, gloom_excess_ratio * gloom_excess_ratio, dtype=np.float32)
    np.clip(gloomiest_sun_scores, MULTIPLICATIVE_SCORE_FLOOR32, 1.0, out=gloomiest_sun_scores)

    sun_scores = (
        average_sun_scores**SUN_PROFILE_AVERAGE_WEIGHT32 * gloomiest_sun_scores**SUN_PROFILE_GLOOM_WEIGHT32
    ).astype(
        np.float32,
        copy=False,
    )
    if timings is not None:
        timings.sun_ms = (perf_counter() - sun_started) * 1000

    combine_started = perf_counter()
    scores = _combine_score_blocks(
        temperature_scores,
        rain_scores,
        sun_scores,
        (temperature_weight, rain_weight, sun_weight),
    )
    if timings is not None:
        timings.combine_ms = (perf_counter() - combine_started) * 1000
    return scores


def normalize_score_array(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale one score vector so the best match lands at 1.0."""
    if scores.size == 0:
        return scores

    max_score = float(scores.max())
    if max_score == 0.0:
        return scores

    normalized_scores = scores.copy()
    np.divide(normalized_scores, np.float32(max_score), out=normalized_scores)
    np.round(normalized_scores, 4, out=normalized_scores)
    return normalized_scores


def score_matrix_row_breakdown(
    climate_matrix: ClimateMatrix,
    row_index: int,
    preferences: PreferenceInputs,
) -> ProbeBreakdown:
    """Build `/probe` metrics from min/max, rain, and cloud rows without cached mean temperatures."""
    if (
        climate_matrix.temperature_min_c is None
        or climate_matrix.temperature_max_c is None
        or climate_matrix.precipitation_mm is None
        or climate_matrix.cloud_cover_pct is None
    ):
        msg = "score_matrix_row_breakdown requires monthly climate arrays"
        raise ValueError(msg)

    typical_high_value = float(np.median(climate_matrix.temperature_max_c[row_index]))
    hottest_month_high_value = float(np.max(climate_matrix.temperature_max_c[row_index]))
    coldest_month_low_value = float(np.min(climate_matrix.temperature_min_c[row_index]))
    typical_high_c = round(typical_high_value, 1)
    hottest_month_high_c = round(hottest_month_high_value, 1)
    coldest_month_low_c = round(coldest_month_low_value, 1)
    typical_score, high_score, low_score = temperature_component_scores(
        typical_high_value,
        coldest_month_low_value,
        hottest_month_high_value,
        preferences,
    )
    avg_precip_mm = round(float(np.mean(climate_matrix.precipitation_mm[row_index])), 1)
    avg_cloud_pct = round(float(np.mean(climate_matrix.cloud_cover_pct[row_index].astype(np.float32))), 1)
    avg_sun_pct = round(100.0 - avg_cloud_pct, 1)
    monthly_precipitation = tuple(float(value) for value in climate_matrix.precipitation_mm[row_index])
    monthly_cloud_cover = tuple(int(value) for value in climate_matrix.cloud_cover_pct[row_index])
    temperature_score_value = weighted_product_score(
        (typical_score, TEMPERATURE_IDEAL_WEIGHT),
        (high_score, TEMPERATURE_HEAT_WEIGHT),
        (low_score, TEMPERATURE_COLD_WEIGHT),
    )
    rain_score = rain_profile_score(monthly_precipitation, preferences.dryness_preference)
    sun_score = sunshine_profile_score(monthly_cloud_cover, preferences.sunshine_preference)
    temperature_weight, rain_weight, sun_weight = preference_block_weights(
        preferences.dryness_preference,
        preferences.sunshine_preference,
    )
    preference_score_value = weighted_product_score(
        (rain_score, rain_weight),
        (sun_score, sun_weight),
    )
    overall_score = round(
        weighted_product_score(
            (temperature_score_value, temperature_weight),
            (preference_score_value, 1 - temperature_weight),
        ),
        4,
    )
    metric_scores = (
        ProbeMetricBreakdown(
            key="temp",
            label="temp",
            value=typical_high_c,
            display_value=f"{typical_high_c:.1f}C",
            score=round(typical_score, 3),
        ),
        ProbeMetricBreakdown(
            key="high",
            label="high",
            value=hottest_month_high_c,
            display_value=f"{hottest_month_high_c:.1f}C",
            score=round(high_score, 3),
        ),
        ProbeMetricBreakdown(
            key="low",
            label="low",
            value=coldest_month_low_c,
            display_value=f"{coldest_month_low_c:.1f}C",
            score=round(low_score, 3),
        ),
        ProbeMetricBreakdown(
            key="rain",
            label="rain",
            value=avg_precip_mm,
            display_value=f"{round(avg_precip_mm)}mm/mo",
            score=round(rain_score, 3),
        ),
        ProbeMetricBreakdown(
            key="sun",
            label="sun",
            value=avg_sun_pct,
            display_value=f"{round(avg_sun_pct)}% sun",
            score=round(sun_score, 3),
        ),
    )
    return ProbeBreakdown(overall_score=overall_score, metrics=metric_scores)


def score_climate_cell_breakdown(cell: ClimateCell, preferences: PreferenceInputs) -> ProbeBreakdown:
    """Build `/probe` metrics from one on-demand climate row."""
    climate_matrix = ClimateMatrix.from_cells((cell,))
    return score_matrix_row_breakdown(climate_matrix, 0, preferences)


def score_preferences(preferences: PreferenceInputs) -> list[CellScorePoint]:
    """Score against the in-repo stub dataset.

    This stays as a small test helper while the main runtime path reads climate
    rows through a repository.
    """
    return score_climate_cells(STUB_CLIMATE_CELLS, preferences)
