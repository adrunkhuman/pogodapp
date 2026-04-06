import numpy as np
import pytest

from backend.scoring import (
    STUB_CLIMATE_CELLS,
    ClimateCell,
    ClimateMatrix,
    PreferenceInputs,
    ProbeBreakdown,
    annual_score,
    cloud_score,
    normalize_score_array,
    rain_score,
    score_climate_matrix,
    score_matrix_row_breakdown,
    temperature_score,
)


def make_preferences(**overrides: int) -> PreferenceInputs:
    base = {
        "preferred_day_temperature": 22,
        "summer_heat_limit": 30,
        "winter_cold_limit": 5,
        "dryness_preference": 60,
        "sunshine_preference": 60,
    }
    base.update(overrides)
    return PreferenceInputs(**base)


def test_temperature_score_keeps_the_comfort_band_at_full_score() -> None:
    preferences = make_preferences()

    assert temperature_score(22.0, 14.0, 26.0, preferences) == 1.0
    assert temperature_score(24.0, 16.0, 28.0, preferences) == 1.0
    assert temperature_score(20.0, 12.0, 24.0, preferences) == 1.0


def test_temperature_score_penalizes_months_far_from_the_preferred_band() -> None:
    preferences = make_preferences()

    assert temperature_score(28.0, 20.0, 32.0, preferences) < 1.0
    assert temperature_score(14.0, 6.0, 18.0, preferences) < 1.0


def test_temperature_score_penalizes_heat_above_the_summer_limit() -> None:
    preferences = make_preferences(summer_heat_limit=26)

    assert temperature_score(26.0, 22.0, 30.0, preferences) < temperature_score(22.0, 18.0, 26.0, preferences)


def test_temperature_score_penalizes_cold_below_the_winter_limit() -> None:
    preferences = make_preferences(winter_cold_limit=10)

    assert temperature_score(8.0, 4.0, 12.0, preferences) < temperature_score(14.0, 10.0, 18.0, preferences)


def test_preference_inputs_require_typical_day_to_stay_within_limits() -> None:
    with pytest.raises(ValueError, match="summer_heat_limit"):
        make_preferences(preferred_day_temperature=31, summer_heat_limit=30)

    with pytest.raises(ValueError, match="winter_cold_limit"):
        make_preferences(preferred_day_temperature=6, winter_cold_limit=7)


def test_cloud_score_penalizes_overcast_months_more_for_sun_seekers() -> None:
    assert cloud_score(40, sunshine_preference=20) == 1.0
    assert cloud_score(40, sunshine_preference=90) < 1.0
    assert cloud_score(90, sunshine_preference=90) < cloud_score(60, sunshine_preference=90)


def test_cloud_score_stays_perfect_at_the_tolerance_threshold_and_saturates_at_full_overcast() -> None:
    assert cloud_score(22, sunshine_preference=90) == 1.0
    assert cloud_score(100, sunshine_preference=90) == 0.0
    assert cloud_score(61, sunshine_preference=90) == pytest.approx(0.75)


def test_rain_score_respects_zero_dryness_preference_and_saturation_point() -> None:
    assert rain_score(150.0, dryness_preference=0) == 1.0
    assert rain_score(150.0, dryness_preference=50) == pytest.approx(0.75)
    assert rain_score(300.0, dryness_preference=100) == 0.0
    assert rain_score(450.0, dryness_preference=100) == 0.0


def test_annual_score_stays_normalized_for_extreme_climate_rows() -> None:
    extreme_cell = ClimateCell(
        lat=0.0,
        lon=0.0,
        temperature_c=(50.0,) * 12,
        temperature_min_c=(40.0,) * 12,
        temperature_max_c=(60.0,) * 12,
        precipitation_mm=(400.0,) * 12,
        cloud_cover_pct=(100,) * 12,
    )

    score = annual_score(
        extreme_cell,
        make_preferences(preferred_day_temperature=15, summer_heat_limit=18, winter_cold_limit=15),
    )

    assert 0.0 < score < 1.0


def test_annual_score_averages_monthly_scores_over_twelve_months() -> None:
    mostly_perfect_cell = ClimateCell(
        lat=0.0,
        lon=0.0,
        temperature_c=(18.0,) * 11 + (45.0,),
        temperature_min_c=(14.0,) * 11 + (35.0,),
        temperature_max_c=(22.0,) * 11 + (50.0,),
        precipitation_mm=(0.0,) * 11 + (300.0,),
        cloud_cover_pct=(15,) * 11 + (100,),
    )

    assert 0.0 < annual_score(mostly_perfect_cell, make_preferences()) < 1.0


def test_annual_score_penalizes_a_single_too_hot_month() -> None:
    mild_cell = ClimateCell(
        lat=0.0,
        lon=0.0,
        temperature_c=(18.0,) * 12,
        temperature_min_c=(12.0,) * 12,
        temperature_max_c=(22.0,) * 12,
        precipitation_mm=(0.0,) * 12,
        cloud_cover_pct=(15,) * 12,
    )
    hot_peak_cell = ClimateCell(
        lat=1.0,
        lon=1.0,
        temperature_c=(18.0,) * 11 + (21.0,),
        temperature_min_c=(12.0,) * 11 + (18.0,),
        temperature_max_c=(22.0,) * 11 + (35.0,),
        precipitation_mm=(0.0,) * 12,
        cloud_cover_pct=(15,) * 12,
    )

    preferences = make_preferences(preferred_day_temperature=22, summer_heat_limit=25, winter_cold_limit=5)

    assert annual_score(hot_peak_cell, preferences) < annual_score(mild_cell, preferences)


@pytest.mark.parametrize(
    ("field_name", "temperature_c", "precipitation_mm", "cloud_cover_pct"),
    [
        ("temperature_c", (20.0,) * 11, (50.0,) * 12, (30,) * 12),
        ("precipitation_mm", (20.0,) * 12, (50.0,) * 11, (30,) * 12),
        ("cloud_cover_pct", (20.0,) * 12, (50.0,) * 12, (30,) * 11),
    ],
)
def test_climate_cell_requires_exactly_twelve_months_per_signal(
    field_name: str,
    temperature_c: tuple[float, ...],
    precipitation_mm: tuple[float, ...],
    cloud_cover_pct: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ClimateCell(
            lat=0.0,
            lon=0.0,
            temperature_c=temperature_c,
            temperature_min_c=(10.0,) * 12,
            temperature_max_c=(20.0,) * 12,
            precipitation_mm=precipitation_mm,
            cloud_cover_pct=cloud_cover_pct,
        )


def test_vectorized_scoring_matches_scalar_scoring_for_stub_cells() -> None:
    climate_matrix = ClimateMatrix(
        latitudes=np.array([cell.lat for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        longitudes=np.array([cell.lon for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        temperature_c=np.array([cell.temperature_c for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        temperature_min_c=np.array([cell.temperature_min_c for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        temperature_max_c=np.array([cell.temperature_max_c for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        precipitation_mm=np.array([cell.precipitation_mm for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        cloud_cover_pct=np.array([cell.cloud_cover_pct for cell in STUB_CLIMATE_CELLS], dtype=np.uint8),
    )

    preferences = make_preferences(summer_heat_limit=28, winter_cold_limit=8)
    scalar_scores = np.array([annual_score(cell, preferences) for cell in STUB_CLIMATE_CELLS], dtype=np.float32)
    vectorized_scores = score_climate_matrix(climate_matrix, preferences)

    assert np.allclose(vectorized_scores, scalar_scores)


def test_normalize_score_array_scales_best_match_to_one() -> None:
    normalized = normalize_score_array(np.array([0.25, 0.5, 0.125], dtype=np.float32))

    assert normalized.tolist() == [0.5, 1.0, 0.25]


def test_score_matrix_row_breakdown_returns_structured_probe_metrics() -> None:
    climate_matrix = ClimateMatrix.from_cells((STUB_CLIMATE_CELLS[0],))
    breakdown = score_matrix_row_breakdown(climate_matrix, 0, make_preferences())

    assert isinstance(breakdown, ProbeBreakdown)
    assert 0 <= breakdown.overall_score <= 1
    assert [metric.key for metric in breakdown.metrics] == ["temp", "high", "low", "rain", "sun"]
    assert [metric.label for metric in breakdown.metrics] == ["temp", "high", "low", "rain", "sun"]
    assert all(metric.display_value for metric in breakdown.metrics)
