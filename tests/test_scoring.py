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


def test_temperature_score_keeps_the_comfort_band_at_full_score() -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=4,
        heat_tolerance=6,
        rain_sensitivity=50,
        sun_preference=60,
    )

    assert temperature_score(22.0, preferences) == 1.0
    assert temperature_score(23.5, preferences) == 1.0
    assert temperature_score(20.5, preferences) == 1.0


def test_temperature_score_uses_cold_and_heat_tolerances_as_separate_slopes() -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=2,
        heat_tolerance=10,
        rain_sensitivity=50,
        sun_preference=60,
    )

    assert temperature_score(15.0, preferences) < temperature_score(29.0, preferences)


def test_temperature_score_uses_the_configured_band_and_slope_width() -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=4,
        heat_tolerance=6,
        rain_sensitivity=50,
        sun_preference=60,
    )

    assert temperature_score(24.0, preferences) == pytest.approx(1 - (0.5 / 12))
    assert temperature_score(20.0, preferences) == pytest.approx(1 - (0.5 / 10))


def test_temperature_score_with_zero_tolerance_still_uses_base_slope() -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=0,
        heat_tolerance=0,
        rain_sensitivity=50,
        sun_preference=60,
    )

    assert temperature_score(24.0, preferences) == pytest.approx(1 - (0.5 / 6))


def test_cloud_score_penalizes_overcast_months_more_for_sun_seekers() -> None:
    assert cloud_score(40, sun_preference=20) == 1.0
    assert cloud_score(40, sun_preference=90) < 1.0
    assert cloud_score(90, sun_preference=90) < cloud_score(60, sun_preference=90)


def test_cloud_score_stays_perfect_at_the_tolerance_threshold_and_saturates_at_full_overcast() -> None:
    assert cloud_score(22, sun_preference=90) == 1.0
    assert cloud_score(100, sun_preference=90) == 0.0
    assert cloud_score(61, sun_preference=90) == pytest.approx(0.75)


def test_rain_score_respects_zero_sensitivity_and_saturation_point() -> None:
    assert rain_score(150.0, sensitivity=0) == 1.0
    assert rain_score(150.0, sensitivity=50) == pytest.approx(0.75)
    assert rain_score(300.0, sensitivity=100) == 0.0
    assert rain_score(450.0, sensitivity=100) == 0.0


def test_annual_score_stays_normalized_for_extreme_climate_rows() -> None:
    extreme_cell = ClimateCell(
        lat=0.0,
        lon=0.0,
        temperature_c=(50.0,) * 12,
        precipitation_mm=(400.0,) * 12,
        cloud_cover_pct=(100,) * 12,
    )
    preferences = PreferenceInputs(
        ideal_temperature=-10,
        cold_tolerance=0,
        heat_tolerance=0,
        rain_sensitivity=100,
        sun_preference=100,
    )

    assert annual_score(extreme_cell, preferences) == 0.0


def test_annual_score_averages_monthly_scores_over_twelve_months() -> None:
    mostly_perfect_cell = ClimateCell(
        lat=0.0,
        lon=0.0,
        temperature_c=(22.0,) * 11 + (50.0,),
        precipitation_mm=(0.0,) * 11 + (300.0,),
        cloud_cover_pct=(15,) * 11 + (100,),
    )
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=0,
        heat_tolerance=0,
        rain_sensitivity=100,
        sun_preference=100,
    )

    assert annual_score(mostly_perfect_cell, preferences) == pytest.approx(11 / 12)


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
            precipitation_mm=precipitation_mm,
            cloud_cover_pct=cloud_cover_pct,
        )


def test_vectorized_scoring_matches_scalar_scoring_for_stub_cells() -> None:
    preferences = PreferenceInputs(
        ideal_temperature=22,
        cold_tolerance=7,
        heat_tolerance=5,
        rain_sensitivity=55,
        sun_preference=60,
    )
    climate_matrix = ClimateMatrix(
        latitudes=np.array([cell.lat for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        longitudes=np.array([cell.lon for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        temperature_c=np.array([cell.temperature_c for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        precipitation_mm=np.array([cell.precipitation_mm for cell in STUB_CLIMATE_CELLS], dtype=np.float32),
        cloud_cover_pct=np.array([cell.cloud_cover_pct for cell in STUB_CLIMATE_CELLS], dtype=np.uint8),
    )

    scalar_scores = np.array([annual_score(cell, preferences) for cell in STUB_CLIMATE_CELLS], dtype=np.float32)
    vectorized_scores = score_climate_matrix(climate_matrix, preferences)

    assert np.allclose(vectorized_scores, scalar_scores)


def test_normalize_score_array_scales_best_match_to_one() -> None:
    normalized = normalize_score_array(np.array([0.25, 0.5, 0.125], dtype=np.float32))

    assert normalized.tolist() == [0.5, 1.0, 0.25]


def test_score_matrix_row_breakdown_returns_structured_probe_metrics() -> None:
    climate_matrix = ClimateMatrix.from_cells((STUB_CLIMATE_CELLS[0],))
    breakdown = score_matrix_row_breakdown(
        climate_matrix,
        0,
        PreferenceInputs(
            ideal_temperature=22,
            cold_tolerance=7,
            heat_tolerance=5,
            rain_sensitivity=55,
            sun_preference=60,
        ),
    )

    assert isinstance(breakdown, ProbeBreakdown)
    assert 0 <= breakdown.overall_score <= 1
    assert [metric.key for metric in breakdown.metrics] == ["temp", "rain", "sun"]
    assert all(metric.display_value for metric in breakdown.metrics)
