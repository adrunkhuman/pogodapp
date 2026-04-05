from io import BytesIO

import numpy as np
from PIL import Image

from backend.heatmap import (
    HEIGHT,
    WIDTH,
    HeatmapProjection,
    _preserve_local_maxima,
    _smooth_styled_heatmap_gray,
    _stylize_heatmap_gray,
    render_heatmap_png_from_arrays,
    render_heatmap_png_from_projection,
)


def test_cached_heatmap_projection_matches_array_renderer() -> None:
    latitudes = np.array([37.5, 41.9, -33.9, 90.0], dtype=np.float32)
    longitudes = np.array([-122.0, 12.5, 18.4, 0.0], dtype=np.float32)
    scores = np.array([1.0, 0.6, 0.8, 0.9], dtype=np.float32)

    projection = HeatmapProjection.from_coordinates(latitudes, longitudes)

    assert render_heatmap_png_from_projection(projection, scores) == render_heatmap_png_from_arrays(
        latitudes,
        longitudes,
        scores,
    )


def test_heatmap_projection_filters_invalid_latitudes_and_keeps_duplicate_pixels() -> None:
    projection = HeatmapProjection.from_coordinates(
        np.array([0.0, 0.0, 90.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert projection.score_indexes.tolist() == [0, 1]
    assert projection.xs.tolist() == [WIDTH // 2, WIDTH // 2]
    assert projection.ys.tolist() == [HEIGHT // 2, HEIGHT // 2]


def test_preserve_local_maxima_lifts_peak_above_blurred_neighbors() -> None:
    base_gray = np.zeros((9, 9), dtype=np.uint8)
    base_gray[4, 4] = 255
    base_gray[4, 3] = 40
    base_gray[4, 5] = 40

    blurred_gray = np.full((9, 9), 72, dtype=np.uint8)
    blurred_gray[4, 4] = 68

    preserved = _preserve_local_maxima(base_gray, blurred_gray)

    assert preserved[4, 4] > blurred_gray[4, 4]
    assert preserved[4, 4] > preserved[4, 3]
    assert preserved[3, 4] < preserved[4, 4]


def test_heatmap_png_keeps_peak_pixel_opaque_when_surrounded_by_weaker_scores() -> None:
    latitudes = np.array([0.0, 0.0, 0.5, -0.5, 0.0], dtype=np.float32)
    longitudes = np.array([0.0, -0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    scores = np.array([1.0, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    projection = HeatmapProjection.from_coordinates(latitudes, longitudes)
    png_bytes = render_heatmap_png_from_projection(projection, scores)
    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    alpha = np.asarray(image, dtype=np.uint8)[..., 3]

    peak_x = int(projection.xs[0])
    peak_y = int(projection.ys[0])

    assert alpha[peak_y, peak_x] >= 165


def test_stylize_heatmap_gray_quantizes_to_limited_band_values() -> None:
    gray = np.array([[0, 20, 60, 110, 170, 230, 255]], dtype=np.uint8)

    styled = _stylize_heatmap_gray(gray)

    assert styled[0, 0] == 0
    assert np.all(np.diff(styled[0].astype(np.int16)) >= 0)
    assert styled[0, 2] < gray[0, 2]
    assert styled[0, -1] == 255


def test_smooth_styled_heatmap_gray_softens_isolated_speckle() -> None:
    gray = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    smoothed = _smooth_styled_heatmap_gray(gray)

    assert 0 < smoothed[1, 1] < 255
