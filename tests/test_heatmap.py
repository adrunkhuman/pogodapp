from io import BytesIO

import numpy as np
from PIL import Image

from backend.heatmap import (
    HEIGHT,
    WIDTH,
    WORK_GRID_SCALE,
    WORK_WIDTH,
    HeatmapProjection,
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

    expected_work_idx = (HEIGHT // 2 // WORK_GRID_SCALE) * WORK_WIDTH + (WIDTH // 2 // WORK_GRID_SCALE)
    assert projection.score_indexes.tolist() == [0, 1]
    assert projection.xs.tolist() == [WIDTH // 2, WIDTH // 2]
    assert projection.ys.tolist() == [HEIGHT // 2, HEIGHT // 2]
    assert projection.work_indexes.tolist() == [expected_work_idx, expected_work_idx]


def test_heatmap_png_hotspot_cluster_outshines_weak_background() -> None:
    # Display-grid renderer works at work-tile granularity, not single-pixel precision.
    # A dense hotspot cluster must produce visibly higher alpha than a distant weak field.
    rng = np.random.default_rng(0)
    bg_lats = rng.uniform(-10.0, 10.0, 300).astype(np.float32)
    bg_lons = rng.uniform(-60.0, 60.0, 300).astype(np.float32)
    hot_lats = rng.uniform(-0.3, 0.3, 30).astype(np.float32)
    hot_lons = rng.uniform(-0.3, 0.3, 30).astype(np.float32)

    lats = np.concatenate([bg_lats, hot_lats])
    lons = np.concatenate([bg_lons, hot_lons])
    scores = np.concatenate([np.full(300, 0.15, dtype=np.float32), np.full(30, 1.0, dtype=np.float32)])

    projection = HeatmapProjection.from_coordinates(latitudes=lats, longitudes=lons)
    alpha = np.asarray(
        Image.open(BytesIO(render_heatmap_png_from_projection(projection, scores))).convert("RGBA"),
        dtype=np.uint8,
    )[..., 3]

    cx, cy = WIDTH // 2, HEIGHT // 2
    hotspot_max = alpha[cy - 6 : cy + 7, cx - 6 : cx + 7].max()
    background_max = alpha[cy, cx + 40 : cx + 60].max()

    assert hotspot_max > background_max


def test_heatmap_png_same_tile_peak_survives_tile_average() -> None:
    latitudes = np.zeros(5, dtype=np.float32)
    longitudes = np.zeros(5, dtype=np.float32)
    mixed_scores = np.array([1.0, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    averaged_scores = np.full(5, mixed_scores.mean(), dtype=np.float32)

    projection = HeatmapProjection.from_coordinates(latitudes, longitudes)
    mixed_alpha = np.asarray(
        Image.open(BytesIO(render_heatmap_png_from_projection(projection, mixed_scores))).convert("RGBA"),
        dtype=np.uint8,
    )[..., 3]
    averaged_alpha = np.asarray(
        Image.open(BytesIO(render_heatmap_png_from_projection(projection, averaged_scores))).convert("RGBA"),
        dtype=np.uint8,
    )[..., 3]

    cy, cx = HEIGHT // 2, WIDTH // 2
    mixed_window = mixed_alpha[cy - 8 : cy + 9, cx - 8 : cx + 9]
    averaged_window = averaged_alpha[cy - 8 : cy + 9, cx - 8 : cx + 9]

    assert mixed_window.mean() > averaged_window.mean()
    assert mixed_window.max() > averaged_window.max()


def test_heatmap_png_keeps_peak_pixel_visible_when_surrounded_by_weaker_scores() -> None:
    latitudes = np.array([0.0, 0.0, 0.5, -0.5, 0.0], dtype=np.float32)
    longitudes = np.array([0.0, -0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    scores = np.array([1.0, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    projection = HeatmapProjection.from_coordinates(latitudes, longitudes)
    alpha = np.asarray(
        Image.open(BytesIO(render_heatmap_png_from_projection(projection, scores))).convert("RGBA"),
        dtype=np.uint8,
    )[..., 3]

    peak_x = int(projection.xs[0])
    peak_y = int(projection.ys[0])

    assert alpha[peak_y, peak_x] >= 165


def test_heatmap_png_uses_configured_raster_dimensions() -> None:
    latitudes = np.array([0.0], dtype=np.float32)
    longitudes = np.array([0.0], dtype=np.float32)
    scores = np.array([1.0], dtype=np.float32)

    projection = HeatmapProjection.from_coordinates(latitudes, longitudes)
    png_bytes = render_heatmap_png_from_projection(projection, scores)
    image = Image.open(BytesIO(png_bytes))

    assert image.size == (WIDTH, HEIGHT)


def test_stylize_heatmap_gray_quantizes_to_limited_band_values() -> None:
    gray = np.array([[0, 20, 60, 110, 170, 230, 255]], dtype=np.uint8)

    styled = _stylize_heatmap_gray(gray)

    assert styled[0, 0] == 0
    assert np.all(np.diff(styled[0].astype(np.int16)) >= 0)
    assert styled[0, 2] < gray[0, 2]
    assert styled[0, -1] == 255
