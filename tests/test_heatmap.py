import numpy as np

from backend.heatmap import HeatmapProjection, render_heatmap_png_from_arrays, render_heatmap_png_from_projection


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
    assert projection.xs.tolist() == [720, 720]
    assert projection.ys.tolist() == [360, 360]
