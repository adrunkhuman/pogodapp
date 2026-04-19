"""Server-side heatmap rendering for climate score data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

from backend.config import MAP_PROJECTION

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .scoring import CellScorePoint

WIDTH = 1600
HEIGHT = 800
WORK_GRID_SCALE = 3
WORK_WIDTH = (WIDTH + WORK_GRID_SCALE - 1) // WORK_GRID_SCALE
WORK_HEIGHT = (HEIGHT + WORK_GRID_SCALE - 1) // WORK_GRID_SCALE
WORK_BLUR_RADIUS = 4.0
NORM_MIN_CELLS = 10  # work tiles with fewer source cells blend toward diluted to suppress single-cell outliers
PEAK_DETAIL_BLUR_RADIUS = 2.4
PEAK_DETAIL_BLEND_WEIGHT = 0.55
UPSCALED_BLEND_BLUR_RADIUS = 0.9
GLOW_RADIUS = 12.0  # px — post-mask bloom so narrow hotspots radiate into surrounding ocean/gaps
SCORE_CURVE_GAMMA = 1.35
_MERCATOR_MAX_RENDER_LATITUDE = MAP_PROJECTION.max_render_latitude

if MAP_PROJECTION.name != "mercator" or _MERCATOR_MAX_RENDER_LATITUDE is None:
    msg = f"Unsupported heatmap projection: {MAP_PROJECTION.name}"
    raise ValueError(msg)

# Mercator y at the maximum latitude, used to normalise y to [0, HEIGHT].
_Y_MAX = math.log(math.tan(math.pi / 4 + _MERCATOR_MAX_RENDER_LATITUDE * math.pi / 360))

# 0.0 stays transparent; higher scores interpolate through the RGBA ramp.
_COLOR_STOPS: list[tuple[float, tuple[int, int, int, int]]] = [
    (0.00, (53, 92, 125, 0)),
    (0.20, (53, 92, 125, 89)),
    (0.45, (127, 179, 213, 140)),
    (0.65, (248, 182, 90, 199)),
    (0.82, (234, 95, 137, 224)),
    (1.00, (121, 40, 202, 235)),
]


@dataclass(frozen=True, slots=True)
class HeatmapProjection:
    """Cached raster coordinates for one fixed climate grid."""

    score_indexes: NDArray[np.int32]
    work_indexes: NDArray[np.int32]
    # Pre-dilated land mask: MaxFilter(7) is grid-fixed, not score-dependent.
    land_mask: NDArray[np.bool_]

    @classmethod
    def from_coordinates(cls, latitudes: np.ndarray, longitudes: np.ndarray) -> HeatmapProjection:
        """Project one fixed set of lat/lon coordinates into heatmap pixels once."""
        valid = np.abs(latitudes) < _MERCATOR_MAX_RENDER_LATITUDE
        valid_indexes = np.flatnonzero(valid).astype(np.int32, copy=False)
        valid_latitudes = latitudes[valid]
        valid_longitudes = longitudes[valid]

        xs = ((valid_longitudes + 180.0) / 360.0 * WIDTH).astype(np.uint16)
        y_merc = np.log(np.tan(np.pi / 4 + np.radians(valid_latitudes) / 2))
        ys = ((_Y_MAX - y_merc) / (2 * _Y_MAX) * HEIGHT).astype(np.uint16)

        in_bounds = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        final_xs = xs[in_bounds]
        final_ys = ys[in_bounds]
        work_xs = (final_xs // WORK_GRID_SCALE).astype(np.int32, copy=False)
        work_ys = (final_ys // WORK_GRID_SCALE).astype(np.int32, copy=False)
        work_indexes = work_ys * WORK_WIDTH + work_xs

        land_mask_raw = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        land_mask_raw[final_ys, final_xs] = 255
        land_mask = np.asarray(Image.fromarray(land_mask_raw, mode="L").filter(ImageFilter.MaxFilter(7))) > 0

        return cls(
            score_indexes=valid_indexes[in_bounds],
            work_indexes=work_indexes,
            land_mask=land_mask,
        )


def _apply_color_ramp(values: np.ndarray) -> np.ndarray:
    """Map a 0-1 float grid to RGBA via piecewise linear interpolation."""
    rgba = np.zeros((*values.shape, 4), dtype=np.float32)

    for i in range(len(_COLOR_STOPS) - 1):
        t0, c0 = _COLOR_STOPS[i]
        t1, c1 = _COLOR_STOPS[i + 1]
        in_band = (values >= t0) & (values <= t1) if i == 0 else (values > t0) & (values <= t1)
        f = np.where(in_band, (values - t0) / (t1 - t0), 0.0)
        for ch in range(4):
            rgba[..., ch] += np.where(in_band, c0[ch] + f * (c1[ch] - c0[ch]), 0.0)

    at_top = values >= _COLOR_STOPS[-1][0]
    for ch in range(4):
        rgba[..., ch] = np.where(at_top, _COLOR_STOPS[-1][1][ch], rgba[..., ch])

    return np.clip(rgba, 0, 255).astype(np.uint8)


def _build_color_ramp_lookup() -> np.ndarray:
    """Precompute RGBA output for every blurred 8-bit grayscale value."""
    grayscale_values = np.arange(256, dtype=np.float32) / 255.0
    return _apply_color_ramp(grayscale_values[:, None])[:, 0, :]


_COLOR_RAMP_LOOKUP = _build_color_ramp_lookup()


def _stylize_heatmap_gray(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Compress the low end while keeping a continuous gradient surface."""
    curved = np.power(gray.astype(np.float32) / 255.0, SCORE_CURVE_GAMMA)
    return np.rint(curved * 255).astype(np.uint8)


def _blur_work_field(field: NDArray[np.float32], radius: float) -> NDArray[np.float32]:
    """Blur one work-grid field through Pillow's fast grayscale path."""
    return (
        np.asarray(
            Image.fromarray((field * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=radius)),
            dtype=np.float32,
        )
        / 255.0
    )


def render_heatmap_png_from_projection(projection: HeatmapProjection, scores: np.ndarray) -> bytes:
    """Rasterize one score vector aligned with the projection's source climate-matrix rows."""
    work_scores = scores[projection.score_indexes]

    summed_grid = (
        np.bincount(
            projection.work_indexes,
            weights=work_scores,
            minlength=WORK_WIDTH * WORK_HEIGHT,
        )
        .astype(np.float32, copy=False)
        .reshape((WORK_HEIGHT, WORK_WIDTH))
    )
    hit_count_grid = np.bincount(
        projection.work_indexes,
        minlength=WORK_WIDTH * WORK_HEIGHT,
    ).reshape((WORK_HEIGHT, WORK_WIDTH))
    work_field = np.divide(
        summed_grid,
        hit_count_grid,
        out=np.zeros_like(summed_grid),
        where=hit_count_grid > 0,
    )
    peak_grid = np.zeros(WORK_WIDTH * WORK_HEIGHT, dtype=np.float32)
    np.maximum.at(peak_grid, projection.work_indexes, work_scores)
    peak_grid = peak_grid.reshape((WORK_HEIGHT, WORK_WIDTH))

    # Normalized blur: blur numerator and mask separately, then divide.
    # Prevents ocean zero-tiles from diluting isolated coastal hotspots.
    blurred_field = _blur_work_field(work_field, WORK_BLUR_RADIUS) * 255.0
    blurred_mask = np.asarray(
        Image.fromarray(((work_field > 0) * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.GaussianBlur(radius=WORK_BLUR_RADIUS)
        ),
        dtype=np.float32,
    )
    normalized = np.divide(blurred_field, blurred_mask, out=np.zeros_like(blurred_field), where=blurred_mask > 0)
    plain = blurred_field / 255.0
    confidence = np.clip(hit_count_grid.astype(np.float32) / NORM_MIN_CELLS, 0.0, 1.0)
    work_smooth = (plain * (1.0 - confidence) + normalized * confidence).clip(0.0, 1.0)
    peak_detail = _blur_work_field(np.clip(peak_grid - work_field, 0.0, 1.0), PEAK_DETAIL_BLUR_RADIUS)
    work_smooth = np.clip(work_smooth + peak_detail * PEAK_DETAIL_BLEND_WEIGHT, 0.0, 1.0)
    upscaled_blurred_gray = np.asarray(
        Image.fromarray((work_smooth * 255).astype(np.uint8), mode="L").resize(
            (WIDTH, HEIGHT), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )
    blended_gray = np.asarray(
        Image.fromarray(upscaled_blurred_gray, mode="L").filter(
            ImageFilter.GaussianBlur(radius=UPSCALED_BLEND_BLUR_RADIUS)
        ),
        dtype=np.uint8,
    )
    styled_gray = (_stylize_heatmap_gray(blended_gray) * projection.land_mask).astype(np.uint8)
    glow_gray = np.asarray(
        Image.fromarray(styled_gray, mode="L").filter(ImageFilter.GaussianBlur(radius=GLOW_RADIUS)),
        dtype=np.uint8,
    )
    styled_gray = np.maximum(styled_gray, glow_gray)
    rgba = _COLOR_RAMP_LOOKUP[styled_gray]

    buf = BytesIO()
    # Lower PNG compression trades a small size increase for materially less CPU per request.
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def render_heatmap_png_from_arrays(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    scores: np.ndarray,
) -> bytes:
    """Rasterize score arrays through the non-cached convenience path."""
    return render_heatmap_png_from_projection(HeatmapProjection.from_coordinates(latitudes, longitudes), scores)


def render_heatmap_png(scores: list[CellScorePoint]) -> bytes:
    """Rasterize scored cells in the configured map projection and return a PNG.

    The server and frontend both read `MAP_PROJECTION`, so projection changes stay
    synchronized. Only Mercator is supported today because the rasterization math
    and MapLibre image corners are projection-specific.
    """
    return render_heatmap_png_from_arrays(
        np.array([point["lat"] for point in scores], dtype=np.float32),
        np.array([point["lon"] for point in scores], dtype=np.float32),
        np.array([point["score"] for point in scores], dtype=np.float32),
    )
