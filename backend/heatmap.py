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

WIDTH = 1440
HEIGHT = 720
BLUR_RADIUS = 14  # px — ~3.5 degrees at this resolution
PEAK_BOOST_THRESHOLD = 0.72
PEAK_BOOST_STRENGTH = 0.8
_MERCATOR_MAX_RENDER_LATITUDE = MAP_PROJECTION.max_render_latitude

if MAP_PROJECTION.name != "mercator" or _MERCATOR_MAX_RENDER_LATITUDE is None:
    msg = f"Unsupported heatmap projection: {MAP_PROJECTION.name}"
    raise ValueError(msg)

# Mercator y at the maximum latitude — used to normalise y to [0, HEIGHT].
_Y_MAX = math.log(math.tan(math.pi / 4 + _MERCATOR_MAX_RENDER_LATITUDE * math.pi / 360))

# Color ramp: stop format (value, (R, G, B, A)). Value 0.0 maps to alpha=0 (transparent).
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
    xs: NDArray[np.int32]
    ys: NDArray[np.int32]

    @classmethod
    def from_coordinates(cls, latitudes: np.ndarray, longitudes: np.ndarray) -> HeatmapProjection:
        """Project one fixed set of lat/lon coordinates into heatmap pixels once."""
        valid = np.abs(latitudes) < _MERCATOR_MAX_RENDER_LATITUDE
        valid_indexes = np.flatnonzero(valid).astype(np.int32, copy=False)
        valid_latitudes = latitudes[valid]
        valid_longitudes = longitudes[valid]

        xs = ((valid_longitudes + 180.0) / 360.0 * WIDTH).astype(np.int32)
        y_merc = np.log(np.tan(np.pi / 4 + np.radians(valid_latitudes) / 2))
        ys = ((_Y_MAX - y_merc) / (2 * _Y_MAX) * HEIGHT).astype(np.int32)

        in_bounds = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        return cls(
            score_indexes=valid_indexes[in_bounds],
            xs=xs[in_bounds],
            ys=ys[in_bounds],
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

    # Pin the final stop exactly
    at_top = values >= _COLOR_STOPS[-1][0]
    for ch in range(4):
        rgba[..., ch] = np.where(at_top, _COLOR_STOPS[-1][1][ch], rgba[..., ch])

    return np.clip(rgba, 0, 255).astype(np.uint8)


def _build_color_ramp_lookup() -> np.ndarray:
    """Precompute RGBA output for every blurred 8-bit grayscale value."""
    grayscale_values = np.arange(256, dtype=np.float32) / 255.0
    return _apply_color_ramp(grayscale_values[:, None])[:, 0, :]


_COLOR_RAMP_LOOKUP = _build_color_ramp_lookup()


def _preserve_local_maxima(base_gray: NDArray[np.uint8], blurred_gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Keep isolated strong cells visible after the soft blur pass.

    The blur gives the surface a continuous look, but it can also flatten a real
    local best match into its weaker surroundings. Only genuinely strong source
    cells get a direct boost so the broader surface stays soft and low-value
    single-cell noise does not leak back into the final image.
    """
    base_float = base_gray.astype(np.float32)
    blurred_float = blurred_gray.astype(np.float32)
    peak_mask = base_float >= PEAK_BOOST_THRESHOLD * 255.0
    peak_floor = np.where(peak_mask, base_float * PEAK_BOOST_STRENGTH, 0.0)
    preserved = np.maximum(blurred_float, peak_floor)
    return np.clip(preserved, 0, 255).astype(np.uint8)


def render_heatmap_png_from_projection(projection: HeatmapProjection, scores: np.ndarray) -> bytes:
    """Rasterize one score vector aligned with the projection's source climate-matrix rows."""
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    np.maximum.at(grid, (projection.ys, projection.xs), scores[projection.score_indexes])

    base_gray = (grid * 255).astype(np.uint8)
    pil_gray = Image.fromarray(base_gray, mode="L")
    pil_gray = pil_gray.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    blurred_gray = np.asarray(pil_gray, dtype=np.uint8)
    rgba = _COLOR_RAMP_LOOKUP[_preserve_local_maxima(base_gray, blurred_gray)]

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
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
