"""Server-side heatmap rendering for climate score data."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter

if TYPE_CHECKING:
    from .scoring import ScorePoint

WIDTH = 1440
HEIGHT = 720
BLUR_RADIUS = 14  # px — ~3.5 degrees at this resolution

# Matches the client-side color ramp; stop format: (value, (R, G, B, A))
_COLOR_STOPS: list[tuple[float, tuple[int, int, int, int]]] = [
    (0.00, (53,   92,  125,   0)),
    (0.20, (53,   92,  125,  89)),
    (0.45, (127, 179,  213, 140)),
    (0.65, (248, 182,   90, 199)),
    (0.82, (234,  95,  137, 224)),
    (1.00, (121,  40,  202, 235)),
]


def _apply_color_ramp(values: np.ndarray) -> np.ndarray:
    """Map a 0-1 float grid to RGBA via piecewise linear interpolation."""
    rgba = np.zeros((*values.shape, 4), dtype=np.float32)

    for i in range(len(_COLOR_STOPS) - 1):
        t0, c0 = _COLOR_STOPS[i]
        t1, c1 = _COLOR_STOPS[i + 1]
        in_band = (values > t0) & (values <= t1)
        f = np.where(in_band, (values - t0) / (t1 - t0), 0.0)
        for ch in range(4):
            rgba[..., ch] += np.where(in_band, c0[ch] + f * (c1[ch] - c0[ch]), 0.0)

    # Pin the final stop exactly
    at_top = values >= _COLOR_STOPS[-1][0]
    for ch in range(4):
        rgba[..., ch] = np.where(at_top, _COLOR_STOPS[-1][1][ch], rgba[..., ch])

    return np.clip(rgba, 0, 255).astype(np.uint8)


def render_heatmap_png(scores: list[ScorePoint]) -> bytes:
    """Rasterize scored cells, apply Gaussian blur, and return a PNG as bytes.

    The image covers the full world extent [-180,-90] to [180,90] in Mercator so
    MapLibre can place it as an image source with the standard world corner coords.
    """
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for p in scores:
        x = int((p["lon"] + 180.0) / 360.0 * WIDTH)
        y = int((90.0 - p["lat"]) / 180.0 * HEIGHT)
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            grid[y, x] = max(grid[y, x], p["score"])

    pil_gray = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    pil_gray = pil_gray.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    blurred = np.array(pil_gray, dtype=np.float32) / 255.0

    rgba = _apply_color_ramp(blurred)

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()
