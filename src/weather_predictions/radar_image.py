"""Render the current reflectivity grid, cropped to a region around a point,
as a 7-color image for a Waveshare 5.65" ACeP e-Paper display (600x448),
with arrows showing which way precipitation cells are moving.

Needs the `display` dependency group (`poetry install --with display`) —
Pillow + OpenCV (for the motion field), no Py-ART. Runs fine on a Pi given
grids synced over from wherever decoding happened (see radar_nowcast.py's
docstring for the same point).

Motion arrows come from the same dense-optical-flow estimate radar_nowcast.py
uses for its forecast, just visualized directly rather than used to advect
the grid forward — this shows current movement, not a future prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from weather_predictions.config import LATITUDE, LONGITUDE
from weather_predictions.radar_nowcast import GRID_DIR, NO_ECHO_DBZ, estimate_motion_field, load_recent_frames

# Waveshare 5.65" ACeP 7-color e-Paper panel.
PANEL_SIZE = (600, 448)

# Standard ACeP palette (approximate manufacturer RGB values).
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE = (255, 128, 0)

# dBZ bins -> one of the 7 panel colors, roughly following the standard NWS
# reflectivity scale but quantized to what a 7-color panel can show. Black
# and white are reserved for arrows/background so they don't get confused
# with reflectivity.
_DBZ_COLOR_BINS: list[tuple[float, tuple[int, int, int]]] = [
    (5.0, COLOR_WHITE),  # no meaningful echo
    (20.0, COLOR_BLUE),  # light
    (30.0, COLOR_GREEN),  # moderate
    (40.0, COLOR_YELLOW),  # heavy
    (50.0, COLOR_ORANGE),  # very heavy
    (float("inf"), COLOR_RED),  # severe
]

_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_AT_EQUATOR = 111.320

# Skip drawing an arrow where there's essentially no echo to move.
_ARROW_MIN_DBZ = 15.0


class OutOfRadarRangeError(RuntimeError):
    pass


@dataclass
class RenderResult:
    output_path: Path
    frame_timestamp: str
    region_radius_km: float


def _km_offset(origin_lat: float, origin_lon: float, lat: float, lon: float) -> tuple[float, float]:
    dy_km = (lat - origin_lat) * _KM_PER_DEG_LAT
    dx_km = (lon - origin_lon) * _KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(origin_lat))
    return dx_km, dy_km


def _pixel_bounds(
    frame: dict[str, Any], center_lat: float, center_lon: float, radius_km: float
) -> tuple[int, int, int, int]:
    """Row/col bounding box (row_min, row_max, col_min, col_max), inclusive,
    for a `radius_km` box centered on (center_lat, center_lon) within `frame`."""
    n = frame["reflectivity_dbz"].shape[0]
    grid_km = frame["grid_km"]
    dx_km, dy_km = _km_offset(frame["latitude"], frame["longitude"], center_lat, center_lon)

    if abs(dx_km) - radius_km > grid_km + 1e-6 or abs(dy_km) - radius_km > grid_km + 1e-6:
        raise OutOfRadarRangeError(
            f"({center_lat}, {center_lon}) +/- {radius_km:.0f}km falls entirely outside "
            f"the radar's {grid_km:.0f}km grid."
        )

    def to_index(km: float) -> float:
        return (km + grid_km) / (2 * grid_km) * (n - 1)

    col_center, row_center = to_index(dx_km), to_index(dy_km)
    half_span_px = radius_km / grid_km * (n - 1) / 2
    row_min = max(0, int(round(row_center - half_span_px)))
    row_max = min(n - 1, int(round(row_center + half_span_px)))
    col_min = max(0, int(round(col_center - half_span_px)))
    col_max = min(n - 1, int(round(col_center + half_span_px)))
    return row_min, row_max, col_min, col_max


def dbz_to_rgb(dbz: np.ndarray) -> np.ndarray:
    """Quantize a dBZ array into the 7-color panel palette, (H, W, 3) uint8."""
    filled = np.nan_to_num(dbz, nan=NO_ECHO_DBZ)
    rgb = np.zeros((*filled.shape, 3), dtype=np.uint8)
    lower = -np.inf
    for upper, color in _DBZ_COLOR_BINS:
        mask = (filled >= lower) & (filled < upper)
        rgb[mask] = color
        lower = upper
    return rgb


def _draw_arrows(image: Image.Image, dbz_crop: np.ndarray, flow_crop: np.ndarray, spacing_px: int = 20) -> None:
    """Overlay black arrows on a coarse grid, one per `spacing_px` block.

    Each arrow summarizes its whole block (max reflectivity for the "is
    there anything here" gate, mean flow vector for direction) rather than
    sampling a single pixel at the block's center — real reflectivity is
    often scattered/speckled rather than one solid cell, so point-sampling
    mostly lands on gaps between echoes and draws almost no arrows even
    when there's clear, real motion in the data.
    """
    draw = ImageDraw.Draw(image)
    scale_x = image.width / dbz_crop.shape[1]
    scale_y = image.height / dbz_crop.shape[0]
    max_len_px = spacing_px * min(scale_x, scale_y) * 0.4

    filled_dbz = np.nan_to_num(dbz_crop, nan=NO_ECHO_DBZ)
    n_rows, n_cols = dbz_crop.shape
    for row in range(0, n_rows, spacing_px):
        for col in range(0, n_cols, spacing_px):
            row_end, col_end = min(row + spacing_px, n_rows), min(col + spacing_px, n_cols)
            block_dbz = filled_dbz[row:row_end, col:col_end]
            if block_dbz.max() < _ARROW_MIN_DBZ:
                continue

            block_flow = flow_crop[row:row_end, col:col_end]
            fx, fy = block_flow[..., 0].mean(), block_flow[..., 1].mean()
            magnitude = math.hypot(fx, fy)
            if magnitude < 1e-3:
                continue

            x0 = (col + (col_end - col) / 2) * scale_x
            y0 = (row + (row_end - row) / 2) * scale_y
            # Floor the length so a real-but-slow motion signal still reads as
            # a visible arrow on a low-contrast e-ink panel, not a speck.
            length = max(10.0, min(max_len_px, magnitude * min(scale_x, scale_y) * 3))
            x1 = x0 + (fx / magnitude) * length
            y1 = y0 + (fy / magnitude) * length

            draw.line([(x0, y0), (x1, y1)], fill=COLOR_BLACK, width=3)
            _draw_arrowhead(draw, (x0, y0), (x1, y1))


def _draw_arrowhead(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float]) -> None:
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len, head_angle = 8.0, math.radians(25)
    for sign in (-1, 1):
        wing_angle = angle + math.pi - sign * head_angle
        wing = (end[0] + head_len * math.cos(wing_angle), end[1] + head_len * math.sin(wing_angle))
        draw.line([end, wing], fill=COLOR_BLACK, width=3)


def render(
    radius_km: float = 50.0,
    center_lat: float = LATITUDE,
    center_lon: float = LONGITUDE,
    grid_dir: Path = GRID_DIR,
    output_path: Path = Path("radar.png"),
    panel_size: tuple[int, int] = PANEL_SIZE,
) -> RenderResult:
    """Render the most recent reflectivity grid, cropped to `radius_km` around
    (center_lat, center_lon), with motion arrows, as a PNG sized for the panel."""
    prev_frame, curr_frame = load_recent_frames(2, grid_dir)
    flow, _interval_minutes = estimate_motion_field(prev_frame, curr_frame)

    row_min, row_max, col_min, col_max = _pixel_bounds(curr_frame, center_lat, center_lon, radius_km)
    dbz_crop = curr_frame["reflectivity_dbz"][row_min : row_max + 1, col_min : col_max + 1]
    flow_crop = flow[row_min : row_max + 1, col_min : col_max + 1]

    rgb = dbz_to_rgb(dbz_crop)
    image = Image.fromarray(rgb).resize(panel_size, Image.NEAREST)
    _draw_arrows(image, dbz_crop, flow_crop)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return RenderResult(output_path=output_path, frame_timestamp=curr_frame["timestamp"], region_radius_km=radius_km)


if __name__ == "__main__":
    result = render()
    print(f"Rendered {result.output_path} from frame {result.frame_timestamp} (+/-{result.region_radius_km:.0f}km)")
