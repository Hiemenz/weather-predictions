"""Render an MRMS national radar region as a color image with motion arrows.

Like radar_image.py but uses the MRMS CONUS composite, so any point in the
country can be rendered — not just ±200km around the nearest NEXRAD site.
The rendered region, dBZ color scale, and arrow style are identical to the
e-ink panel rendering so the output can be used interchangeably.

Needs `poetry install --with display` for OpenCV (motion field) and Pillow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from weather_predictions.config import LATITUDE, LONGITUDE, MRMS_DATA_DIR
from weather_predictions.mrms_nowcast import GRID_DIR, load_recent_mrms_frames
from weather_predictions.mrms_processing import OutOfMrmsRangeError, crop_to_region
from weather_predictions.radar_image import PANEL_SIZE, COLOR_BLACK, dbz_to_rgb, draw_arrows
from weather_predictions.radar_nowcast import estimate_motion_field

_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_AT_EQUATOR = 111.320


@dataclass
class MrmsRenderResult:
    output_path: Path
    frame_timestamp: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    radius_km: float


def render(
    radius_km: float = 300.0,
    center_lat: float = LATITUDE,
    center_lon: float = LONGITUDE,
    grid_dir: Path = GRID_DIR,
    output_path: Path = MRMS_DATA_DIR / "mrms_radar.png",
    panel_size: tuple[int, int] = PANEL_SIZE,
) -> MrmsRenderResult:
    """Render the most recent MRMS reflectivity, cropped to `radius_km` around
    (center_lat, center_lon), with motion arrows showing storm movement.

    The bounding box of the rendered region is included in the result so callers
    can overlay it on a map or display it alongside coordinates.
    """
    prev_frame, curr_frame = load_recent_mrms_frames(2, grid_dir)
    prev_crop = crop_to_region(prev_frame, center_lat, center_lon, radius_km)
    curr_crop = crop_to_region(curr_frame, center_lat, center_lon, radius_km)

    flow, _interval_minutes = estimate_motion_field(prev_crop, curr_crop)

    dbz = curr_crop["reflectivity_dbz"]
    rgb = dbz_to_rgb(dbz)
    image = Image.fromarray(rgb).resize(panel_size, Image.NEAREST)
    draw_arrows(image, dbz, flow)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return MrmsRenderResult(
        output_path=output_path,
        frame_timestamp=curr_frame["timestamp"],
        lat_min=curr_crop["lat_min"],
        lat_max=curr_crop["lat_max"],
        lon_min=curr_crop["lon_min"],
        lon_max=curr_crop["lon_max"],
        radius_km=radius_km,
    )


if __name__ == "__main__":
    result = render()
    print(
        f"Rendered {result.output_path} from frame {result.frame_timestamp} | "
        f"bbox lat [{result.lat_min:.3f}, {result.lat_max:.3f}] "
        f"lon [{result.lon_min:.3f}, {result.lon_max:.3f}]"
    )
