"""Identify and describe discrete storm cells in an MRMS region.

Thresholds the reflectivity crop, finds connected components (OpenCV), and
uses the optical-flow motion field (already computed for nowcasting) to give
each cell a velocity — so a cell can be reported as e.g. "38km SW of you,
moving NE at 45 km/h, peak 52dBZ".

Needs `poetry install --with display` for OpenCV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from weather_predictions.config import LATITUDE, LONGITUDE
from weather_predictions.mrms_home_check import DEFAULT_RADIUS_KM
from weather_predictions.mrms_nowcast import GRID_DIR, load_recent_mrms_frames
from weather_predictions.mrms_processing import crop_to_region
from weather_predictions.radar_nowcast import estimate_motion_field
from weather_predictions.radar_nowcast_evaluate import RAIN_THRESHOLD_DBZ

_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_AT_EQUATOR = 111.320

# Ignore blobs smaller than this (km^2 at 1km resolution = pixel count) —
# single-pixel speckle isn't a storm cell.
MIN_CELL_AREA_KM2 = 20

_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


@dataclass
class StormCell:
    center_lat: float
    center_lon: float
    area_km2: float
    peak_dbz: float
    mean_dbz: float
    distance_km: float  # from the reference point
    bearing: str  # compass direction from the reference point to the cell
    speed_kmh: float
    heading: str  # compass direction the cell is moving toward
    approaching: bool  # is the cell's motion reducing its distance to the point


def _bearing_to_compass(bearing_deg: float) -> str:
    return _COMPASS[int(round(bearing_deg / 22.5)) % 16]


def detect_cells(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    radius_km: float = DEFAULT_RADIUS_KM,
    threshold_dbz: float = RAIN_THRESHOLD_DBZ,
    grid_dir: Path = GRID_DIR,
) -> list[StormCell]:
    """Detect storm cells in the latest MRMS frame around (lat, lon), sorted
    nearest first. Raises InsufficientFramesError / OutOfMrmsRangeError."""
    prev_frame, curr_frame = load_recent_mrms_frames(2, grid_dir)
    prev_crop = crop_to_region(prev_frame, lat, lon, radius_km)
    curr_crop = crop_to_region(curr_frame, lat, lon, radius_km)
    flow, interval_minutes = estimate_motion_field(prev_crop, curr_crop)

    dbz = np.nan_to_num(curr_crop["reflectivity_dbz"], nan=-32.0)
    mask = (dbz >= threshold_dbz).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    res_lat_km = (curr_crop["lat_max"] - curr_crop["lat_min"]) / (curr_crop["nlat"] - 1) * _KM_PER_DEG_LAT
    res_lon_km = (
        (curr_crop["lon_max"] - curr_crop["lon_min"])
        / (curr_crop["nlon"] - 1)
        * _KM_PER_DEG_LON_AT_EQUATOR
        * math.cos(math.radians(lat))
    )
    px_area_km2 = res_lat_km * res_lon_km

    cells: list[StormCell] = []
    for label in range(1, n_labels):  # label 0 is background
        area_km2 = stats[label, cv2.CC_STAT_AREA] * px_area_km2
        if area_km2 < MIN_CELL_AREA_KM2:
            continue

        cell_mask = labels == label
        col_c, row_c = centroids[label]
        cell_lat = curr_crop["lat_min"] + row_c / (curr_crop["nlat"] - 1) * (curr_crop["lat_max"] - curr_crop["lat_min"])
        cell_lon = curr_crop["lon_min"] + col_c / (curr_crop["nlon"] - 1) * (curr_crop["lon_max"] - curr_crop["lon_min"])

        # Offset from the reference point to the cell, in km.
        dy_km = (cell_lat - lat) * _KM_PER_DEG_LAT
        dx_km = (cell_lon - lon) * _KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat))
        distance_km = math.hypot(dx_km, dy_km)
        bearing_deg = math.degrees(math.atan2(dx_km, dy_km)) % 360

        # Mean flow over the cell, converted from px/frame-interval to km/h.
        # Flow x = columns (east), flow y = rows; rows increase northward in
        # these south-to-north grids, so +y flow is northward motion.
        fx = float(flow[cell_mask, 0].mean()) * res_lon_km / interval_minutes * 60
        fy = float(flow[cell_mask, 1].mean()) * res_lat_km / interval_minutes * 60
        speed_kmh = math.hypot(fx, fy)
        heading_deg = math.degrees(math.atan2(fx, fy)) % 360

        # Approaching if velocity has a component pointing from cell toward the point.
        to_point = (-dx_km, -dy_km)
        approaching = distance_km > 1e-6 and (fx * to_point[0] + fy * to_point[1]) > 0

        cells.append(
            StormCell(
                center_lat=round(cell_lat, 4),
                center_lon=round(cell_lon, 4),
                area_km2=round(area_km2, 1),
                peak_dbz=float(dbz[cell_mask].max()),
                mean_dbz=float(dbz[cell_mask].mean()),
                distance_km=round(distance_km, 1),
                bearing=_bearing_to_compass(bearing_deg),
                speed_kmh=round(speed_kmh, 1),
                heading=_bearing_to_compass(heading_deg),
                approaching=approaching,
            )
        )

    return sorted(cells, key=lambda c: c.distance_km)
