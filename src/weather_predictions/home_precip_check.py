"""Check whether the radar nowcast (radar_nowcast.py) shows precipitation
reaching a specific point — e.g. LATITUDE/LONGITUDE from config.py — within
its lead time.

Best-effort geolocation, not survey-grade: the saved grid is a flat
Cartesian projection centered on the radar site (Py-ART's grid_from_radars
convention — x increases eastward, y increases northward, row/col 0 is the
grid's south/west edge), and the target point's lat/lon is converted to a
pixel offset via a flat-earth (equirectangular) approximation. That's fine
at the ~100-200km scale and 1km resolution these grids use, but this is
explicitly an experimental supplement to NWS's own alerts, not a
survey-grade tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from weather_predictions.config import LATITUDE, LONGITUDE
from weather_predictions.radar_nowcast import GRID_DIR, load_recent_frames, optical_flow_forecast
from weather_predictions.radar_nowcast_evaluate import RAIN_THRESHOLD_DBZ

_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_AT_EQUATOR = 111.320


class OutOfRadarRangeError(RuntimeError):
    pass


@dataclass
class HomeCheckResult:
    valid_at: str
    lead_minutes: float
    reflectivity_dbz: float
    rain_expected: bool
    threshold_dbz: float = RAIN_THRESHOLD_DBZ


def _km_offset(origin_lat: float, origin_lon: float, lat: float, lon: float) -> tuple[float, float]:
    """(east-west km, north-south km) offset from the radar's origin to (lat, lon)."""
    dy_km = (lat - origin_lat) * _KM_PER_DEG_LAT
    dx_km = (lon - origin_lon) * _KM_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(origin_lat))
    return dx_km, dy_km


def reflectivity_at_point(
    dbz: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    grid_km: float,
    point_lat: float,
    point_lon: float,
) -> float:
    """Sample the grid's dBZ value nearest (point_lat, point_lon)."""
    dx_km, dy_km = _km_offset(origin_lat, origin_lon, point_lat, point_lon)
    if abs(dx_km) > grid_km + 1e-6 or abs(dy_km) > grid_km + 1e-6:
        raise OutOfRadarRangeError(
            f"({point_lat}, {point_lon}) is outside the radar's {grid_km:.0f}km grid "
            f"(offset: {dx_km:.0f}km E/W, {dy_km:.0f}km N/S)."
        )

    n = dbz.shape[0]
    col = int(round((dx_km + grid_km) / (2 * grid_km) * (n - 1)))
    row = int(round((dy_km + grid_km) / (2 * grid_km) * (n - 1)))
    col = min(max(col, 0), n - 1)
    row = min(max(row, 0), n - 1)
    return float(dbz[row, col])


def check_home(
    lead_minutes: float = 30.0,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    grid_dir: Path = GRID_DIR,
) -> HomeCheckResult:
    """Run a fresh optical-flow nowcast and check whether it shows rain
    reaching (lat, lon) within `lead_minutes`. Raises InsufficientFramesError
    (from radar_nowcast) if fewer than 2 decoded frames exist yet, and
    OutOfRadarRangeError if the point falls outside the radar's grid."""
    prev_frame, curr_frame = load_recent_frames(2, grid_dir)
    forecast_dbz = optical_flow_forecast(prev_frame, curr_frame, lead_minutes)

    value = reflectivity_at_point(
        forecast_dbz, curr_frame["latitude"], curr_frame["longitude"], curr_frame["grid_km"], lat, lon
    )
    valid_at = datetime.fromisoformat(curr_frame["timestamp"]) + timedelta(minutes=lead_minutes)
    return HomeCheckResult(
        valid_at=valid_at.isoformat(),
        lead_minutes=lead_minutes,
        reflectivity_dbz=value,
        rain_expected=value >= RAIN_THRESHOLD_DBZ,
    )
