"""Check whether MRMS nowcasts show precipitation reaching a point, and when.

MRMS counterpart to home_precip_check.py, with one upgrade: instead of a
single yes/no at one lead time, `estimate_arrival` advects the field at a
ladder of lead times and reports the first one where reflectivity at the
target point crosses the rain threshold — i.e. "rain arrives in ~20 min",
not just "rain within 30 min: yes".

Because MRMS covers the whole CONUS, the out-of-range failure mode of the
single-station check only triggers for points outside the continental US.

Needs `poetry install --with display` for OpenCV (optical flow).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from weather_predictions.config import LATITUDE, LONGITUDE
from weather_predictions.mrms_nowcast import GRID_DIR, load_recent_mrms_frames
from weather_predictions.mrms_processing import crop_to_region
from weather_predictions.radar_nowcast import optical_flow_forecast
from weather_predictions.radar_nowcast_evaluate import RAIN_THRESHOLD_DBZ

# Lead times (minutes) probed when estimating arrival, nearest first.
ARRIVAL_LEADS_MINUTES = (10.0, 20.0, 30.0, 45.0, 60.0)

# How far around the point to crop before running optical flow. Weather moving
# at ~100 km/h covers 100 km in the 60-minute max lead, so 300 km of context
# comfortably contains everything that could arrive within the ladder.
DEFAULT_RADIUS_KM = 300.0


@dataclass
class MrmsArrivalResult:
    as_of: str  # timestamp of the frame the forecast was made from
    rain_now: bool  # already raining at the point in the current frame
    arrival_lead_minutes: float | None  # first probed lead with rain, None if none
    arrival_at: str | None  # wall-clock time of that lead
    reflectivity_by_lead: dict[float, float]  # lead minutes -> forecast dBZ at the point
    threshold_dbz: float = RAIN_THRESHOLD_DBZ


def reflectivity_at_point(crop: dict[str, Any], dbz_array, lat: float, lon: float) -> float:
    """Sample `dbz_array` (same shape/bounds as `crop`) at (lat, lon)."""
    row = (lat - crop["lat_min"]) / (crop["lat_max"] - crop["lat_min"]) * (crop["nlat"] - 1)
    col = (lon - crop["lon_min"]) / (crop["lon_max"] - crop["lon_min"]) * (crop["nlon"] - 1)
    row = min(max(int(round(row)), 0), crop["nlat"] - 1)
    col = min(max(int(round(col)), 0), crop["nlon"] - 1)
    return float(dbz_array[row, col])


def estimate_arrival(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    radius_km: float = DEFAULT_RADIUS_KM,
    leads_minutes: tuple[float, ...] = ARRIVAL_LEADS_MINUTES,
    grid_dir: Path = GRID_DIR,
) -> MrmsArrivalResult:
    """Advect the MRMS field at each lead time and report the first one where
    rain-threshold reflectivity reaches (lat, lon).

    Raises InsufficientFramesError (fewer than 2 MRMS frames collected) or
    OutOfMrmsRangeError (point outside CONUS coverage).
    """
    prev_frame, curr_frame = load_recent_mrms_frames(2, grid_dir)
    prev_crop = crop_to_region(prev_frame, lat, lon, radius_km)
    curr_crop = crop_to_region(curr_frame, lat, lon, radius_km)

    curr_ts = datetime.fromisoformat(curr_frame["timestamp"])
    rain_now = reflectivity_at_point(curr_crop, curr_crop["reflectivity_dbz"], lat, lon) >= RAIN_THRESHOLD_DBZ

    reflectivity_by_lead: dict[float, float] = {}
    arrival_lead: float | None = None
    for lead in leads_minutes:
        forecast_dbz = optical_flow_forecast(prev_crop, curr_crop, lead)
        value = reflectivity_at_point(curr_crop, forecast_dbz, lat, lon)
        reflectivity_by_lead[lead] = value
        if arrival_lead is None and value >= RAIN_THRESHOLD_DBZ:
            arrival_lead = lead

    return MrmsArrivalResult(
        as_of=curr_frame["timestamp"],
        rain_now=rain_now,
        arrival_lead_minutes=arrival_lead,
        arrival_at=(curr_ts + timedelta(minutes=arrival_lead)).isoformat() if arrival_lead else None,
        reflectivity_by_lead=reflectivity_by_lead,
    )
