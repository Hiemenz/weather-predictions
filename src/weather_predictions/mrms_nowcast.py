"""Short-term nowcasting from MRMS national radar frames.

Same optical-flow approach as radar_nowcast.py but uses the MRMS CONUS
composite instead of a single NEXRAD station — so any point in the country
can be forecast, not just the ±200km around Nashville's KOHX site.

Workflow:
  1. Collect 2+ MRMS frames with `weather mrms-fetch` (repeat every few minutes).
  2. Run `weather mrms-nowcast` to produce a predicted reflectivity grid for
     the configured lat/lon ±radius_km, N minutes ahead.

Needs `poetry install --with display` for OpenCV (optical flow).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from weather_predictions.config import LATITUDE, LONGITUDE, MRMS_DATA_DIR
from weather_predictions.mrms_processing import OutOfMrmsRangeError, crop_to_region, load_mrms_grid
from weather_predictions.radar_nowcast import (
    InsufficientFramesError,
    METHOD_OPTICAL_FLOW,
    METHOD_PERSISTENCE,
    estimate_motion_field,
    optical_flow_forecast,
    persistence_forecast,
)
from weather_predictions.storage import upsert_radar_nowcasts

# `station` value used in the shared radar_nowcasts table to distinguish MRMS
# nowcasts from single-NEXRAD-station ones (e.g. "KOHX").
MRMS_STATION = "MRMS_CONUS"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GRID_DIR = MRMS_DATA_DIR / "grids"
NOWCAST_DIR = MRMS_DATA_DIR / "nowcasts"


@dataclass
class MrmsNowcastResult:
    predicted_at: str
    valid_at: str
    lead_minutes: float
    center_lat: float
    center_lon: float
    radius_km: float
    grid_paths: dict[str, str]


def load_recent_mrms_frames(n: int = 2, grid_dir: Path = GRID_DIR) -> list[dict[str, Any]]:
    """Load the `n` most recent decoded MRMS frames, oldest first."""
    files = sorted(grid_dir.glob("MRMS_CONUS_*.npz"))
    if len(files) < n:
        raise InsufficientFramesError(
            f"Need at least {n} MRMS frame(s) in {grid_dir}, found {len(files)}. "
            "Run `weather mrms-fetch` to collect more."
        )
    return [load_mrms_grid(f) for f in files[-n:]]


def _save_nowcast_grid(
    dbz: np.ndarray, valid_at: datetime, method: str, dest_dir: Path, crop: dict[str, Any]
) -> Path:
    """Save a forecast grid along with the lat/lon bounds of the crop it was
    made from, so the evaluator can cut the actual national frame to the same
    region when scoring."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts_compact = valid_at.isoformat().replace(":", "").replace("-", "")
    dest_path = dest_dir / f"MRMS_CONUS_{ts_compact}_{method}.npz"
    np.savez_compressed(
        dest_path,
        reflectivity_dbz=dbz.astype(np.float32),
        lat_min=crop["lat_min"],
        lat_max=crop["lat_max"],
        lon_min=crop["lon_min"],
        lon_max=crop["lon_max"],
    )
    return dest_path


def nowcast(
    lead_minutes: float = 30.0,
    center_lat: float = LATITUDE,
    center_lon: float = LONGITUDE,
    radius_km: float = 300.0,
    grid_dir: Path = GRID_DIR,
    dest_dir: Path = NOWCAST_DIR,
) -> MrmsNowcastResult:
    """Forecast MRMS reflectivity `lead_minutes` ahead for the region around
    (center_lat, center_lon) ± radius_km, using optical flow + persistence."""
    prev_frame, curr_frame = load_recent_mrms_frames(2, grid_dir)
    prev_crop = crop_to_region(prev_frame, center_lat, center_lon, radius_km)
    curr_crop = crop_to_region(curr_frame, center_lat, center_lon, radius_km)

    curr_ts = datetime.fromisoformat(curr_frame["timestamp"])
    valid_at = curr_ts + timedelta(minutes=lead_minutes)

    forecasts = {
        METHOD_PERSISTENCE: persistence_forecast(curr_crop),
        METHOD_OPTICAL_FLOW: optical_flow_forecast(prev_crop, curr_crop, lead_minutes),
    }

    grid_paths: dict[str, str] = {}
    rows = []
    for method, dbz in forecasts.items():
        saved_path = _save_nowcast_grid(dbz, valid_at, method, dest_dir, curr_crop)
        grid_paths[method] = str(saved_path)
        rows.append(
            {
                "predicted_at": curr_frame["timestamp"],
                "valid_at": valid_at.isoformat(),
                "lead_minutes": lead_minutes,
                "method": method,
                "station": MRMS_STATION,
                "grid_path": str(saved_path),
            }
        )
        log.info("[%s] MRMS nowcast for %s -> %s", method, valid_at.isoformat(), saved_path)

    upsert_radar_nowcasts(rows)
    return MrmsNowcastResult(
        predicted_at=curr_frame["timestamp"],
        valid_at=valid_at.isoformat(),
        lead_minutes=lead_minutes,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        grid_paths=grid_paths,
    )
