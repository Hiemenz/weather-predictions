"""Short-term precipitation nowcasting from stored reflectivity grids.

Needs the `radar` dependency group (`poetry install --with radar`) — same
constraint as radar.py, plus OpenCV for optical flow.

There's no accumulated time series of frames yet (radar collection just
started), so a data-hungry model like a ConvLSTM would just be overfitting
noise. Optical-flow extrapolation needs only two consecutive frames: it
estimates a motion field between them with dense optical flow, then advects
the most recent frame forward along that field. Same "does it beat naive"
framing as the tabular model — persistence (assume nothing moves) is stored
alongside it as the baseline, scored by radar_nowcast_evaluate.py once real
outcomes are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from weather_predictions.config import RADAR_DATA_DIR, RADAR_STATION_ID
from weather_predictions.radar_processing import load_grid
from weather_predictions.storage import upsert_radar_nowcasts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GRID_DIR = RADAR_DATA_DIR / "grids"
NOWCAST_DIR = RADAR_DATA_DIR / "nowcasts"

# dBZ range the grid is clipped to before converting to an 8-bit image for
# optical flow — chosen to cover NEXRAD's typical reflectivity range without
# wasting most of the 0-255 range on values that never occur.
_DBZ_MIN = -30.0
_DBZ_MAX = 70.0

# Value written where a grid cell has no radar return (was NaN). Below any
# reasonable "is it raining" threshold, so it reads as "clear" everywhere
# downstream (flow estimation, MAE, CSI) without special-casing NaN.
NO_ECHO_DBZ = -32.0

METHOD_PERSISTENCE = "persistence"
METHOD_OPTICAL_FLOW = "optical_flow"


class InsufficientFramesError(RuntimeError):
    pass


@dataclass
class NowcastResult:
    predicted_at: str  # timestamp of the most recent observed frame
    valid_at: str  # timestamp this forecast is for
    lead_minutes: float
    station: str
    grid_paths: dict[str, str]  # method -> saved .npz path


def _list_grid_files(grid_dir: Path) -> list[Path]:
    return sorted(grid_dir.glob("*.npz"))


def load_recent_frames(n: int = 2, grid_dir: Path = GRID_DIR) -> list[dict[str, Any]]:
    """Load the `n` most recent decoded frames, oldest first."""
    files = _list_grid_files(grid_dir)
    if len(files) < n:
        raise InsufficientFramesError(
            f"Need at least {n} decoded radar frame(s) in {grid_dir}, found {len(files)}. "
            "Run `weather radar-fetch` / `radar-decode-pending` to accumulate more."
        )
    return [load_grid(f) for f in files[-n:]]


def _to_image(dbz: np.ndarray) -> np.ndarray:
    """dBZ array (NaN = no echo) -> normalized uint8 image for optical flow."""
    filled = np.nan_to_num(dbz, nan=NO_ECHO_DBZ)
    clipped = np.clip(filled, _DBZ_MIN, _DBZ_MAX)
    return ((clipped - _DBZ_MIN) / (_DBZ_MAX - _DBZ_MIN) * 255).astype(np.uint8)


def _dense_flow(prev_img: np.ndarray, curr_img: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow, prev -> curr, in pixels/frame-interval."""
    return cv2.calcOpticalFlowFarneback(
        prev_img, curr_img, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def _advect(dbz: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    """Shift `dbz` along `flow` scaled by `scale` (e.g. lead time / frame interval).

    Cells advected in from outside the grid have no data, so they're filled
    with NO_ECHO_DBZ rather than fabricating a value.
    """
    h, w = dbz.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[..., 0] * scale
    map_y = grid_y - flow[..., 1] * scale
    filled = np.nan_to_num(dbz, nan=NO_ECHO_DBZ).astype(np.float32)
    return cv2.remap(
        filled, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=NO_ECHO_DBZ
    )


def persistence_forecast(latest_frame: dict[str, Any]) -> np.ndarray:
    """Naive baseline: assume the field doesn't move at all."""
    return np.nan_to_num(latest_frame["reflectivity_dbz"], nan=NO_ECHO_DBZ).astype(np.float32)


def optical_flow_forecast(prev_frame: dict[str, Any], curr_frame: dict[str, Any], lead_minutes: float) -> np.ndarray:
    """Estimate motion between the last two frames and advect the latest one forward."""
    prev_ts = datetime.fromisoformat(prev_frame["timestamp"])
    curr_ts = datetime.fromisoformat(curr_frame["timestamp"])
    interval_minutes = (curr_ts - prev_ts).total_seconds() / 60
    if interval_minutes <= 0:
        raise InsufficientFramesError("Frames must be strictly increasing in time to estimate motion.")

    flow = _dense_flow(_to_image(prev_frame["reflectivity_dbz"]), _to_image(curr_frame["reflectivity_dbz"]))
    return _advect(curr_frame["reflectivity_dbz"], flow, scale=lead_minutes / interval_minutes)


def _save_forecast_grid(dbz: np.ndarray, station: str, valid_at: datetime, method: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts_compact = valid_at.isoformat().replace(":", "").replace("-", "")
    dest_path = dest_dir / f"{station}_{ts_compact}_{method}.npz"
    np.savez_compressed(dest_path, reflectivity_dbz=dbz.astype(np.float32))
    return dest_path


def nowcast(
    lead_minutes: float = 30.0,
    grid_dir: Path = GRID_DIR,
    dest_dir: Path = NOWCAST_DIR,
    station: str = RADAR_STATION_ID,
) -> NowcastResult:
    """Forecast the reflectivity grid `lead_minutes` ahead using both the
    optical-flow and persistence methods, saving both and recording metadata
    so radar_nowcast_evaluate.py can score them once the real outcome shows up."""
    prev_frame, curr_frame = load_recent_frames(2, grid_dir)
    curr_ts = datetime.fromisoformat(curr_frame["timestamp"])
    valid_at = curr_ts + timedelta(minutes=lead_minutes)

    forecasts = {
        METHOD_PERSISTENCE: persistence_forecast(curr_frame),
        METHOD_OPTICAL_FLOW: optical_flow_forecast(prev_frame, curr_frame, lead_minutes),
    }

    grid_paths: dict[str, str] = {}
    rows = []
    for method, dbz in forecasts.items():
        saved_path = _save_forecast_grid(dbz, station, valid_at, method, dest_dir)
        grid_paths[method] = str(saved_path)
        rows.append(
            {
                "predicted_at": curr_frame["timestamp"],
                "valid_at": valid_at.isoformat(),
                "lead_minutes": lead_minutes,
                "method": method,
                "station": station,
                "grid_path": str(saved_path),
            }
        )
        log.info("[%s] forecast for %s -> %s", method, valid_at.isoformat(), saved_path)

    upsert_radar_nowcasts(rows)
    return NowcastResult(
        predicted_at=curr_frame["timestamp"],
        valid_at=valid_at.isoformat(),
        lead_minutes=lead_minutes,
        station=station,
        grid_paths=grid_paths,
    )


if __name__ == "__main__":
    result = nowcast()
    log.info("nowcast valid at %s (t+%.0f min): %s", result.valid_at, result.lead_minutes, result.grid_paths)
