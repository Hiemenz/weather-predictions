"""Score past MRMS nowcasts against the real national frame that arrived.

Same framing as radar_nowcast_evaluate.py, with two MRMS-specific twists:
  - Stored MRMS nowcasts are regional crops, so the actual national frame is
    cut down to the crop's stored lat/lon bounds before comparing.
  - MRMS updates every ~2 minutes (vs ~5 for a NEXRAD volume scan), so the
    match tolerance is tighter.

Scores land in the same radar_nowcast_performance table; the method names
carry an "mrms_" prefix so they don't mix with single-station results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from weather_predictions.mrms_nowcast import GRID_DIR, MRMS_STATION
from weather_predictions.mrms_processing import OutOfMrmsRangeError, crop_to_bounds, load_mrms_grid
from weather_predictions.radar_nowcast import NO_ECHO_DBZ
from weather_predictions.radar_nowcast_evaluate import EvaluationResult, _score
from weather_predictions.storage import fetch_all_radar_nowcasts, upsert_radar_nowcast_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# MRMS produces a frame every ~2 minutes; half that cadence.
DEFAULT_TOLERANCE_SECONDS = 60.0


def _find_actual_frame(valid_at: datetime, grid_dir: Path, tolerance_seconds: float) -> Path | None:
    best_path, best_delta = None, tolerance_seconds
    for path in grid_dir.glob("MRMS_CONUS_*.npz"):
        try:
            ts = datetime.fromisoformat(load_mrms_grid(path)["timestamp"])
        except Exception:
            continue
        delta = abs((ts - valid_at).total_seconds())
        if delta <= best_delta:
            best_path, best_delta = path, delta
    return best_path


def evaluate(
    grid_dir: Path = GRID_DIR, tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS
) -> tuple[list[EvaluationResult], int]:
    """Returns (scored results grouped by method/lead_minutes, count still pending)."""
    nowcasts = [r for r in fetch_all_radar_nowcasts() if r["station"] == MRMS_STATION]

    groups: dict[tuple[str, float], list[tuple[float, float]]] = {}
    pending = 0
    for row in nowcasts:
        valid_at = datetime.fromisoformat(row["valid_at"])
        actual_path = _find_actual_frame(valid_at, grid_dir, tolerance_seconds)
        if actual_path is None:
            pending += 1
            continue

        with np.load(row["grid_path"]) as forecast_data:
            predicted = forecast_data["reflectivity_dbz"]
            bounds = {k: float(forecast_data[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")}

        actual_frame = load_mrms_grid(actual_path)
        try:
            actual_crop = crop_to_bounds(actual_frame, **bounds)
        except OutOfMrmsRangeError as e:
            log.warning("skipping nowcast %s: %s", row["grid_path"], e)
            continue

        actual = np.nan_to_num(actual_crop["reflectivity_dbz"], nan=NO_ECHO_DBZ)
        if actual.shape != predicted.shape:
            log.warning(
                "skipping nowcast %s: shape mismatch predicted %s vs actual %s",
                row["grid_path"], predicted.shape, actual.shape,
            )
            continue

        mae, csi = _score(predicted, actual)
        key = (f"mrms_{row['method']}", row["lead_minutes"])
        groups.setdefault(key, []).append((mae, csi))

    evaluated_at = datetime.now(timezone.utc).isoformat()
    results: list[EvaluationResult] = []
    performance_rows = []
    for (method, lead_minutes), scores in groups.items():
        maes, csis = zip(*scores)
        result = EvaluationResult(
            method=method,
            lead_minutes=lead_minutes,
            n_samples=len(scores),
            mae_dbz=float(np.mean(maes)),
            csi=float(np.mean(csis)),
        )
        results.append(result)
        performance_rows.append(
            {
                "evaluated_at": evaluated_at,
                "method": result.method,
                "lead_minutes": result.lead_minutes,
                "n_samples": result.n_samples,
                "mae_dbz": result.mae_dbz,
                "csi": result.csi,
                "csi_threshold_dbz": result.csi_threshold_dbz,
            }
        )

    if performance_rows:
        upsert_radar_nowcast_performance(performance_rows)

    return sorted(results, key=lambda r: (r.lead_minutes, r.method)), pending
