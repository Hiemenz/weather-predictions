"""Score past radar nowcasts against the real grid that eventually arrived.

Same predict -> evaluate framing as evaluate.py: a nowcast made for some
`valid_at` timestamp is only scorable once a real decoded grid exists close
to that time. Results are grouped by (method, lead_minutes) so the
optical-flow forecast can be compared against the persistence baseline it's
meant to beat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from weather_predictions.radar_nowcast import GRID_DIR, NO_ECHO_DBZ
from weather_predictions.radar_processing import load_grid
from weather_predictions.storage import fetch_all_radar_nowcasts, upsert_radar_nowcast_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# How close an actual grid's timestamp must be to a nowcast's valid_at to
# count as "the outcome for that forecast" — half the ~5-minute scan cadence.
DEFAULT_TOLERANCE_SECONDS = 150.0

# Reflectivity above this is treated as "precipitation present" for CSI.
RAIN_THRESHOLD_DBZ = 20.0


@dataclass
class EvaluationResult:
    method: str
    lead_minutes: float
    n_samples: int
    mae_dbz: float
    csi: float
    csi_threshold_dbz: float = RAIN_THRESHOLD_DBZ


def _find_actual_grid(valid_at: datetime, grid_dir: Path, tolerance_seconds: float) -> Path | None:
    """Find the decoded grid (see radar_processing.save_grid) closest to `valid_at`,
    reading its timestamp from the frame's own stored metadata — a saved grid's
    filename doesn't follow the raw NEXRAD naming convention parse_scan_timestamp expects."""
    best_path, best_delta = None, tolerance_seconds
    for path in grid_dir.glob("*.npz"):
        try:
            ts = datetime.fromisoformat(load_grid(path)["timestamp"])
        except Exception:
            continue
        delta = abs((ts - valid_at).total_seconds())
        if delta <= best_delta:
            best_path, best_delta = path, delta
    return best_path


def _score(predicted_dbz: np.ndarray, actual_dbz: np.ndarray, threshold: float = RAIN_THRESHOLD_DBZ) -> tuple[float, float]:
    mae = float(np.mean(np.abs(predicted_dbz - actual_dbz)))

    pred_echo = predicted_dbz >= threshold
    actual_echo = actual_dbz >= threshold
    tp = int(np.logical_and(pred_echo, actual_echo).sum())
    fp = int(np.logical_and(pred_echo, ~actual_echo).sum())
    fn = int(np.logical_and(~pred_echo, actual_echo).sum())
    csi = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    return mae, csi


def evaluate(
    grid_dir: Path = GRID_DIR, tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS
) -> tuple[list[EvaluationResult], int]:
    """Returns (scored results grouped by method/lead_minutes, count still pending)."""
    nowcasts = fetch_all_radar_nowcasts()

    groups: dict[tuple[str, float], list[tuple[float, float]]] = {}
    pending = 0
    for row in nowcasts:
        valid_at = datetime.fromisoformat(row["valid_at"])
        actual_path = _find_actual_grid(valid_at, grid_dir, tolerance_seconds)
        if actual_path is None:
            pending += 1
            continue

        predicted = np.load(row["grid_path"])["reflectivity_dbz"]
        actual = np.nan_to_num(load_grid(actual_path)["reflectivity_dbz"], nan=NO_ECHO_DBZ)
        mae, csi = _score(predicted, actual)

        key = (row["method"], row["lead_minutes"])
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


if __name__ == "__main__":
    scored, pending = evaluate()
    for r in scored:
        log.info(
            "method=%s lead=%.0fmin n=%d mae=%.2fdBZ csi=%.2f (threshold %.0fdBZ)",
            r.method,
            r.lead_minutes,
            r.n_samples,
            r.mae_dbz,
            r.csi,
            r.csi_threshold_dbz,
        )
    log.info("pending (no actual grid yet): %d", pending)
