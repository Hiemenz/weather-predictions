"""Score past hurricane forecasts against subsequent HURDAT2 best-track fixes.

HURDAT2 is a finalized historical record that only refreshes roughly once a
year, after the season ends — so a forecast made for an active storm today
will typically show up as "pending" for a long time before it's scorable.
Same honest framing as radar_nowcast_evaluate.py waiting on future decoded
frames: the pending count is expected, not a bug.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from weather_predictions.geo import haversine_km
from weather_predictions.storage import (
    fetch_all_hurricane_fixes,
    fetch_all_hurricane_predictions,
    upsert_hurricane_performance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# HURDAT2's normal cadence is 6-hourly; half of that is a safe match window.
MATCH_TOLERANCE_HOURS = 1.5


@dataclass
class EvaluationResult:
    model_trained_at: str
    horizon_hours: int
    n_samples: int
    track_error_km: float
    wind_mae_kt: float


def _index_fixes_by_storm(fixes: list[dict]) -> dict[str, list[dict]]:
    by_storm: dict[str, list[dict]] = {}
    for fx in fixes:
        by_storm.setdefault(fx["storm_id"], []).append(fx)
    for storm_fixes in by_storm.values():
        storm_fixes.sort(key=lambda f: f["timestamp"])
    return by_storm


def _find_actual_fix(storm_fixes: list[dict], valid_at: datetime, tolerance_hours: float) -> dict | None:
    best, best_delta = None, tolerance_hours * 3600
    for fx in storm_fixes:
        delta = abs((datetime.fromisoformat(fx["timestamp"]) - valid_at).total_seconds())
        if delta <= best_delta:
            best, best_delta = fx, delta
    return best


def evaluate(tolerance_hours: float = MATCH_TOLERANCE_HOURS) -> tuple[list[EvaluationResult], int]:
    """Returns (scored results grouped by model/horizon, count still pending)."""
    predictions = fetch_all_hurricane_predictions()
    fixes_by_storm = _index_fixes_by_storm(fetch_all_hurricane_fixes())

    groups: dict[tuple[str, int], list[tuple[float, float]]] = {}
    pending = 0
    for pred in predictions:
        storm_fixes = fixes_by_storm.get(pred["storm_id"], [])
        actual = _find_actual_fix(storm_fixes, datetime.fromisoformat(pred["valid_at"]), tolerance_hours)
        if actual is None:
            pending += 1
            continue

        track_err = float(haversine_km(pred["lat_pred"], pred["lon_pred"], actual["lat"], actual["lon"]))
        wind_err = abs(pred["wind_pred_kt"] - actual["max_wind_kt"])
        key = (pred["model_trained_at"], pred["horizon_hours"])
        groups.setdefault(key, []).append((track_err, wind_err))

    evaluated_at = datetime.now(timezone.utc).isoformat()
    results: list[EvaluationResult] = []
    performance_rows = []
    for (model_trained_at, horizon_hours), scores in groups.items():
        track_errs, wind_errs = zip(*scores)
        result = EvaluationResult(
            model_trained_at=model_trained_at,
            horizon_hours=horizon_hours,
            n_samples=len(scores),
            track_error_km=float(sum(track_errs) / len(track_errs)),
            wind_mae_kt=float(sum(wind_errs) / len(wind_errs)),
        )
        results.append(result)
        performance_rows.append(
            {
                "evaluated_at": evaluated_at,
                "model_trained_at": result.model_trained_at,
                "horizon_hours": result.horizon_hours,
                "n_samples": result.n_samples,
                "track_error_km": result.track_error_km,
                "wind_mae_kt": result.wind_mae_kt,
            }
        )

    if performance_rows:
        upsert_hurricane_performance(performance_rows)

    return sorted(results, key=lambda r: (r.horizon_hours, r.model_trained_at)), pending


if __name__ == "__main__":
    scored, pending = evaluate()
    for r in scored:
        log.info(
            "model=%s h=%dh n=%d track_err=%.0fkm wind_mae=%.1fkt",
            r.model_trained_at,
            r.horizon_hours,
            r.n_samples,
            r.track_error_km,
            r.wind_mae_kt,
        )
    log.info("pending (no actual fix yet): %d", pending)
