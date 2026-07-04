"""Score past predictions against what actually happened.

Joins the `predictions` table to `daily_observations` on target_date. Only
predictions whose target_date has already occurred (and been recorded) can
be scored — everything else is still pending. Results are grouped by
(horizon_days, model_trained_at) so you can see whether a newer model
version is actually doing better than the one before it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sklearn.metrics import accuracy_score, mean_absolute_error

from weather_predictions.storage import fetch_all_daily, fetch_all_predictions, upsert_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_trained_at: str
    horizon_days: int
    n_samples: int
    rain_accuracy: float
    rain_brier: float
    temp_max_mae: float
    temp_min_mae: float


def evaluate() -> tuple[list[EvaluationResult], int]:
    """Returns (scored results, count of still-pending predictions)."""
    predictions = fetch_all_predictions()
    actual_by_date = {d["date"]: d for d in fetch_all_daily()}

    groups: dict[tuple[str, int], list[dict]] = {}
    pending = 0
    for pred in predictions:
        actual = actual_by_date.get(pred["target_date"])
        if actual is None or actual["temp_max_c"] is None:
            pending += 1
            continue
        key = (pred["model_trained_at"], pred["horizon_days"])
        groups.setdefault(key, []).append({**pred, **{f"actual_{k}": v for k, v in actual.items()}})

    evaluated_at = datetime.now(timezone.utc).isoformat()
    results: list[EvaluationResult] = []
    performance_rows = []
    for (model_trained_at, horizon_days), rows in groups.items():
        actual_rain = [r["actual_rain"] for r in rows]
        pred_rain = [r["rain_predicted"] for r in rows]
        pred_proba = [r["rain_probability"] for r in rows]
        actual_tmax = [r["actual_temp_max_c"] for r in rows]
        pred_tmax = [r["temp_max_pred_c"] for r in rows]
        actual_tmin = [r["actual_temp_min_c"] for r in rows]
        pred_tmin = [r["temp_min_pred_c"] for r in rows]

        result = EvaluationResult(
            model_trained_at=model_trained_at,
            horizon_days=horizon_days,
            n_samples=len(rows),
            rain_accuracy=float(accuracy_score(actual_rain, pred_rain)),
            rain_brier=float(sum((a - p) ** 2 for a, p in zip(actual_rain, pred_proba)) / len(rows)),
            temp_max_mae=float(mean_absolute_error(actual_tmax, pred_tmax)),
            temp_min_mae=float(mean_absolute_error(actual_tmin, pred_tmin)),
        )
        results.append(result)
        performance_rows.append(
            {
                "evaluated_at": evaluated_at,
                "model_trained_at": model_trained_at,
                "horizon_days": horizon_days,
                "n_samples": result.n_samples,
                "rain_accuracy": result.rain_accuracy,
                "rain_brier": result.rain_brier,
                "temp_max_mae": result.temp_max_mae,
                "temp_min_mae": result.temp_min_mae,
            }
        )

    if performance_rows:
        upsert_performance(performance_rows)

    return sorted(results, key=lambda r: (r.model_trained_at, r.horizon_days)), pending


if __name__ == "__main__":
    scored, pending = evaluate()
    for r in scored:
        log.info(
            "model=%s h=%dd n=%d rain_acc=%.2f rain_brier=%.3f tmax_mae=%.2f tmin_mae=%.2f",
            r.model_trained_at,
            r.horizon_days,
            r.n_samples,
            r.rain_accuracy,
            r.rain_brier,
            r.temp_max_mae,
            r.temp_min_mae,
        )
    log.info("pending (no actual outcome yet): %d", pending)
