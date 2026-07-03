"""Train a next-day rain/no-rain classifier from accumulated observations.

Uses a time-ordered split (not random) so evaluation reflects genuine
forecasting skill rather than leaking future days into training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from weather_predictions.config import MIN_TRAINING_DAYS, MODEL_PATH, MODELS_DIR
from weather_predictions.features import (
    FEATURE_COLUMNS,
    build_daily_features,
    build_training_frame,
    raw_to_frame,
)
from weather_predictions.storage import fetch_all_observations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class NotEnoughDataError(RuntimeError):
    pass


@dataclass
class TrainResult:
    trained_at: str
    n_days: int
    n_train: int
    n_test: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    roc_auc: float | None
    baseline_accuracy: float | None
    feature_columns: list[str]


def _evaluate(y_true, y_pred, y_proba) -> dict:
    if len(set(y_true)) < 2:
        # Can't compute most metrics meaningfully with only one class present.
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": None,
            "recall": None,
            "roc_auc": None,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train(min_training_days: int = MIN_TRAINING_DAYS) -> TrainResult:
    raw_records = fetch_all_observations()
    raw_df = raw_to_frame(raw_records)
    daily = build_daily_features(raw_df)

    if len(daily) < min_training_days:
        raise NotEnoughDataError(
            f"Only {len(daily)} day(s) of aggregated history; need at least "
            f"{min_training_days}. Keep the fetch job running and try again later."
        )

    X, y = build_training_frame(daily)
    if len(X) < min_training_days:
        raise NotEnoughDataError(
            f"Only {len(X)} labeled day(s) after dropping incomplete rows; need at least "
            f"{min_training_days}. Keep the fetch job running and try again later."
        )

    # Time-based split: last 20% (min 1, at least a couple days if possible) held out.
    n_test = max(1, int(len(X) * 0.2))
    n_train = len(X) - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = _evaluate(y_test, y_pred, y_proba)

    # Persistence baseline: "tomorrow will look like today".
    baseline_pred = X_test["rain_today"]
    baseline_accuracy = float(accuracy_score(y_test, baseline_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)

    result = TrainResult(
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_days=len(daily),
        n_train=n_train,
        n_test=n_test,
        baseline_accuracy=baseline_accuracy,
        feature_columns=FEATURE_COLUMNS,
        **metrics,
    )
    log.info("trained model on %d days -> %s", len(daily), MODEL_PATH)
    log.info("metrics: %s", asdict(result))
    return result


if __name__ == "__main__":
    try:
        train()
    except NotEnoughDataError as e:
        log.warning(str(e))
