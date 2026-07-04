"""Train next-1/2/3-day rain and temperature models from accumulated daily history.

Trains one rain classifier and two temperature regressors (max, min) per
forecast horizon, all sharing the same feature set. Uses a time-ordered
split (not random) per horizon so evaluation reflects genuine forecasting
skill rather than leaking future days into training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

from weather_predictions.config import FORECAST_HORIZONS, MIN_TRAINING_DAYS, MODEL_PATH, MODELS_DIR
from weather_predictions.features import FEATURE_COLUMNS, build_daily_features, build_training_frame
from weather_predictions.storage import fetch_all_daily

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class NotEnoughDataError(RuntimeError):
    pass


@dataclass
class HorizonResult:
    horizon_days: int
    n_train: int
    n_test: int
    rain_accuracy: float
    rain_baseline_accuracy: float
    rain_roc_auc: float | None
    temp_max_mae: float
    temp_max_baseline_mae: float
    temp_min_mae: float
    temp_min_baseline_mae: float


@dataclass
class TrainResult:
    trained_at: str
    n_days: int
    horizons: list[HorizonResult]


def _train_one_horizon(daily, horizon: int, min_training_days: int) -> tuple[dict, HorizonResult]:
    X, y_rain, y_tmax, y_tmin = build_training_frame(daily, horizon)
    if len(X) < min_training_days:
        raise NotEnoughDataError(
            f"Only {len(X)} labeled day(s) available for the {horizon}-day horizon; need at "
            f"least {min_training_days}. Keep collecting data and try again later."
        )

    n_test = max(1, int(len(X) * 0.2))
    n_train = len(X) - n_test
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]

    rain_clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42, class_weight="balanced"
    )
    rain_clf.fit(X_train, y_rain.iloc[:n_train])
    rain_pred = rain_clf.predict(X_test)
    rain_true = y_rain.iloc[n_train:]
    rain_accuracy = float(accuracy_score(rain_true, rain_pred))
    rain_baseline_accuracy = float(accuracy_score(rain_true, X_test["rain"]))
    rain_roc_auc = None
    if len(set(rain_true)) > 1:
        rain_proba = rain_clf.predict_proba(X_test)[:, 1]
        rain_roc_auc = float(roc_auc_score(rain_true, rain_proba))

    tmax_reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42)
    tmax_reg.fit(X_train, y_tmax.iloc[:n_train])
    tmax_true = y_tmax.iloc[n_train:]
    temp_max_mae = float(mean_absolute_error(tmax_true, tmax_reg.predict(X_test)))
    temp_max_baseline_mae = float(mean_absolute_error(tmax_true, X_test["temp_max_c"]))

    tmin_reg = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42)
    tmin_reg.fit(X_train, y_tmin.iloc[:n_train])
    tmin_true = y_tmin.iloc[n_train:]
    temp_min_mae = float(mean_absolute_error(tmin_true, tmin_reg.predict(X_test)))
    temp_min_baseline_mae = float(mean_absolute_error(tmin_true, X_test["temp_min_c"]))

    models = {"rain_clf": rain_clf, "tmax_reg": tmax_reg, "tmin_reg": tmin_reg}
    result = HorizonResult(
        horizon_days=horizon,
        n_train=n_train,
        n_test=n_test,
        rain_accuracy=rain_accuracy,
        rain_baseline_accuracy=rain_baseline_accuracy,
        rain_roc_auc=rain_roc_auc,
        temp_max_mae=temp_max_mae,
        temp_max_baseline_mae=temp_max_baseline_mae,
        temp_min_mae=temp_min_mae,
        temp_min_baseline_mae=temp_min_baseline_mae,
    )
    return models, result


def train(min_training_days: int = MIN_TRAINING_DAYS) -> TrainResult:
    daily_records = fetch_all_daily()
    daily = build_daily_features(daily_records)

    if len(daily) < min_training_days:
        raise NotEnoughDataError(
            f"Only {len(daily)} day(s) of history; need at least {min_training_days}. "
            "Keep collecting data and try again later."
        )

    trained_at = datetime.now(timezone.utc).isoformat()
    horizon_models: dict[int, dict] = {}
    horizon_results: list[HorizonResult] = []
    for horizon in FORECAST_HORIZONS:
        models, result = _train_one_horizon(daily, horizon, min_training_days)
        horizon_models[horizon] = models
        horizon_results.append(result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "trained_at": trained_at,
            "feature_columns": FEATURE_COLUMNS,
            "horizons": horizon_models,
        },
        MODEL_PATH,
    )

    result = TrainResult(trained_at=trained_at, n_days=len(daily), horizons=horizon_results)
    log.info("trained model bundle on %d days -> %s", len(daily), MODEL_PATH)
    for hr in horizon_results:
        log.info(
            "h=%dd rain_acc=%.2f (baseline %.2f) tmax_mae=%.2f (baseline %.2f) tmin_mae=%.2f (baseline %.2f)",
            hr.horizon_days,
            hr.rain_accuracy,
            hr.rain_baseline_accuracy,
            hr.temp_max_mae,
            hr.temp_max_baseline_mae,
            hr.temp_min_mae,
            hr.temp_min_baseline_mae,
        )
    return result


if __name__ == "__main__":
    try:
        train()
    except NotEnoughDataError as e:
        log.warning(str(e))
