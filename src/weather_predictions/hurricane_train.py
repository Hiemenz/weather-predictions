"""Train hurricane track/intensity forecasting models from HURDAT2 best-track history.

This is a statistical model — current position/motion vector/intensity/
climatology predicting position and wind at t+12/24/48/72h — the same
class of model NHC's own historical baselines (CLIPER/SHIFOR) use, not a
reanalysis-scale deep model this project has no path to training. One
RandomForestRegressor per horizon for track (lat/lon, multi-output) and one
for wind, evaluated against a straight-line-motion baseline (persist the
current bearing/speed) for track and persistence for wind — same "does it
beat naive" framing as the tabular model.

Split by storm season, not a random row split: holding out the most recent
`test_years` seasons keeps the test set genuinely forward-looking and
avoids leaking one storm's other fixes across train/test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from weather_predictions.config import HURRICANE_MODEL_PATH, MODELS_DIR
from weather_predictions.geo import destination_point, haversine_km
from weather_predictions.hurricane_features import (
    FEATURE_COLUMNS,
    FORECAST_HORIZONS_HOURS,
    build_storm_features,
    build_training_frame,
)
from weather_predictions.storage import fetch_all_hurricane_fixes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 50
DEFAULT_TEST_YEARS = 5


class NotEnoughDataError(RuntimeError):
    pass


@dataclass
class HorizonResult:
    horizon_hours: int
    n_train: int
    n_test: int
    track_error_km: float
    track_baseline_error_km: float
    wind_mae_kt: float
    wind_baseline_mae_kt: float


@dataclass
class TrainResult:
    trained_at: str
    n_fixes: int
    horizons: list[HorizonResult]


def _bearing_from_sincos(sin_b: np.ndarray, cos_b: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(sin_b, cos_b)) + 360) % 360


def _train_one_horizon(df: pd.DataFrame, horizon: int, min_samples: int, test_years: int) -> tuple[dict, HorizonResult]:
    X, y_lat, y_lon, y_wind, timestamps = build_training_frame(df, horizon)
    if len(X) < min_samples:
        raise NotEnoughDataError(
            f"Only {len(X)} labeled fix(es) for the {horizon}h horizon; need at least {min_samples}. "
            "Keep backfilling/waiting for more storms and try again later."
        )

    order = timestamps.argsort().to_numpy()
    X = X.iloc[order].reset_index(drop=True)
    y_lat = y_lat.iloc[order].reset_index(drop=True)
    y_lon = y_lon.iloc[order].reset_index(drop=True)
    y_wind = y_wind.iloc[order].reset_index(drop=True)
    years = timestamps.iloc[order].dt.year.reset_index(drop=True)

    cutoff_year = years.max() - test_years + 1
    train_mask = (years < cutoff_year).to_numpy()
    test_mask = ~train_mask
    if train_mask.sum() < min_samples or test_mask.sum() < 1:
        raise NotEnoughDataError(
            f"Not enough storms outside the most recent {test_years} season(s) to both train and "
            f"test the {horizon}h horizon."
        )

    X_train, X_test = X[train_mask], X[test_mask]
    y_track_train = pd.concat([y_lat[train_mask], y_lon[train_mask]], axis=1)

    track_reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
    track_reg.fit(X_train, y_track_train)
    track_pred = track_reg.predict(X_test)

    wind_reg = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)
    wind_reg.fit(X_train, y_wind[train_mask])
    wind_pred = wind_reg.predict(X_test)

    true_lat, true_lon = y_lat[test_mask].to_numpy(), y_lon[test_mask].to_numpy()
    track_error_km = float(np.mean(haversine_km(track_pred[:, 0], track_pred[:, 1], true_lat, true_lon)))

    bearing = _bearing_from_sincos(X_test["motion_bearing_sin"].to_numpy(), X_test["motion_bearing_cos"].to_numpy())
    baseline_dist_km = X_test["motion_speed_kmh"].to_numpy() * horizon
    baseline_lat, baseline_lon = destination_point(
        X_test["lat"].to_numpy(), X_test["lon"].to_numpy(), bearing, baseline_dist_km
    )
    track_baseline_error_km = float(np.mean(haversine_km(baseline_lat, baseline_lon, true_lat, true_lon)))

    true_wind = y_wind[test_mask].to_numpy()
    wind_mae_kt = float(mean_absolute_error(true_wind, wind_pred))
    wind_baseline_mae_kt = float(mean_absolute_error(true_wind, X_test["max_wind_kt"].to_numpy()))

    models = {"track_reg": track_reg, "wind_reg": wind_reg}
    result = HorizonResult(
        horizon_hours=horizon,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        track_error_km=track_error_km,
        track_baseline_error_km=track_baseline_error_km,
        wind_mae_kt=wind_mae_kt,
        wind_baseline_mae_kt=wind_baseline_mae_kt,
    )
    return models, result


def train(min_samples: int = MIN_TRAINING_SAMPLES, test_years: int = DEFAULT_TEST_YEARS) -> TrainResult:
    fixes = fetch_all_hurricane_fixes()
    if not fixes:
        raise NotEnoughDataError("No hurricane fixes stored — run `weather hurricane-backfill` first.")

    df = build_storm_features(fixes)
    trained_at = datetime.now(timezone.utc).isoformat()

    horizon_models: dict[int, dict] = {}
    horizon_results: list[HorizonResult] = []
    for horizon in FORECAST_HORIZONS_HOURS:
        models, result = _train_one_horizon(df, horizon, min_samples, test_years)
        horizon_models[horizon] = models
        horizon_results.append(result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"trained_at": trained_at, "feature_columns": FEATURE_COLUMNS, "horizons": horizon_models},
        HURRICANE_MODEL_PATH,
    )

    result = TrainResult(trained_at=trained_at, n_fixes=len(df), horizons=horizon_results)
    log.info("trained hurricane model bundle on %d fixes -> %s", len(df), HURRICANE_MODEL_PATH)
    for hr in horizon_results:
        log.info(
            "h=%dh track_err=%.0fkm (baseline %.0fkm) wind_mae=%.1fkt (baseline %.1fkt)",
            hr.horizon_hours,
            hr.track_error_km,
            hr.track_baseline_error_km,
            hr.wind_mae_kt,
            hr.wind_baseline_mae_kt,
        )
    return result


if __name__ == "__main__":
    try:
        train()
    except NotEnoughDataError as e:
        log.warning(str(e))
