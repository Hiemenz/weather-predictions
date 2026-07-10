"""Forecast currently active tropical cyclones using the trained hurricane model.

Needs `weather hurricane-train` to have been run at least once. Uses NHC's
live feed (hurricane_client.get_active_storms) for the "as of now" snapshot
— that feed already reports current movement direction/speed directly, so
no historical fixes are needed to build a live feature row, unlike
backtesting against HURDAT2 where the motion vector comes from consecutive
fixes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from weather_predictions.config import HURRICANE_MODEL_PATH
from weather_predictions.hurricane_client import get_active_storms
from weather_predictions.hurricane_features import FORECAST_HORIZONS_HOURS
from weather_predictions.storage import upsert_hurricane_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_MPH_TO_KMH = 1.60934


class ModelNotTrainedError(RuntimeError):
    pass


@dataclass
class HorizonForecast:
    horizon_hours: int
    valid_at: str
    lat_pred: float
    lon_pred: float
    wind_pred_kt: float


@dataclass
class StormForecast:
    storm_id: str
    name: str
    as_of: str
    classification: str | None
    horizons: list[HorizonForecast] = field(default_factory=list)


def _feature_row(storm: dict) -> pd.DataFrame:
    """One live snapshot -> one feature row. `storm_age_hours` is unknown
    from a single snapshot (no fix history), so it's set to 0 — climatology
    (doy_sin/cos) and current position/motion/intensity carry the signal."""
    bearing = storm.get("movement_dir_deg") or 0.0
    speed_kmh = (storm.get("movement_speed_mph") or 0.0) * _MPH_TO_KMH
    timestamp = datetime.fromisoformat(storm["last_update"].replace("Z", "+00:00"))
    doy = timestamp.timetuple().tm_yday

    return pd.DataFrame(
        [
            {
                "lat": storm["lat"],
                "lon": storm["lon"],
                "max_wind_kt": storm.get("intensity_kt") or 0.0,
                "min_pressure_mb": storm.get("pressure_mb") or 1010.0,
                "motion_bearing_sin": np.sin(np.radians(bearing)),
                "motion_bearing_cos": np.cos(np.radians(bearing)),
                "motion_speed_kmh": speed_kmh,
                "storm_age_hours": 0.0,
                "doy_sin": np.sin(2 * np.pi * doy / 365.25),
                "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            }
        ]
    )


def predict() -> list[StormForecast]:
    if not HURRICANE_MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained hurricane model at {HURRICANE_MODEL_PATH}. Run `weather hurricane-train` first."
        )

    bundle = joblib.load(HURRICANE_MODEL_PATH)
    feature_columns = bundle["feature_columns"]

    forecasts: list[StormForecast] = []
    prediction_rows = []
    for storm in get_active_storms():
        if storm["lat"] is None or storm["lon"] is None or not storm["last_update"]:
            log.warning("skipping storm %s: missing position or timestamp in live feed", storm.get("id"))
            continue

        X = _feature_row(storm)[feature_columns]
        as_of = storm["last_update"]
        forecast = StormForecast(
            storm_id=storm["id"], name=storm["name"], as_of=as_of, classification=storm["classification"]
        )

        for horizon in FORECAST_HORIZONS_HOURS:
            models = bundle["horizons"][horizon]
            lat_pred, lon_pred = models["track_reg"].predict(X)[0]
            wind_pred = float(models["wind_reg"].predict(X)[0])
            as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            valid_at = as_of_dt + timedelta(hours=horizon)

            forecast.horizons.append(
                HorizonForecast(horizon, valid_at.isoformat(), float(lat_pred), float(lon_pred), wind_pred)
            )
            prediction_rows.append(
                {
                    "predicted_at": as_of,
                    "storm_id": storm["id"],
                    "storm_name": storm["name"],
                    "horizon_hours": horizon,
                    "valid_at": valid_at.isoformat(),
                    "lat_pred": float(lat_pred),
                    "lon_pred": float(lon_pred),
                    "wind_pred_kt": wind_pred,
                    "model_trained_at": bundle["trained_at"],
                }
            )
        forecasts.append(forecast)

    if prediction_rows:
        upsert_hurricane_predictions(prediction_rows)
    return forecasts


if __name__ == "__main__":
    results = predict()
    if not results:
        log.info("no active tropical cyclones right now")
    for f in results:
        log.info("%s (%s) as of %s:", f.name, f.storm_id, f.as_of)
        for hp in f.horizons:
            log.info("  t+%dh (%s): lat=%.1f lon=%.1f wind=%.0fkt", hp.horizon_hours, hp.valid_at, hp.lat_pred, hp.lon_pred, hp.wind_pred_kt)
