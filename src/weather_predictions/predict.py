"""Predict rain probability and high/low temperature for the next 1-3 days.

Predictions are stored (see storage.upsert_predictions) so `evaluate.py` can
later compare them against what actually happened. Also surfaces the
official NWS forecast for the same days as a point of comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import joblib

from weather_predictions.config import FORECAST_HORIZONS, LATITUDE, LONGITUDE, MODEL_PATH
from weather_predictions.features import build_daily_features, latest_feature_row
from weather_predictions.nws_client import NWSClientError, get_forecast, get_point_metadata
from weather_predictions.storage import fetch_all_daily, upsert_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


class NoUsableDataError(RuntimeError):
    pass


@dataclass
class HorizonPrediction:
    horizon_days: int
    target_date: str
    rain_probability: float
    rain_predicted: bool
    temp_max_pred_c: float
    temp_min_pred_c: float


@dataclass
class PredictionSet:
    as_of_date: str
    model_trained_at: str
    horizons: list[HorizonPrediction] = field(default_factory=list)
    nws_daytime_forecasts: list[str] = field(default_factory=list)


def _nws_upcoming_forecasts(n: int = 3) -> list[str]:
    try:
        point = get_point_metadata(LATITUDE, LONGITUDE)
        forecast = get_forecast(point["properties"]["forecast"])
        periods = [p for p in forecast["properties"]["periods"] if p.get("isDaytime")]
        return [f"{p['name']}: {p['detailedForecast']}" for p in periods[:n]]
    except (NWSClientError, KeyError, IndexError) as e:
        log.warning("could not fetch NWS comparison forecast: %s", e)
        return []


def predict() -> PredictionSet:
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `weather train` once enough "
            "history has accumulated (see `weather status`)."
        )

    bundle = joblib.load(MODEL_PATH)
    feature_columns = bundle["feature_columns"]

    daily = build_daily_features(fetch_all_daily())
    row = latest_feature_row(daily)
    if row is None:
        raise NoUsableDataError("No complete day of features available yet to predict from.")

    as_of_date = row["date"].iloc[0]
    X = row[feature_columns]

    result = PredictionSet(
        as_of_date=str(as_of_date.date()),
        model_trained_at=bundle["trained_at"],
        nws_daytime_forecasts=_nws_upcoming_forecasts(len(FORECAST_HORIZONS)),
    )

    prediction_rows = []
    for horizon in FORECAST_HORIZONS:
        models = bundle["horizons"][horizon]
        rain_proba = float(models["rain_clf"].predict_proba(X)[0, 1])
        temp_max = float(models["tmax_reg"].predict(X)[0])
        temp_min = float(models["tmin_reg"].predict(X)[0])
        target_date = as_of_date + timedelta(days=horizon)

        result.horizons.append(
            HorizonPrediction(
                horizon_days=horizon,
                target_date=str(target_date.date()),
                rain_probability=rain_proba,
                rain_predicted=rain_proba >= 0.5,
                temp_max_pred_c=temp_max,
                temp_min_pred_c=temp_min,
            )
        )
        prediction_rows.append(
            {
                "predicted_date": str(as_of_date.date()),
                "horizon_days": horizon,
                "target_date": str(target_date.date()),
                "rain_probability": rain_proba,
                "rain_predicted": int(rain_proba >= 0.5),
                "temp_max_pred_c": temp_max,
                "temp_min_pred_c": temp_min,
                "model_trained_at": bundle["trained_at"],
            }
        )

    upsert_predictions(prediction_rows)
    return result


if __name__ == "__main__":
    for hp in predict().horizons:
        log.info("t+%d (%s): %s", hp.horizon_days, hp.target_date, hp)
