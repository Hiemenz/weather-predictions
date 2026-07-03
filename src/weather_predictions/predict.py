"""Predict tomorrow's rain probability from the trained model, using the most
recent day of accumulated observations. Also surfaces the official NWS
forecast for the same location as a point of comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib

from weather_predictions.config import LATITUDE, LONGITUDE, MODEL_PATH, STATION_ID
from weather_predictions.features import build_daily_features, latest_feature_row, raw_to_frame
from weather_predictions.nws_client import NWSClientError, get_forecast, get_point_metadata
from weather_predictions.storage import fetch_all_observations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


class NoUsableDataError(RuntimeError):
    pass


@dataclass
class Prediction:
    as_of_local_date: str
    rain_probability: float
    rain_predicted: bool
    nws_forecast_summary: str | None
    nws_forecast_pop_pct: int | None


def _nws_tomorrow_forecast() -> tuple[str | None, int | None]:
    try:
        point = get_point_metadata(LATITUDE, LONGITUDE)
        forecast = get_forecast(point["properties"]["forecast"])
        periods = forecast["properties"]["periods"]
        # First overnight/daytime period is "today"; take the next one as "tomorrow".
        period = periods[1] if len(periods) > 1 else periods[0]
        pop = period.get("probabilityOfPrecipitation", {}).get("value")
        return period.get("detailedForecast"), pop
    except (NWSClientError, KeyError, IndexError) as e:
        log.warning("could not fetch NWS comparison forecast: %s", e)
        return None, None


def predict(station_id: str = STATION_ID) -> Prediction:
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run training once enough "
            "history has accumulated (see `weather status`)."
        )

    bundle = joblib.load(MODEL_PATH)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    raw_df = raw_to_frame(fetch_all_observations())
    daily = build_daily_features(raw_df)
    row = latest_feature_row(daily)
    if row is None:
        raise NoUsableDataError("No complete day of features available yet to predict from.")

    proba = float(model.predict_proba(row[feature_columns])[0, 1])
    nws_summary, nws_pop = _nws_tomorrow_forecast()

    return Prediction(
        as_of_local_date=str(daily["local_date"].iloc[-1].date()),
        rain_probability=proba,
        rain_predicted=proba >= 0.5,
        nws_forecast_summary=nws_summary,
        nws_forecast_pop_pct=nws_pop,
    )


if __name__ == "__main__":
    result = predict()
    log.info("prediction: %s", result)
