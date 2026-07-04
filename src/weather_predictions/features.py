"""Feature engineering for multi-day-ahead rain and temperature forecasting.

Everything is built from `daily_observations` — one row per calendar date
with temp_max_c, temp_min_c, precip_mm, and a derived `rain` flag. That table
is populated from two sources (see storage.py): CDO/GHCND for historical
bulk, and a live METAR-derived aggregate (see `compute_live_daily_aggregate`)
for the last day or two before CDO catches up.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from weather_predictions.config import FORECAST_HORIZONS, RAIN_THRESHOLD_MM

LOCAL_TZ = ZoneInfo("America/Chicago")

FEATURE_COLUMNS = [
    "temp_max_c",
    "temp_min_c",
    "precip_mm",
    "rain",
    "rain_yesterday",
    "temp_max_trend_c",
    "temp_max_3d_mean_c",
    "temp_min_3d_mean_c",
    "precip_3d_mean_mm",
    "precip_7d_mean_mm",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]


def raw_to_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten raw METAR observation rows and attach a local calendar date."""
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["local_date"] = df["timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    return df


def compute_live_daily_aggregate(raw_df: pd.DataFrame) -> list[dict]:
    """Aggregate raw METAR reports into daily_observations-shaped rows.

    Precipitation is approximated by summing `precip_last_hour_mm` across the
    day (missing readings treated as 0) — METAR stations don't populate that
    field on every report, so this is a lower bound, not an exact total.
    """
    if raw_df.empty:
        return []

    grouped = raw_df.groupby("local_date")
    agg = grouped.agg(
        temp_max_c=("temperature_c", "max"),
        temp_min_c=("temperature_c", "min"),
        precip_mm=("precip_last_hour_mm", lambda s: s.fillna(0).sum()),
    ).reset_index()

    agg["rain"] = (agg["precip_mm"] >= RAIN_THRESHOLD_MM).astype(int)
    agg["source"] = "metar_live"
    agg["date"] = agg["local_date"].astype(str)
    return agg[["date", "source", "temp_max_c", "temp_min_c", "precip_mm", "rain"]].to_dict("records")


def build_daily_features(daily_records: list[dict]) -> pd.DataFrame:
    """Turn stored daily_observations rows into a feature-engineered frame."""
    if not daily_records:
        return pd.DataFrame(columns=["date", *FEATURE_COLUMNS])

    df = pd.DataFrame.from_records(daily_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    month = df["date"].dt.month
    doy = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["rain_yesterday"] = df["rain"].shift(1)
    df["temp_max_trend_c"] = df["temp_max_c"] - df["temp_max_c"].shift(1)
    df["temp_max_3d_mean_c"] = df["temp_max_c"].rolling(3, min_periods=1).mean()
    df["temp_min_3d_mean_c"] = df["temp_min_c"].rolling(3, min_periods=1).mean()
    df["precip_3d_mean_mm"] = df["precip_mm"].rolling(3, min_periods=1).mean()
    df["precip_7d_mean_mm"] = df["precip_mm"].rolling(7, min_periods=1).mean()

    return df


def add_targets(daily: pd.DataFrame, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
    """Attach rain/temp_max/temp_min targets for each forecast horizon."""
    out = daily.copy()
    for h in horizons:
        out[f"rain_h{h}"] = out["rain"].shift(-h)
        out[f"temp_max_h{h}"] = out["temp_max_c"].shift(-h)
        out[f"temp_min_h{h}"] = out["temp_min_c"].shift(-h)
    return out


def build_training_frame(daily: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return (X, y_rain, y_temp_max, y_temp_min) for one horizon, dropping incomplete rows."""
    labeled = add_targets(daily, (horizon,))
    target_cols = [f"rain_h{horizon}", f"temp_max_h{horizon}", f"temp_min_h{horizon}"]
    labeled = labeled.dropna(subset=[*FEATURE_COLUMNS, *target_cols])
    X = labeled[FEATURE_COLUMNS]
    y_rain = labeled[f"rain_h{horizon}"].astype(int)
    y_tmax = labeled[f"temp_max_h{horizon}"]
    y_tmin = labeled[f"temp_min_h{horizon}"]
    return X, y_rain, y_tmax, y_tmin


def latest_feature_row(daily: pd.DataFrame) -> pd.DataFrame | None:
    """The most recent fully-formed day's features, used as the basis for predictions."""
    usable = daily.dropna(subset=FEATURE_COLUMNS)
    if usable.empty:
        return None
    return usable.iloc[[-1]]
