"""Turn raw hourly-ish observations into a daily feature table for modeling.

Design notes:
- Days are bucketed in local time (America/Chicago) so "today"/"tomorrow"
  line up with how a person would read a forecast.
- Daily precipitation is approximated by summing `precip_last_hour_mm`
  across the day, treating missing readings as 0. METAR stations don't
  always populate that field on every report, so this is a lower bound,
  not a perfectly calibrated total.
- The label is whether *tomorrow* sees measurable rain, so every row's
  target comes from the following day's aggregate (see `add_target`).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from weather_predictions.config import RAIN_THRESHOLD_MM

LOCAL_TZ = ZoneInfo("America/Chicago")

FEATURE_COLUMNS = [
    "temp_max_c",
    "temp_min_c",
    "temp_mean_c",
    "dewpoint_mean_c",
    "humidity_mean_pct",
    "pressure_mean_pa",
    "pressure_delta_pa",
    "wind_speed_mean_kmh",
    "wind_gust_max_kmh",
    "precip_total_mm",
    "rain_today",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "precip_3d_mean_mm",
    "pressure_3d_delta_pa",
    "rain_yesterday",
]


def raw_to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["local_date"] = df["timestamp"].dt.tz_convert(LOCAL_TZ).dt.date
    return df


def build_daily_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw observations (one row per report) into one row per day."""
    if raw_df.empty:
        return pd.DataFrame(columns=["local_date", *FEATURE_COLUMNS])

    grouped = raw_df.groupby("local_date")
    daily = grouped.agg(
        temp_max_c=("temperature_c", "max"),
        temp_min_c=("temperature_c", "min"),
        temp_mean_c=("temperature_c", "mean"),
        dewpoint_mean_c=("dewpoint_c", "mean"),
        humidity_mean_pct=("relative_humidity_pct", "mean"),
        pressure_mean_pa=("barometric_pressure_pa", "mean"),
        wind_speed_mean_kmh=("wind_speed_kmh", "mean"),
        wind_gust_max_kmh=("wind_gust_kmh", "max"),
        precip_total_mm=("precip_last_hour_mm", lambda s: s.fillna(0).sum()),
        n_observations=("timestamp", "count"),
    ).reset_index()

    # Same-day pressure trend: last reading minus first reading.
    pressure_delta = grouped["barometric_pressure_pa"].agg(
        lambda s: s.dropna().iloc[-1] - s.dropna().iloc[0] if s.dropna().size >= 2 else np.nan
    )
    daily["pressure_delta_pa"] = pressure_delta.to_numpy()

    daily["rain_today"] = (daily["precip_total_mm"] >= RAIN_THRESHOLD_MM).astype(int)
    daily["local_date"] = pd.to_datetime(daily["local_date"])
    daily = daily.sort_values("local_date").reset_index(drop=True)

    month = daily["local_date"].dt.month
    doy = daily["local_date"].dt.dayofyear
    daily["month_sin"] = np.sin(2 * np.pi * month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * month / 12)
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    daily["precip_3d_mean_mm"] = daily["precip_total_mm"].rolling(3, min_periods=1).mean()
    daily["pressure_3d_delta_pa"] = daily["pressure_delta_pa"].rolling(3, min_periods=1).mean()
    daily["rain_yesterday"] = daily["rain_today"].shift(1)

    return daily


def add_target(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach `rain_tomorrow`, the label for next-day precipitation."""
    out = daily.copy()
    out["rain_tomorrow"] = out["rain_today"].shift(-1)
    return out


def build_training_frame(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for sklearn, dropping rows with no label or missing features."""
    labeled = add_target(daily).dropna(subset=["rain_tomorrow", *FEATURE_COLUMNS])
    X = labeled[FEATURE_COLUMNS]
    y = labeled["rain_tomorrow"].astype(int)
    return X, y


def latest_feature_row(daily: pd.DataFrame) -> pd.DataFrame | None:
    """The most recent fully-formed day's features, used to predict tomorrow."""
    usable = daily.dropna(subset=FEATURE_COLUMNS)
    if usable.empty:
        return None
    return usable.iloc[[-1]][FEATURE_COLUMNS]
