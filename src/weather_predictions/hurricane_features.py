"""Feature engineering for hurricane track/intensity forecasting from
HURDAT2 best-track fixes.

Same shape as features.py: per-fix engineered features (here: motion
bearing/speed derived from consecutive fixes, storm age, climatology),
plus targets for each forecast horizon built by matching each fix to the
nearest later fix *within the same storm* close to (timestamp + horizon).
Fixes aren't perfectly 6-hourly (special/landfall records break the
cadence), so a naive `.shift(-h)` would mismatch — this matches by nearest
timestamp within a tolerance instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_predictions.geo import bearing_deg, haversine_km

FORECAST_HORIZONS_HOURS = (12, 24, 48, 72)

FEATURE_COLUMNS = [
    "lat",
    "lon",
    "max_wind_kt",
    "min_pressure_mb",
    "motion_bearing_sin",
    "motion_bearing_cos",
    "motion_speed_kmh",
    "storm_age_hours",
    "doy_sin",
    "doy_cos",
]

# How close an actual fix must be to (timestamp + horizon) to count as the
# outcome for that horizon — HURDAT2's normal cadence is 6-hourly, so half
# of that comfortably matches the intended fix without drifting onto the
# next one.
_TARGET_MATCH_TOLERANCE_HOURS = 1.5


def build_storm_features(fixes: list[dict]) -> pd.DataFrame:
    """Turn stored hurricane_fixes rows into a feature-engineered frame, one
    row per fix, with motion vector/age/climatology features attached."""
    if not fixes:
        return pd.DataFrame(columns=["storm_id", "name", "timestamp", *FEATURE_COLUMNS])

    df = pd.DataFrame.from_records(fixes)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["storm_id", "timestamp"]).drop_duplicates(subset=["storm_id", "timestamp"], keep="last")

    groups = []
    for _, g in df.groupby("storm_id", sort=False):
        g = g.reset_index(drop=True)
        lat, lon, ts = g["lat"].to_numpy(), g["lon"].to_numpy(), g["timestamp"]

        bearing = np.full(len(g), np.nan)
        speed_kmh = np.full(len(g), np.nan)
        for i in range(1, len(g)):
            dt_hours = (ts.iloc[i] - ts.iloc[i - 1]).total_seconds() / 3600
            if dt_hours <= 0:
                continue
            bearing[i] = bearing_deg(lat[i - 1], lon[i - 1], lat[i], lon[i])
            speed_kmh[i] = haversine_km(lat[i - 1], lon[i - 1], lat[i], lon[i]) / dt_hours

        g["motion_bearing_sin"] = np.sin(np.radians(bearing))
        g["motion_bearing_cos"] = np.cos(np.radians(bearing))
        g["motion_speed_kmh"] = speed_kmh
        g["storm_age_hours"] = (ts - ts.iloc[0]).dt.total_seconds() / 3600
        groups.append(g)

    result = pd.concat(groups, ignore_index=True)
    doy = result["timestamp"].dt.dayofyear
    result["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    result["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return result


def _match_horizon_targets(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """For each fix in storm `g`, find the fix nearest to (timestamp + horizon)
    within tolerance, returning lat/lon/wind columns aligned to g's row order."""
    target_time = g["timestamp"] + pd.Timedelta(hours=horizon)
    lookup = g[["timestamp", "lat", "lon", "max_wind_kt"]].rename(columns={"timestamp": "match_time"})

    left = pd.DataFrame({"_orig_idx": g.index, "target_time": target_time}).sort_values("target_time")
    merged = pd.merge_asof(
        left,
        lookup.sort_values("match_time"),
        left_on="target_time",
        right_on="match_time",
        direction="nearest",
        tolerance=pd.Timedelta(hours=_TARGET_MATCH_TOLERANCE_HOURS),
    ).sort_values("_orig_idx")

    return pd.DataFrame(
        {
            f"lat_h{horizon}": merged["lat"].to_numpy(),
            f"lon_h{horizon}": merged["lon"].to_numpy(),
            f"wind_h{horizon}": merged["max_wind_kt"].to_numpy(),
        },
        index=g.index,
    )


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...] = FORECAST_HORIZONS_HOURS) -> pd.DataFrame:
    """Attach lat/lon/wind targets for each forecast horizon, matched per-storm."""
    parts = []
    for _, g in df.groupby("storm_id", sort=False):
        targets = pd.concat([_match_horizon_targets(g, h) for h in horizons], axis=1)
        parts.append(pd.concat([g, targets], axis=1))
    return pd.concat(parts, ignore_index=True) if parts else df


def build_training_frame(
    df: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (X, y_lat, y_lon, y_wind, timestamps) for one horizon, dropping incomplete rows."""
    labeled = add_targets(df, (horizon,))
    target_cols = [f"lat_h{horizon}", f"lon_h{horizon}", f"wind_h{horizon}"]
    labeled = labeled.dropna(subset=[*FEATURE_COLUMNS, *target_cols]).reset_index(drop=True)
    X = labeled[FEATURE_COLUMNS]
    y_lat = labeled[f"lat_h{horizon}"]
    y_lon = labeled[f"lon_h{horizon}"]
    y_wind = labeled[f"wind_h{horizon}"]
    return X, y_lat, y_lon, y_wind, labeled["timestamp"]
