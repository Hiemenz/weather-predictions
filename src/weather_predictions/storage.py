"""SQLite storage.

Three tables:
- `observations` — raw NWS/METAR reports, deduplicated on (station_id, timestamp).
  Used to derive a live aggregate for "today" before CDO's data catches up.
- `daily_observations` — one row per calendar date (temp max/min, precip),
  the primary table features/training/prediction are built from. Populated
  by CDO backfill (authoritative, wins ties) and by a live METAR-derived
  aggregate for very recent days CDO hasn't published yet (fills gaps only).
- `predictions` / `model_performance` — forecasts made by the model and their
  scored accuracy once the real outcome is known, so skill can be tracked
  over time.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from weather_predictions.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    station_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    text_description TEXT,
    temperature_c REAL,
    dewpoint_c REAL,
    wind_direction_deg REAL,
    wind_speed_kmh REAL,
    wind_gust_kmh REAL,
    barometric_pressure_pa REAL,
    sea_level_pressure_pa REAL,
    visibility_m REAL,
    max_temp_last_24h_c REAL,
    min_temp_last_24h_c REAL,
    precip_last_hour_mm REAL,
    precip_last_3h_mm REAL,
    precip_last_6h_mm REAL,
    relative_humidity_pct REAL,
    wind_chill_c REAL,
    heat_index_c REAL,
    PRIMARY KEY (station_id, timestamp)
);

CREATE TABLE IF NOT EXISTS daily_observations (
    date TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    temp_max_c REAL,
    temp_min_c REAL,
    precip_mm REAL,
    rain INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    predicted_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    target_date TEXT NOT NULL,
    rain_probability REAL,
    rain_predicted INTEGER,
    temp_max_pred_c REAL,
    temp_min_pred_c REAL,
    model_trained_at TEXT,
    PRIMARY KEY (predicted_date, horizon_days)
);

CREATE TABLE IF NOT EXISTS model_performance (
    evaluated_at TEXT NOT NULL,
    model_trained_at TEXT,
    horizon_days INTEGER NOT NULL,
    n_samples INTEGER,
    rain_accuracy REAL,
    rain_brier REAL,
    temp_max_mae REAL,
    temp_min_mae REAL,
    PRIMARY KEY (evaluated_at, horizon_days)
);
"""

_OBS_COLUMNS = [
    "station_id",
    "timestamp",
    "text_description",
    "temperature_c",
    "dewpoint_c",
    "wind_direction_deg",
    "wind_speed_kmh",
    "wind_gust_kmh",
    "barometric_pressure_pa",
    "sea_level_pressure_pa",
    "visibility_m",
    "max_temp_last_24h_c",
    "min_temp_last_24h_c",
    "precip_last_hour_mm",
    "precip_last_3h_mm",
    "precip_last_6h_mm",
    "relative_humidity_pct",
    "wind_chill_c",
    "heat_index_c",
]

_DAILY_COLUMNS = ["date", "source", "temp_max_c", "temp_min_c", "precip_mm", "rain"]

_PREDICTION_COLUMNS = [
    "predicted_date",
    "horizon_days",
    "target_date",
    "rain_probability",
    "rain_predicted",
    "temp_max_pred_c",
    "temp_min_pred_c",
    "model_trained_at",
]

_PERFORMANCE_COLUMNS = [
    "evaluated_at",
    "model_trained_at",
    "horizon_days",
    "n_samples",
    "rain_accuracy",
    "rain_brier",
    "temp_max_mae",
    "temp_min_mae",
]


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _upsert(conn: sqlite3.Connection, table: str, columns: list[str], records: list[dict[str, Any]], replace: bool) -> int:
    if not records:
        return 0
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    placeholders = ", ".join(f":{c}" for c in columns)
    sql = f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    before = conn.total_changes
    conn.executemany(sql, records)
    return conn.total_changes - before


def upsert_observations(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Insert raw observations, ignoring ones already stored."""
    with connect(db_path) as conn:
        return _upsert(conn, "observations", _OBS_COLUMNS, records, replace=False)


def fetch_all_observations(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM observations ORDER BY station_id, timestamp").fetchall()
        return [dict(r) for r in rows]


def count_observations(db_path: Path = DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]


def upsert_daily_from_cdo(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """CDO/GHCND data is authoritative — always overwrites existing rows for that date."""
    with connect(db_path) as conn:
        return _upsert(conn, "daily_observations", _DAILY_COLUMNS, records, replace=True)


def upsert_daily_from_metar(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Live METAR-derived aggregate — only fills gaps for dates CDO hasn't published yet."""
    with connect(db_path) as conn:
        return _upsert(conn, "daily_observations", _DAILY_COLUMNS, records, replace=False)


def fetch_all_daily(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM daily_observations ORDER BY date").fetchall()
        return [dict(r) for r in rows]


def count_daily(db_path: Path = DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM daily_observations").fetchone()[0]


def upsert_predictions(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Re-running predict for the same day/horizon replaces the earlier prediction."""
    with connect(db_path) as conn:
        return _upsert(conn, "predictions", _PREDICTION_COLUMNS, records, replace=True)


def fetch_all_predictions(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM predictions ORDER BY predicted_date, horizon_days").fetchall()
        return [dict(r) for r in rows]


def upsert_performance(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    with connect(db_path) as conn:
        return _upsert(conn, "model_performance", _PERFORMANCE_COLUMNS, records, replace=True)


def fetch_all_performance(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM model_performance ORDER BY evaluated_at, horizon_days").fetchall()
        return [dict(r) for r in rows]
