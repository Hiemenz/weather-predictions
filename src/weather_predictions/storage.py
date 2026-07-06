"""SQLite storage.

Four tables:
- `observations` — raw NWS/METAR reports, deduplicated on (station_id, timestamp).
  Used to derive a live aggregate for "today" before CDO/LCD data catches up.
- `daily_observations` — one row per calendar date, the primary table
  features/training/prediction are built from. temp_max_c/temp_min_c/
  precip_mm/rain/source are owned by CDO/GHCND backfill (authoritative) or
  the live METAR-derived aggregate (fills gaps only, never overwrites
  GHCND). humidity_pct/pressure_hpa/wind_speed_kmh are filled in separately
  by LCD enrichment (or the live aggregate), and are never clobbered by a
  GHCND re-run since GHCND doesn't carry those fields at all.
- `predictions` / `model_performance` — forecasts made by the model and their
  scored accuracy once the real outcome is known, so skill can be tracked
  over time.
- `radar_nowcasts` / `radar_nowcast_performance` — same predict/evaluate
  pattern as above, for radar-based nowcasts (see radar_nowcast.py /
  radar_nowcast_evaluate.py) instead of the tabular rain/temp model.
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
    rain INTEGER,
    humidity_pct REAL,
    pressure_hpa REAL,
    wind_speed_kmh REAL
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

CREATE TABLE IF NOT EXISTS radar_nowcasts (
    predicted_at TEXT NOT NULL,
    valid_at TEXT NOT NULL,
    lead_minutes REAL NOT NULL,
    method TEXT NOT NULL,
    station TEXT NOT NULL,
    grid_path TEXT NOT NULL,
    PRIMARY KEY (predicted_at, valid_at, method)
);

CREATE TABLE IF NOT EXISTS radar_nowcast_performance (
    evaluated_at TEXT NOT NULL,
    method TEXT NOT NULL,
    lead_minutes REAL NOT NULL,
    n_samples INTEGER,
    mae_dbz REAL,
    csi REAL,
    csi_threshold_dbz REAL,
    PRIMARY KEY (evaluated_at, method, lead_minutes)
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

_ENRICHMENT_COLUMNS = ["humidity_pct", "pressure_hpa", "wind_speed_kmh"]

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

_RADAR_NOWCAST_COLUMNS = [
    "predicted_at",
    "valid_at",
    "lead_minutes",
    "method",
    "station",
    "grid_path",
]

_RADAR_NOWCAST_PERFORMANCE_COLUMNS = [
    "evaluated_at",
    "method",
    "lead_minutes",
    "n_samples",
    "mae_dbz",
    "csi",
    "csi_threshold_dbz",
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database's initial creation."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(daily_observations)")}
    for column in _ENRICHMENT_COLUMNS:
        if column not in existing:
            conn.execute(f"ALTER TABLE daily_observations ADD COLUMN {column} REAL")


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        _migrate(conn)
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
    """CDO/GHCND owns temp/precip/rain/source and always overwrites them for that date.

    Never touches humidity_pct/pressure_hpa/wind_speed_kmh, so re-running a
    GHCND backfill can't wipe out LCD enrichment for the same dates.
    """
    if not records:
        return 0
    sql = """
    INSERT INTO daily_observations (date, source, temp_max_c, temp_min_c, precip_mm, rain)
    VALUES (:date, :source, :temp_max_c, :temp_min_c, :precip_mm, :rain)
    ON CONFLICT(date) DO UPDATE SET
        source = excluded.source,
        temp_max_c = excluded.temp_max_c,
        temp_min_c = excluded.temp_min_c,
        precip_mm = excluded.precip_mm,
        rain = excluded.rain
    """
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(sql, records)
        return conn.total_changes - before


def upsert_daily_from_metar(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Live METAR-derived temp/precip/rain for "today"/"yesterday" before GHCND catches up.

    Refines its own row as more observations come in through the day, but
    never overwrites a date GHCND has already made authoritative. Pressure/
    humidity/wind are handled separately by `upsert_daily_enrichment`, since
    GHCND never owns those fields — a date can be GHCND-authoritative for
    temp/precip while still needing its pressure/humidity filled in live.
    """
    if not records:
        return 0
    sql = """
    INSERT INTO daily_observations (date, source, temp_max_c, temp_min_c, precip_mm, rain)
    VALUES (:date, :source, :temp_max_c, :temp_min_c, :precip_mm, :rain)
    ON CONFLICT(date) DO UPDATE SET
        temp_max_c = excluded.temp_max_c,
        temp_min_c = excluded.temp_min_c,
        precip_mm = excluded.precip_mm,
        rain = excluded.rain
    WHERE daily_observations.source != 'ghcnd'
    """
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(sql, records)
        return conn.total_changes - before


def upsert_daily_enrichment(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Fill in humidity/pressure/wind for existing dates without touching other columns.

    Used by LCD enrichment, which supplies fields GHCND's daily summaries and
    the METAR-derived aggregate keep separately. If no row exists yet for a
    date, inserts a bare one tagged source='lcd'.
    """
    if not records:
        return 0
    sql = """
    INSERT INTO daily_observations (date, source, humidity_pct, pressure_hpa, wind_speed_kmh)
    VALUES (:date, 'lcd', :humidity_pct, :pressure_hpa, :wind_speed_kmh)
    ON CONFLICT(date) DO UPDATE SET
        humidity_pct = excluded.humidity_pct,
        pressure_hpa = excluded.pressure_hpa,
        wind_speed_kmh = excluded.wind_speed_kmh
    """
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(sql, records)
        return conn.total_changes - before


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


def upsert_radar_nowcasts(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Re-running a nowcast for the same predicted_at/valid_at/method replaces the earlier one."""
    with connect(db_path) as conn:
        return _upsert(conn, "radar_nowcasts", _RADAR_NOWCAST_COLUMNS, records, replace=True)


def fetch_all_radar_nowcasts(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM radar_nowcasts ORDER BY predicted_at, method").fetchall()
        return [dict(r) for r in rows]


def upsert_radar_nowcast_performance(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    with connect(db_path) as conn:
        return _upsert(conn, "radar_nowcast_performance", _RADAR_NOWCAST_PERFORMANCE_COLUMNS, records, replace=True)


def fetch_all_radar_nowcast_performance(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM radar_nowcast_performance ORDER BY evaluated_at, method").fetchall()
        return [dict(r) for r in rows]
