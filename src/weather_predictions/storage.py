"""SQLite storage for raw observations, deduplicated on (station_id, timestamp)."""

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
"""

_COLUMNS = [
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


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_observations(records: list[dict[str, Any]], db_path: Path = DB_PATH) -> int:
    """Insert observations, ignoring ones already stored. Returns rows newly inserted."""
    if not records:
        return 0

    placeholders = ", ".join(f":{c}" for c in _COLUMNS)
    sql = f"""
    INSERT OR IGNORE INTO observations ({', '.join(_COLUMNS)})
    VALUES ({placeholders})
    """
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.executemany(sql, records)
        return conn.total_changes - before


def fetch_all_observations(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observations ORDER BY station_id, timestamp"
        ).fetchall()
        return [dict(r) for r in rows]


def count_observations(db_path: Path = DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
