"""Collect MRMS hourly rainfall accumulation (QPE) at the home point.

MultiSensor QPE is NOAA's gauge-corrected "how much rain actually fell" grid —
much better ground truth for scoring rain predictions than GHCND's once-daily
totals, and hourly rather than daily. One value is sampled at LATITUDE/LONGITUDE
per hourly file and stored in SQLite (qpe_hourly table); the full grid is
discarded after sampling.

Needs the `mrms` dependency group (cfgrib/eccodes) to decode the GRIB2 files.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from weather_predictions.config import LATITUDE, LONGITUDE, MRMS_DATA_DIR, MRMS_QPE_PRODUCT
from weather_predictions.mrms_client import download_mrms_scan, latest_mrms_scan_key, list_mrms_scans
from weather_predictions.mrms_processing import decode_mrms_grib2, parse_mrms_timestamp
from weather_predictions.storage import upsert_qpe_hourly

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = MRMS_DATA_DIR / "qpe_raw"


def _sample_at_point(frame: dict[str, Any], lat: float, lon: float) -> float | None:
    """Nearest-cell value at (lat, lon); None where MRMS has no data (NaN)."""
    import numpy as np

    row = (lat - frame["lat_min"]) / (frame["lat_max"] - frame["lat_min"]) * (frame["nlat"] - 1)
    col = (lon - frame["lon_min"]) / (frame["lon_max"] - frame["lon_min"]) * (frame["nlon"] - 1)
    row = min(max(int(round(row)), 0), frame["nlat"] - 1)
    col = min(max(int(round(col)), 0), frame["nlon"] - 1)
    value = frame["reflectivity_dbz"][row, col]  # decode reuses this key for any MRMS grid
    return None if np.isnan(value) else float(value)


def _fetch_and_store(key: str, lat: float, lon: float) -> float | None:
    raw_path = download_mrms_scan(key, RAW_DIR)
    try:
        frame = decode_mrms_grib2(raw_path)
    finally:
        raw_path.unlink(missing_ok=True)
    precip_mm = _sample_at_point(frame, lat, lon)
    upsert_qpe_hourly(
        [{"valid_at": frame["timestamp"], "latitude": lat, "longitude": lon, "precip_mm": precip_mm}]
    )
    return precip_mm


def fetch_latest(lat: float = LATITUDE, lon: float = LONGITUDE) -> float | None:
    """Store the most recent hourly QPE value at (lat, lon). Returns the mm value."""
    key = latest_mrms_scan_key(MRMS_QPE_PRODUCT)
    if key is None:
        log.warning("no QPE files found for today or yesterday")
        return None
    precip_mm = _fetch_and_store(key, lat, lon)
    log.info("QPE %s -> %s mm at (%.4f, %.4f)", key, precip_mm, lat, lon)
    return precip_mm


def backfill_range(
    start: datetime,
    end: datetime,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
) -> int:
    """Store hourly QPE at (lat, lon) for every file in [start, end] (UTC).
    24 files/day, ~1MB each."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    stored = 0
    day = start.date()
    while day <= end.date():
        for key in list_mrms_scans(day, MRMS_QPE_PRODUCT):
            scan_ts = parse_mrms_timestamp(Path(key))
            if not (start <= scan_ts <= end):
                continue
            try:
                _fetch_and_store(key, lat, lon)
                stored += 1
            except Exception as e:
                log.warning("skipped %s: %s", key, e)
        day += timedelta(days=1)

    log.info("QPE backfill complete: %d hour(s) stored", stored)
    return stored
