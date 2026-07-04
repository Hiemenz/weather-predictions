"""Fetch + decode NEXRAD volume scans into stored reflectivity grids.

Raw volume scans (~12-15MB each) are downloaded to a scratch directory,
decoded into a much smaller gridded array (~100-150KB), then deleted by
default — NOAA's archive is permanent, so there's no need to hoard raw
files locally, especially on space-constrained devices like a Pi.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from weather_predictions.config import RADAR_DATA_DIR, RADAR_STATION_ID
from weather_predictions.radar_client import download_scan, latest_scan_key, list_scans
from weather_predictions.radar_processing import decode_reflectivity_grid, save_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_RAW_DIR = RADAR_DATA_DIR / "raw"
_GRID_DIR = RADAR_DATA_DIR / "grids"


def _fetch_and_decode(key: str, keep_raw: bool) -> str:
    raw_path = download_scan(key, _RAW_DIR)
    try:
        frame = decode_reflectivity_grid(raw_path)
        saved_path = save_grid(frame, _GRID_DIR)
        return str(saved_path)
    finally:
        if not keep_raw:
            raw_path.unlink(missing_ok=True)


def fetch_latest(station: str = RADAR_STATION_ID, keep_raw: bool = False) -> str | None:
    key = latest_scan_key(station)
    if key is None:
        log.warning("no volume scans found for %s today or yesterday", station)
        return None
    saved_path = _fetch_and_decode(key, keep_raw)
    log.info("decoded latest scan %s -> %s", key, saved_path)
    return saved_path


def backfill_range(
    start: datetime,
    end: datetime,
    station: str = RADAR_STATION_ID,
    keep_raw: bool = False,
) -> int:
    """Fetch + decode every scan in [start, end] (UTC). ~12 scans/hour, ~12-15MB
    raw download each — a full day is roughly 3-4GB of transient downloads."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    n_days = (end.date() - start.date()).days + 1
    log.info("backfilling radar for %s from %s to %s (%d day(s))", station, start, end, n_days)

    saved_count = 0
    day = start.date()
    while day <= end.date():
        keys = list_scans(day, station)
        for key in keys:
            ts_str = key.rsplit("/", 1)[-1].split("_")[1]  # HHMMSS from the filename
            scan_dt = datetime.combine(day, datetime.strptime(ts_str, "%H%M%S").time(), tzinfo=timezone.utc)
            if not (start <= scan_dt <= end):
                continue
            try:
                saved_path = _fetch_and_decode(key, keep_raw)
                saved_count += 1
                log.info("[%d] decoded %s -> %s", saved_count, key, saved_path)
            except Exception as e:
                log.warning("skipped %s: %s", key, e)
        day += timedelta(days=1)

    log.info("radar backfill complete: %d scan(s) decoded", saved_count)
    return saved_count


if __name__ == "__main__":
    fetch_latest()
