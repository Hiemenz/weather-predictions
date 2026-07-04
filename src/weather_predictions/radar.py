"""Decode NEXRAD volume scans into stored reflectivity grids.

Needs the `radar` dependency group (`poetry install --with radar`) — this
imports Py-ART, which pulls in Cartopy. That has no prebuilt ARM wheels, so
this module should only run on a machine where that group is installed
(e.g. the Mac), not the Pi. See radar_raw.py for the Pi-safe raw-download-only
counterpart.

Raw volume scans (~12-15MB each) are downloaded to a scratch directory,
decoded into a much smaller gridded array (~100-150KB), then deleted by
default — NOAA's archive is permanent, so there's no need to hoard raw
files locally, especially on space-constrained devices like a Pi.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weather_predictions.config import RADAR_DATA_DIR, RADAR_STATION_ID
from weather_predictions.radar_client import download_scan, latest_scan_key, list_scans
from weather_predictions.radar_processing import decode_reflectivity_grid, save_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = RADAR_DATA_DIR / "raw"
GRID_DIR = RADAR_DATA_DIR / "grids"


def decode_file(raw_path: Path, keep_raw: bool = False) -> str:
    """Decode one already-downloaded raw scan into a stored grid."""
    frame = decode_reflectivity_grid(raw_path)
    saved_path = save_grid(frame, GRID_DIR)
    if not keep_raw:
        raw_path.unlink(missing_ok=True)
    return str(saved_path)


def decode_pending(raw_dir: Path = RAW_DIR, keep_raw: bool = False) -> int:
    """Decode every raw scan sitting in `raw_dir` — e.g. files synced over
    from the Pi's raw-only collector (radar_raw.py). Returns count decoded."""
    raw_files = sorted(p for p in raw_dir.glob("*_V0*") if p.is_file())
    decoded = 0
    for raw_path in raw_files:
        try:
            saved_path = decode_file(raw_path, keep_raw)
            decoded += 1
            log.info("[%d/%d] decoded %s -> %s", decoded, len(raw_files), raw_path.name, saved_path)
        except Exception as e:
            log.warning("skipped %s: %s", raw_path.name, e)
    log.info("decode_pending complete: %d scan(s) decoded", decoded)
    return decoded


def _fetch_and_decode(key: str, keep_raw: bool) -> str:
    raw_path = download_scan(key, RAW_DIR)
    return decode_file(raw_path, keep_raw)


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
