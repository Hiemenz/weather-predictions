"""Raw NEXRAD scan collection only — no Py-ART/Cartopy import, deliberately.

This is the module the Pi's cron job should call: it needs nothing beyond
the `aws` CLI and the stdlib, so it can run on a Raspberry Pi without ever
touching the radar-decoding dependency group (arm-pyart pulls in Cartopy,
which has no prebuilt ARM wheels — see pyproject.toml). Decoding happens
separately, on whichever machine has `poetry install --with radar` (see
radar.py), against raw files synced over from wherever this ran.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weather_predictions.config import RADAR_DATA_DIR, RADAR_STATION_ID
from weather_predictions.radar_client import download_scan, latest_scan_key, list_scans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = RADAR_DATA_DIR / "raw"


def fetch_latest_raw(station: str = RADAR_STATION_ID, dest_dir: Path = RAW_DIR) -> Path | None:
    """Download the latest scan if we don't already have it. Safe to run on
    any cadence — skips the download entirely once caught up."""
    key = latest_scan_key(station)
    if key is None:
        log.warning("no volume scans found for %s today or yesterday", station)
        return None

    dest_path = dest_dir / Path(key).name
    if dest_path.exists():
        log.info("already have latest scan %s", key)
        return None

    saved_path = download_scan(key, dest_dir)
    log.info("downloaded %s -> %s", key, saved_path)
    return saved_path


def backfill_raw(
    start: datetime,
    end: datetime,
    station: str = RADAR_STATION_ID,
    dest_dir: Path = RAW_DIR,
) -> int:
    """Download every raw scan in [start, end] (UTC) without decoding. Same
    volume math as the decode-including backfill: ~12 scans/hour, ~12-15MB each."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    n_days = (end.date() - start.date()).days + 1
    log.info("raw radar backfill for %s from %s to %s (%d day(s))", station, start, end, n_days)

    downloaded = 0
    day = start.date()
    while day <= end.date():
        for key in list_scans(day, station):
            ts_str = key.rsplit("/", 1)[-1].split("_")[1]
            scan_dt = datetime.combine(day, datetime.strptime(ts_str, "%H%M%S").time(), tzinfo=timezone.utc)
            if not (start <= scan_dt <= end):
                continue
            dest_path = dest_dir / Path(key).name
            if dest_path.exists():
                continue
            try:
                download_scan(key, dest_dir)
                downloaded += 1
                log.info("[%d] downloaded %s", downloaded, key)
            except Exception as e:
                log.warning("skipped %s: %s", key, e)
        day += timedelta(days=1)

    log.info("raw radar backfill complete: %d scan(s) downloaded", downloaded)
    return downloaded


if __name__ == "__main__":
    fetch_latest_raw()
