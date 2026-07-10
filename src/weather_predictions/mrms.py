"""Fetch and decode MRMS national radar composite scans.

Needs the `mrms` dependency group (`poetry install --with mrms`) — cfgrib
requires the eccodes C library (brew install eccodes / apt install libeccodes-dev).

Raw .grib2.gz files (~1.5MB each) are downloaded to a scratch directory,
decoded into compressed .npz grids (~1-5MB depending on storm coverage),
then deleted by default. NOAA's MRMS archive on S3 goes back to 2020.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from weather_predictions.config import MRMS_DATA_DIR
from weather_predictions.mrms_client import download_mrms_scan, latest_mrms_scan_key, list_mrms_scans
from weather_predictions.mrms_processing import decode_mrms_grib2, parse_mrms_timestamp, save_mrms_grid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = MRMS_DATA_DIR / "raw"
GRID_DIR = MRMS_DATA_DIR / "grids"


def _fetch_and_decode(key: str, keep_raw: bool = False) -> str:
    raw_path = download_mrms_scan(key, RAW_DIR)
    frame = decode_mrms_grib2(raw_path)
    saved_path = save_mrms_grid(frame, GRID_DIR)
    if not keep_raw:
        raw_path.unlink(missing_ok=True)
    return str(saved_path)


def fetch_latest(keep_raw: bool = False) -> str | None:
    """Download + decode the most recent MRMS national scan."""
    key = latest_mrms_scan_key()
    if key is None:
        log.warning("no MRMS scans found for today or yesterday")
        return None
    saved_path = _fetch_and_decode(key, keep_raw)
    log.info("decoded latest MRMS scan %s -> %s", key, saved_path)
    return saved_path


def backfill_range(
    start: datetime,
    end: datetime,
    keep_raw: bool = False,
) -> int:
    """Fetch + decode every MRMS scan in [start, end] (UTC).

    ~30 scans/hour at ~1.5MB raw each — a full day is roughly 1GB of transient
    downloads (much less than the equivalent NEXRAD multi-station approach).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    n_days = (end.date() - start.date()).days + 1
    log.info("backfilling MRMS from %s to %s (%d day(s))", start, end, n_days)

    saved_count = 0
    day = start.date()
    while day <= end.date():
        keys = list_mrms_scans(day)
        for key in keys:
            from pathlib import Path as _Path

            scan_ts = parse_mrms_timestamp(_Path(key))
            if not (start <= scan_ts <= end):
                continue
            try:
                saved_path = _fetch_and_decode(key, keep_raw)
                saved_count += 1
                log.info("[%d] decoded %s -> %s", saved_count, key, saved_path)
            except Exception as e:
                log.warning("skipped %s: %s", key, e)
        day += timedelta(days=1)

    log.info("MRMS backfill complete: %d scan(s) decoded", saved_count)
    return saved_count


if __name__ == "__main__":
    fetch_latest()
