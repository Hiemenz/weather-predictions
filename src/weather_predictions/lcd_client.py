"""Client for NOAA's Local Climatological Data (LCD) bulk CSV files.

Unlike CDO/GHCND, these are plain static files with no token, no rate limit,
and a few seconds per year to download — but they only go back to when the
station started reporting hourly (varies by station), and each file bundles
hourly + daily rows together, so we filter to the daily summary rows.

Used specifically for pressure/humidity/wind, which GHCND's daily summaries
don't carry. Docs: https://www.ncei.noaa.gov/data/local-climatological-data/
"""

from __future__ import annotations

import csv
import io
import logging
import shutil
import subprocess

import requests

from weather_predictions.config import LCD_STATION_ID

log = logging.getLogger(__name__)

_TIMEOUT = 60
_LCD_BASE = "https://www.ncei.noaa.gov/data/local-climatological-data/access"
_DAILY_REPORT_TYPE = "SOD"


class LCDClientError(RuntimeError):
    pass


def _to_float(raw: str) -> float | None:
    if raw is None:
        return None
    raw = raw.strip().rstrip("sVs*")  # LCD sometimes suffixes flags like "s"
    if raw in ("", "T"):  # "T" = trace amount
        return 0.0 if raw == "T" else None
    try:
        return float(raw)
    except ValueError:
        return None


def _download(url: str) -> tuple[int, str]:
    """Fetch a URL's body, preferring curl when available.

    Some sandboxed/dev environments throttle Python's own HTTP stack far
    below what `curl` gets on the same network for large files — these LCD
    files are ~10MB each — so shelling out to curl when present avoids
    multi-minute downloads that a plain `requests.get()` would otherwise
    take. Falls back to `requests` if curl isn't on PATH.
    """
    if shutil.which("curl"):
        result = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code}", "--max-time", str(_TIMEOUT), url],
            capture_output=True,
            text=True,
        )
        body, _, status = result.stdout.rpartition("\n")
        return int(status or 0), body

    resp = requests.get(url, timeout=_TIMEOUT)
    return resp.status_code, resp.text


def fetch_year_csv(year: int, station_id: str = LCD_STATION_ID) -> str:
    url = f"{_LCD_BASE}/{year}/{station_id}.csv"
    status_code, body = _download(url)
    if status_code == 404:
        raise LCDClientError(f"No LCD file for station {station_id} in {year} (404).")
    if status_code != 200:
        raise LCDClientError(f"GET {url} -> {status_code}: {body[:300]}")
    return body


def parse_daily_rows(csv_text: str) -> list[dict]:
    """Extract one row per calendar day (REPORT_TYPE == SOD) with converted units."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for r in reader:
        if r.get("REPORT_TYPE", "").strip() != _DAILY_REPORT_TYPE:
            continue

        humidity = _to_float(r.get("DailyAverageRelativeHumidity", ""))
        pressure_inhg = _to_float(r.get("DailyAverageSeaLevelPressure", "")) or _to_float(
            r.get("DailyAverageStationPressure", "")
        )
        wind_mph = _to_float(r.get("DailyAverageWindSpeed", ""))

        rows.append(
            {
                "date": r["DATE"][:10],
                "humidity_pct": humidity,
                "pressure_hpa": pressure_inhg * 33.8639 if pressure_inhg is not None else None,
                "wind_speed_kmh": wind_mph * 1.60934 if wind_mph is not None else None,
            }
        )
    return rows


def get_daily_pressure_humidity(year: int, station_id: str = LCD_STATION_ID) -> list[dict]:
    csv_text = fetch_year_csv(year, station_id)
    return parse_daily_rows(csv_text)
