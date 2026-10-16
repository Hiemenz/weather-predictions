"""Client for NOAA/NHC hurricane data: historical best-track (HURDAT2) and
the live active-storms feed.

HURDAT2 is a plain static text file — same curl-preferring download as
`lcd_client.py` (sandboxed environments have been observed to throttle
Python's own HTTP stack far below what curl gets for the same file).

The live feed's schema is confirmed against NHC's own "Tropical Cyclone
Status JSON File Reference" (nhc.noaa.gov/productexamples/), not guessed —
field names like `latitude_numeric`/`movementDir`/`movementSpeed` are exact.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from weather_predictions.config import NHC_CURRENT_STORMS_URL, NHC_HURDAT2_URL
from weather_predictions.storage import upsert_hurricane_fixes

_TIMEOUT = 120
_MISSING_SENTINEL = -999.0


class HurricaneClientError(RuntimeError):
    pass


def _download(url: str, timeout: int = _TIMEOUT) -> str:
    """Fetch a URL's body, preferring curl when available (see lcd_client._download)."""
    if shutil.which("curl"):
        result = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code}", "--max-time", str(timeout), url],
            capture_output=True,
            text=True,
        )
        body, _, status = result.stdout.rpartition("\n")
        status_code = int(status or 0)
    else:
        resp = requests.get(url, timeout=timeout)
        status_code, body = resp.status_code, resp.text

    if status_code != 200:
        raise HurricaneClientError(f"GET {url} -> {status_code}")
    return body


def download_hurdat2(dest_path: Path, url: str = NHC_HURDAT2_URL) -> Path:
    body = _download(url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(body)
    return dest_path


def _to_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    value = float(raw)
    return None if value == _MISSING_SENTINEL else value


def _parse_lat(raw: str) -> float:
    raw = raw.strip()
    sign = -1.0 if raw.endswith("S") else 1.0
    return sign * float(raw[:-1])


def _parse_lon(raw: str) -> float:
    raw = raw.strip()
    sign = -1.0 if raw.endswith("W") else 1.0
    return sign * float(raw[:-1])


def parse_hurdat2(text: str) -> list[dict[str, Any]]:
    """Parse HURDAT2's format: a header line per storm (id, name, record
    count) followed by that many 6-hourly data lines (date, time, record
    identifier, status, lat, lon, wind, pressure, then wind-radii fields
    this project doesn't need). A data line's first field is always an
    8-digit date; a header line's first field never parses as an int —
    that's what distinguishes the two without relying on line position.
    """
    records: list[dict[str, Any]] = []
    storm_id: str | None = None
    storm_name: str | None = None

    for line in text.splitlines():
        fields = [f.strip() for f in line.strip().split(",")]
        if not fields or not fields[0]:
            continue

        try:
            int(fields[0])
        except ValueError:
            storm_id, storm_name = fields[0], fields[1]
            continue

        date_str, time_str, _record_id, status, lat_str, lon_str, wind_str, pressure_str = fields[:8]
        timestamp = datetime.strptime(date_str + time_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        records.append(
            {
                "storm_id": storm_id,
                "name": storm_name,
                "timestamp": timestamp.isoformat(),
                "lat": _parse_lat(lat_str),
                "lon": _parse_lon(lon_str),
                "max_wind_kt": _to_float(wind_str),
                "min_pressure_mb": _to_float(pressure_str),
                "status": status,
            }
        )
    return records


def sync_hurdat2(dest_path: Path, url: str = NHC_HURDAT2_URL) -> int:
    """Download HURDAT2 and store every fix. Run once (or whenever the
    yearly refresh is worth pulling in again — re-running is safe, fixes
    are upserted by (storm_id, timestamp))."""
    download_hurdat2(dest_path, url)
    records = parse_hurdat2(dest_path.read_text())
    return upsert_hurricane_fixes(records)


def get_active_storms(timeout: int = _TIMEOUT) -> list[dict[str, Any]]:
    """Fetch NHC's live active-storms feed. Field names match the official
    schema (nhc.noaa.gov/productexamples/NHC_Tropical_Cyclone_Status_JSON_File_Reference.pdf)."""
    resp = requests.get(NHC_CURRENT_STORMS_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    storms = []
    for s in data.get("activeStorms", []):
        storms.append(
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "classification": s.get("classification"),
                "lat": s.get("latitude_numeric"),
                "lon": s.get("longitude_numeric"),
                "intensity_kt": s.get("intensity"),
                "pressure_mb": s.get("pressure"),
                "movement_dir_deg": s.get("movementDir"),
                "movement_speed_mph": s.get("movementSpeed"),
                "last_update": s.get("lastUpdate"),
            }
        )
    return storms
