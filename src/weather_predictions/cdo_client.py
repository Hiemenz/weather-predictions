"""Client for NOAA's Climate Data Online (CDO) API v2.

Used for bulk historical backfill (GHCND daily summaries), since api.weather.gov
only retains a rolling window of raw observations. Requires a free token from
https://www.ncdc.noaa.gov/cdo-web/token, set as NOAA_CDO_TOKEN in .env.

Docs: https://www.ncei.noaa.gov/cdo-web/webservices/v2
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import requests

from weather_predictions.config import CDO_API_BASE, CDO_TOKEN

_TIMEOUT = 30
_PAGE_LIMIT = 1000  # CDO API max results per request
_MAX_RANGE_DAYS = 365  # CDO API caps date ranges per request to one year

# GHCND datatypes we care about for daily temp/precip modeling.
DATATYPES = ["TMAX", "TMIN", "PRCP"]


class CDOClientError(RuntimeError):
    pass


class CDOTokenMissingError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if not CDO_TOKEN:
        raise CDOTokenMissingError(
            "NOAA_CDO_TOKEN is not set. Get a free token at "
            "https://www.ncdc.noaa.gov/cdo-web/token and put it in a .env file "
            "as NOAA_CDO_TOKEN=<token>."
        )
    return {"token": CDO_TOKEN}


def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    resp = requests.get(f"{CDO_API_BASE}{path}", headers=_headers(), params=params, timeout=_TIMEOUT)
    if resp.status_code == 429:
        raise CDOClientError("CDO API rate limit hit (max ~5 req/sec, 10,000 req/day).")
    if not resp.ok:
        raise CDOClientError(f"GET {resp.url} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def find_stations_near(lat: float, lon: float, radius_deg: float = 0.25) -> list[dict[str, Any]]:
    """List GHCND stations within a bounding box around a point, for verifying station IDs."""
    extent = f"{lat - radius_deg},{lon - radius_deg},{lat + radius_deg},{lon + radius_deg}"
    data = _get(
        "/stations",
        {"datasetid": "GHCND", "extent": extent, "limit": 25, "sortfield": "datacoverage", "sortorder": "desc"},
    )
    return data.get("results", [])


def iter_date_chunks(start: date, end: date):
    """CDO caps date ranges per request to one year — split [start, end] accordingly."""
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=_MAX_RANGE_DAYS - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def get_daily_summaries_chunk(
    station_id: str,
    start: date,
    end: date,
    datatypes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch GHCND daily summaries for a single chunk (<=1 year), paginating as needed.

    Returns raw CDO records: one row per (date, datatype), e.g.
    {"date": "2020-01-01T00:00:00", "datatype": "TMAX", "station": "...", "value": 89}
    Values come back in the units passed (we always request "metric": °C, mm).
    """
    datatypes = datatypes or DATATYPES
    results_out: list[dict[str, Any]] = []
    offset = 1
    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": datatypes,
            "startdate": start.isoformat(),
            "enddate": end.isoformat(),
            "units": "metric",
            "limit": _PAGE_LIMIT,
            "offset": offset,
        }
        data = _get("/data", params)
        results = data.get("results", [])
        results_out.extend(results)

        count = data.get("metadata", {}).get("resultset", {}).get("count", 0)
        if offset + len(results) > count or not results:
            break
        offset += _PAGE_LIMIT
        time.sleep(0.25)  # stay well under the ~5 req/sec rate limit

    return results_out


def get_daily_summaries(
    station_id: str,
    start: date,
    end: date,
    datatypes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch GHCND daily summaries for a station over [start, end], inclusive.

    Convenience wrapper over get_daily_summaries_chunk for small ranges/tests.
    For bulk backfill, prefer driving iter_date_chunks + get_daily_summaries_chunk
    directly so progress can be persisted incrementally.
    """
    all_results: list[dict[str, Any]] = []
    for chunk_start, chunk_end in iter_date_chunks(start, end):
        all_results.extend(get_daily_summaries_chunk(station_id, chunk_start, chunk_end, datatypes))
    return all_results


def pivot_daily_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Reshape [{"date":..., "datatype": "TMAX", "value": ...}, ...] into
    {date_str: {"TMAX": value, "TMIN": value, "PRCP": value}}.
    """
    by_date: dict[str, dict[str, float]] = {}
    for rec in records:
        day = rec["date"][:10]
        by_date.setdefault(day, {})[rec["datatype"]] = rec["value"]
    return by_date
