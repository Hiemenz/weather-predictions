"""Thin client for the NOAA National Weather Service API (api.weather.gov).

No API key required. Docs: https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from weather_predictions.config import API_BASE, USER_AGENT

_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
_TIMEOUT = 30


class NWSClientError(RuntimeError):
    pass


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = path if path.startswith("http") else f"{API_BASE}{path}"
    resp = requests.get(url, headers=_HEADERS, params=params, timeout=_TIMEOUT)
    if not resp.ok:
        raise NWSClientError(f"GET {resp.url} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _value(field: dict[str, Any] | None) -> float | None:
    if not field:
        return None
    return field.get("value")


def get_point_metadata(lat: float, lon: float) -> dict[str, Any]:
    """Resolve a lat/lon into forecast office, gridpoint, and forecast URLs."""
    return _get(f"/points/{lat},{lon}")


def get_forecast(office_grid_url: str) -> dict[str, Any]:
    """Fetch the forecast for a gridpoint URL (from get_point_metadata)."""
    return _get(office_grid_url)


def parse_observation(feature: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single GeoJSON observation feature into a flat record."""
    props = feature["properties"]
    return {
        "station_id": props["stationId"],
        "timestamp": props["timestamp"],
        "text_description": props.get("textDescription"),
        "temperature_c": _value(props.get("temperature")),
        "dewpoint_c": _value(props.get("dewpoint")),
        "wind_direction_deg": _value(props.get("windDirection")),
        "wind_speed_kmh": _value(props.get("windSpeed")),
        "wind_gust_kmh": _value(props.get("windGust")),
        "barometric_pressure_pa": _value(props.get("barometricPressure")),
        "sea_level_pressure_pa": _value(props.get("seaLevelPressure")),
        "visibility_m": _value(props.get("visibility")),
        "max_temp_last_24h_c": _value(props.get("maxTemperatureLast24Hours")),
        "min_temp_last_24h_c": _value(props.get("minTemperatureLast24Hours")),
        "precip_last_hour_mm": _value(props.get("precipitationLastHour")),
        "precip_last_3h_mm": _value(props.get("precipitationLast3Hours")),
        "precip_last_6h_mm": _value(props.get("precipitationLast6Hours")),
        "relative_humidity_pct": _value(props.get("relativeHumidity")),
        "wind_chill_c": _value(props.get("windChill")),
        "heat_index_c": _value(props.get("heatIndex")),
    }


def get_observations(
    station_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Fetch observations for a station, optionally bounded by [start, end).

    NWS retains only a rolling window of raw observations per station
    (commonly ~1-2 days), regardless of how far back `start` is set.
    """
    params: dict[str, Any] = {"limit": limit}
    if start is not None:
        params["start"] = start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if end is not None:
        params["end"] = end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    data = _get(f"/stations/{station_id}/observations", params=params)
    return [parse_observation(f) for f in data.get("features", [])]


def get_latest_observation(station_id: str) -> dict[str, Any]:
    data = _get(f"/stations/{station_id}/observations/latest")
    return parse_observation(data)
