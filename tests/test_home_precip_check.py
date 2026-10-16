"""Tests for the home-coordinate radar check: pixel geolocation math and the
end-to-end check_home() flow against synthetic frames."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import weather_predictions.home_precip_check as home_precip_check_module
from weather_predictions.home_precip_check import OutOfRadarRangeError, check_home, reflectivity_at_point
from weather_predictions.radar_nowcast import InsufficientFramesError
from weather_predictions.radar_processing import save_grid

_ORIGIN_LAT, _ORIGIN_LON = 36.0, -87.0
_GRID_KM = 100.0


def _grid(n=101, value_at=None, value=45.0):
    dbz = np.full((n, n), -32.0)
    if value_at:
        dbz[value_at] = value
    return dbz


def test_reflectivity_at_point_returns_center_pixel_at_origin():
    n = 101
    dbz = _grid(n, value_at=(n // 2, n // 2))
    value = reflectivity_at_point(dbz, _ORIGIN_LAT, _ORIGIN_LON, _GRID_KM, _ORIGIN_LAT, _ORIGIN_LON)
    assert value == 45.0


def test_reflectivity_at_point_northeast_offset_hits_northeast_corner():
    n = 101
    dbz = _grid(n, value_at=(n - 1, n - 1))  # row=north edge, col=east edge (this module's convention)
    # A point exactly at the grid's edge, north and east of the origin ->
    # should map to the (n-1, n-1) corner pixel.
    ne_lat = _ORIGIN_LAT + _GRID_KM / 110.574
    ne_lon = _ORIGIN_LON + _GRID_KM / (111.320 * np.cos(np.radians(_ORIGIN_LAT)))
    value = reflectivity_at_point(dbz, _ORIGIN_LAT, _ORIGIN_LON, _GRID_KM, ne_lat, ne_lon)
    assert value == 45.0


def test_reflectivity_at_point_raises_when_out_of_range():
    dbz = _grid(101)
    with pytest.raises(OutOfRadarRangeError):
        reflectivity_at_point(dbz, _ORIGIN_LAT, _ORIGIN_LON, _GRID_KM, _ORIGIN_LAT + 5.0, _ORIGIN_LON)


def _blob_frame(timestamp, x0, size=60, station="KOHX"):
    grid = np.full((size, size), np.nan, dtype=np.float32)
    grid[20:35, x0 : x0 + 15] = 40.0
    return {
        "station": station,
        "timestamp": timestamp.isoformat(),
        "grid_km": 60,
        "resolution_km": 1,
        "latitude": _ORIGIN_LAT,
        "longitude": _ORIGIN_LON,
        "reflectivity_dbz": grid,
    }


def test_check_home_detects_rain_when_forecast_blob_covers_home(tmp_path, monkeypatch):
    grid_dir = tmp_path / "grids"
    t0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    # Blob sits over the grid's center column and is moving further east —
    # forecasting 5 more minutes ahead should still cover the origin (home).
    save_grid(_blob_frame(t0, x0=20), grid_dir)
    save_grid(_blob_frame(t1, x0=22), grid_dir)

    monkeypatch.setattr(home_precip_check_module, "GRID_DIR", grid_dir)
    result = check_home(lead_minutes=5, lat=_ORIGIN_LAT, lon=_ORIGIN_LON, grid_dir=grid_dir)

    assert result.lead_minutes == 5
    assert result.rain_expected is True
    assert result.reflectivity_dbz > 30.0  # home sits inside the blob in both frames


def test_check_home_no_rain_when_blob_is_far_from_home(tmp_path, monkeypatch):
    grid_dir = tmp_path / "grids"
    t0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    # Blob confined to the far edge of the grid, moving away from center.
    save_grid(_blob_frame(t0, x0=0), grid_dir)
    save_grid(_blob_frame(t1, x0=0), grid_dir)

    monkeypatch.setattr(home_precip_check_module, "GRID_DIR", grid_dir)
    result = check_home(lead_minutes=5, lat=_ORIGIN_LAT, lon=_ORIGIN_LON, grid_dir=grid_dir)

    assert result.rain_expected is False


def test_check_home_raises_insufficient_frames(tmp_path):
    with pytest.raises(InsufficientFramesError):
        check_home(grid_dir=tmp_path)
