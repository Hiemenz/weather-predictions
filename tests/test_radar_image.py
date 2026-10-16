"""Tests for e-ink radar image rendering: dBZ->color quantization, region
cropping math, and an end-to-end render against synthetic frames."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from weather_predictions.radar_image import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    OutOfRadarRangeError,
    PANEL_SIZE,
    _pixel_bounds,
    dbz_to_rgb,
    render,
)
from weather_predictions.radar_processing import save_grid

_ORIGIN_LAT, _ORIGIN_LON = 36.0, -87.0


def test_dbz_to_rgb_bins_match_expected_colors():
    dbz = np.array([[np.nan, 10.0, 25.0], [35.0, 45.0, 60.0]])
    rgb = dbz_to_rgb(dbz)
    assert tuple(rgb[0, 0]) == COLOR_WHITE  # NaN -> no echo -> white
    assert tuple(rgb[0, 1]) == COLOR_BLUE  # 10 dBZ -> light
    assert tuple(rgb[0, 2]) == COLOR_GREEN  # 25 dBZ -> moderate
    assert tuple(rgb[1, 0]) == COLOR_YELLOW  # 35 dBZ -> heavy
    assert tuple(rgb[1, 1]) == COLOR_ORANGE  # 45 dBZ -> very heavy
    assert tuple(rgb[1, 2]) == COLOR_RED  # 60 dBZ -> severe


def _frame(n=101, grid_km=100.0):
    return {
        "reflectivity_dbz": np.full((n, n), np.nan),
        "grid_km": grid_km,
        "latitude": _ORIGIN_LAT,
        "longitude": _ORIGIN_LON,
    }


def test_pixel_bounds_centered_on_origin_gives_symmetric_box():
    frame = _frame(n=101, grid_km=100.0)
    row_min, row_max, col_min, col_max = _pixel_bounds(frame, _ORIGIN_LAT, _ORIGIN_LON, radius_km=20.0)
    # Center of a 101-wide, 100km grid is index 50; +/-20km is +/-20% of the half-width.
    assert row_min < 50 < row_max
    assert col_min < 50 < col_max
    assert (row_max - row_min) == (col_max - col_min)  # square region


def test_pixel_bounds_raises_when_entirely_out_of_range():
    frame = _frame(n=101, grid_km=100.0)
    with pytest.raises(OutOfRadarRangeError):
        _pixel_bounds(frame, _ORIGIN_LAT + 5.0, _ORIGIN_LON, radius_km=10.0)


def _blob_frame(timestamp, x0, size=120, station="KOHX", grid_km=60.0):
    grid = np.full((size, size), np.nan, dtype=np.float32)
    grid[50:70, x0 : x0 + 20] = 45.0
    return {
        "station": station,
        "timestamp": timestamp.isoformat(),
        "grid_km": grid_km,
        "resolution_km": grid_km * 2 / size,
        "latitude": _ORIGIN_LAT,
        "longitude": _ORIGIN_LON,
        "reflectivity_dbz": grid,
    }


def test_render_produces_correctly_sized_panel_image(tmp_path):
    grid_dir = tmp_path / "grids"
    t0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    save_grid(_blob_frame(t0, x0=50), grid_dir)
    save_grid(_blob_frame(t1, x0=52), grid_dir)

    output_path = tmp_path / "radar.png"
    result = render(
        radius_km=30.0, center_lat=_ORIGIN_LAT, center_lon=_ORIGIN_LON, grid_dir=grid_dir, output_path=output_path
    )

    assert output_path.exists()
    with Image.open(output_path) as img:
        assert img.size == PANEL_SIZE
        assert img.mode == "RGB"
    assert result.region_radius_km == 30.0
