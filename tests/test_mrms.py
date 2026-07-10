"""Tests for the MRMS pipeline that don't require network access or a real
GRIB2 file — decoding itself needs cfgrib/eccodes and a real file.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from weather_predictions.mrms_client import list_mrms_scans
from weather_predictions.mrms_processing import MrmsProcessingError, load_mrms_grid, parse_mrms_timestamp, save_mrms_grid


def test_parse_mrms_timestamp():
    ts = parse_mrms_timestamp(
        Path("MRMS_MergedReflectivityQCComposite_00.50_20260710-000040.grib2.gz")
    )
    assert ts == datetime(2026, 7, 10, 0, 0, 40, tzinfo=timezone.utc)


def test_parse_mrms_timestamp_without_gz():
    ts = parse_mrms_timestamp(
        Path("MRMS_MergedReflectivityQCComposite_00.50_20260710-123456.grib2")
    )
    assert ts == datetime(2026, 7, 10, 12, 34, 56, tzinfo=timezone.utc)


def test_parse_mrms_timestamp_rejects_bad_filename():
    with pytest.raises(MrmsProcessingError):
        parse_mrms_timestamp(Path("not_a_mrms_file.txt"))


def test_save_and_load_mrms_grid_roundtrip(tmp_path):
    frame = {
        "source": "MRMS_CONUS",
        "timestamp": "2026-07-10T00:00:40+00:00",
        "lat_min": 20.005,
        "lat_max": 54.995,
        "lon_min": -129.995,
        "lon_max": -60.005,
        "nlat": 3,
        "nlon": 4,
        "reflectivity_dbz": np.array(
            [[np.nan, 10.0, 20.0, 30.0], [5.0, np.nan, 25.0, 35.0], [0.0, 15.0, np.nan, 45.0]],
            dtype=np.float32,
        ),
    }
    saved_path = save_mrms_grid(frame, tmp_path)
    assert saved_path.exists()
    assert saved_path.name == "MRMS_CONUS_20260710T000040+0000.npz"

    loaded = load_mrms_grid(saved_path)
    assert loaded["source"] == "MRMS_CONUS"
    assert loaded["timestamp"] == frame["timestamp"]
    assert loaded["lat_min"] == pytest.approx(20.005)
    assert loaded["lon_min"] == pytest.approx(-129.995)
    assert loaded["nlat"] == 3
    assert loaded["nlon"] == 4
    np.testing.assert_array_equal(loaded["reflectivity_dbz"], frame["reflectivity_dbz"])


def test_list_mrms_scans_parses_keys(monkeypatch):
    fake_ls_output = (
        "2026-07-10 00:01:36    1594987 MRMS_MergedReflectivityQCComposite_00.50_20260710-000040.grib2.gz\n"
        "2026-07-10 00:03:30    1594124 MRMS_MergedReflectivityQCComposite_00.50_20260710-000239.grib2.gz\n"
        "2026-07-10 00:05:28    1583315 MRMS_MergedReflectivityQCComposite_00.50_20260710-000441.grib2.gz\n"
    )

    class FakeResult:
        returncode = 0
        stdout = fake_ls_output
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    monkeypatch.setattr("weather_predictions.mrms_client.shutil.which", lambda _: "/usr/bin/aws")

    keys = list_mrms_scans(date(2026, 7, 10))
    assert len(keys) == 3
    assert keys[0].endswith("MRMS_MergedReflectivityQCComposite_00.50_20260710-000040.grib2.gz")
    assert all("CONUS/MergedReflectivityQCComposite_00.50/20260710/" in k for k in keys)


def test_mrms_module_never_imports_cfgrib_at_top_level():
    """mrms_processing.py defers cfgrib import inside decode_mrms_grib2 so that
    load_mrms_grid/save_mrms_grid work without the mrms dep group installed."""
    import sys

    was_loaded_before = "cfgrib" in sys.modules
    import weather_predictions.mrms_processing  # noqa: F401

    if not was_loaded_before:
        assert "cfgrib" not in sys.modules
