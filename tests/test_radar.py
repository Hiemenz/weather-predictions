"""Tests for the radar pipeline that don't require network access or a real
NEXRAD file — decoding itself was verified manually against a live scan
(see the module docstrings), since it needs Py-ART and a real ~15MB file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weather_predictions.radar_client import list_scans
from weather_predictions.radar_processing import RadarProcessingError, load_grid, parse_scan_timestamp, save_grid


def test_parse_scan_timestamp():
    station, ts = parse_scan_timestamp(Path("KOHX20260704_124343_V06"))
    assert station == "KOHX"
    assert ts.isoformat() == "2026-07-04T12:43:43+00:00"


def test_parse_scan_timestamp_rejects_bad_filename():
    with pytest.raises(RadarProcessingError):
        parse_scan_timestamp(Path("not_a_radar_file.txt"))


def test_save_and_load_grid_roundtrip(tmp_path):
    frame = {
        "station": "KOHX",
        "timestamp": "2026-07-04T12:43:43+00:00",
        "grid_km": 200,
        "resolution_km": 1,
        "latitude": 36.24,
        "longitude": -86.56,
        "reflectivity_dbz": np.array([[1.0, np.nan], [3.5, -20.0]], dtype=np.float32),
    }
    saved_path = save_grid(frame, tmp_path)
    assert saved_path.exists()

    loaded = load_grid(saved_path)
    assert loaded["station"] == "KOHX"
    assert loaded["timestamp"] == frame["timestamp"]
    np.testing.assert_array_equal(loaded["reflectivity_dbz"], frame["reflectivity_dbz"])


def test_list_scans_filters_metadata_sidecars(monkeypatch):
    fake_ls_output = (
        "2026-07-04 12:01:09   15000000 KOHX20260704_120109_V06\n"
        "2026-07-04 12:01:20     600000 KOHX20260704_120109_V06_MDM\n"
        "2026-07-04 12:06:00   15100000 KOHX20260704_120600_V06\n"
    )

    class FakeResult:
        returncode = 0
        stdout = fake_ls_output
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    monkeypatch.setattr("weather_predictions.radar_client.shutil.which", lambda _: "/usr/bin/aws")

    from datetime import date

    keys = list_scans(date(2026, 7, 4))
    assert keys == [
        "2026/07/04/KOHX/KOHX20260704_120109_V06",
        "2026/07/04/KOHX/KOHX20260704_120600_V06",
    ]
