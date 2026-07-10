"""Tests for radar nowcasting: synthetic frames with a moving reflectivity
blob, so optical-flow extrapolation can be checked against ground truth
without needing real accumulated radar history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from weather_predictions import radar_nowcast, radar_nowcast_evaluate
from weather_predictions.radar_nowcast import (
    NO_ECHO_DBZ,
    InsufficientFramesError,
    load_recent_frames,
    optical_flow_forecast,
    persistence_forecast,
)
from weather_predictions.radar_processing import save_grid

_GRID_SIZE = 60
_BLOB_DBZ = 40.0


def _blob_frame(timestamp: datetime, x0: int) -> dict:
    grid = np.full((_GRID_SIZE, _GRID_SIZE), np.nan, dtype=np.float32)
    grid[20:35, x0 : x0 + 15] = _BLOB_DBZ
    return {
        "station": "KOHX",
        "timestamp": timestamp.isoformat(),
        "grid_km": 60,
        "resolution_km": 1,
        "latitude": 36.24,
        "longitude": -86.56,
        "reflectivity_dbz": grid,
    }


def test_load_recent_frames_requires_minimum_count(tmp_path):
    with pytest.raises(InsufficientFramesError):
        load_recent_frames(2, grid_dir=tmp_path)


def test_persistence_forecast_is_unchanged_latest_frame():
    frame = _blob_frame(datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc), x0=10)
    forecast = persistence_forecast(frame)
    assert forecast[25, 15] == _BLOB_DBZ
    assert forecast[0, 0] == NO_ECHO_DBZ


def test_optical_flow_forecast_tracks_moving_blob_better_than_persistence():
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    prev_frame = _blob_frame(t0, x0=5)
    curr_frame = _blob_frame(t1, x0=10)  # blob moved +5px in 5 minutes

    # Forecast another 5 minutes ahead -> blob should keep moving at the same rate.
    truth = _blob_frame(t1 + timedelta(minutes=5), x0=15)["reflectivity_dbz"]
    truth = np.nan_to_num(truth, nan=NO_ECHO_DBZ)

    flow_forecast = optical_flow_forecast(prev_frame, curr_frame, lead_minutes=5)
    persistence = persistence_forecast(curr_frame)

    flow_error = np.mean(np.abs(flow_forecast - truth))
    persistence_error = np.mean(np.abs(persistence - truth))

    assert flow_error < persistence_error


def test_optical_flow_forecast_rejects_non_increasing_timestamps():
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    frame = _blob_frame(t0, x0=5)
    with pytest.raises(InsufficientFramesError):
        optical_flow_forecast(frame, frame, lead_minutes=5)


def test_nowcast_saves_both_methods_and_records_metadata(tmp_path, monkeypatch):
    grid_dir = tmp_path / "grids"
    dest_dir = tmp_path / "nowcasts"
    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=5)
    save_grid(_blob_frame(t0, x0=5), grid_dir)
    save_grid(_blob_frame(t1, x0=10), grid_dir)

    recorded_rows = []
    monkeypatch.setattr(radar_nowcast, "upsert_radar_nowcasts", lambda rows: recorded_rows.extend(rows))

    result = radar_nowcast.nowcast(lead_minutes=5, grid_dir=grid_dir, dest_dir=dest_dir)

    assert set(result.grid_paths) == {"persistence", "optical_flow"}
    for path in result.grid_paths.values():
        assert Path(path).exists()
    assert len(recorded_rows) == 2
    assert {r["method"] for r in recorded_rows} == {"persistence", "optical_flow"}
    assert all(r["lead_minutes"] == 5 for r in recorded_rows)


def test_radar_nowcast_evaluate_scores_matched_forecast_and_counts_pending(tmp_path, monkeypatch):
    grid_dir = tmp_path / "grids"
    t_actual = datetime(2026, 7, 4, 12, 30, 0, tzinfo=timezone.utc)
    save_grid(_blob_frame(t_actual, x0=20), grid_dir)

    predicted_dbz = np.nan_to_num(_blob_frame(t_actual, x0=20)["reflectivity_dbz"], nan=NO_ECHO_DBZ)
    predicted_path = tmp_path / "predicted.npz"
    np.savez_compressed(predicted_path, reflectivity_dbz=predicted_dbz)

    nowcasts = [
        {
            "predicted_at": "2026-07-04T12:25:00+00:00",
            "valid_at": t_actual.isoformat(),
            "lead_minutes": 5.0,
            "method": "optical_flow",
            "station": "KOHX",
            "grid_path": str(predicted_path),
        },
        {
            # No matching actual grid exists for this valid_at -> pending.
            "predicted_at": "2026-07-04T13:00:00+00:00",
            "valid_at": "2026-07-04T13:30:00+00:00",
            "lead_minutes": 5.0,
            "method": "optical_flow",
            "station": "KOHX",
            "grid_path": str(predicted_path),
        },
    ]
    monkeypatch.setattr(radar_nowcast_evaluate, "fetch_all_radar_nowcasts", lambda: nowcasts)
    monkeypatch.setattr(radar_nowcast_evaluate, "upsert_radar_nowcast_performance", lambda rows: None)

    scored, pending = radar_nowcast_evaluate.evaluate(grid_dir=grid_dir)

    assert pending == 1
    assert len(scored) == 1
    assert scored[0].method == "optical_flow"
    assert scored[0].n_samples == 1
    assert scored[0].mae_dbz == pytest.approx(0.0, abs=1e-4)
    assert scored[0].csi == pytest.approx(1.0)
