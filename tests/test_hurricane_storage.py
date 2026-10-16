"""Roundtrip tests for the new hurricane_* tables against a real (tmp) SQLite db."""

from __future__ import annotations

from weather_predictions.storage import (
    fetch_all_hurricane_fixes,
    fetch_all_hurricane_performance,
    fetch_all_hurricane_predictions,
    upsert_hurricane_fixes,
    upsert_hurricane_performance,
    upsert_hurricane_predictions,
)


def test_hurricane_fixes_roundtrip_and_replace(tmp_path):
    db_path = tmp_path / "test.sqlite"
    fix = {
        "storm_id": "AL01",
        "name": "Test",
        "timestamp": "2026-08-01T00:00:00+00:00",
        "lat": 20.0,
        "lon": -70.0,
        "max_wind_kt": 60.0,
        "min_pressure_mb": 995.0,
        "status": "HU",
    }
    assert upsert_hurricane_fixes([fix], db_path=db_path) == 1
    assert fetch_all_hurricane_fixes(db_path=db_path) == [fix]

    updated = {**fix, "max_wind_kt": 65.0}
    upsert_hurricane_fixes([updated], db_path=db_path)
    rows = fetch_all_hurricane_fixes(db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["max_wind_kt"] == 65.0


def test_hurricane_predictions_and_performance_roundtrip(tmp_path):
    db_path = tmp_path / "test.sqlite"
    prediction = {
        "predicted_at": "2026-08-01T00:00:00+00:00",
        "storm_id": "AL01",
        "storm_name": "Test",
        "horizon_hours": 24,
        "valid_at": "2026-08-02T00:00:00+00:00",
        "lat_pred": 21.0,
        "lon_pred": -71.0,
        "wind_pred_kt": 65.0,
        "model_trained_at": "v1",
    }
    assert upsert_hurricane_predictions([prediction], db_path=db_path) == 1
    assert fetch_all_hurricane_predictions(db_path=db_path) == [prediction]

    performance = {
        "evaluated_at": "2026-08-03T00:00:00+00:00",
        "model_trained_at": "v1",
        "horizon_hours": 24,
        "n_samples": 1,
        "track_error_km": 12.5,
        "wind_mae_kt": 3.0,
    }
    assert upsert_hurricane_performance([performance], db_path=db_path) == 1
    assert fetch_all_hurricane_performance(db_path=db_path) == [performance]
