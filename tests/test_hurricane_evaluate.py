from __future__ import annotations

import pytest

import weather_predictions.hurricane_evaluate as hurricane_evaluate_module
from weather_predictions.hurricane_evaluate import evaluate


def test_evaluate_scores_matched_prediction_and_counts_pending(monkeypatch):
    predictions = [
        {
            "predicted_at": "2026-08-01T00:00:00+00:00",
            "storm_id": "AL01",
            "storm_name": "Test",
            "horizon_hours": 24,
            "valid_at": "2026-08-02T00:00:00+00:00",
            "lat_pred": 20.0,
            "lon_pred": -70.0,
            "wind_pred_kt": 80.0,
            "model_trained_at": "v1",
        },
        {
            # No fix exists near this valid_at -> pending.
            "predicted_at": "2026-08-05T00:00:00+00:00",
            "storm_id": "AL01",
            "storm_name": "Test",
            "horizon_hours": 24,
            "valid_at": "2026-08-06T00:00:00+00:00",
            "lat_pred": 22.0,
            "lon_pred": -72.0,
            "wind_pred_kt": 85.0,
            "model_trained_at": "v1",
        },
    ]
    fixes = [
        {
            "storm_id": "AL01",
            "timestamp": "2026-08-02T00:00:00+00:00",
            "lat": 20.0,
            "lon": -70.0,
            "max_wind_kt": 85.0,
        },
    ]
    monkeypatch.setattr(hurricane_evaluate_module, "fetch_all_hurricane_predictions", lambda: predictions)
    monkeypatch.setattr(hurricane_evaluate_module, "fetch_all_hurricane_fixes", lambda: fixes)
    monkeypatch.setattr(hurricane_evaluate_module, "upsert_hurricane_performance", lambda rows: None)

    scored, pending = evaluate()

    assert pending == 1
    assert len(scored) == 1
    result = scored[0]
    assert result.horizon_hours == 24
    assert result.n_samples == 1
    assert result.track_error_km == pytest.approx(0.0, abs=1e-6)
    assert result.wind_mae_kt == pytest.approx(5.0)
