"""Tests for live hurricane forecasting — no network access, no active storm
required (mocks NHC's live feed with a synthetic snapshot)."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

import weather_predictions.hurricane_predict as hurricane_predict_module
from weather_predictions.hurricane_features import FEATURE_COLUMNS, FORECAST_HORIZONS_HOURS
from weather_predictions.hurricane_predict import ModelNotTrainedError, predict


class _FakeTrackReg:
    def __init__(self, lat, lon):
        self._lat, self._lon = lat, lon

    def predict(self, X):
        return np.array([[self._lat, self._lon]])


class _FakeWindReg:
    def __init__(self, wind):
        self._wind = wind

    def predict(self, X):
        return np.array([self._wind])


def _fake_bundle():
    return {
        "trained_at": "2026-01-01T00:00:00+00:00",
        "feature_columns": FEATURE_COLUMNS,
        "horizons": {h: {"track_reg": _FakeTrackReg(20.0 + h / 10, -70.0 - h / 10), "wind_reg": _FakeWindReg(80.0)} for h in FORECAST_HORIZONS_HOURS},
    }


def _synthetic_storm():
    return {
        "id": "al992026",
        "name": "Testphoon",
        "classification": "HU",
        "lat": 18.0,
        "lon": -65.0,
        "intensity_kt": 75,
        "pressure_mb": 970,
        "movement_dir_deg": 300,
        "movement_speed_mph": 12,
        "last_update": "2026-08-01T12:00:00.000Z",
    }


def test_predict_raises_when_model_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(hurricane_predict_module, "HURRICANE_MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(ModelNotTrainedError):
        predict()


def test_predict_returns_forecast_per_horizon_and_stores_rows(tmp_path, monkeypatch):
    model_path = tmp_path / "hurricane_model.joblib"
    joblib.dump(_fake_bundle(), model_path)
    monkeypatch.setattr(hurricane_predict_module, "HURRICANE_MODEL_PATH", model_path)
    monkeypatch.setattr(hurricane_predict_module, "get_active_storms", lambda: [_synthetic_storm()])

    recorded = []
    monkeypatch.setattr(hurricane_predict_module, "upsert_hurricane_predictions", lambda rows: recorded.extend(rows))

    forecasts = predict()

    assert len(forecasts) == 1
    forecast = forecasts[0]
    assert forecast.storm_id == "al992026"
    assert len(forecast.horizons) == len(FORECAST_HORIZONS_HOURS)
    assert len(recorded) == len(FORECAST_HORIZONS_HOURS)
    assert all(r["storm_id"] == "al992026" for r in recorded)


def test_predict_skips_storm_missing_position(tmp_path, monkeypatch):
    model_path = tmp_path / "hurricane_model.joblib"
    joblib.dump(_fake_bundle(), model_path)
    monkeypatch.setattr(hurricane_predict_module, "HURRICANE_MODEL_PATH", model_path)

    broken_storm = {**_synthetic_storm(), "lat": None}
    monkeypatch.setattr(hurricane_predict_module, "get_active_storms", lambda: [broken_storm])
    monkeypatch.setattr(hurricane_predict_module, "upsert_hurricane_predictions", lambda rows: None)

    assert predict() == []


def test_predict_no_active_storms_returns_empty(tmp_path, monkeypatch):
    model_path = tmp_path / "hurricane_model.joblib"
    joblib.dump(_fake_bundle(), model_path)
    monkeypatch.setattr(hurricane_predict_module, "HURRICANE_MODEL_PATH", model_path)
    monkeypatch.setattr(hurricane_predict_module, "get_active_storms", lambda: [])

    assert predict() == []
