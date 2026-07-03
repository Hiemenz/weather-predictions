"""Smoke tests for the feature/train/predict pipeline using synthetic data.

These don't touch the real accumulated database — they verify the modeling
code runs end-to-end and produces sane shapes/types, not that predictions
are accurate (accuracy depends on real accumulated history).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from weather_predictions import train as train_module
from weather_predictions.features import build_daily_features, build_training_frame, raw_to_frame


def _synthetic_records(n_days: int = 30) -> list[dict]:
    random.seed(0)
    records = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(n_days):
        raining = random.random() < 0.3
        for hour in range(0, 24, 3):
            ts = start + timedelta(days=day, hours=hour)
            records.append(
                {
                    "station_id": "TEST",
                    "timestamp": ts.isoformat(),
                    "text_description": "Rain" if raining else "Clear",
                    "temperature_c": 10 + random.uniform(-5, 5),
                    "dewpoint_c": 5 + random.uniform(-5, 5),
                    "wind_direction_deg": random.uniform(0, 360),
                    "wind_speed_kmh": random.uniform(0, 20),
                    "wind_gust_kmh": random.uniform(0, 30),
                    "barometric_pressure_pa": 101000 + random.uniform(-500, 500),
                    "sea_level_pressure_pa": None,
                    "visibility_m": 16000,
                    "max_temp_last_24h_c": None,
                    "min_temp_last_24h_c": None,
                    "precip_last_hour_mm": random.uniform(0.5, 3) if raining else 0.0,
                    "precip_last_3h_mm": None,
                    "precip_last_6h_mm": None,
                    "relative_humidity_pct": random.uniform(30, 90),
                    "wind_chill_c": None,
                    "heat_index_c": None,
                }
            )
    return records


def test_build_daily_features_shapes():
    raw_df = raw_to_frame(_synthetic_records(30))
    daily = build_daily_features(raw_df)
    # UTC-to-Chicago conversion can split the first/last UTC day across an
    # extra local-date bucket, so allow for a boundary day.
    assert len(daily) in (30, 31)
    assert set(daily["rain_today"].unique()) <= {0, 1}


def test_build_training_frame_drops_unlabeled_tail():
    raw_df = raw_to_frame(_synthetic_records(30))
    daily = build_daily_features(raw_df)
    X, y = build_training_frame(daily)
    # Last day has no "tomorrow" label, first day has no "yesterday" lag -> dropped.
    assert len(X) == len(daily) - 2
    assert len(X) == len(y)


def test_train_end_to_end(monkeypatch, tmp_path):
    records = _synthetic_records(30)
    monkeypatch.setattr(train_module, "fetch_all_observations", lambda: records)
    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    result = train_module.train(min_training_days=14)

    assert (tmp_path / "model.joblib").exists()
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_train + result.n_test == result.n_days - 2


def test_train_raises_when_not_enough_data(monkeypatch, tmp_path):
    records = _synthetic_records(5)
    monkeypatch.setattr(train_module, "fetch_all_observations", lambda: records)
    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    with pytest.raises(train_module.NotEnoughDataError):
        train_module.train(min_training_days=14)
