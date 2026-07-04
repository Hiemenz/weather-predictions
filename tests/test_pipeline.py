"""Smoke tests for the feature/train/predict/evaluate pipeline using synthetic data.

These don't touch real accumulated data — they verify the modeling code runs
end-to-end and produces sane shapes/types, not that predictions are accurate
(accuracy depends on real accumulated history).
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

import pytest

from weather_predictions import evaluate as evaluate_module
from weather_predictions import train as train_module
from weather_predictions.features import (
    build_daily_features,
    build_training_frame,
    compute_live_daily_aggregate,
    raw_to_frame,
)


def _synthetic_daily_records(n_days: int = 40, start: date = date(2026, 1, 1)) -> list[dict]:
    random.seed(0)
    records = []
    for i in range(n_days):
        raining = random.random() < 0.3
        tmax = 15 + 10 * random.uniform(-1, 1)
        records.append(
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "source": "ghcnd",
                "temp_max_c": tmax,
                "temp_min_c": tmax - random.uniform(3, 10),
                "precip_mm": random.uniform(1, 10) if raining else 0.0,
                "rain": int(raining),
            }
        )
    return records


def _synthetic_raw_observations(n_days: int = 2) -> list[dict]:
    random.seed(1)
    records = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(n_days):
        for hour in range(0, 24, 3):
            ts = start + timedelta(days=day, hours=hour)
            records.append(
                {
                    "station_id": "TEST",
                    "timestamp": ts.isoformat(),
                    "temperature_c": 10 + random.uniform(-5, 5),
                    "precip_last_hour_mm": random.choice([0.0, 0.0, 0.0, 2.0]),
                }
            )
    return records


def test_compute_live_daily_aggregate():
    raw_df = raw_to_frame(_synthetic_raw_observations(2))
    rows = compute_live_daily_aggregate(raw_df)
    assert len(rows) >= 2
    assert {"date", "source", "temp_max_c", "temp_min_c", "precip_mm", "rain"} <= rows[0].keys()
    assert all(r["source"] == "metar_live" for r in rows)


def test_build_daily_features_shapes():
    daily = build_daily_features(_synthetic_daily_records(40))
    assert len(daily) == 40
    assert set(daily["rain"].unique()) <= {0, 1}


def test_build_training_frame_per_horizon():
    daily = build_daily_features(_synthetic_daily_records(40))
    for horizon in (1, 2, 3):
        X, y_rain, y_tmax, y_tmin = build_training_frame(daily, horizon)
        # Loses 1 row for rain_yesterday lag at the start, `horizon` rows for the target at the end.
        assert len(X) == len(daily) - 1 - horizon
        assert len(X) == len(y_rain) == len(y_tmax) == len(y_tmin)


def test_train_end_to_end(monkeypatch, tmp_path):
    records = _synthetic_daily_records(40)
    monkeypatch.setattr(train_module, "fetch_all_daily", lambda: records)
    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    result = train_module.train(min_training_days=14)

    assert (tmp_path / "model.joblib").exists()
    assert len(result.horizons) == 3
    for hr in result.horizons:
        assert 0.0 <= hr.rain_accuracy <= 1.0
        assert hr.temp_max_mae >= 0.0


def test_train_raises_when_not_enough_data(monkeypatch, tmp_path):
    records = _synthetic_daily_records(5)
    monkeypatch.setattr(train_module, "fetch_all_daily", lambda: records)
    monkeypatch.setattr(train_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    with pytest.raises(train_module.NotEnoughDataError):
        train_module.train(min_training_days=14)


def test_evaluate_scores_predictions_with_known_outcomes(monkeypatch):
    daily = _synthetic_daily_records(10, start=date(2026, 1, 1))
    predictions = [
        {
            "predicted_date": "2026-01-05",
            "horizon_days": 1,
            "target_date": "2026-01-06",
            "rain_probability": 0.8,
            "rain_predicted": 1,
            "temp_max_pred_c": 20.0,
            "temp_min_pred_c": 10.0,
            "model_trained_at": "v1",
        },
        {
            # target_date not yet in daily -> should be counted as pending.
            "predicted_date": "2026-01-09",
            "horizon_days": 1,
            "target_date": "2026-02-01",
            "rain_probability": 0.5,
            "rain_predicted": 0,
            "temp_max_pred_c": 18.0,
            "temp_min_pred_c": 9.0,
            "model_trained_at": "v1",
        },
    ]
    monkeypatch.setattr(evaluate_module, "fetch_all_daily", lambda: daily)
    monkeypatch.setattr(evaluate_module, "fetch_all_predictions", lambda: predictions)
    monkeypatch.setattr(evaluate_module, "upsert_performance", lambda rows: None)

    scored, pending = evaluate_module.evaluate()

    assert pending == 1
    assert len(scored) == 1
    assert scored[0].n_samples == 1
    assert 0.0 <= scored[0].rain_accuracy <= 1.0
