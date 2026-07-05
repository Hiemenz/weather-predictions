"""Test that the trained hurricane track model actually beats the
straight-line-motion baseline on a systematically curving track — a
straight-line extrapolation is blind to a turn, but a model trained on many
storms sharing the same turning pattern should learn it."""

from __future__ import annotations

import numpy as np
import pytest

import weather_predictions.hurricane_train as hurricane_train_module
from weather_predictions.geo import destination_point
from weather_predictions.hurricane_train import NotEnoughDataError, train


def _curving_storm(storm_id: str, year: int, start_lat: float, start_lon: float) -> list[dict]:
    """A storm that turns 15 degrees to the right every 6 hours at constant
    speed — same pattern for every synthetic storm, so it's learnable."""
    lat, lon = start_lat, start_lon
    bearing = 270.0  # starts moving due west
    speed_kmh = 20.0
    fixes = []
    for step in range(16):
        ts = f"{year}-08-{1 + step // 4:02d}T{(step % 4) * 6:02d}:00:00+00:00"
        fixes.append(
            {
                "storm_id": storm_id,
                "name": "SYN",
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "max_wind_kt": 60.0,
                "min_pressure_mb": 985.0,
                "status": "HU",
            }
        )
        lat, lon = destination_point(lat, lon, bearing, speed_kmh * 6)
        bearing = (bearing + 15) % 360
    return fixes


def test_train_beats_straight_line_baseline_on_curving_tracks(tmp_path, monkeypatch):
    fixes = []
    for year in range(2015, 2030):
        for i, (start_lat, start_lon) in enumerate([(15.0, -40.0), (20.0, -50.0), (25.0, -60.0)]):
            fixes.extend(_curving_storm(f"AL{i}{year}", year, start_lat, start_lon))

    monkeypatch.setattr(hurricane_train_module, "fetch_all_hurricane_fixes", lambda: fixes)
    monkeypatch.setattr(hurricane_train_module, "HURRICANE_MODEL_PATH", tmp_path / "hurricane_model.joblib")
    monkeypatch.setattr(hurricane_train_module, "MODELS_DIR", tmp_path)

    result = train(min_samples=20, test_years=5)

    for hr in result.horizons:
        assert hr.track_error_km < hr.track_baseline_error_km, (
            f"h={hr.horizon_hours}: model {hr.track_error_km:.0f}km should beat "
            f"straight-line baseline {hr.track_baseline_error_km:.0f}km on a systematically curving track"
        )


def test_train_raises_when_not_enough_data(monkeypatch):
    import weather_predictions.hurricane_train as hurricane_train_module

    monkeypatch.setattr(hurricane_train_module, "fetch_all_hurricane_fixes", lambda: [])
    with pytest.raises(NotEnoughDataError):
        hurricane_train_module.train()
