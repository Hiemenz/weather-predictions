"""Tests for hurricane feature engineering — motion vector derivation and
horizon-target matching by nearest-timestamp-within-tolerance (not a naive
`.shift(-h)`, since HURDAT2 isn't perfectly 6-hourly)."""

from __future__ import annotations

import numpy as np
import pytest

from weather_predictions.geo import bearing_deg, haversine_km
from weather_predictions.hurricane_features import add_targets, build_storm_features, build_training_frame


def _fix(storm_id, ts, lat, lon, wind=60.0, pressure=990.0, status="HU"):
    return {
        "storm_id": storm_id,
        "name": "TEST",
        "timestamp": ts,
        "lat": lat,
        "lon": lon,
        "max_wind_kt": wind,
        "min_pressure_mb": pressure,
        "status": status,
    }


def test_build_storm_features_computes_motion_vector():
    fixes = [
        _fix("AL01", "2026-08-01T00:00:00+00:00", 20.0, -60.0),
        _fix("AL01", "2026-08-01T06:00:00+00:00", 20.0, -61.0),  # due west, 6h later
    ]
    df = build_storm_features(fixes)
    assert len(df) == 2
    assert np.isnan(df.iloc[0]["motion_speed_kmh"])  # first fix: no previous fix

    second = df.iloc[1]
    expected_bearing = bearing_deg(20.0, -60.0, 20.0, -61.0)
    expected_speed = haversine_km(20.0, -60.0, 20.0, -61.0) / 6
    recovered_bearing = np.degrees(np.arctan2(second["motion_bearing_sin"], second["motion_bearing_cos"])) % 360
    assert recovered_bearing == pytest.approx(expected_bearing, abs=1e-4)
    assert second["motion_speed_kmh"] == pytest.approx(expected_speed, rel=1e-6)
    assert second["storm_age_hours"] == 6.0


def test_add_targets_matches_nearest_fix_within_tolerance_despite_irregular_cadence():
    # A storm with an off-cadence extra fix (e.g. a landfall record) between
    # regular 6-hourly fixes — the +24h target for fix 0 should match fix 4
    # (also at +24h), not get thrown off by the irregular fix at +3h.
    fixes = [
        _fix("AL02", "2026-09-01T00:00:00+00:00", 15.0, -70.0, wind=50),
        _fix("AL02", "2026-09-01T03:00:00+00:00", 15.2, -70.5, wind=55),  # off-cadence
        _fix("AL02", "2026-09-01T06:00:00+00:00", 15.4, -71.0, wind=60),
        _fix("AL02", "2026-09-01T12:00:00+00:00", 16.0, -72.0, wind=70),
        _fix("AL02", "2026-09-02T00:00:00+00:00", 18.0, -75.0, wind=90),  # +24h from fix 0
    ]
    df = build_storm_features(fixes)
    labeled = add_targets(df, (24,))

    row0 = labeled.iloc[0]
    assert row0["lat_h24"] == 18.0
    assert row0["lon_h24"] == -75.0
    assert row0["wind_h24"] == 90

    # fix 1 (03:00) is only 21h from fix 4 (24:00 next day) -> outside the
    # 1.5h tolerance -> no match -> NaN, not a wrong nearby value.
    row1 = labeled.iloc[1]
    assert np.isnan(row1["lat_h24"])


def test_build_training_frame_drops_incomplete_rows():
    fixes = [
        _fix("AL03", "2026-09-01T00:00:00+00:00", 10.0, -50.0),
        _fix("AL03", "2026-09-01T06:00:00+00:00", 10.5, -51.0),
        _fix("AL03", "2026-09-01T12:00:00+00:00", 11.0, -52.0),
    ]
    df = build_storm_features(fixes)
    X, y_lat, y_lon, y_wind, timestamps = build_training_frame(df, 6)

    # First fix has no motion vector (dropped); last fix has no +6h target (dropped).
    assert len(X) == 1
    assert y_lat.iloc[0] == 11.0
    assert len(timestamps) == 1
