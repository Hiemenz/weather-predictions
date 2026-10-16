from __future__ import annotations

import pytest

from weather_predictions.geo import bearing_deg, destination_point, haversine_km


def test_haversine_km_known_distance():
    # Nashville (BNA) to Atlanta (ATL), roughly 340km great-circle.
    dist = haversine_km(36.1245, -86.6782, 33.6407, -84.4277)
    assert dist == pytest.approx(340, abs=15)


def test_haversine_km_zero_for_same_point():
    assert haversine_km(30.0, -80.0, 30.0, -80.0) == pytest.approx(0.0, abs=1e-9)


def test_bearing_deg_due_east():
    # Moving east along the equator: bearing should be ~90 degrees.
    assert bearing_deg(0.0, -80.0, 0.0, -79.0) == pytest.approx(90.0, abs=0.5)


def test_bearing_deg_due_north():
    assert bearing_deg(20.0, -80.0, 21.0, -80.0) == pytest.approx(0.0, abs=0.5)


def test_destination_point_round_trip():
    lat, lon = 25.0, -70.0
    dist_km = haversine_km(lat, lon, 26.0, -70.0)
    bearing = bearing_deg(lat, lon, 26.0, -70.0)
    dest_lat, dest_lon = destination_point(lat, lon, bearing, dist_km)
    assert dest_lat == pytest.approx(26.0, abs=0.05)
    assert dest_lon == pytest.approx(-70.0, abs=0.05)
