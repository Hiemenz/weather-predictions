"""Great-circle geodesy helpers shared by hurricane track features/training/
prediction/evaluation: distance and bearing between two points, and
projecting a point forward along a bearing (used for the straight-line
motion baseline track/intensity forecasting is compared against).

Implemented with plain numpy trig rather than a dependency — accurate
enough at storm-track scales (haversine/spherical-earth, not ellipsoidal).
All functions are elementwise, so they work on scalars or numpy arrays/
pandas Series interchangeably.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between (lat1, lon1) and (lat2, lon2)."""
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing (degrees from true north, 0-360) from point 1 to point 2."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def destination_point(lat, lon, bearing, distance_km):
    """Project (lat, lon) forward `distance_km` along `bearing` (degrees from true north).

    Returns (lat2, lon2). Used for the "storm keeps moving the way it's
    currently moving" baseline that the trained track model is compared
    against.
    """
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    theta = np.radians(bearing)
    delta = np.asarray(distance_km) / EARTH_RADIUS_KM

    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(theta))
    lon2 = lon1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(lat1), np.cos(delta) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), (np.degrees(lon2) + 540) % 360 - 180
