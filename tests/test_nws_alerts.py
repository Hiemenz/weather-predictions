"""Tests for the NWS active-alerts parser — no network access."""

from __future__ import annotations

from weather_predictions.nws_client import get_active_alerts

_FIXTURE = {
    "features": [
        {
            "properties": {
                "id": "urn:oid:2.49.0.1.840.0.example",
                "event": "Severe Thunderstorm Warning",
                "severity": "Severe",
                "headline": "Severe Thunderstorm Warning issued for Davidson County",
                "areaDesc": "Davidson, TN",
                "effective": "2026-08-01T14:00:00-05:00",
                "expires": "2026-08-01T15:00:00-05:00",
            }
        }
    ]
}


def test_get_active_alerts_parses_features(monkeypatch):
    class _FakeResponse:
        ok = True
        url = "https://api.weather.gov/alerts/active"

        def json(self):
            return _FIXTURE

    monkeypatch.setattr("weather_predictions.nws_client.requests.get", lambda *a, **k: _FakeResponse())

    alerts = get_active_alerts(36.16, -86.78)
    assert len(alerts) == 1
    assert alerts[0]["event"] == "Severe Thunderstorm Warning"
    assert alerts[0]["severity"] == "Severe"
    assert alerts[0]["area_desc"] == "Davidson, TN"


def test_get_active_alerts_empty(monkeypatch):
    class _FakeResponse:
        ok = True
        url = "https://api.weather.gov/alerts/active"

        def json(self):
            return {"features": []}

    monkeypatch.setattr("weather_predictions.nws_client.requests.get", lambda *a, **k: _FakeResponse())
    assert get_active_alerts(36.16, -86.78) == []
