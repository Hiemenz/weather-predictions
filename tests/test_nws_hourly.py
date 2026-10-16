"""Tests for the NWS hourly-forecast parser — no network access."""

from __future__ import annotations

import pytest

from weather_predictions.nws_client import parse_hourly_periods

_FIXTURE = {
    "properties": {
        "periods": [
            {
                "number": 1,
                "startTime": "2026-08-01T14:00:00-05:00",
                "endTime": "2026-08-01T15:00:00-05:00",
                "isDaytime": True,
                "temperature": 86,
                "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 40},
                "windSpeed": "10 mph",
                "windDirection": "SW",
                "shortForecast": "Chance Showers And Thunderstorms",
            },
            {
                "number": 2,
                "startTime": "2026-08-01T15:00:00-05:00",
                "endTime": "2026-08-01T16:00:00-05:00",
                "isDaytime": True,
                "temperature": 88,
                "temperatureUnit": "F",
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": None},
                "windSpeed": "12 mph",
                "windDirection": "SW",
                "shortForecast": "Sunny",
            },
        ]
    }
}


def test_parse_hourly_periods_converts_fahrenheit_and_extracts_fields():
    periods = parse_hourly_periods(_FIXTURE, hours=24)
    assert len(periods) == 2

    first = periods[0]
    assert first["start_time"] == "2026-08-01T14:00:00-05:00"
    assert first["temperature_c"] == pytest.approx(30.0, abs=0.1)
    assert first["precip_probability_pct"] == 40
    assert first["wind_speed"] == "10 mph"
    assert first["short_forecast"] == "Chance Showers And Thunderstorms"

    second = periods[1]
    assert second["precip_probability_pct"] is None


def test_parse_hourly_periods_respects_hours_limit():
    periods = parse_hourly_periods(_FIXTURE, hours=1)
    assert len(periods) == 1
