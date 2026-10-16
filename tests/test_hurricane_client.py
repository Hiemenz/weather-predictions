"""Tests for HURDAT2 parsing and the live NHC feed parser — no network access.

The HURDAT2 sample below is a trimmed, realistic two-storm excerpt (real
header/fix formatting, invented positions). The live-feed fixture matches
the exact schema from NHC's own "Tropical Cyclone Status JSON File
Reference" (field names verified against that doc, not guessed).
"""

from __future__ import annotations

from weather_predictions.hurricane_client import get_active_storms, parse_hurdat2

_SAMPLE_HURDAT2 = """\
AL011851,              UNNAMED,     3,
18510625, 0000,  , HU, 28.0N,  94.8W,  80,-999,
18510625, 0600,  , HU, 28.1N,  95.4W,  80,-999,
18510625, 1200,  , HU, 28.2N,  96.0W,  70, 985,
AL092021,                 IDA,     2,
20210829, 1200,  , HU, 29.1N,  90.2W, 130, 930,
20210829, 1800, L, HU, 29.5N,  90.7W, 130, 931,
"""


def test_parse_hurdat2_header_and_fixes():
    records = parse_hurdat2(_SAMPLE_HURDAT2)
    assert len(records) == 5

    first = records[0]
    assert first["storm_id"] == "AL011851"
    assert first["name"] == "UNNAMED"
    assert first["timestamp"] == "1851-06-25T00:00:00+00:00"
    assert first["lat"] == 28.0
    assert first["lon"] == -94.8
    assert first["max_wind_kt"] == 80
    assert first["min_pressure_mb"] is None  # -999 sentinel -> None
    assert first["status"] == "HU"

    third = records[2]
    assert third["min_pressure_mb"] == 985

    ida_fixes = [r for r in records if r["storm_id"] == "AL092021"]
    assert len(ida_fixes) == 2
    assert ida_fixes[0]["name"] == "IDA"
    assert ida_fixes[1]["lat"] == 29.5
    assert ida_fixes[1]["lon"] == -90.7


def test_get_active_storms_parses_official_schema(monkeypatch):
    fixture = {
        "activeStorms": [
            {
                "id": "al112017",
                "binNumber": "AT1",
                "name": "Irma",
                "classification": "HU",
                "intensity": 125,
                "pressure": 941,
                "latitude": "22.9N",
                "longitude": "79.9W",
                "latitude_numeric": 22.9,
                "longitude_numeric": -79.9,
                "movementDir": 280,
                "movementSpeed": 9,
                "lastUpdate": "2017-09-09T16:00:00.000Z",
            }
        ]
    }

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return fixture

    monkeypatch.setattr("weather_predictions.hurricane_client.requests.get", lambda *a, **k: _FakeResponse())

    storms = get_active_storms()
    assert len(storms) == 1
    storm = storms[0]
    assert storm["id"] == "al112017"
    assert storm["name"] == "Irma"
    assert storm["lat"] == 22.9
    assert storm["lon"] == -79.9
    assert storm["intensity_kt"] == 125
    assert storm["pressure_mb"] == 941
    assert storm["movement_dir_deg"] == 280
    assert storm["movement_speed_mph"] == 9
    assert storm["last_update"] == "2017-09-09T16:00:00.000Z"


def test_get_active_storms_empty_feed(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"activeStorms": []}

    monkeypatch.setattr("weather_predictions.hurricane_client.requests.get", lambda *a, **k: _FakeResponse())
    assert get_active_storms() == []
