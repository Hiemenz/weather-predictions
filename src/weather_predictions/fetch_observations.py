"""Fetch the latest observations from NWS and append them to local storage.

Meant to be run repeatedly (e.g. every few hours via cron/launchd). Safe to
run as often as you like — inserts are deduplicated on (station_id, timestamp).
NWS only retains a rolling ~1-2 day window of raw observations per station, so
running this on a gap longer than that window will silently miss data.
"""

from __future__ import annotations

import logging

from weather_predictions.config import STATION_ID
from weather_predictions.nws_client import get_observations
from weather_predictions.storage import count_observations, upsert_observations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run(station_id: str = STATION_ID) -> int:
    records = get_observations(station_id, limit=500)
    inserted = upsert_observations(records)
    total = count_observations()
    log.info(
        "fetched=%d new=%d total_stored=%d station=%s",
        len(records),
        inserted,
        total,
        station_id,
    )
    return inserted


if __name__ == "__main__":
    run()
