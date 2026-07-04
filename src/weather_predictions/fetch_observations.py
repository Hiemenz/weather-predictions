"""Fetch the latest observations from NWS and append them to local storage.

Meant to be run repeatedly (e.g. every few hours via cron/launchd). Safe to
run as often as you like — inserts are deduplicated on (station_id, timestamp).
NWS only retains a rolling ~1-2 day window of raw observations per station, so
running this on a gap longer than that window will silently miss data.

Also derives a live daily aggregate for the last couple of days and stores it
in `daily_observations`, since CDO/GHCND/LCD data all lag a few days before
they're published — this keeps "today"/"yesterday" available for feature
engineering in the meantime. Temp/precip/rain only fill gaps GHCND hasn't
covered yet; humidity/pressure/wind always get filled in live, since GHCND
never provides those at all (LCD is the only other source, and lags too).
"""

from __future__ import annotations

import logging

from weather_predictions.config import STATION_ID
from weather_predictions.features import compute_live_daily_aggregate, raw_to_frame
from weather_predictions.nws_client import get_observations
from weather_predictions.storage import (
    count_observations,
    fetch_all_observations,
    upsert_daily_enrichment,
    upsert_daily_from_metar,
    upsert_observations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run(station_id: str = STATION_ID) -> int:
    records = get_observations(station_id, limit=500)
    inserted = upsert_observations(records)

    raw_df = raw_to_frame(fetch_all_observations())
    daily_rows = compute_live_daily_aggregate(raw_df)
    daily_inserted = upsert_daily_from_metar(daily_rows)
    upsert_daily_enrichment(daily_rows)

    total = count_observations()
    log.info(
        "fetched=%d new=%d total_stored=%d daily_gap_filled=%d station=%s",
        len(records),
        inserted,
        total,
        daily_inserted,
        station_id,
    )
    return inserted


if __name__ == "__main__":
    run()
