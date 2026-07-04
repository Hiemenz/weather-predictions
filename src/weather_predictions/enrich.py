"""Enrich existing daily_observations rows with pressure/humidity/wind from LCD.

GHCND's daily summaries only carry temp/precip. Pressure and humidity trends
are the classic cheap predictors of incoming rain, so this fills them in from
NOAA's Local Climatological Data bulk files, one year at a time (persisting
after each year, same lesson learned from the CDO backfill being slow and
initially silent).
"""

from __future__ import annotations

import logging
from datetime import date

from weather_predictions.config import LCD_STATION_ID
from weather_predictions.lcd_client import LCDClientError, get_daily_pressure_humidity
from weather_predictions.storage import upsert_daily_enrichment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run(start: date, end: date | None = None, station_id: str = LCD_STATION_ID) -> int:
    end = end or date.today()
    years = list(range(start.year, end.year + 1))
    log.info("enriching %s from %d to %d (%d year(s))", station_id, years[0], years[-1], len(years))

    total_inserted = 0
    for i, year in enumerate(years, start=1):
        try:
            rows = get_daily_pressure_humidity(year, station_id)
        except LCDClientError as e:
            log.warning("year %d/%d (%d): skipped — %s", i, len(years), year, e)
            continue
        rows = [r for r in rows if start.isoformat() <= r["date"] <= end.isoformat()]
        inserted = upsert_daily_enrichment(rows)
        total_inserted += inserted
        log.info("year %d/%d (%d): %d day-records, %d upserted", i, len(years), year, len(rows), inserted)

    log.info("enrichment complete: %d rows upserted total", total_inserted)
    return total_inserted


if __name__ == "__main__":
    import sys

    start_arg = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2005, 1, 1)
    run(start_arg)
