"""One-off (or periodic) bulk historical backfill from NOAA CDO/GHCND.

Persists after each ~1-year chunk (rather than batching the whole range in
memory) so progress survives an interruption and is visible while it runs —
the CDO API can take several seconds per request, and a multi-year backfill
is a lot of requests.
"""

from __future__ import annotations

import logging
from datetime import date

from weather_predictions.cdo_client import get_daily_summaries_chunk, iter_date_chunks, pivot_daily_summaries
from weather_predictions.config import GHCND_STATION_ID, RAIN_THRESHOLD_MM
from weather_predictions.storage import upsert_daily_from_cdo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _to_daily_rows(records: list[dict]) -> list[dict]:
    pivoted = pivot_daily_summaries(records)
    rows = []
    for day_str, values in pivoted.items():
        precip = values.get("PRCP")
        rows.append(
            {
                "date": day_str,
                "source": "ghcnd",
                "temp_max_c": values.get("TMAX"),
                "temp_min_c": values.get("TMIN"),
                "precip_mm": precip,
                "rain": int(precip is not None and precip >= RAIN_THRESHOLD_MM),
            }
        )
    return rows


def run(start: date, end: date | None = None, station_id: str = GHCND_STATION_ID) -> int:
    end = end or date.today()
    chunks = list(iter_date_chunks(start, end))
    log.info("backfilling %s from %s to %s (%d chunk(s))", station_id, start, end, len(chunks))

    total_inserted = 0
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        records = get_daily_summaries_chunk(station_id, chunk_start, chunk_end)
        rows = _to_daily_rows(records)
        inserted = upsert_daily_from_cdo(rows)
        total_inserted += inserted
        log.info(
            "chunk %d/%d (%s to %s): %d day-records, %d upserted",
            i,
            len(chunks),
            chunk_start,
            chunk_end,
            len(rows),
            inserted,
        )

    log.info("backfill complete: %d rows upserted total", total_inserted)
    return total_inserted


if __name__ == "__main__":
    import sys

    start_arg = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2000, 1, 1)
    run(start_arg)
