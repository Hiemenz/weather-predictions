# weather-predictions

Collects weather observations from NOAA's National Weather Service API
(`api.weather.gov`, no key required) for Nashville, TN, accumulates a local
history, and trains a model to predict next-day rain.

## Why a local history?

`api.weather.gov` only retains a rolling ~1-2 days of raw observations per
station — not enough to train on. This project runs a small fetch job on a
schedule (see below) that polls the API and appends new observations to a
local SQLite database, so real history builds up over time. Once at least
`MIN_TRAINING_DAYS` (default 14) days of daily-aggregated history exist,
training becomes possible.

## Setup

```bash
poetry install
```

## Usage

```bash
poetry run weather status    # how much history is collected, is training possible yet
poetry run weather fetch     # pull latest observations, store them (idempotent)
poetry run weather train     # train/retrain the rain/no-rain model
poetry run weather predict   # predict tomorrow's rain probability, vs. the NWS forecast
```

## Deployment: Pi collects, Mac trains

The intended setup is an always-on device (e.g. a Raspberry Pi) running
`weather fetch` on a schedule to accumulate history, while training and
prediction happen wherever you actually work (e.g. this Mac) against a
synced copy of the same sqlite database. See `scripts/README.md` for the
full setup, including the cron job (`scripts/install_cron.sh`) that runs
`weather fetch` every 3 hours on the collector device.

## How it works

- `nws_client.py` — thin wrapper around the NWS API (observations, forecast).
- `storage.py` — SQLite storage for raw observations, deduplicated on
  `(station_id, timestamp)` so re-running fetch is always safe.
- `features.py` — aggregates raw reports into one row per local day
  (temp min/max/mean, precip total, pressure trend, rolling features,
  calendar seasonality) and builds the "rain tomorrow" label.
- `train.py` — trains a `RandomForestClassifier` on a time-ordered
  train/test split (no shuffling, to avoid leaking future days into
  training) and compares it against a naive "tomorrow looks like today"
  persistence baseline.
- `predict.py` — loads the trained model, predicts from the latest day of
  features, and fetches the official NWS forecast for comparison.

## Location

Configured in `src/weather_predictions/config.py` — currently Nashville, TN
(station `KBNA`, Nashville International Airport). Change `LATITUDE`,
`LONGITUDE`, and `STATION_ID` there to point elsewhere.

## Caveats

- Precipitation totals are approximated from the `precipitationLastHour`
  field, which METAR stations don't populate on every report — treat it as
  a lower bound, not an exact total.
- Model quality depends entirely on how much history has accumulated. With
  only a couple weeks of data, expect it to do little better than the
  persistence baseline; it should improve as more seasons of data build up.
