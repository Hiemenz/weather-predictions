# weather-predictions

Collects weather observations from NOAA (`api.weather.gov` for live data,
Climate Data Online/GHCND for bulk history) for Nashville, TN, and trains a
model to predict rain probability and high/low temperature 1-3 days out.

## Data sources

- **CDO/GHCND (`weather backfill`)** — decades of finalized daily
  temp/precip history. This is what makes a *large* training dataset
  possible immediately, rather than waiting years for it to accumulate.
  Requires a free token: sign up at https://www.ncdc.noaa.gov/cdo-web/token
  (instant, email only) and put it in a `.env` file as `NOAA_CDO_TOKEN=...`.
  GHCND typically lags a few days behind real time before data is finalized.
- **NWS (`weather fetch`)** — live observations for the last ~1-2 days,
  used to fill the gap between "now" and whatever GHCND has finalized so
  far. Also used to fetch the official forecast for comparison.

Both feed into one `daily_observations` table (date, temp_max_c, temp_min_c,
precip_mm, rain). GHCND is authoritative and always wins if both sources
have a row for the same date; the NWS-derived row only fills the gap.

## Setup

```bash
poetry install
```

## Usage

```bash
poetry run weather backfill --start 2000-01-01   # bulk historical load from CDO (run once)
poetry run weather fetch                          # pull latest NWS observations (run on a schedule)
poetry run weather status                         # how much history is collected, ready to train?
poetry run weather train                           # train the 1/2/3-day rain + temp models
poetry run weather predict                         # predict next 3 days, stores predictions for scoring
poetry run weather evaluate                        # score past predictions against what actually happened
```

## Deployment: Pi collects, Mac trains

The intended setup is an always-on device (e.g. a Raspberry Pi) running
`weather fetch` on a schedule to keep the live gap filled, while backfill,
training, and prediction happen wherever you actually work (e.g. this Mac)
against a synced copy of the same sqlite database. See `scripts/README.md`
for the full setup, including the cron job (`scripts/install_cron.sh`) that
runs `weather fetch` every 3 hours on the collector device.

## The predict → evaluate feedback loop

Every `weather predict` run writes one row per horizon (1/2/3 days out) to
the `predictions` table, tagged with which model version (`trained_at`)
made it. Once real data for a `target_date` shows up in
`daily_observations` (from backfill or from the NWS gap-filler), `weather
evaluate` joins the two, scores rain accuracy/Brier score and temperature
MAE, and stores the result in `model_performance` — grouped by
`(model_trained_at, horizon_days)`. Run `weather predict` daily and
`weather evaluate` periodically (e.g. weekly, or after each retrain) to
build up a track record and see whether retraining on more data actually
improves skill over the previous model version, not just whether the model
beats the naive persistence baseline.

## How it works

- `nws_client.py` — wrapper around the NWS API (live observations, forecast).
- `cdo_client.py` — wrapper around NOAA CDO/GHCND (bulk historical daily data).
- `backfill.py` — pulls a date range of GHCND daily summaries and stores them.
- `storage.py` — SQLite storage: raw NWS observations, unified daily
  observations (source-tagged), predictions, and model performance history.
- `features.py` — turns `daily_observations` into a feature-engineered
  frame (rolling means, day-over-day trend, calendar seasonality) and
  builds the rain/temp-max/temp-min targets for each forecast horizon.
- `train.py` — trains one rain classifier + two temperature regressors
  (max, min) per horizon, using a time-ordered train/test split (no
  shuffling, to avoid leaking future days into training), evaluated
  against a naive "tomorrow looks like today" persistence baseline.
- `predict.py` — loads the trained model bundle, predicts 1/2/3 days out,
  stores the predictions, and fetches the official NWS forecast for
  comparison.
- `evaluate.py` — scores stored predictions against real outcomes once
  they're known, tracked per model version over time.

## Location

Configured in `src/weather_predictions/config.py` — currently Nashville, TN
(`STATION_ID` for live NWS data, `GHCND_STATION_ID` for CDO backfill, both
Nashville International Airport). Change `LATITUDE`, `LONGITUDE`, and both
station IDs there to point elsewhere.

## Caveats

- The NWS-derived daily aggregate approximates precipitation from the
  `precipitationLastHour` field, which METAR stations don't populate on
  every report — treat it as a lower bound. GHCND backfilled data doesn't
  have this issue since it's the finalized daily total.
- Model quality depends on how much real history has accumulated for each
  horizon (`weather status` shows this) — check `weather evaluate` results
  rather than assuming accuracy from training metrics alone.
