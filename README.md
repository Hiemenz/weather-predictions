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
- **NOAA LCD (`weather enrich`)** — decades of daily pressure/humidity/wind,
  which GHCND's daily summaries don't carry. Plain static CSV files, no
  token needed. Pressure trend is one of the cheapest, most useful signals
  for predicting incoming rain, so this is what pushed the t+2/t+3 rain
  classifier from losing to the persistence baseline to beating or matching
  it (see "Results" below). Also lags a few days behind real time.
- **NWS (`weather fetch`)** — live observations for the last ~1-2 days,
  used to fill the gap between "now" and whatever GHCND/LCD have finalized
  so far. Also used to fetch the official forecast for comparison.

All three feed into one `daily_observations` table. GHCND owns temp_max_c/
temp_min_c/precip_mm/rain (authoritative, always wins); LCD and the NWS
live aggregate both write humidity_pct/pressure_hpa/wind_speed_kmh, since
GHCND never provides those at all — whichever ran more recently wins for
those columns, since there's no real "authoritative" source to defer to.

- **NEXRAD Level II (`weather radar-fetch` / `radar-backfill`)** — raw radar
  reflectivity sweeps, not currently used by the tabular rain/temp model.
  This is the foundation for a *separate*, future radar-based nowcasting
  model (see "Radar" below) — a genuinely different kind of model (spatial
  image sequences, not daily tabular features).

## Setup

```bash
poetry install
```

Radar backfill/fetch also needs the AWS CLI on PATH (`brew install awscli` /
`apt install awscli`) — no AWS account or credentials required, the NEXRAD
archive bucket is public.

## Usage

```bash
poetry run weather backfill --start 2000-01-01   # bulk historical temp/precip from CDO (run once)
poetry run weather enrich --start 2000-01-01     # bulk historical pressure/humidity/wind from LCD (run once)
poetry run weather fetch                          # pull latest NWS observations (run on a schedule)
poetry run weather status                         # how much history is collected, ready to train?
poetry run weather train                           # train the 1/2/3-day rain + temp models
poetry run weather predict                         # predict next 3 days, stores predictions for scoring
poetry run weather evaluate                        # score past predictions against what actually happened

poetry run weather radar-fetch                     # download + decode the latest radar scan
poetry run weather radar-backfill START END        # e.g. 2026-07-04T00:00:00 2026-07-04T06:00:00
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

## Results (Nashville, 2005-2026, 7,854 days)

| Horizon | Rain accuracy (baseline) | Temp max MAE (baseline) | Temp min MAE (baseline) |
|---|---|---|---|
| t+1d | 0.65 (0.65 — tied) | 2.7°C (3.2°C) | 2.1°C (2.7°C) |
| t+2d | 0.56 (0.55 — **beats it**) | 3.6°C (4.4°C) | 3.0°C (3.9°C) |
| t+3d | 0.55 (0.57 — close) | 3.8°C (4.9°C) | 3.5°C (4.5°C) |

Before adding pressure/humidity/wind (`weather enrich`), t+2d/t+3d rain
accuracy was 0.53/0.51 — clearly losing to persistence. Pressure trend
closed most of that gap. Temperature forecasting beats the naive baseline
at every horizon either way.

## Radar

Raw NEXRAD Level II volume scans for KOHX (Nashville radar) come from
NOAA's public archive on AWS (`unidata-nexrad-level2`, no credentials
needed — free/open data). Decoding the lowest-elevation reflectivity sweep
with [Py-ART](https://arm-doe.github.io/pyart/) projects it onto a 400x400
grid (200km x 200km, 1km resolution) centered on the radar, saved as a
compressed `.npz` (~100-150KB vs. ~12-15MB raw) in `data/radar/grids/`.

Py-ART pulls in Cartopy, which has **no prebuilt ARM wheels** — a poor fit
for a Raspberry Pi. So collection and decoding are split, same idea as the
CDO/LCD/NWS split above:

- `weather radar-fetch-raw` — downloads only, no decoding. Needs just the
  `aws` CLI, nothing from the `radar` poetry group. This is what runs
  continuously on the Pi (`scripts/install_cron.sh` sets this up alongside
  the tabular `weather fetch`, every 5 minutes to match the scan cadence).
- `weather radar-decode-pending` — decodes whatever raw files it finds in
  `data/radar/raw/` (e.g. synced over from the Pi) and deletes them after.
  Needs `poetry install --with radar`. Run this wherever Py-ART is
  installed (the Mac).
- `weather radar-fetch` / `radar-backfill` — download + decode in one step,
  for convenience when running directly on a machine that has the `radar`
  group installed (i.e. not the Pi).

This is **data collection only** — there's no radar-based prediction model
yet. That would be a genuinely different model from the tabular one above:
a sequence of these reflectivity grids over time, fed into something like
optical-flow extrapolation or a ConvLSTM, to predict where precipitation
moves next (nowcasting). Building that requires first accumulating a real
time series of frames, the same way the tabular model needed accumulated
daily history before it could train.

## How it works

- `nws_client.py` — wrapper around the NWS API (live observations, forecast).
- `cdo_client.py` — wrapper around NOAA CDO/GHCND (bulk historical temp/precip).
- `lcd_client.py` — downloads NOAA LCD's per-year CSVs (bulk historical
  pressure/humidity/wind); shells out to `curl` for the actual download
  since some sandboxed environments throttle Python's own HTTP stack far
  below what curl gets for the same ~10MB files.
- `backfill.py` / `enrich.py` — drive the CDO/LCD clients over a date range,
  persisting after each year so progress survives an interruption instead
  of being lost if something goes wrong mid-run.
- `storage.py` — SQLite storage: raw NWS observations, unified daily
  observations (source-tagged, with GHCND/LCD/live-METAR column ownership
  kept separate so none of them can clobber a field they don't own),
  predictions, and model performance history.
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
- `radar_client.py` — lists/downloads NEXRAD volume scans from the public
  S3 archive via the `aws` CLI (faster and dependency-conflict-free vs.
  `boto3`/`s3fs` here).
- `radar_processing.py` — decodes a volume scan with Py-ART into a fixed
  Cartesian reflectivity grid, with save/load for the compressed `.npz`
  format frames are stored in.
- `radar.py` — orchestrates fetch/backfill: download → decode → save →
  delete the raw file, with progress logging per scan.

## Location

Configured in `src/weather_predictions/config.py` — currently Nashville, TN
(`STATION_ID` for live NWS data, `GHCND_STATION_ID` for CDO backfill,
`LCD_STATION_ID` for LCD enrichment, `RADAR_STATION_ID` for NEXRAD — all
Nashville, just identified differently by each NOAA system;
`RADAR_STATION_ID` is the actual radar site (KOHX), a few miles from the
airport station the others use). Change `LATITUDE`, `LONGITUDE`, and all
station IDs there to point elsewhere.

## Caveats

- The NWS-derived daily aggregate approximates precipitation from the
  `precipitationLastHour` field, which METAR stations don't populate on
  every report — treat it as a lower bound. GHCND backfilled data doesn't
  have this issue since it's the finalized daily total.
- Model quality depends on how much real history has accumulated for each
  horizon (`weather status` shows this) — check `weather evaluate` results
  rather than assuming accuracy from training metrics alone.
