# Deployment: Pi collects, Mac trains

This project is meant to run split across two machines:

- **Raspberry Pi (or any always-on Linux box)** — runs `weather fetch` on a
  schedule to continuously accumulate observation history into
  `data/observations.sqlite`.
- **Mac (or wherever you develop)** — runs `weather train` / `weather predict`
  against a copy of that same sqlite file.

## On the Pi

```bash
git clone <this-repo> && cd weather_predicitons
poetry install
scripts/install_cron.sh   # runs `weather fetch` every 3 hours via cron
```

Check it's working:

```bash
tail -f data/fetch.log
poetry run weather status
```

Uninstall with `scripts/uninstall_cron.sh`.

## Getting data to the Mac

`install_cron.sh` only handles fetching — it does not sync anything anywhere.
Copy `data/observations.sqlite` from the Pi to the same path in your Mac
checkout by whatever means you prefer (`scp`, `rsync`, Syncthing, a shared
drive, etc.) before running `weather train` or `weather predict` on the Mac.
A one-off pull looks like:

```bash
scp pi@<pi-host>:/path/to/weather_predicitons/data/observations.sqlite ./data/
```

## On the Mac

```bash
poetry install
poetry run weather status              # confirm the synced data shows up
poetry run weather backfill --start 2000-01-01   # one-time bulk temp/precip (needs NOAA_CDO_TOKEN in .env)
poetry run weather enrich --start 2000-01-01     # one-time bulk pressure/humidity/wind (no token needed)
poetry run weather train               # once enough history has accumulated
poetry run weather predict
```

`backfill`/`enrich` only need to be run once (plus occasionally again to
pick up newly-published years) — they're not part of the Pi's recurring
cron job, since they cover history, not the live gap.

## Radar: Pi downloads raw, Mac decodes

Radar decoding needs Py-ART, which pulls in Cartopy — Cartopy has no
prebuilt ARM wheels, so it's a poor fit for the Pi. The split:

- **Pi**: `poetry install` (default — the `radar` dependency group is
  optional and skipped), then `weather radar-fetch-raw` on a schedule.
  Needs only the `aws` CLI, no Python geospatial deps at all.
- **Mac**: `poetry install --with radar` to get Py-ART, then sync the raw
  files over (same idea as the sqlite sync above) and run
  `weather radar-decode-pending` to turn them into stored grids.

```bash
# On the Pi
scripts/install_cron.sh                 # add a radar-fetch-raw line, see below
poetry run weather radar-fetch-raw       # or run manually / via its own cron entry

# Sync raw scans to the Mac
rsync -av pi@<pi-host>:/path/to/weather_predicitons/data/radar/raw/ ./data/radar/raw/

# On the Mac
poetry install --with radar
poetry run weather radar-decode-pending  # decodes + deletes the synced raw files
```

`install_cron.sh` only wires up the tabular `weather fetch` job today —
add a second crontab line for `radar-fetch-raw` at whatever cadence you
want (scans arrive every ~5 minutes; every 5-10 minutes matches that,
every hour is enough if you just want a lower-cost archive).
