#!/bin/bash
# Run this ON THE DEVICE THAT SHOULD COLLECT DATA (e.g. the Raspberry Pi).
# Installs two crontab entries: `weather fetch` every 3 hours (tabular
# temp/precip/pressure/etc.) and `weather radar-fetch-raw` every 5 minutes
# (raw NEXRAD scans, no decoding — safe without the `radar` poetry group).
# Safe to re-run — it removes any previous entries for this project first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
FETCH_LOG="$PROJECT_ROOT/data/fetch.log"
RADAR_LOG="$PROJECT_ROOT/data/radar_fetch.log"
FETCH_MARKER="# weather-predictions:fetch"
RADAR_MARKER="# weather-predictions:radar-fetch-raw"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "No virtualenv found at $VENV_PYTHON — run 'poetry install' first." >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/data"

FETCH_LINE="0 */3 * * * cd $PROJECT_ROOT && $VENV_PYTHON -m weather_predictions.fetch_observations >> $FETCH_LOG 2>&1 $FETCH_MARKER"
RADAR_LINE="*/5 * * * * cd $PROJECT_ROOT && $VENV_PYTHON -m weather_predictions.radar_raw >> $RADAR_LOG 2>&1 $RADAR_MARKER"

( crontab -l 2>/dev/null | grep -vF "$FETCH_MARKER" | grep -vF "$RADAR_MARKER"; echo "$FETCH_LINE"; echo "$RADAR_LINE" ) | crontab -

echo "Installed cron jobs: weather fetch (every 3h), radar-fetch-raw (every 5min)."
echo "Radar note: every-5-minute raw scans are ~12-15MB each (~3.5GB/day) until"
echo "synced elsewhere and decoded — see scripts/README.md's Radar section."
echo "Logs: $FETCH_LOG, $RADAR_LOG"
echo "Uninstall with: scripts/uninstall_cron.sh"
