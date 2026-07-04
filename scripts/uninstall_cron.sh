#!/bin/bash
# Removes the crontab entries installed by install_cron.sh.
set -euo pipefail

FETCH_MARKER="# weather-predictions:fetch"
RADAR_MARKER="# weather-predictions:radar-fetch-raw"

crontab -l 2>/dev/null | grep -vF "$FETCH_MARKER" | grep -vF "$RADAR_MARKER" | crontab - || true

echo "Removed weather-predictions cron entries (if they existed)."
