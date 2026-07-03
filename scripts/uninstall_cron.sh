#!/bin/bash
# Removes the crontab entry installed by install_cron.sh.
set -euo pipefail

MARKER="# weather-predictions:fetch"

crontab -l 2>/dev/null | grep -vF "$MARKER" | crontab - || true

echo "Removed weather-predictions cron entry (if it existed)."
