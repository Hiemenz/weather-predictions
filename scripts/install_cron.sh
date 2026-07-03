#!/bin/bash
# Run this ON THE DEVICE THAT SHOULD COLLECT DATA (e.g. the Raspberry Pi).
# Installs a crontab entry that runs `weather fetch` every 3 hours, so
# observation history accumulates continuously. Safe to re-run — it removes
# any previous entry for this project before adding the new one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_FILE="$PROJECT_ROOT/data/fetch.log"
MARKER="# weather-predictions:fetch"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "No virtualenv found at $VENV_PYTHON — run 'poetry install' first." >&2
    exit 1
fi

mkdir -p "$PROJECT_ROOT/data"

CRON_LINE="0 */3 * * * cd $PROJECT_ROOT && $VENV_PYTHON -m weather_predictions.fetch_observations >> $LOG_FILE 2>&1 $MARKER"

( crontab -l 2>/dev/null | grep -vF "$MARKER" ; echo "$CRON_LINE" ) | crontab -

echo "Installed cron job (every 3 hours)."
echo "Logs: $LOG_FILE"
echo "Uninstall with: scripts/uninstall_cron.sh"
