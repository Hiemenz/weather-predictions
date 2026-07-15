"""Push notifications via ntfy.sh — zero-infrastructure phone alerts.

Set NTFY_TOPIC in .env (pick any hard-to-guess string, e.g.
"weather-hiemenz-x7k2m") and subscribe to that topic in the ntfy app
(https://ntfy.sh, iOS/Android, free). Every call here is a no-op when
NTFY_TOPIC is unset, so alerting is strictly opt-in and nothing else in the
pipeline depends on it.
"""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

NTFY_TOPIC = os.getenv("NTFY_TOPIC")
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh")

_TIMEOUT = 15

# ntfy priority levels: 1=min ... 5=urgent.
PRIORITY_DEFAULT = "3"
PRIORITY_HIGH = "4"
PRIORITY_URGENT = "5"


def send_notification(message: str, title: str = "Weather", priority: str = PRIORITY_DEFAULT) -> bool:
    """POST a notification to the configured ntfy topic. Returns True if sent,
    False if NTFY_TOPIC is unset or the request failed (never raises — an
    alerting failure must not break the check that triggered it)."""
    if not NTFY_TOPIC:
        return False
    try:
        response = requests.post(
            f"{NTFY_SERVER}/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "cloud_with_rain"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("ntfy notification failed: %s", e)
        return False
