"""Push the MRMS radar image to a Waveshare 5.65" ACeP 7-color e-Paper display.

Renders the current MRMS regional radar frame with motion arrows (mrms_image.py)
and pushes it to the panel via the Waveshare driver library. Degrades
gracefully at every step:
  - If no MRMS frames are available, falls back to the NEXRAD station grid.
  - If the Waveshare driver isn't installed (not on the Pi), saves the image to
    disk and logs a warning instead of crashing.

Waveshare driver install (Pi only):
  pip install waveshare-epaper
  # or clone https://github.com/waveshare/e-Paper and copy the IT8951 libs

Called from the cron job installed by scripts/install_cron.sh.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from weather_predictions.config import LATITUDE, LONGITUDE, MRMS_DATA_DIR, RADAR_DATA_DIR

log = logging.getLogger(__name__)

_DEFAULT_OUTPUT = MRMS_DATA_DIR / "eink_radar.png"
_FALLBACK_OUTPUT = RADAR_DATA_DIR / "eink_radar.png"


def _render_mrms(radius_km: float, output_path: Path) -> str | None:
    """Render via MRMS; return frame timestamp on success, None on failure."""
    try:
        from weather_predictions.mrms_image import render

        result = render(radius_km=radius_km, output_path=output_path)
        return result.frame_timestamp
    except Exception as e:
        log.warning("MRMS render failed (%s), trying NEXRAD fallback", e)
        return None


def _render_nexrad(radius_km: float, output_path: Path) -> str | None:
    """Render via single-station NEXRAD grid; return timestamp or None."""
    try:
        from weather_predictions.radar_image import OutOfRadarRangeError, render

        result = render(radius_km=radius_km, output_path=output_path)
        return result.frame_timestamp
    except Exception as e:
        log.warning("NEXRAD render also failed (%s); nothing to display", e)
        return None


def _push_to_panel(image_path: Path) -> bool:
    """Push a PNG to the Waveshare ACeP panel. Returns True if sent."""
    try:
        from PIL import Image
        from waveshare_epaper import epd5in65f

        epd = epd5in65f.EPD()
        epd.init()
        img = Image.open(image_path)
        epd.display(epd.getbuffer(img))
        epd.sleep()
        return True
    except ImportError:
        log.warning(
            "waveshare_epaper not installed — image saved to %s but not pushed to panel. "
            "Install on the Pi with: pip install waveshare-epaper",
            image_path,
        )
        return False
    except Exception as e:
        log.error("e-ink display push failed: %s", e)
        return False


def update_display(
    radius_km: float = 300.0,
    output_path: Path = _DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Render and push the latest radar image to the e-ink panel.

    Returns a status dict suitable for logging / the dashboard.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = _render_mrms(radius_km, output_path)
    source = "mrms"

    if timestamp is None:
        output_path = _FALLBACK_OUTPUT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = _render_nexrad(radius_km, output_path)
        source = "nexrad"

    if timestamp is None:
        return {"success": False, "reason": "no radar frames available"}

    pushed = _push_to_panel(output_path)
    result = {
        "success": True,
        "source": source,
        "frame_timestamp": timestamp,
        "image_path": str(output_path),
        "pushed_to_panel": pushed,
    }
    log.info("e-ink update: %s", result)
    return result
