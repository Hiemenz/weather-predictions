"""Hurricane + MRMS radar fusion.

When NHC reports active tropical cyclones, renders MRMS imagery centered on
each storm's current and forecast positions and produces optical-flow nowcasts
for the landfall-risk area. Only runs for storms within (or approaching) CONUS
MRMS coverage — offshore systems without continental radar coverage are skipped.

Typical use: run after `weather hurricane-predict` when a storm is active.
Needs `poetry install --with display` (OpenCV/Pillow for rendering).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weather_predictions.config import MRMS_DATA_DIR

log = logging.getLogger(__name__)

# MRMS CONUS lat/lon bounds (from mrms_processing.py grid specs).
_CONUS_LAT_MIN = 20.005
_CONUS_LAT_MAX = 54.995
_CONUS_LON_MIN = -129.995
_CONUS_LON_MAX = -60.005

# How far outside CONUS a storm's forecast position can be while still
# generating partial radar coverage worth rendering.
_CONUS_MARGIN_DEG = 5.0

# Region radius for storm-centred renders — wide enough to show the whole
# inner-core + feeder bands of a major hurricane.
DEFAULT_STORM_RADIUS_KM = 500.0

OUTPUT_DIR = MRMS_DATA_DIR / "hurricane"


@dataclass
class StormRadarResult:
    storm_id: str
    name: str
    rendered_positions: list[dict[str, Any]]  # one per horizon (current + forecasts)
    skipped_reason: str | None = None


def _within_conus(lat: float, lon: float, margin: float = _CONUS_MARGIN_DEG) -> bool:
    return (
        _CONUS_LAT_MIN - margin <= lat <= _CONUS_LAT_MAX + margin
        and _CONUS_LON_MIN - margin <= lon <= _CONUS_LON_MAX + margin
    )


def _render_position(
    lat: float,
    lon: float,
    label: str,
    output_path: Path,
    radius_km: float = DEFAULT_STORM_RADIUS_KM,
) -> dict[str, Any]:
    """Render + nowcast one position. Returns a status dict."""
    from weather_predictions.mrms_image import render
    from weather_predictions.mrms_nowcast import GRID_DIR, nowcast
    from weather_predictions.mrms_processing import OutOfMrmsRangeError
    from weather_predictions.radar_nowcast import InsufficientFramesError

    result: dict[str, Any] = {"label": label, "lat": lat, "lon": lon}

    try:
        render_result = render(
            radius_km=radius_km,
            center_lat=lat,
            center_lon=lon,
            output_path=output_path,
        )
        result["image_path"] = str(render_result.output_path)
        result["frame_timestamp"] = render_result.frame_timestamp
    except (InsufficientFramesError, OutOfMrmsRangeError) as e:
        result["image_path"] = None
        result["render_error"] = str(e)
        return result

    try:
        nowcast_result = nowcast(
            lead_minutes=60.0,
            center_lat=lat,
            center_lon=lon,
            radius_km=radius_km,
        )
        result["nowcast_valid_at"] = nowcast_result.valid_at
        result["nowcast_paths"] = nowcast_result.grid_paths
    except (InsufficientFramesError, OutOfMrmsRangeError) as e:
        result["nowcast_error"] = str(e)

    return result


def render_active_storms(radius_km: float = DEFAULT_STORM_RADIUS_KM) -> list[StormRadarResult]:
    """Render MRMS radar imagery and produce nowcasts for all currently active
    NHC tropical cyclones that are within or approaching CONUS coverage.

    Returns one StormRadarResult per active storm (skipped storms included with
    a skipped_reason explaining why they were not rendered).
    """
    from weather_predictions.hurricane_predict import ModelNotTrainedError, predict

    try:
        forecasts = predict()
    except ModelNotTrainedError:
        log.warning("hurricane model not trained — run `weather hurricane-train` first")
        return []

    if not forecasts:
        log.info("no active tropical cyclones")
        return []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[StormRadarResult] = []

    for forecast in forecasts:
        # Collect all positions: current (from as_of) + forecast horizons.
        positions: list[tuple[float, float, str]] = []

        # The `as_of` position is the current fix from the live NHC feed.
        # We don't have lat/lon for it directly here (hurricane_predict stores
        # predictions, not the live fix itself), so we start with the first
        # forecast horizon as the "current" position reference.
        for hp in forecast.horizons:
            label = f"t+{hp.horizon_hours}h ({hp.valid_at[:10]})"
            positions.append((hp.lat_pred, hp.lon_pred, label))

        # Skip if no forecast position is within CONUS + margin.
        near_conus = any(_within_conus(lat, lon) for lat, lon, _ in positions)
        if not near_conus:
            log.info("skipping %s (%s) — all forecast positions outside CONUS coverage", forecast.name, forecast.storm_id)
            results.append(
                StormRadarResult(
                    storm_id=forecast.storm_id,
                    name=forecast.name,
                    rendered_positions=[],
                    skipped_reason="all forecast positions outside CONUS MRMS coverage",
                )
            )
            continue

        rendered: list[dict[str, Any]] = []
        for lat, lon, label in positions:
            if not _within_conus(lat, lon):
                rendered.append({"label": label, "lat": lat, "lon": lon, "skipped": "outside CONUS"})
                continue

            safe_label = label.replace(" ", "_").replace(":", "").replace("+", "p")
            output_path = OUTPUT_DIR / f"{forecast.storm_id}_{safe_label}.png"
            pos_result = _render_position(lat, lon, label, output_path, radius_km)
            rendered.append(pos_result)
            log.info("%s %s: %s", forecast.name, label, pos_result.get("image_path", "failed"))

        results.append(
            StormRadarResult(
                storm_id=forecast.storm_id,
                name=forecast.name,
                rendered_positions=rendered,
            )
        )

    return results
