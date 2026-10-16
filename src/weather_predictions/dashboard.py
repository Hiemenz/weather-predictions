"""FastAPI web dashboard served from the Pi (or Mac during development).

Exposes a single HTML page at / showing:
  - Radar image (MRMS regional render, auto-refreshed every 2 min)
  - Storm cells near home (distance, heading, speed)
  - Precipitation arrival estimate
  - Active NWS alerts
  - Latest 1-3 day rain/temp model prediction
  - Recent evaluation scores (tabular model + MRMS nowcast)

All data is assembled fresh on each page load — no background worker needed;
the Pi runs cron jobs that keep the underlying data files current.

Start with: `weather dashboard` or `uvicorn weather_predictions.dashboard:app`
Needs: `poetry install --with web display`
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def _require_fastapi() -> None:
    if not _FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: poetry install --with web")


def _radar_png_b64(radius_km: float = 300.0) -> str | None:
    """Render the latest MRMS image and return it as a base64 PNG string."""
    import tempfile
    try:
        from weather_predictions.mrms_image import render
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        render(radius_km=radius_km, output_path=tmp_path)
        data = tmp_path.read_bytes()
        tmp_path.unlink(missing_ok=True)
        return base64.b64encode(data).decode()
    except Exception as e:
        log.warning("radar render failed: %s", e)
        return None


def _collect_data(radius_km: float = 300.0) -> dict:
    from weather_predictions.config import LATITUDE, LONGITUDE

    data: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "lat": LATITUDE,
        "lon": LONGITUDE,
        "alerts": [],
        "arrival": None,
        "cells": [],
        "prediction": None,
        "nowcast_scores": [],
        "model_scores": [],
        "radar_png_b64": None,
    }

    # NWS alerts
    try:
        from weather_predictions.nws_client import get_active_alerts
        data["alerts"] = get_active_alerts(LATITUDE, LONGITUDE)
    except Exception as e:
        log.warning("alerts fetch failed: %s", e)

    # Arrival estimate
    try:
        from weather_predictions.mrms_home_check import estimate_arrival
        arr = estimate_arrival()
        data["arrival"] = {
            "rain_now": arr.rain_now,
            "arrival_lead_minutes": arr.arrival_lead_minutes,
            "arrival_at": arr.arrival_at,
            "leads": {str(int(k)): round(v, 1) for k, v in arr.reflectivity_by_lead.items()},
        }
    except Exception as e:
        log.warning("arrival estimate failed: %s", e)

    # Storm cells
    try:
        from weather_predictions.mrms_cells import detect_cells
        cells = detect_cells(radius_km=radius_km)
        data["cells"] = [
            {
                "distance_km": c.distance_km,
                "bearing": c.bearing,
                "speed_kmh": c.speed_kmh,
                "heading": c.heading,
                "approaching": c.approaching,
                "peak_dbz": round(c.peak_dbz, 1),
                "area_km2": c.area_km2,
            }
            for c in cells[:8]
        ]
    except Exception as e:
        log.warning("cell detection failed: %s", e)

    # Tabular model prediction
    try:
        from weather_predictions.predict import ModelNotTrainedError, NoUsableDataError, predict
        result = predict()
        data["prediction"] = {
            "as_of": result.as_of_date,
            "horizons": [
                {
                    "target_date": hp.target_date,
                    "horizon_days": hp.horizon_days,
                    "rain_probability": round(hp.rain_probability * 100),
                    "temp_max_c": round(hp.temp_max_pred_c, 1),
                    "temp_min_c": round(hp.temp_min_pred_c, 1),
                }
                for hp in result.horizons
            ],
        }
    except Exception as e:
        log.warning("prediction failed: %s", e)

    # MRMS nowcast evaluation scores (most recent per method)
    try:
        from weather_predictions.storage import fetch_all_radar_nowcast_performance
        rows = fetch_all_radar_nowcast_performance()
        mrms_rows = [r for r in rows if r["method"].startswith("mrms_")]
        seen: dict[tuple, dict] = {}
        for r in sorted(mrms_rows, key=lambda x: x["evaluated_at"]):
            seen[(r["method"], r["lead_minutes"])] = r
        data["nowcast_scores"] = list(seen.values())
    except Exception as e:
        log.warning("nowcast scores failed: %s", e)

    # Radar image (last so a slow render doesn't block the rest)
    data["radar_png_b64"] = _radar_png_b64(radius_km)

    return data


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="120">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weather Dashboard</title>
<style>
  :root {{ --bg:#0f1117; --card:#1a1d27; --text:#e2e8f0; --muted:#94a3b8;
           --green:#4ade80; --yellow:#facc15; --red:#f87171; --blue:#60a5fa; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:system-ui,sans-serif;
          padding:1rem; }}
  h1 {{ font-size:1.25rem; margin-bottom:1rem; color:var(--blue); }}
  h2 {{ font-size:.9rem; font-weight:600; color:var(--muted); text-transform:uppercase;
        letter-spacing:.05em; margin-bottom:.5rem; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
           gap:1rem; }}
  .card {{ background:var(--card); border-radius:.75rem; padding:1rem; }}
  .card.wide {{ grid-column:1/-1; }}
  .meta {{ font-size:.75rem; color:var(--muted); margin-bottom:.75rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th {{ text-align:left; color:var(--muted); font-weight:500; padding:.25rem .5rem; }}
  td {{ padding:.25rem .5rem; border-top:1px solid #2d3147; }}
  .badge {{ display:inline-block; padding:.1rem .4rem; border-radius:.25rem;
            font-size:.75rem; font-weight:600; }}
  .green {{ color:var(--green); }} .yellow {{ color:var(--yellow); }}
  .red {{ color:var(--red); }} .blue {{ color:var(--blue); }}
  img.radar {{ width:100%; border-radius:.5rem; }}
  .alert-item {{ padding:.4rem 0; border-bottom:1px solid #2d3147; font-size:.85rem; }}
  .alert-item:last-child {{ border-bottom:none; }}
</style>
</head>
<body>
<h1>Weather Dashboard</h1>
<p class="meta">Nashville, TN ({lat:.4f}, {lon:.4f}) &mdash; updated {generated_at}</p>
<div class="grid">

<div class="card">
  <h2>Precipitation Outlook</h2>
  {arrival_html}
</div>

<div class="card">
  <h2>Storm Cells</h2>
  {cells_html}
</div>

<div class="card">
  <h2>NWS Alerts</h2>
  {alerts_html}
</div>

<div class="card">
  <h2>3-Day Forecast (Model)</h2>
  {prediction_html}
</div>

<div class="card">
  <h2>MRMS Nowcast Skill</h2>
  {nowcast_html}
</div>

{radar_html}

</div>
</body>
</html>"""


def _render_arrival(arrival: dict | None) -> str:
    if arrival is None:
        return "<p class='meta'>No MRMS frames available yet.</p>"
    if arrival["rain_now"]:
        return "<p class='green'>&#9928; Raining at your location now.</p>"
    if arrival["arrival_lead_minutes"] is not None:
        m = arrival["arrival_lead_minutes"]
        at = arrival["arrival_at"] or ""
        dbz = arrival["leads"].get(str(int(m)), "?")
        return f"<p class='yellow'>&#9928; Rain arriving in ~{m:.0f} min (around {at[:19]}, {dbz} dBZ)</p>"
    leads = arrival["leads"]
    max_lead = max(int(k) for k in leads)
    return f"<p class='green'>&#9728; No rain expected within {max_lead} min.</p>"


def _render_cells(cells: list) -> str:
    if not cells:
        return "<p class='meta'>No storm cells detected.</p>"
    rows = "".join(
        f"<tr><td>{c['distance_km']:.0f}km {c['bearing']}</td>"
        f"<td>{'&#8599;' if c['approaching'] else '&#8600;'} {c['heading']} {c['speed_kmh']:.0f}km/h</td>"
        f"<td class='{'red' if c['peak_dbz']>=50 else 'yellow' if c['peak_dbz']>=35 else 'green'}'>"
        f"{c['peak_dbz']:.0f} dBZ</td></tr>"
        for c in cells
    )
    return f"<table><tr><th>Location</th><th>Motion</th><th>Intensity</th></tr>{rows}</table>"


def _render_alerts(alerts: list) -> str:
    if not alerts:
        return "<p class='green'>No active NWS alerts.</p>"
    return "".join(
        f"<div class='alert-item'><span class='red'>{a['event']}</span> &mdash; {a.get('headline','')[:80]}</div>"
        for a in alerts
    )


def _render_prediction(pred: dict | None) -> str:
    if pred is None:
        return "<p class='meta'>Model not trained yet. Run `weather train`.</p>"
    rows = "".join(
        f"<tr><td>{h['target_date']}</td>"
        f"<td class='{'red' if h['rain_probability']>=60 else 'yellow' if h['rain_probability']>=30 else 'green'}'>"
        f"{h['rain_probability']}%</td>"
        f"<td>{h['temp_max_c']:.0f}&deg; / {h['temp_min_c']:.0f}&deg;C</td></tr>"
        for h in pred["horizons"]
    )
    return f"<p class='meta'>As of {pred['as_of']}</p><table><tr><th>Date</th><th>Rain</th><th>Hi/Lo</th></tr>{rows}</table>"


def _render_nowcast(scores: list) -> str:
    if not scores:
        return "<p class='meta'>No scored nowcasts yet. Run `weather mrms-nowcast-evaluate`.</p>"
    rows = "".join(
        f"<tr><td>{s['method'].replace('mrms_','')}</td>"
        f"<td>{s['lead_minutes']:.0f} min</td>"
        f"<td>{s['mae_dbz']:.1f} dBZ</td>"
        f"<td>{s['csi']:.2f}</td>"
        f"<td>{s['n_samples']}</td></tr>"
        for s in scores
    )
    return f"<table><tr><th>Method</th><th>Lead</th><th>MAE</th><th>CSI</th><th>n</th></tr>{rows}</table>"


def _render_radar(b64: str | None) -> str:
    if b64 is None:
        return "<div class='card wide'><h2>Radar</h2><p class='meta'>No frames available yet.</p></div>"
    return f"<div class='card wide'><h2>Radar (MRMS)</h2><img class='radar' src='data:image/png;base64,{b64}'></div>"


def _build_html(data: dict) -> str:
    return _HTML_TEMPLATE.format(
        lat=data["lat"],
        lon=data["lon"],
        generated_at=data["generated_at"],
        arrival_html=_render_arrival(data["arrival"]),
        cells_html=_render_cells(data["cells"]),
        alerts_html=_render_alerts(data["alerts"]),
        prediction_html=_render_prediction(data["prediction"]),
        nowcast_html=_render_nowcast(data["nowcast_scores"]),
        radar_html=_render_radar(data["radar_png_b64"]),
    )


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="Weather Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _build_html(_collect_data())

    @app.get("/data")
    async def data_json() -> dict:
        d = _collect_data()
        d.pop("radar_png_b64", None)  # too large for the JSON endpoint
        return d

    @app.get("/radar.png")
    async def radar_image() -> Response:
        import tempfile
        from weather_predictions.mrms_image import render
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)
        render(output_path=tmp_path)
        data = tmp_path.read_bytes()
        tmp_path.unlink(missing_ok=True)
        return Response(content=data, media_type="image/png")
