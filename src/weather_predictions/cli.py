"""Command-line entrypoint: `weather fetch|backfill|enrich|radar-fetch-raw|radar-backfill-raw|radar-decode-pending|radar-fetch|radar-backfill|radar-nowcast|radar-nowcast-evaluate|radar-image|mrms-fetch|mrms-backfill|mrms-nowcast|mrms-image|storm-check|hurricane-backfill|hurricane-train|hurricane-predict|hurricane-evaluate|train|predict|evaluate|status`."""

from __future__ import annotations

from datetime import date, datetime

import typer

from weather_predictions.config import MIN_TRAINING_DAYS, MODEL_PATH
from weather_predictions.features import build_daily_features
from weather_predictions.predict import ModelNotTrainedError, NoUsableDataError, predict as run_predict
from weather_predictions.storage import fetch_all_daily
from weather_predictions.train import NotEnoughDataError, train as run_train

app = typer.Typer(help="Fetch NOAA weather data, train, and predict rain/temperature 1-3 days out.")


@app.command()
def fetch() -> None:
    """Pull the latest observations from NWS and store them locally."""
    from weather_predictions.fetch_observations import run

    inserted = run()
    typer.echo(f"Inserted {inserted} new observation(s).")


@app.command()
def backfill(
    start: str = typer.Option("2000-01-01", help="Start date, YYYY-MM-DD."),
    end: str = typer.Option(None, help="End date, YYYY-MM-DD (defaults to today)."),
) -> None:
    """Bulk-load historical daily data from NOAA CDO/GHCND."""
    from weather_predictions.backfill import run as run_backfill
    from weather_predictions.cdo_client import CDOTokenMissingError

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else None
    try:
        inserted = run_backfill(start_date, end_date)
    except CDOTokenMissingError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)
    typer.echo(f"Upserted {inserted} day(s) of historical data.")


@app.command()
def enrich(
    start: str = typer.Option("2005-01-01", help="Start date, YYYY-MM-DD."),
    end: str = typer.Option(None, help="End date, YYYY-MM-DD (defaults to today)."),
) -> None:
    """Fill in pressure/humidity/wind for existing days from NOAA LCD (no token needed)."""
    from weather_predictions.enrich import run as run_enrich

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else None
    inserted = run_enrich(start_date, end_date)
    typer.echo(f"Upserted {inserted} day(s) of pressure/humidity/wind data.")


@app.command()
def radar_fetch_raw() -> None:
    """Download the latest NEXRAD scan (no decoding) — safe on a Pi, needs only the `aws` CLI."""
    from weather_predictions.radar_raw import fetch_latest_raw

    saved_path = fetch_latest_raw()
    typer.echo(f"Downloaded {saved_path}" if saved_path else "Already up to date.")


@app.command()
def radar_backfill_raw(
    start: str = typer.Argument(..., help="Start, ISO 8601 e.g. 2026-07-04T00:00:00."),
    end: str = typer.Argument(..., help="End, ISO 8601."),
) -> None:
    """Download every raw NEXRAD scan in a UTC time range, no decoding (~12/hour, ~12-15MB each)."""
    from weather_predictions.radar_raw import backfill_raw

    downloaded = backfill_raw(datetime.fromisoformat(start), datetime.fromisoformat(end))
    typer.echo(f"Downloaded {downloaded} raw radar scan(s).")


@app.command()
def radar_decode_pending(keep_raw: bool = typer.Option(False, help="Keep raw files after decoding.")) -> None:
    """Decode raw scans sitting in data/radar/raw/ (e.g. synced over from the Pi). Needs `poetry install --with radar`."""
    from weather_predictions.radar import decode_pending

    decoded = decode_pending(keep_raw=keep_raw)
    typer.echo(f"Decoded {decoded} radar scan(s).")


@app.command()
def radar_fetch(keep_raw: bool = typer.Option(False, help="Keep the raw ~12-15MB volume scan after decoding.")) -> None:
    """Download + decode the latest NEXRAD scan in one step. Needs `poetry install --with radar`."""
    from weather_predictions.radar import fetch_latest

    saved_path = fetch_latest(keep_raw=keep_raw)
    if saved_path is None:
        typer.echo("No scans available yet.")
        raise typer.Exit(code=1)
    typer.echo(f"Saved {saved_path}")


@app.command()
def radar_backfill(
    start: str = typer.Argument(..., help="Start, ISO 8601 e.g. 2026-07-04T00:00:00."),
    end: str = typer.Argument(..., help="End, ISO 8601."),
    keep_raw: bool = typer.Option(False, help="Keep raw volume scans after decoding."),
) -> None:
    """Download + decode every NEXRAD scan in a UTC time range in one step. Needs `poetry install --with radar`."""
    from weather_predictions.radar import backfill_range

    saved_count = backfill_range(datetime.fromisoformat(start), datetime.fromisoformat(end), keep_raw=keep_raw)
    typer.echo(f"Decoded {saved_count} radar scan(s).")


@app.command()
def radar_nowcast(
    lead_minutes: float = typer.Option(30.0, help="How far ahead to forecast the reflectivity grid."),
) -> None:
    """Forecast the reflectivity grid N minutes ahead (optical flow + persistence baseline). Needs `poetry install --with radar`."""
    from weather_predictions.radar_nowcast import InsufficientFramesError
    from weather_predictions.radar_nowcast import nowcast as run_radar_nowcast

    try:
        result = run_radar_nowcast(lead_minutes=lead_minutes)
    except InsufficientFramesError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"Nowcast for {result.valid_at} (t+{result.lead_minutes:.0f}min, from {result.predicted_at}):")
    for method, path in result.grid_paths.items():
        typer.echo(f"  {method}: {path}")


@app.command()
def radar_nowcast_evaluate() -> None:
    """Score past radar nowcasts against the real grid that arrived, and log skill over time."""
    from weather_predictions.radar_nowcast_evaluate import evaluate as run_radar_nowcast_evaluate

    scored, pending = run_radar_nowcast_evaluate()
    if not scored:
        typer.echo("No nowcasts old enough to score yet.")
    for r in scored:
        typer.echo(
            f"method={r.method} | lead={r.lead_minutes:.0f}min | n={r.n_samples} | "
            f"mae={r.mae_dbz:.2f}dBZ csi={r.csi:.2f} (threshold {r.csi_threshold_dbz:.0f}dBZ)"
        )
    typer.echo(f"Pending (no actual grid yet): {pending}")


@app.command()
def radar_image(
    radius_km: float = typer.Option(50.0, help="Region radius (km) around LATITUDE/LONGITUDE to render."),
    output: str = typer.Option("data/radar/radar.png", help="Where to save the rendered PNG."),
) -> None:
    """Render the current reflectivity grid + motion arrows as a 7-color PNG for a Waveshare ACeP e-ink panel.

    Needs `poetry install --with display` (Pillow + OpenCV, no Py-ART) —
    works on the Pi given grids synced over from wherever decoding happened.
    """
    from pathlib import Path

    from weather_predictions.radar_image import OutOfRadarRangeError, render as run_render
    from weather_predictions.radar_nowcast import InsufficientFramesError

    try:
        result = run_render(radius_km=radius_km, output_path=Path(output))
    except (InsufficientFramesError, OutOfRadarRangeError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"Rendered {result.output_path} from frame {result.frame_timestamp} (+/-{result.region_radius_km:.0f}km).")


@app.command()
def mrms_fetch(keep_raw: bool = typer.Option(False, help="Keep the raw .grib2.gz after decoding.")) -> None:
    """Download + decode the latest MRMS national radar composite. Needs `poetry install --with mrms` and `brew install eccodes`."""
    from weather_predictions.mrms import fetch_latest

    saved_path = fetch_latest(keep_raw=keep_raw)
    if saved_path is None:
        typer.echo("No MRMS scans available yet.")
        raise typer.Exit(code=1)
    typer.echo(f"Saved {saved_path}")


@app.command()
def mrms_backfill(
    start: str = typer.Argument(..., help="Start, ISO 8601 e.g. 2026-07-04T00:00:00."),
    end: str = typer.Argument(..., help="End, ISO 8601."),
    keep_raw: bool = typer.Option(False, help="Keep raw .grib2.gz files after decoding."),
) -> None:
    """Download + decode every MRMS national scan in a UTC time range (~30/hour, ~1.5MB each). Needs `poetry install --with mrms`."""
    from weather_predictions.mrms import backfill_range

    saved_count = backfill_range(datetime.fromisoformat(start), datetime.fromisoformat(end), keep_raw=keep_raw)
    typer.echo(f"Decoded {saved_count} MRMS scan(s).")


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", help="Host to bind."),
    port: int = typer.Option(8080, help="Port to listen on."),
) -> None:
    """Start the web dashboard (radar, alerts, predictions, evaluation). Needs `poetry install --with web display`."""
    try:
        import uvicorn
        from weather_predictions.dashboard import app as dash_app
    except ImportError:
        typer.echo("FastAPI/uvicorn not installed. Run: poetry install --with web")
        raise typer.Exit(code=1)
    typer.echo(f"Dashboard at http://{host}:{port}  (Ctrl-C to stop)")
    uvicorn.run(dash_app, host=host, port=port)


@app.command()
def convlstm_train(
    epochs: int = typer.Option(20, help="Training epochs."),
    radius_km: float = typer.Option(300.0, help="Region radius (km) used to crop MRMS frames."),
) -> None:
    """Train the ConvLSTM nowcaster on stored MRMS frames. Needs `poetry install --with convlstm` and 2+ weeks of frames."""
    from weather_predictions.config import LATITUDE, LONGITUDE
    from weather_predictions.mrms_convlstm import NotEnoughFramesError
    from weather_predictions.mrms_convlstm import train as run_convlstm_train

    try:
        result = run_convlstm_train(lat=LATITUDE, lon=LONGITUDE, radius_km=radius_km, epochs=epochs)
    except (NotEnoughFramesError, ImportError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(
        f"Trained on {result.n_samples} samples over {result.n_epochs} epochs. "
        f"Final loss: {result.final_loss:.5f}. Model saved to {result.model_path}"
    )


@app.command()
def qpe_fetch() -> None:
    """Store the latest MRMS hourly rainfall accumulation (gauge-corrected) at your location. Needs `poetry install --with mrms`."""
    from weather_predictions.mrms_qpe import fetch_latest as run_qpe_fetch

    precip_mm = run_qpe_fetch()
    if precip_mm is None:
        typer.echo("No QPE value available (no data at your location or no files yet).")
    else:
        typer.echo(f"Last hour's rainfall at your location: {precip_mm:.2f} mm")


@app.command()
def qpe_backfill(
    start: str = typer.Argument(..., help="Start, ISO 8601 e.g. 2026-07-04T00:00:00."),
    end: str = typer.Argument(..., help="End, ISO 8601."),
) -> None:
    """Store MRMS hourly rainfall at your location for a UTC time range (24 files/day). Needs `poetry install --with mrms`."""
    from weather_predictions.mrms_qpe import backfill_range as run_qpe_backfill

    stored = run_qpe_backfill(datetime.fromisoformat(start), datetime.fromisoformat(end))
    typer.echo(f"Stored {stored} hour(s) of QPE data.")


@app.command()
def mrms_nowcast(
    lead_minutes: float = typer.Option(30.0, help="Minutes ahead to forecast."),
    radius_km: float = typer.Option(300.0, help="Region radius (km) around LATITUDE/LONGITUDE."),
) -> None:
    """Forecast MRMS national radar N minutes ahead (optical flow + persistence). Needs `poetry install --with display`."""
    from weather_predictions.mrms_nowcast import MrmsNowcastResult
    from weather_predictions.mrms_nowcast import nowcast as run_mrms_nowcast
    from weather_predictions.mrms_processing import OutOfMrmsRangeError
    from weather_predictions.radar_nowcast import InsufficientFramesError

    try:
        result = run_mrms_nowcast(lead_minutes=lead_minutes, radius_km=radius_km)
    except (InsufficientFramesError, OutOfMrmsRangeError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(
        f"MRMS nowcast for {result.valid_at} (t+{result.lead_minutes:.0f}min, "
        f"from {result.predicted_at}, ±{result.radius_km:.0f}km):"
    )
    for method, path in result.grid_paths.items():
        typer.echo(f"  {method}: {path}")


@app.command()
def mrms_nowcast_evaluate() -> None:
    """Score past MRMS nowcasts against the real national frame that arrived."""
    from weather_predictions.mrms_nowcast_evaluate import evaluate as run_mrms_nowcast_evaluate

    scored, pending = run_mrms_nowcast_evaluate()
    if not scored:
        typer.echo("No MRMS nowcasts old enough to score yet.")
    for r in scored:
        typer.echo(
            f"method={r.method} | lead={r.lead_minutes:.0f}min | n={r.n_samples} | "
            f"mae={r.mae_dbz:.2f}dBZ csi={r.csi:.2f} (threshold {r.csi_threshold_dbz:.0f}dBZ)"
        )
    typer.echo(f"Pending (no actual frame yet): {pending}")


@app.command()
def mrms_image(
    radius_km: float = typer.Option(300.0, help="Region radius (km) around LATITUDE/LONGITUDE to render."),
    output: str = typer.Option("data/mrms/mrms_radar.png", help="Where to save the rendered PNG."),
) -> None:
    """Render the current MRMS national radar region + motion arrows as a PNG.

    Needs `poetry install --with display`. Output includes the lat/lon bounding
    box of the rendered region so it can be overlaid on a map.
    """
    from pathlib import Path

    from weather_predictions.mrms_image import render as run_mrms_render
    from weather_predictions.mrms_processing import OutOfMrmsRangeError
    from weather_predictions.radar_nowcast import InsufficientFramesError

    try:
        result = run_mrms_render(radius_km=radius_km, output_path=Path(output))
    except (InsufficientFramesError, OutOfMrmsRangeError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"Rendered {result.output_path} from frame {result.frame_timestamp}")
    typer.echo(
        f"  bbox: lat [{result.lat_min:.3f}, {result.lat_max:.3f}] "
        f"lon [{result.lon_min:.3f}, {result.lon_max:.3f}]"
    )


@app.command()
def mrms_cells(
    radius_km: float = typer.Option(300.0, help="Search radius (km) around LATITUDE/LONGITUDE."),
    max_cells: int = typer.Option(10, help="Show at most this many cells, nearest first."),
) -> None:
    """List discrete storm cells near you with distance, movement, and intensity. Needs `poetry install --with display`."""
    from weather_predictions.mrms_cells import detect_cells
    from weather_predictions.mrms_processing import OutOfMrmsRangeError
    from weather_predictions.radar_nowcast import InsufficientFramesError

    try:
        cells = detect_cells(radius_km=radius_km)
    except (InsufficientFramesError, OutOfMrmsRangeError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    if not cells:
        typer.echo(f"No storm cells within {radius_km:.0f}km.")
        return
    for c in cells[:max_cells]:
        toward = "approaching" if c.approaching else "moving away"
        typer.echo(
            f"{c.distance_km:.0f}km {c.bearing} | heading {c.heading} at {c.speed_kmh:.0f}km/h ({toward}) | "
            f"peak {c.peak_dbz:.0f}dBZ | {c.area_km2:.0f}km²"
        )


@app.command()
def eink_update(
    radius_km: float = typer.Option(300.0, help="Region radius (km) to render."),
    output: str = typer.Option(None, help="Override output path for the PNG."),
) -> None:
    """Render the latest MRMS radar frame and push it to the Waveshare ACeP e-ink panel.

    Falls back to the NEXRAD station grid if no MRMS frames exist.
    Requires `poetry install --with display`. On the Pi, also needs the
    waveshare-epaper driver (pip install waveshare-epaper).
    """
    from pathlib import Path

    from weather_predictions.eink_display import update_display

    kwargs: dict = {"radius_km": radius_km}
    if output:
        kwargs["output_path"] = Path(output)
    result = update_display(**kwargs)
    if not result["success"]:
        typer.echo(f"e-ink update failed: {result.get('reason', 'unknown')}")
        raise typer.Exit(code=1)
    typer.echo(
        f"{'Pushed to panel' if result['pushed_to_panel'] else 'Image saved (no panel driver)'}: "
        f"{result['image_path']} (source={result['source']}, frame={result['frame_timestamp']})"
    )


@app.command()
def storm_check(
    lead_minutes: float = typer.Option(30.0, help="How far ahead the radar nowcast half of this check looks."),
) -> None:
    """Check for active NWS watches/warnings and (if enough radar frames exist) whether rain is headed your way."""
    from weather_predictions.config import LATITUDE, LONGITUDE
    from weather_predictions.home_precip_check import OutOfRadarRangeError, check_home
    from weather_predictions.nws_client import get_active_alerts
    from weather_predictions.radar_nowcast import InsufficientFramesError

    from weather_predictions.notify import PRIORITY_HIGH, PRIORITY_URGENT, send_notification

    alerts = get_active_alerts(LATITUDE, LONGITUDE)
    if not alerts:
        typer.echo("NWS: no active watches/warnings for your location.")
    for a in alerts:
        typer.echo(f"NWS: {a['event']} ({a['severity']}) until {a['expires']} — {a['headline']}")
        priority = PRIORITY_URGENT if a["severity"] in ("Severe", "Extreme") else PRIORITY_HIGH
        send_notification(a["headline"] or a["event"], title=f"NWS: {a['event']}", priority=priority)

    # Prefer MRMS (national coverage + arrival-time ladder); fall back to the
    # single-NEXRAD-station check when no MRMS frames have been collected.
    from weather_predictions.mrms_home_check import estimate_arrival
    from weather_predictions.mrms_processing import OutOfMrmsRangeError

    try:
        arrival = estimate_arrival()
    except (InsufficientFramesError, OutOfMrmsRangeError):
        arrival = None

    if arrival is not None:
        if arrival.rain_now:
            typer.echo("MRMS: it's raining at your location right now.")
        elif arrival.arrival_lead_minutes is not None:
            typer.echo(
                f"MRMS: rain arriving in ~{arrival.arrival_lead_minutes:.0f} min "
                f"(around {arrival.arrival_at}, "
                f"{arrival.reflectivity_by_lead[arrival.arrival_lead_minutes]:.0f}dBZ forecast)."
            )
            send_notification(
                f"Rain arriving in ~{arrival.arrival_lead_minutes:.0f} min", title="Rain inbound"
            )
        else:
            max_lead = max(arrival.reflectivity_by_lead)
            typer.echo(f"MRMS: no rain expected at your location within {max_lead:.0f} min.")
        return

    try:
        result = check_home(lead_minutes=lead_minutes)
    except InsufficientFramesError as e:
        typer.echo(f"Radar (experimental): {e}")
        return
    except OutOfRadarRangeError as e:
        typer.echo(f"Radar (experimental): {e}")
        return

    if result.rain_expected:
        typer.echo(
            f"Radar (experimental): rain likely reaching you by {result.valid_at} "
            f"(t+{result.lead_minutes:.0f}min, {result.reflectivity_dbz:.0f}dBZ forecast)."
        )
    else:
        typer.echo(
            f"Radar (experimental): no rain expected at your location within {result.lead_minutes:.0f}min "
            f"({result.reflectivity_dbz:.0f}dBZ forecast)."
        )


@app.command()
def hurricane_backfill(
    dest: str = typer.Option(None, help="Where to save the downloaded HURDAT2 file (defaults to data/hurdat2.txt)."),
) -> None:
    """Download NOAA/NHC's Atlantic best-track history (HURDAT2) and store every fix. Run once."""
    from pathlib import Path

    from weather_predictions.config import DATA_DIR
    from weather_predictions.hurricane_client import sync_hurdat2

    dest_path = Path(dest) if dest else DATA_DIR / "hurdat2.txt"
    inserted = sync_hurdat2(dest_path)
    typer.echo(f"Upserted {inserted} hurricane fix(es) from {dest_path}.")


@app.command()
def hurricane_train(
    test_years: int = typer.Option(5, help="Most recent N storm seasons held out as the test set."),
) -> None:
    """Train (or retrain) the hurricane track/intensity model on all backfilled HURDAT2 history."""
    from weather_predictions.hurricane_train import NotEnoughDataError as HurricaneNotEnoughDataError
    from weather_predictions.hurricane_train import train as run_hurricane_train

    try:
        result = run_hurricane_train(test_years=test_years)
    except HurricaneNotEnoughDataError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"Trained on {result.n_fixes} fixes.")
    for hr in result.horizons:
        typer.echo(
            f"  t+{hr.horizon_hours}h: track_err={hr.track_error_km:.0f}km (baseline {hr.track_baseline_error_km:.0f}km)"
            f"  wind_mae={hr.wind_mae_kt:.1f}kt (baseline {hr.wind_baseline_mae_kt:.1f}kt)"
        )


@app.command()
def hurricane_predict() -> None:
    """Forecast currently active tropical cyclones' track/intensity; stores predictions for later scoring."""
    from weather_predictions.hurricane_predict import ModelNotTrainedError as HurricaneModelNotTrainedError
    from weather_predictions.hurricane_predict import predict as run_hurricane_predict

    try:
        forecasts = run_hurricane_predict()
    except HurricaneModelNotTrainedError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    if not forecasts:
        typer.echo("No active tropical cyclones right now.")
        return
    for f in forecasts:
        typer.echo(f"{f.name} ({f.storm_id}, {f.classification}) as of {f.as_of}:")
        for hp in f.horizons:
            typer.echo(
                f"  t+{hp.horizon_hours}h ({hp.valid_at}): lat={hp.lat_pred:.1f} lon={hp.lon_pred:.1f} "
                f"wind={hp.wind_pred_kt:.0f}kt"
            )


@app.command()
def hurricane_radar(
    radius_km: float = typer.Option(500.0, help="Region radius (km) centred on each storm position."),
) -> None:
    """Render MRMS radar + nowcasts centred on active tropical cyclones' forecast positions.

    Skips storms outside CONUS MRMS coverage. Needs `poetry install --with display`.
    """
    from weather_predictions.hurricane_radar import render_active_storms
    from weather_predictions.hurricane_predict import ModelNotTrainedError

    try:
        results = render_active_storms(radius_km=radius_km)
    except ModelNotTrainedError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    if not results:
        typer.echo("No active tropical cyclones.")
        return

    for r in results:
        typer.echo(f"\n{r.name} ({r.storm_id}):")
        if r.skipped_reason:
            typer.echo(f"  Skipped: {r.skipped_reason}")
            continue
        for pos in r.rendered_positions:
            if pos.get("skipped"):
                typer.echo(f"  {pos['label']}: skipped ({pos['skipped']})")
            elif pos.get("image_path"):
                typer.echo(f"  {pos['label']}: {pos['image_path']}")
            else:
                typer.echo(f"  {pos['label']}: render failed — {pos.get('render_error', '?')}")


@app.command()
def hurricane_evaluate() -> None:
    """Compare past hurricane forecasts against subsequent HURDAT2 fixes, and log skill over time."""
    from weather_predictions.hurricane_evaluate import evaluate as run_hurricane_evaluate

    scored, pending = run_hurricane_evaluate()
    if not scored:
        typer.echo("No hurricane forecasts old enough to score yet.")
    for r in scored:
        typer.echo(
            f"model {r.model_trained_at} | t+{r.horizon_hours}h | n={r.n_samples} | "
            f"track_err={r.track_error_km:.0f}km wind_mae={r.wind_mae_kt:.1f}kt"
        )
    typer.echo(f"Pending (no actual fix yet): {pending}")


@app.command()
def status() -> None:
    """Show how much history has been collected and whether training is possible yet."""
    daily = build_daily_features(fetch_all_daily())
    n_days = len(daily)
    typer.echo(f"Days of daily history: {n_days} (need {MIN_TRAINING_DAYS} to train)")
    if not daily.empty:
        typer.echo(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
        typer.echo(daily["source"].value_counts().to_string())
        enriched = daily["pressure_hpa"].notna().sum() if "pressure_hpa" in daily else 0
        typer.echo(f"Days with pressure/humidity/wind (via `weather enrich`): {enriched}")
    typer.echo(f"Model trained: {'yes' if MODEL_PATH.exists() else 'no'} ({MODEL_PATH})")
    if n_days < MIN_TRAINING_DAYS:
        typer.echo(f"Need {MIN_TRAINING_DAYS - n_days} more day(s) — run `weather backfill` for bulk history.")


@app.command()
def train() -> None:
    """Train (or retrain) the 1/2/3-day rain + temperature models on all accumulated history."""
    try:
        result = run_train()
    except NotEnoughDataError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"Trained on {result.n_days} days of history.")
    for hr in result.horizons:
        typer.echo(
            f"  t+{hr.horizon_days}d: rain_acc={hr.rain_accuracy:.2f} (baseline {hr.rain_baseline_accuracy:.2f})"
            f"  tmax_mae={hr.temp_max_mae:.1f}C (baseline {hr.temp_max_baseline_mae:.1f}C)"
            f"  tmin_mae={hr.temp_min_mae:.1f}C (baseline {hr.temp_min_baseline_mae:.1f}C)"
        )


@app.command()
def predict() -> None:
    """Predict rain probability + high/low temp for the next 1-3 days; stores predictions for later scoring."""
    try:
        result = run_predict()
    except (ModelNotTrainedError, NoUsableDataError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"As of {result.as_of_date} (model trained {result.model_trained_at}):")
    for hp in result.horizons:
        typer.echo(
            f"  {hp.target_date} (t+{hp.horizon_days}d): {hp.rain_probability:.0%} rain, "
            f"high {hp.temp_max_pred_c:.0f}C / low {hp.temp_min_pred_c:.0f}C"
        )
    if result.nws_daytime_forecasts:
        typer.echo("NWS forecast for comparison:")
        for line in result.nws_daytime_forecasts:
            typer.echo(f"  {line}")


@app.command()
def evaluate() -> None:
    """Compare past predictions against what actually happened, and log accuracy over time."""
    from weather_predictions.evaluate import evaluate as run_evaluate

    scored, pending = run_evaluate()
    if not scored:
        typer.echo("No predictions old enough to score yet.")
    for r in scored:
        typer.echo(
            f"model {r.model_trained_at} | t+{r.horizon_days}d | n={r.n_samples} | "
            f"rain_acc={r.rain_accuracy:.2f} rain_brier={r.rain_brier:.3f} "
            f"tmax_mae={r.temp_max_mae:.1f}C tmin_mae={r.temp_min_mae:.1f}C"
        )
    typer.echo(f"Pending (no outcome yet): {pending}")


if __name__ == "__main__":
    app()
