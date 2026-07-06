"""Command-line entrypoint: `weather fetch|backfill|enrich|radar-fetch-raw|radar-backfill-raw|radar-decode-pending|radar-fetch|radar-backfill|radar-nowcast|radar-nowcast-evaluate|radar-image|train|predict|evaluate|status`."""

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
