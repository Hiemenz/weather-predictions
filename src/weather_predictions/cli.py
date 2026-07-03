"""Command-line entrypoint: `weather fetch|train|predict|status`."""

from __future__ import annotations

import typer

from weather_predictions.config import MIN_TRAINING_DAYS, MODEL_PATH
from weather_predictions.features import build_daily_features, raw_to_frame
from weather_predictions.predict import ModelNotTrainedError, NoUsableDataError, predict as run_predict
from weather_predictions.storage import count_observations, fetch_all_observations
from weather_predictions.train import NotEnoughDataError, train as run_train

app = typer.Typer(help="Fetch NOAA/NWS weather data, train, and predict next-day rain.")


@app.command()
def fetch() -> None:
    """Pull the latest observations from NWS and store them locally."""
    from weather_predictions.fetch_observations import run

    inserted = run()
    typer.echo(f"Inserted {inserted} new observation(s).")


@app.command()
def status() -> None:
    """Show how much history has been collected and whether training is possible yet."""
    n_raw = count_observations()
    daily = build_daily_features(raw_to_frame(fetch_all_observations()))
    n_days = len(daily)
    typer.echo(f"Raw observations stored: {n_raw}")
    typer.echo(f"Aggregated days: {n_days} (need {MIN_TRAINING_DAYS} to train)")
    typer.echo(f"Model trained: {'yes' if MODEL_PATH.exists() else 'no'} ({MODEL_PATH})")
    if n_days < MIN_TRAINING_DAYS:
        remaining = MIN_TRAINING_DAYS - n_days
        typer.echo(f"Keep the fetch job running — roughly {remaining} more day(s) needed.")


@app.command()
def train() -> None:
    """Train (or retrain) the rain/no-rain model on all accumulated history."""
    try:
        result = run_train()
    except NotEnoughDataError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)
    typer.echo(f"Trained on {result.n_days} days ({result.n_train} train / {result.n_test} test).")
    typer.echo(f"Test accuracy: {result.accuracy:.2f} (persistence baseline: {result.baseline_accuracy:.2f})")
    if result.roc_auc is not None:
        typer.echo(f"ROC-AUC: {result.roc_auc:.2f}  Precision: {result.precision:.2f}  Recall: {result.recall:.2f}")


@app.command()
def predict() -> None:
    """Predict tomorrow's rain probability and show the official NWS forecast alongside it."""
    try:
        result = run_predict()
    except (ModelNotTrainedError, NoUsableDataError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)

    typer.echo(f"As of {result.as_of_local_date}:")
    typer.echo(f"  Model:  {result.rain_probability:.0%} chance of rain tomorrow "
               f"({'RAIN' if result.rain_predicted else 'NO RAIN'})")
    if result.nws_forecast_pop_pct is not None:
        typer.echo(f"  NWS:    {result.nws_forecast_pop_pct}% chance of precipitation")
    if result.nws_forecast_summary:
        typer.echo(f"  NWS says: {result.nws_forecast_summary}")


if __name__ == "__main__":
    app()
