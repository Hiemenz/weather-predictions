"""Central configuration for location, station, and storage paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Nashville, TN
LATITUDE = 36.1627
LONGITUDE = -86.7816

# Nearest reporting NWS/ASOS station (used for live obs + forecast).
STATION_ID = "KBNA"

# Nearest GHCND station with long-running daily history (used for CDO backfill).
# Nashville International Airport. Verified via /cdo-web/api/v2/stations lookup.
GHCND_STATION_ID = "GHCND:USW00013897"

USER_AGENT = "weather-predictions (mckevinaaa24@gmail.com)"
API_BASE = "https://api.weather.gov"

CDO_API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
CDO_TOKEN = os.getenv("NOAA_CDO_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "observations.sqlite"
MODEL_PATH = MODELS_DIR / "precip_model.joblib"

# Minimum days of daily history required before training is allowed.
MIN_TRAINING_DAYS = 14

# Precipitation threshold (mm) above which a day counts as "rain" (~0.01 in).
RAIN_THRESHOLD_MM = 0.254

# How many days ahead the model predicts.
FORECAST_HORIZONS = (1, 2, 3)
