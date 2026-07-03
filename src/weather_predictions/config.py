"""Central configuration for location, station, and storage paths."""

from pathlib import Path

# Nashville, TN
LATITUDE = 36.1627
LONGITUDE = -86.7816

# Nashville International Airport — nearest reporting NWS/ASOS station.
STATION_ID = "KBNA"

USER_AGENT = "weather-predictions (mckevinaaa24@gmail.com)"
API_BASE = "https://api.weather.gov"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "observations.sqlite"
MODEL_PATH = MODELS_DIR / "precip_model.joblib"

# Minimum days of daily-aggregated history required before training is allowed.
MIN_TRAINING_DAYS = 14

# Precipitation threshold (mm) above which a day counts as "rain" (~0.01 in).
RAIN_THRESHOLD_MM = 0.254
