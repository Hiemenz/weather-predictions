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

# Same station's USAF-WBAN id, used for NOAA Local Climatological Data (LCD)
# bulk CSV files — the source for daily pressure/humidity/wind, which GHCND's
# daily summaries don't carry. Verified against the LCD file directory listing.
LCD_STATION_ID = "72327013897"

# Nearest NEXRAD radar site (raw Level II reflectivity/velocity sweeps),
# verified via api.weather.gov's `radarStation` field for this lat/lon.
RADAR_STATION_ID = "KOHX"

# Public NEXRAD Level II archive on AWS Open Data. Migrated from the
# deprecated `noaa-nexrad-level2` bucket in September 2025.
RADAR_S3_BUCKET = "unidata-nexrad-level2"

# NHC's Atlantic best-track history (comma-delimited, 6-hourly fixes back to
# 1851). The filename embeds the file's own last-update date and changes
# roughly once a year after hurricane season ends (verified current as of
# this writing — https://www.nhc.noaa.gov/data/ lists the current filename
# if this 404s).
NHC_HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"

# Live feed of active tropical cyclones, updated every ~2 minutes. Static
# URL, `{"activeStorms": []}` when nothing is active.
NHC_CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"

USER_AGENT = "weather-predictions (mckevinaaa24@gmail.com)"
API_BASE = "https://api.weather.gov"

CDO_API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
CDO_TOKEN = os.getenv("NOAA_CDO_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RADAR_DATA_DIR = DATA_DIR / "radar"

# NOAA MRMS (Multi-Radar Multi-Sensor) national composite on AWS Open Data.
# Pre-mosaiced CONUS coverage at 1km / 2-minute resolution — one file instead
# of ~160 per-station NEXRAD downloads. Public bucket, no credentials needed.
MRMS_S3_BUCKET = "noaa-mrms-pds"
MRMS_PRODUCT = "MergedReflectivityQCComposite_00.50"
MRMS_REGION = "CONUS"
MRMS_DATA_DIR = DATA_DIR / "mrms"

DB_PATH = DATA_DIR / "observations.sqlite"
MODEL_PATH = MODELS_DIR / "precip_model.joblib"
HURRICANE_MODEL_PATH = MODELS_DIR / "hurricane_model.joblib"

# Minimum days of daily history required before training is allowed.
MIN_TRAINING_DAYS = 14

# Precipitation threshold (mm) above which a day counts as "rain" (~0.01 in).
RAIN_THRESHOLD_MM = 0.254

# How many days ahead the model predicts.
FORECAST_HORIZONS = (1, 2, 3)
