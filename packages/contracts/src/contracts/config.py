import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# The config utility should provide these as Path objects.
POWER_DATA_PATH = Path(os.getenv("POWER_DATA_PATH", "data/power.parquet"))
WEATHER_DATA_PATH = Path(os.getenv("WEATHER_DATA_PATH", "data/weather.parquet"))
FORECAST_DATA_PATH = Path(os.getenv("FORECAST_DATA_PATH", "data/forecast.parquet"))
METRICS_DATA_PATH = Path(os.getenv("METRICS_DATA_PATH", "data/metrics.parquet"))
