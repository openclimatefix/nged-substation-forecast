import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# The config utility should provide these as Path objects.
POWER_DATA_PATH = Path(os.getenv("POWER_DATA_PATH", "data/power.parquet"))
NWP_DATA_PATH = Path(os.getenv("NWP_DATA_PATH", "data/nwp"))
FORECAST_DATA_PATH = Path(os.getenv("FORECAST_DATA_PATH", "data/forecast.parquet"))
METRICS_DATA_PATH = Path(os.getenv("METRICS_DATA_PATH", "data/metrics.parquet"))
