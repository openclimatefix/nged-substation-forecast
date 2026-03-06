import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# TODO: All config should probably be in a Pydantic class
NGED_DATA_PATH = Path(os.getenv("NGED_DATA_PATH", "data/NGED"))
NWP_DATA_PATH = Path(os.getenv("NWP_DATA_PATH", "data/NWP"))
POWER_FORECASTS_DATA_PATH = Path(os.getenv("POWER_FORECASTS_DATA_PATH", "data/power_forecasts"))
FORECAST_METRICS_DATA_PATH = Path(os.getenv("FORECAST_METRICS_DATA_PATH", "data/forecast_metrics"))
TRAINED_ML_MODEL_PARAMS_BASE_PATH = Path(
    os.getenv("TRAINED_ML_MODEL_PARAMS_BASE_PATH", "data/trained_ML_model_params")
)
