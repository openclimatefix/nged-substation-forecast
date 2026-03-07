from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, HttpUrl
from pathlib import Path


class Settings(BaseSettings):
    # NGED Connected Data
    NGED_CKAN_TOKEN: SecretStr = SecretStr("")

    # S3 Storage
    NGED_S3_BUCKET_URL: HttpUrl
    NGED_S3_BUCKET_ACCESS_KEY: SecretStr
    NGED_S3_BUCKET_SECRET: SecretStr

    # Paths
    NGED_DATA_PATH: Path = Path("data/NGED")
    NWP_DATA_PATH: Path = Path("data/NWP")
    POWER_FORECASTS_DATA_PATH: Path = Path("data/power_forecasts")
    FORECAST_METRICS_DATA_PATH: Path = Path("data/forecast_metrics")
    TRAINED_ML_MODEL_PARAMS_BASE_PATH: Path = Path("data/trained_ML_model_params")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
