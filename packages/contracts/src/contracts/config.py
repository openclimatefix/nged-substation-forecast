from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # NGED Connected Data
    NGED_CKAN_TOKEN: str = ""

    # S3 Storage
    NGED_S3_BUCKET_URL: str = "https://connecteddata.nationalgrid.co.uk"
    NGED_S3_BUCKET_ACCESS_KEY: str = ""
    NGED_S3_BUCKET_SECRET: str = ""

    # Paths
    NGED_DATA_PATH: str = "data/NGED"
    NWP_DATA_PATH: str = "data/NWP"
    POWER_FORECASTS_DATA_PATH: str = "data/power_forecasts"
    FORECAST_METRICS_DATA_PATH: str = "data/forecast_metrics"
    TRAINED_ML_MODEL_PARAMS_BASE_PATH: str = "data/trained_ML_model_params"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Global instance for backward compatibility during migration
settings = Settings()

# Constants for backward compatibility (using types as defined in class)
NGED_DATA_PATH = settings.NGED_DATA_PATH
NWP_DATA_PATH = settings.NWP_DATA_PATH
POWER_FORECASTS_DATA_PATH = settings.POWER_FORECASTS_DATA_PATH
FORECAST_METRICS_DATA_PATH = settings.FORECAST_METRICS_DATA_PATH
TRAINED_ML_MODEL_PARAMS_BASE_PATH = settings.TRAINED_ML_MODEL_PARAMS_BASE_PATH
