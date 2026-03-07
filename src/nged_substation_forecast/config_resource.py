from dagster import ConfigurableResource
from contracts.config import Settings


class NgedConfig(ConfigurableResource):
    """Dagster resource for NGED substation forecast configuration."""

    NGED_CKAN_TOKEN: str  # NGED Connected Data CKAN Token
    NGED_S3_BUCKET_URL: str  # S3 Bucket URL
    NGED_S3_BUCKET_ACCESS_KEY: str  # S3 Bucket Access Key
    NGED_S3_BUCKET_SECRET: str  # S3 Bucket Secret
    NGED_DATA_PATH: str  # Path to NGED data
    NWP_DATA_PATH: str  # Path to NWP data
    POWER_FORECASTS_DATA_PATH: str  # Path to power forecasts data
    FORECAST_METRICS_DATA_PATH: str  # Path to forecast metrics data
    TRAINED_ML_MODEL_PARAMS_BASE_PATH: str  # Path to trained ML model parameters

    def to_settings(self) -> Settings:
        return Settings.model_validate(self.model_dump())
