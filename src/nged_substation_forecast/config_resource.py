from contracts.config import Settings
from dagster import ConfigurableResource
from pydantic import Field


class NgedConfig(ConfigurableResource):
    """Dagster resource for NGED substation forecast configuration."""

    # This is a repetition of the fields in packages/contracts/config/Settings
    # because Dagster ConfigurableResource can't handle the rich Pydantic types.

    NGED_CKAN_TOKEN: str = Field(...)  # NGED Connected Data CKAN Token
    NGED_S3_BUCKET_URL: str = Field(...)  # S3 Bucket URL
    NGED_S3_BUCKET_ACCESS_KEY: str = Field(...)  # S3 Bucket Access Key
    NGED_S3_BUCKET_SECRET: str = Field(...)  # S3 Bucket Secret
    NGED_DATA_PATH: str = Field(...)  # Path to NGED data
    NWP_DATA_PATH: str = Field(...)  # Path to NWP data
    POWER_FORECASTS_DATA_PATH: str = Field(...)  # Path to power forecasts data
    FORECAST_METRICS_DATA_PATH: str = Field(...)  # Path to forecast metrics data
    TRAINED_ML_MODEL_PARAMS_BASE_PATH: str = Field(...)  # Path to trained ML model parameters

    def to_settings(self) -> Settings:
        return Settings.model_validate(self.model_dump())
