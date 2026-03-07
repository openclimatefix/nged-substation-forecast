from dagster import ConfigurableResource
from contracts.config import Settings


class NgedConfig(ConfigurableResource):
    """Dagster resource for NGED substation forecast configuration."""

    NGED_CKAN_TOKEN: str = ""
    NGED_S3_BUCKET_URL: str
    NGED_S3_BUCKET_ACCESS_KEY: str
    NGED_S3_BUCKET_SECRET: str
    NGED_DATA_PATH: str = "data/NGED"
    NWP_DATA_PATH: str = "data/NWP"
    POWER_FORECASTS_DATA_PATH: str = "data/power_forecasts"
    FORECAST_METRICS_DATA_PATH: str = "data/forecast_metrics"
    TRAINED_ML_MODEL_PARAMS_BASE_PATH: str = "data/trained_ML_model_params"

    def to_settings(self) -> Settings:
        return Settings(**self.model_dump())
