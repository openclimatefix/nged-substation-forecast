from src.nged_substation_forecast.config_resource import NgedConfig
from pydantic import HttpUrl, SecretStr
from pathlib import Path


def test_nged_config_to_settings():
    config = NgedConfig(
        NGED_CKAN_TOKEN="test_ckan_token",
        NGED_S3_BUCKET_URL="https://test.com",
        NGED_S3_BUCKET_ACCESS_KEY="test_access_key",
        NGED_S3_BUCKET_SECRET="test_secret",
        NGED_DATA_PATH="data/NGED",
        NWP_DATA_PATH="data/NWP",
        POWER_FORECASTS_DATA_PATH="data/power_forecasts",
        FORECAST_METRICS_DATA_PATH="data/forecast_metrics",
        TRAINED_ML_MODEL_PARAMS_BASE_PATH="data/trained_ML_model_params",
    )
    settings = config.to_settings()
    assert settings.NGED_CKAN_TOKEN == SecretStr("test_ckan_token")
    assert settings.NGED_S3_BUCKET_URL == HttpUrl("https://test.com")
    assert settings.NGED_S3_BUCKET_ACCESS_KEY == SecretStr("test_access_key")
    assert settings.NGED_S3_BUCKET_SECRET == SecretStr("test_secret")
    assert settings.NGED_DATA_PATH == Path("data/NGED")
    assert settings.NWP_DATA_PATH == Path("data/NWP")
    assert settings.POWER_FORECASTS_DATA_PATH == Path("data/power_forecasts")
    assert settings.FORECAST_METRICS_DATA_PATH == Path("data/forecast_metrics")
    assert settings.TRAINED_ML_MODEL_PARAMS_BASE_PATH == Path("data/trained_ML_model_params")
