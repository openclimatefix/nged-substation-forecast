from contracts.settings import Settings
from pathlib import Path


def test_settings_instantiation():
    settings = Settings(
        nged_ckan_token="test_ckan_token",
        nged_s3_bucket_url="https://test.com",
        nged_s3_bucket_access_key="test_access_key",
        nged_s3_bucket_secret="test_secret",
        nged_data_path=Path("data/NGED"),
        nwp_data_path=Path("data/NWP"),
        power_forecasts_data_path=Path("data/power_forecasts"),
        forecast_metrics_data_path=Path("data/forecast_metrics"),
        trained_ml_model_params_base_path=Path("data/trained_ML_model_params"),
    )
    assert settings.nged_ckan_token == "test_ckan_token"
    assert settings.nged_s3_bucket_url == "https://test.com"
    assert settings.nged_s3_bucket_access_key == "test_access_key"
    assert settings.nged_s3_bucket_secret == "test_secret"
    assert settings.nged_data_path == Path("data/NGED")
    assert settings.nwp_data_path == Path("data/NWP")
    assert settings.power_forecasts_data_path == Path("data/power_forecasts")
    assert settings.forecast_metrics_data_path == Path("data/forecast_metrics")
    assert settings.trained_ml_model_params_base_path == Path("data/trained_ML_model_params")
