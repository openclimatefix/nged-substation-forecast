import pytest
from contracts.settings import DataQualitySettings, Settings
from pydantic import ValidationError


def test_data_quality_settings_defaults():
    settings = DataQualitySettings()
    assert settings.stuck_std_threshold == 0.01
    assert settings.max_mw_threshold == 150.0
    assert settings.min_mw_threshold == -50.0


def test_settings_validation_invalid_url():
    with pytest.raises(ValidationError, match="Input should be a valid URL"):
        Settings(
            nged_s3_bucket_url="not-a-url",
            nged_s3_bucket_access_key="key",
            nged_s3_bucket_secret="secret",
        )


def test_paths_derive_from_data_root():
    """Unset data-table paths derive from data_path; nested tables sit under nged_data_path."""
    settings = Settings(
        data_path="/srv/data",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.nwp_data_path == "/srv/data/NWP"
    assert settings.nged_data_path == "/srv/data/NGED"
    assert settings.power_time_series_data_path == "/srv/data/NGED/power_time_series.delta"
    assert settings.metadata_path == "/srv/data/NGED/metadata.parquet"
    assert settings.h3_grid_weights_path == "/srv/data/h3_grid_weights.parquet"


def test_remote_data_path_keeps_artifacts_local():
    """A remote data_path yields s3:// data tables while artifacts stay under local_artifacts_path."""
    settings = Settings(
        data_path="s3://bucket/data",
        local_artifacts_path="/local/artifacts",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.power_forecasts_data_path == "s3://bucket/data/power_forecasts"
    assert settings.power_time_series_data_path == "s3://bucket/data/NGED/power_time_series.delta"
    assert settings.model_cache_base_path == "/local/artifacts/model_cache"
    assert settings.production_model_path == "/local/artifacts/production_model"
    assert settings.plots_data_path == "/local/artifacts/plots"


def test_explicit_path_overrides_derivation():
    """An explicitly set path wins over derivation from its root."""
    settings = Settings(
        data_path="s3://bucket/data",
        plots_data_path="s3://other-bucket/plots",
        nwp_data_path="/mnt/fast/NWP",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.plots_data_path == "s3://other-bucket/plots"
    assert settings.nwp_data_path == "/mnt/fast/NWP"
    # Siblings still derive normally.
    assert settings.power_forecasts_data_path == "s3://bucket/data/power_forecasts"


def test_storage_options_empty_without_dev_credentials():
    """With no ``data_store_*`` credentials set, ``storage_options`` is empty (the AWS default)."""
    settings = Settings(
        data_path="s3://bucket/data",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.storage_options == {}


def test_storage_options_populated_from_dev_credentials():
    """The ``data_store_*`` settings map onto the shared ``aws_*`` object_store option keys."""
    settings = Settings(
        data_path="s3://bucket/data",
        data_store_endpoint_url="http://localhost:9000",
        data_store_access_key_id="minio",
        data_store_secret_access_key="minio123",
        data_store_region="us-east-1",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.storage_options == {
        "aws_endpoint_url": "http://localhost:9000",
        "aws_allow_http": "true",
        "aws_access_key_id": "minio",
        "aws_secret_access_key": "minio123",
        "aws_region": "us-east-1",
    }
