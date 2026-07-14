from typing import cast

import pytest
from contracts.settings import DataQualitySettings, Settings
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _isolate_settings_from_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every settings construction in this module hermetic.

    ``Settings`` reads ``PROJECT_ROOT/.env`` and process environment variables by default, so a
    developer's real local configuration — e.g. the ``DATA_STORE_*`` credentials that
    ``docs/live_service/setup.md`` has them put in ``.env`` — would otherwise leak into tests
    that assert on unset-field defaults (``storage_options == {}``, path derivation, …).
    """
    # cast: SettingsConfigDict is a TypedDict, whose per-key value types don't unify with
    # monkeypatch.setitem's Mapping[K, V] signature.
    monkeypatch.setitem(cast("dict[str, object]", Settings.model_config), "env_file", None)
    for settings_cls in (Settings, DataQualitySettings):
        for field_name in settings_cls.model_fields:
            monkeypatch.delenv(field_name.upper(), raising=False)


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
    """Unset data-table paths derive from data_path_internal; nested tables sit under nged_data_path."""
    settings = Settings(
        data_path_internal="/srv/data",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.nwp_data_path == "/srv/data/NWP"
    assert settings.nged_data_path == "/srv/data/NGED"
    assert settings.power_time_series_data_path == "/srv/data/NGED/power_time_series.delta"
    assert settings.metadata_path == "/srv/data/NGED/metadata.parquet"
    assert settings.h3_grid_weights_path == "/srv/data/h3_grid_weights.parquet"


def test_delivery_paths_derive_from_delivery_root():
    """Delivery-table paths derive from data_path_delivery, independently of data_path_internal."""
    settings = Settings(
        data_path_internal="s3://internal-bucket/data",
        data_path_delivery="s3://delivery-bucket/data",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.power_forecasts_data_path == "s3://delivery-bucket/data/power_forecasts"
    assert settings.effective_capacity_data_path == "s3://delivery-bucket/data/effective_capacity"
    assert settings.nwp_data_path == "s3://internal-bucket/data/NWP"


def test_delivery_fields_are_exactly_the_known_delivery_tables():
    """Guards against a new delivery table being added but never wired into _derive_unset_paths.

    If this test fails after adding a new *_data_path field, decide whether it belongs in the
    NGED-facing delivery bucket (see docs/roadmap/delivery-tables.md) and update either this set
    or Settings._derive_unset_paths accordingly.
    """
    known_delivery_fields = {"power_forecasts_data_path", "effective_capacity_data_path"}
    # Always-local artifacts derive from local_artifacts_path, not either data-table root, so
    # they're excluded from this internal-vs-delivery check even though their names also end in
    # "_data_path".
    local_artifact_fields = {"plots_data_path", "model_cache_base_path", "production_model_path"}
    settings = Settings(
        data_path_internal="/internal",
        data_path_delivery="/delivery",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    managed_data_table_fields = {
        name
        for name in Settings.model_fields
        if (name.endswith("_data_path") or name in {"metadata_path", "h3_grid_weights_path"})
        and name not in local_artifact_fields
    }
    for field_name in managed_data_table_fields:
        value = getattr(settings, field_name)
        if field_name in known_delivery_fields:
            assert value.startswith("/delivery"), f"{field_name} should derive from /delivery"
        else:
            assert value.startswith("/internal"), f"{field_name} should derive from /internal"


def test_remote_data_path_keeps_artifacts_local():
    """A remote data_path_internal yields s3:// data tables while artifacts stay under local_artifacts_path."""
    settings = Settings(
        data_path_internal="s3://bucket/data",
        data_path_delivery="s3://bucket/data",
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
        data_path_internal="s3://bucket/data",
        data_path_delivery="s3://bucket/data",
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
        data_path_internal="s3://bucket/data",
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    )
    assert settings.storage_options == {}


def test_storage_options_populated_from_dev_credentials():
    """The ``data_store_*`` settings map onto the shared ``aws_*`` object_store option keys."""
    settings = Settings(
        data_path_internal="s3://bucket/data",
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
