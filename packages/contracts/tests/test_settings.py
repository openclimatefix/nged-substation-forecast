import subprocess
import sys
from typing import cast

import pytest
from contracts.settings import DataQualitySettings, Settings, get_settings
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _isolate_settings_from_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every settings construction in this module hermetic.

    ``Settings`` reads ``PROJECT_ROOT/.env`` and process environment variables by default, so a
    developer's real local configuration — e.g. the ``DATA_STORE_*`` credentials that
    <https://openclimatefix.github.io/nged-substation-forecast/live_service/setup/> has them put
    in ``.env`` — would otherwise leak into tests that assert on unset-field defaults
    (``storage_options == {}``, path derivation, …).
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
    """Liberal about a missing source-bucket URL, strict about a malformed one."""
    with pytest.raises(ValidationError, match="Input should be a valid URL"):
        Settings(nged_s3_bucket_url="not-a-url")


def test_paths_derive_from_data_root():
    """Unset data-table paths derive from data_path_internal; nested ones from nged_data_path."""
    settings = Settings(
        data_path_internal="/srv/data",
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
    )
    assert settings.power_forecasts_data_path == "s3://delivery-bucket/data/power_forecasts"
    assert settings.effective_capacity_data_path == "s3://delivery-bucket/data/effective_capacity"
    assert settings.nwp_data_path == "s3://internal-bucket/data/NWP"


def test_delivery_fields_are_exactly_the_known_delivery_tables():
    """Guards against a new delivery table being added but never wired into _derive_unset_paths.

    If this test fails after adding a new *_data_path field, decide whether it belongs in the
    NGED-facing delivery bucket
    (<https://openclimatefix.github.io/nged-substation-forecast/roadmap/delivery-tables/>) and
    update either this set or Settings._derive_unset_paths accordingly.
    """
    known_delivery_fields = {"power_forecasts_data_path", "effective_capacity_data_path"}
    # Always-local artifacts derive from local_artifacts_path, not either data-table root, so
    # they're excluded from this internal-vs-delivery check even though their names also end in
    # "_data_path".
    local_artifact_fields = {"production_model_path"}
    settings = Settings(
        data_path_internal="/internal",
        data_path_delivery="/delivery",
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
    """A remote data_path_internal yields s3:// tables; artifacts stay under
    local_artifacts_path.
    """
    settings = Settings(
        data_path_internal="s3://bucket/data",
        data_path_delivery="s3://bucket/data",
        local_artifacts_path="/local/artifacts",
    )
    assert settings.power_forecasts_data_path == "s3://bucket/data/power_forecasts"
    assert settings.power_time_series_data_path == "s3://bucket/data/NGED/power_time_series.delta"
    assert settings.production_model_path == "/local/artifacts/production_model"


def test_explicit_path_overrides_derivation():
    """An explicitly set path wins over derivation from its root."""
    settings = Settings(
        data_path_internal="s3://bucket/data",
        data_path_delivery="s3://bucket/data",
        production_model_path="/mnt/models/prod",
        nwp_data_path="/mnt/fast/NWP",
    )
    assert settings.production_model_path == "/mnt/models/prod"
    assert settings.nwp_data_path == "/mnt/fast/NWP"
    # Siblings still derive normally.
    assert settings.power_forecasts_data_path == "s3://bucket/data/power_forecasts"


def test_storage_options_empty_without_dev_credentials():
    """With no ``data_store_*`` credentials set, ``storage_options`` is empty (the AWS default)."""
    settings = Settings(
        data_path_internal="s3://bucket/data",
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
    )
    assert settings.storage_options == {
        "aws_endpoint_url": "http://localhost:9000",
        "aws_allow_http": "true",
        "aws_access_key_id": "minio",
        "aws_secret_access_key": "minio123",
        "aws_region": "us-east-1",
    }


def test_get_settings_is_cached():
    """get_settings returns a single shared instance; cache_clear forces a rebuild."""
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
        first = get_settings()
        get_settings.cache_clear()
        assert get_settings() is not first
    finally:
        # Don't leak this module's hermetic instance into other test modules' cache.
        get_settings.cache_clear()


def test_importing_contracts_constructs_no_settings():
    """Importing the schema modules must not construct Settings (no import-time .env read).

    Constructing Settings reads .env and the environment, so an import-time construction would
    freeze whatever configuration happened to be present when the module first loaded, before a
    test (or an entry point) could set it. Run in a fresh interpreter because this process has
    already imported (and used) these modules. See contracts.settings.get_settings.

    The guard patches ``Settings.__init__`` to raise, so it catches *any* import-time construction —
    a direct module-level ``Settings()`` as well as one routed through ``get_settings()`` — not just
    a populated cache. ``contracts/__init__.py`` imports nothing, so the patch lands before any
    schema module loads.
    """
    probe = (
        "import contracts.settings as s\n"
        "def _boom(*args, **kwargs):\n"
        "    raise AssertionError('Settings() constructed at import time')\n"
        "s.Settings.__init__ = _boom\n"
        "import contracts.weather_schemas\n"
        "import contracts.geo_schemas\n"
        "import contracts.power_schemas\n"
        "import contracts.ml_schemas\n"
        "assert s.get_settings.cache_info().currsize == 0\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_settings_builds_without_nged_source_credentials():
    """A clone with no `.env` must still get a usable `Settings`.

    `.env` is gitignored and holds third-party credentials for NGED's *source* bucket, so
    requiring them would mean a fresh checkout could not run the test suite, train a model, or
    open a dashboard — none of which read NGED's bucket. That is the laptop-exercisable principle
    applied to configuration.
    """
    settings = Settings()

    assert settings.nged_s3_bucket_url == ""
    assert settings.nged_s3_bucket_access_key == ""
    assert settings.nged_s3_bucket_secret == ""


@pytest.mark.parametrize(
    ("url", "access_key", "secret", "expected_missing"),
    [
        ("", "", "", "NGED_S3_BUCKET_URL, NGED_S3_BUCKET_ACCESS_KEY, NGED_S3_BUCKET_SECRET"),
        ("https://example.com", "", "", "NGED_S3_BUCKET_ACCESS_KEY, NGED_S3_BUCKET_SECRET"),
        ("https://example.com", "key", "", "NGED_S3_BUCKET_SECRET"),
        ("", "key", "secret", "NGED_S3_BUCKET_URL"),
    ],
)
def test_require_nged_source_credentials_names_what_is_missing(
    url: str, access_key: str, secret: str, expected_missing: str
):
    """The error has to name the unset variables — that is the whole point of moving the check
    from construction (where pydantic named them) to the point of use."""
    settings = Settings(
        nged_s3_bucket_url=url,
        nged_s3_bucket_access_key=access_key,
        nged_s3_bucket_secret=secret,
    )

    with pytest.raises(ValueError, match=expected_missing):
        settings.require_nged_source_credentials()


def test_require_nged_source_credentials_passes_when_all_three_are_set():
    Settings(
        nged_s3_bucket_url="https://example.com",
        nged_s3_bucket_access_key="key",
        nged_s3_bucket_secret="secret",
    ).require_nged_source_credentials()


def test_get_nged_s3_store_refuses_to_build_a_store_without_credentials():
    """Pins the *wiring*, not just the check.

    Moving the credential check out of construction only helps if something still performs it
    before the credentials are used. Every test that reaches ingest monkeypatches
    ``get_nged_s3_store`` wholesale, so without this test the one line that calls
    ``require_nged_source_credentials`` could be deleted and the suite would stay green.
    """
    with pytest.raises(ValueError, match="NGED_S3_BUCKET_URL"):
        Settings().get_nged_s3_store()
