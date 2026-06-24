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
