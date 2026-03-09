import pytest
from contracts.config import Settings
from pydantic import HttpUrl, SecretStr


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("NGED_CKAN_TOKEN", "test_ckan_token")
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "http://test.s3.bucket")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "test_access_key")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "test_secret")

    settings = Settings(**{})

    assert settings.NGED_CKAN_TOKEN == SecretStr("test_ckan_token")
    assert settings.NGED_S3_BUCKET_URL == HttpUrl("http://test.s3.bucket")
    assert settings.NGED_S3_BUCKET_ACCESS_KEY == SecretStr("test_access_key")
    assert settings.NGED_S3_BUCKET_SECRET == SecretStr("test_secret")


def test_settings_no_defaults(monkeypatch):
    # Ensure no env vars are interfering
    monkeypatch.delenv("NGED_CKAN_TOKEN", raising=False)
    monkeypatch.delenv("NGED_S3_BUCKET_URL", raising=False)
    monkeypatch.delenv("NGED_S3_BUCKET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("NGED_S3_BUCKET_SECRET", raising=False)

    from pydantic import ValidationError
    from pydantic_settings import SettingsConfigDict

    # Force a non-existent env file to ensure no loading from disk
    class IsolatedSettings(Settings):
        model_config = SettingsConfigDict(env_file="/non/existent/path")

    with pytest.raises(ValidationError, match="NGED_S3_BUCKET_ACCESS_KEY"):
        IsolatedSettings(**{})
