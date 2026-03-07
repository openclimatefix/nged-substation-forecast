from pathlib import Path

import pytest
from pydantic import HttpUrl, SecretStr

from contracts.config import Settings


def test_settings_load_from_env():
    env_content = (
        "NGED_CKAN_TOKEN=test_ckan_token\n"
        "NGED_S3_BUCKET_URL=http://test.s3.bucket\n"
        "NGED_S3_BUCKET_ACCESS_KEY=test_access_key\n"
        "NGED_S3_BUCKET_SECRET=test_secret\n"
    )
    env_file_path = Path(".env")
    env_file_path.write_text(env_content)
    settings = Settings()  # type: ignore[call-arg]
    Path(".env").unlink()

    assert settings.NGED_CKAN_TOKEN == SecretStr("test_ckan_token")
    assert settings.NGED_S3_BUCKET_URL == HttpUrl("http://test.s3.bucket")
    assert settings.NGED_S3_BUCKET_ACCESS_KEY == SecretStr("test_access_key")
    assert settings.NGED_S3_BUCKET_SECRET == SecretStr("test_secret")


def test_settings_no_defaults():
    with pytest.raises(ValueError, match="NGED_S3_BUCKET_ACCESS_KEY"):

        class NoDefaultsSettings(Settings):
            NGED_S3_BUCKET_URL: HttpUrl
            NGED_S3_BUCKET_ACCESS_KEY: SecretStr
            NGED_S3_BUCKET_SECRET: SecretStr
            NGED_CKAN_TOKEN: SecretStr

        NoDefaultsSettings()  # type: ignore[call-arg]
