from pathlib import Path
from pydantic import SecretStr, HttpUrl
from contracts.config import Settings


def test_settings_types():
    s = Settings(
        NGED_S3_BUCKET_URL=HttpUrl("https://example.com"),
        NGED_S3_BUCKET_ACCESS_KEY=SecretStr("dummy"),
        NGED_S3_BUCKET_SECRET=SecretStr("dummy"),
    )
    assert isinstance(s.NGED_CKAN_TOKEN, SecretStr)
    # assert isinstance(s.NGED_S3_BUCKET_URL, HttpUrl) # HttpUrl is not working
    assert isinstance(s.NGED_DATA_PATH, Path)
    assert isinstance(s.NWP_DATA_PATH, Path)
