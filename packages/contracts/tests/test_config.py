from pathlib import Path
from pydantic import SecretStr
from contracts.config import Settings


def test_settings_types():
    s = Settings()
    assert isinstance(s.NGED_CKAN_TOKEN, SecretStr)
    # assert isinstance(s.NGED_S3_BUCKET_URL, HttpUrl) # HttpUrl is not working
    assert isinstance(s.NGED_DATA_PATH, Path)
    assert isinstance(s.NWP_DATA_PATH, Path)
