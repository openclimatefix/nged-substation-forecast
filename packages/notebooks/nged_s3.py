from contracts.config import Settings


def test_s3_access():
    settings = Settings()  # type: ignore[call-arg]
    str(settings.NGED_S3_BUCKET_URL)
    settings.NGED_S3_BUCKET_ACCESS_KEY.get_secret_value()
    # ... logic using settings ...
