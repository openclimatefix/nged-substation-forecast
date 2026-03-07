from contracts.config import Settings

settings = Settings()


def test_s3_access():
    str(settings.NGED_S3_BUCKET_URL)
    settings.NGED_S3_BUCKET_ACCESS_KEY.get_secret_value()
    # ... logic using settings ...
