from nged_substation_forecast.config_resource import NgedConfig


def test_nged_config_initialization():
    config = NgedConfig(
        NGED_S3_BUCKET_URL="https://example.com",
        NGED_S3_BUCKET_ACCESS_KEY="dummy",
        NGED_S3_BUCKET_SECRET="dummy",
    )
    assert config is not None
