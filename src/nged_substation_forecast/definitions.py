"""Dagster definitions entry point."""

from dagster import Definitions, load_assets_from_modules
from contracts.settings import Settings

from nged_substation_forecast.defs import assets, schedules
from nged_substation_forecast.resources import S3Resource

all_assets = load_assets_from_modules([assets])

settings = Settings()

defs = Definitions(
    assets=all_assets,
    schedules=[schedules.power_time_series_and_metadata_schedule],
    resources={
        "s3": S3Resource(
            bucket_url=settings.nged_s3_bucket_url,
            access_key=settings.nged_s3_bucket_access_key,
            secret_key=settings.nged_s3_bucket_secret,
        )
    },
)
