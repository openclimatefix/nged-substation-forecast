"""Dagster definitions entry point."""

from dagster import Definitions, load_assets_from_modules

from nged_substation_forecast.defs import assets, schedules

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    schedules=[schedules.power_time_series_and_metadata_schedule],
)
