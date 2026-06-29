"""Dagster definitions entry point."""

from dagster import Definitions, load_assets_from_modules
from contracts.settings import Settings

from nged_substation_forecast.defs import assets, cv_assets, jobs, plot_assets, schedules

all_assets = load_assets_from_modules([assets, cv_assets, plot_assets])

settings = Settings()

defs = Definitions(
    assets=all_assets,
    jobs=[jobs.register_experiment_job],
    schedules=[schedules.power_time_series_and_metadata_schedule],
)
