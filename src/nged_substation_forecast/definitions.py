"""Dagster definitions entry point."""

from contracts.settings import get_settings
from dagster import Definitions, load_assets_from_modules

from nged_substation_forecast._sentry import init_sentry
from nged_substation_forecast.defs import (
    assets,
    checks,
    cv_assets,
    jobs,
    production_assets,
    schedules,
)

all_assets = load_assets_from_modules([assets, cv_assets, production_assets])

# Initialise Sentry once per process. This module is imported by every Dagster process — the
# daemon, the webserver, and each run worker — so error telemetry and the live_forecasts
# heartbeat are active wherever code runs. A no-op unless a Sentry DSN is configured.
init_sentry(get_settings())

defs = Definitions(
    assets=all_assets,
    asset_checks=[checks.power_data_is_fresh],
    jobs=[jobs.register_experiment_job],
    schedules=[
        schedules.power_time_series_and_metadata_schedule,
        schedules.ecmwf_ens_schedule,
        schedules.live_forecasts_schedule,
    ],
)
