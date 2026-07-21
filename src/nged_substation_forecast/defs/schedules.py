"""Dagster schedules for the NGED Substation Forecast project."""

from dagster import (
    AssetSelection,
    RunRequest,
    ScheduleDefinition,
    ScheduleEvaluationContext,
    build_schedule_from_partitioned_job,
    define_asset_job,
    schedule,
)

from nged_substation_forecast._sentry import sentry_capture_failure
from nged_substation_forecast.defs.assets import ecmwf_ens_partitions
from nged_substation_forecast.defs.production_assets import live_forecast_partitions

# Define a job that targets the power_time_series_and_metadata asset
power_time_series_and_metadata_job = define_asset_job(
    name="power_time_series_and_metadata_job",
    selection=AssetSelection.assets("power_time_series_and_metadata"),
    hooks={sentry_capture_failure},
)

power_time_series_and_metadata_schedule = ScheduleDefinition(
    name="power_time_series_and_metadata_schedule",
    job=power_time_series_and_metadata_job,
    cron_schedule="55 * * * *",
)
"""Fires at :55 past every hour — 5 minutes *before* the top of the hour — so this hour's pull
has landed by the time ``live_forecasts_schedule`` ticks at 00/06/12/18 UTC.
``live_forecasts`` depends on ``power_time_series_and_metadata`` (``defs/production_assets.py``)
but the two run as separate jobs on separate schedules, so nothing actually enforces the
ordering; this offset is a cheap mitigation, not a guarantee.

TODO: explore a more rigorous fix — e.g. a run-status sensor that only fires
``live_forecasts_job`` once that hour's ``power_time_series_and_metadata_job`` has actually
succeeded, rather than assuming a fixed offset is always enough."""

ecmwf_ens_job = define_asset_job(
    "ecmwf_ens_job",
    selection=AssetSelection.assets("ecmwf_ens"),
    partitions_def=ecmwf_ens_partitions,
    hooks={sentry_capture_failure},
)


@schedule(job=ecmwf_ens_job, cron_schedule="30 8 * * *", execution_timezone="UTC")
def ecmwf_ens_schedule(context: ScheduleEvaluationContext) -> RunRequest:
    """Materialise today's ``ecmwf_ens`` partition daily at 08:30 UTC.

    08:30 UTC is a safety margin past the 00Z run's expected publication time (roughly 08:00
    UTC / 9am BST); ``ecmwf_ens_partitions``' ``end_offset=1`` means today's partition key
    already exists by this point. If the run isn't published yet, ``ecmwf_ens`` retries every
    30 minutes for up to 4 hours (``NwpRunNotYetAvailable`` → ``RetryRequested`` in
    ``defs/assets.py``) rather than failing outright; any other error still fails immediately.
    Live inference (``live_forecasts``) always uses the freshest run genuinely present
    regardless of this schedule's exact timing.
    """
    return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))


live_forecasts_job = define_asset_job(
    "live_forecasts_job",
    selection=AssetSelection.assets("live_forecasts"),
    partitions_def=live_forecast_partitions,
    hooks={sentry_capture_failure},
)

live_forecasts_schedule = build_schedule_from_partitioned_job(live_forecasts_job)
"""Ticks at 00/06/12/18 UTC, materialising the just-completed window with default run config
(``availability_mode="live"``) — the schedule is always live; replays are manual, launched from
the UI with ``availability_mode="replay"``. See ``power_time_series_and_metadata_schedule``'s
docstring above for how the two schedules' relative timing is (loosely) coordinated."""
