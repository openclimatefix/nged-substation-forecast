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

from nged_substation_forecast.defs.assets import ecmwf_ens_partitions
from nged_substation_forecast.defs.production_assets import live_forecast_partitions

# Define a job that targets the power_time_series_and_metadata asset
power_time_series_and_metadata_job = define_asset_job(
    name="power_time_series_and_metadata_job",
    selection=AssetSelection.assets("power_time_series_and_metadata"),
)

# Define a schedule that runs the job every hour
power_time_series_and_metadata_schedule = ScheduleDefinition(
    name="power_time_series_and_metadata_schedule",
    job=power_time_series_and_metadata_job,
    cron_schedule="0 * * * *",
)

ecmwf_ens_job = define_asset_job(
    "ecmwf_ens_job",
    selection=AssetSelection.assets("ecmwf_ens"),
    partitions_def=ecmwf_ens_partitions,
)


@schedule(job=ecmwf_ens_job, cron_schedule="30 7 * * *", execution_timezone="UTC")
def ecmwf_ens_schedule(context: ScheduleEvaluationContext) -> RunRequest:
    """Materialise today's ``ecmwf_ens`` partition daily at 07:30 UTC.

    07:30 UTC is roughly the 00Z run's init time plus Dynamical's publication/ingest delay;
    ``ecmwf_ens_partitions``' ``end_offset=1`` means today's partition key already exists by
    this point. A too-early run fails cleanly and is re-materialisable from the UI; live
    inference (``live_forecasts``) always uses the freshest run genuinely present regardless of
    this schedule's exact timing.
    """
    return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))


live_forecasts_job = define_asset_job(
    "live_forecasts_job",
    selection=AssetSelection.assets("live_forecasts"),
    partitions_def=live_forecast_partitions,
)

live_forecasts_schedule = build_schedule_from_partitioned_job(live_forecasts_job)
"""Ticks at 00/06/12/18 UTC, materialising the just-completed window with default run config
(``availability_mode="live"``) — the schedule is always live; replays are manual, launched from
the UI with ``availability_mode="replay"``."""
