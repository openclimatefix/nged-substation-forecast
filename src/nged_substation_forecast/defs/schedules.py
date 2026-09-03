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

# Define a job that targets the power_time_series_and_metadata asset
power_time_series_and_metadata_job = define_asset_job(
    name="power_time_series_and_metadata_job",
    selection=AssetSelection.assets("power_time_series_and_metadata"),
    hooks={sentry_capture_failure},
    description=(
        "Pull the latest NGED telemetry from S3 into the power_time_series Delta table and"
        " upsert the substation metadata parquet. Runs hourly at :55, 5 minutes before"
        " live_forecasts_schedule ticks; see power_time_series_and_metadata_schedule for why a"
        " missed pull still lets live_forecasts run on time."
    ),
)

power_time_series_and_metadata_schedule = ScheduleDefinition(
    name="power_time_series_and_metadata_schedule",
    job=power_time_series_and_metadata_job,
    cron_schedule="55 * * * *",
    description=(
        "Fires at :55 past every hour, 5 minutes before live_forecasts_schedule ticks at"
        " 00/06/12/18 UTC, so this hour's telemetry has landed first. A missed or late pull does"
        " not hold live_forecasts back: the two schedules couple through data at rest, never"
        " through run status, so the forecast runs on time against whatever telemetry is already"
        " on disk."
    ),
)
"""Fires at :55 past every hour — 5 minutes *before* the top of the hour — so this hour's pull
has landed by the time ``live_forecasts_schedule`` ticks at 00/06/12/18 UTC.

The two schedules are deliberately not ordered against each other: production jobs couple through
data at rest, never through run status, so a missed or late pull here leaves ``live_forecasts``
running on time against whatever telemetry is already on disk rather than suppressing the
forecast. Telemetry staleness is not yet recorded on the forecast row itself — only
``nwp_init_time`` is. See
<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#14-production-jobs-are-coupled-through-data-at-rest-never-through-run-status>."""

ecmwf_ens_job = define_asset_job(
    "ecmwf_ens_job",
    selection=AssetSelection.assets("ecmwf_ens"),
    hooks={sentry_capture_failure},
    description=(
        "Download the day's ECMWF ENS NWP run and write it to the nwp Delta table, replacing"
        " that (nwp_model_id, init_time) partition. Runs daily at 08:30 UTC; see"
        " ecmwf_ens_schedule for the retry behaviour when the run is not yet published."
    ),
)


@schedule(job=ecmwf_ens_job, cron_schedule="30 8 * * *", execution_timezone="UTC")
def ecmwf_ens_schedule(context: ScheduleEvaluationContext) -> RunRequest:
    """Materialise today's ``ecmwf_ens`` partition daily at 08:30 UTC.

    08:30 UTC is a safety margin past the 00Z run's expected publication time (roughly 08:00
    UTC / 9am BST); ``ecmwf_ens_partitions``' ``end_offset=1`` means today's partition key
    already exists by this point. If the run isn't usable yet — absent from the catalog, or
    present with a weather variable still wholesale empty — ``ecmwf_ens`` retries every 30
    minutes, up to 8 times (``NwpRunNotYetAvailable`` / ``NwpVariableWhollyMissing`` →
    ``RetryRequested`` in ``defs/assets.py``) rather than failing outright; any other error still
    fails immediately.
    Live inference (``live_forecasts``) always uses the freshest run genuinely present
    regardless of this schedule's exact timing.

    Further reading:
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#a-wholly-missing-variable-is-retried-not-failed-outright>
    — why a wholly-missing variable is retried rather than failed.
    """
    return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))


live_forecasts_job = define_asset_job(
    "live_forecasts_job",
    selection=AssetSelection.assets("live_forecasts"),
    hooks={sentry_capture_failure},
    description=(
        "Production inference: forecast from the latest NWP for one 6-hourly slot and write it"
        " to the power_forecasts Delta table. Runs at 00/06/12/18 UTC with"
        " availability_mode='live'; see the live_forecasts asset docstring for partition"
        " semantics and what a degraded run looks like."
    ),
)

# `name` is explicit because `build_schedule_from_partitioned_job` otherwise derives the
# registered name from the job's — `live_forecasts_job_schedule` — which is not the name this
# variable, the runbook, and the operator's Dagster UI all use.
live_forecasts_schedule = build_schedule_from_partitioned_job(
    live_forecasts_job,
    name="live_forecasts_schedule",
    description=(
        "Ticks at 00/06/12/18 UTC and materialises the just-completed window with"
        " availability_mode='live'. The schedule is always live; replays are manual, launched"
        " from the UI with availability_mode='replay'. The slot fires on the clock whether or"
        " not the ingest jobs succeeded."
    ),
)
"""Ticks at 00/06/12/18 UTC, materialising the just-completed window with default run config
(``availability_mode="live"``) — the schedule is always live; replays are manual, launched from
the UI with ``availability_mode="replay"``. This slot fires on the clock regardless of whether
the ingest jobs succeeded; see ``power_time_series_and_metadata_schedule``'s docstring above for
why the two schedules are deliberately not ordered against each other."""
