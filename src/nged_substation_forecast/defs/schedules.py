"""Dagster schedules for the NGED Substation Forecast project."""

from dagster import (
    AssetSelection,
    RunRequest,
    ScheduleDefinition,
    ScheduleEvaluationContext,
    define_asset_job,
    schedule,
)

from nged_substation_forecast._sentry import sentry_capture_failure

# Define a job that targets the power_time_series_and_metadata asset and the clean_nged_power_data
# asset that cleans its output
power_time_series_and_metadata_job = define_asset_job(
    name="power_time_series_and_metadata_job",
    selection=AssetSelection.assets("power_time_series_and_metadata", "clean_nged_power_data"),
    hooks={sentry_capture_failure},
    description=(
        "Pull the latest NGED telemetry from S3 into the power_time_series Delta table, upsert"
        " the substation metadata parquet, then rebuild the cleaned_power_time_series Delta table"
        " if new telemetry arrived. Runs hourly at :55."
    ),
)

power_time_series_and_metadata_schedule = ScheduleDefinition(
    name="power_time_series_and_metadata_schedule",
    job=power_time_series_and_metadata_job,
    cron_schedule="55 * * * *",
    description=(
        "Fires at :55 past every hour. Nothing downstream waits on this job's run status: an"
        " asset that reads the cleaned telemetry runs on its own schedule against whatever"
        " cleaned telemetry is already on disk, so a missed or late pull never holds it back."
    ),
)
"""Fires at :55 past every hour, so this hour's pull has landed, and been cleaned by
``clean_nged_power_data``, a few minutes before the top of the hour.

No downstream job waits on this job's run status, so a failed or late pull never holds back an
asset that reads the cleaned telemetry. Rule: "never make one production job's run status a
precondition for another's" —
<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules>.
The principle in full, with the trade it makes:
<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#14-production-jobs-are-coupled-through-data-at-rest-never-through-run-status>."""

ecmwf_ens_job = define_asset_job(
    "ecmwf_ens_job",
    selection=AssetSelection.assets("ecmwf_ens"),
    hooks={sentry_capture_failure},
    description=(
        "Download the day's ECMWF ENS NWP run and write it to the nwp Delta table, replacing"
        " that (nwp_model_id, init_time) partition. Runs daily at 10:30 UTC; see"
        " ecmwf_ens_schedule for the retry behaviour when the run is not yet published."
    ),
)


@schedule(job=ecmwf_ens_job, cron_schedule="30 10 * * *", execution_timezone="UTC")
def ecmwf_ens_schedule(context: ScheduleEvaluationContext) -> RunRequest:
    """Materialise today's ``ecmwf_ens`` partition daily at 10:30 UTC.

    10:30 UTC is a safety margin past the 00Z run's usual arrival on ECMWF's bucket and
    Dynamical.org's publication (08:05 to 08:20 UTC on a normal day). ``ecmwf_ens_partitions``'
    ``end_offset=1`` means today's partition key already exists by this point. A run isn't usable
    yet when it is absent from the catalog, when whole (member, lead time) slices of an
    instantaneous variable are still empty, or when a de-accumulated variable is still wholly
    empty. The same retry covers a failed fetch of the catalog itself. ``ecmwf_ens`` then retries
    every 30 minutes, up to 8 times (``NwpRunNotYetAvailable`` / ``NwpVariableWhollyMissing`` /
    ``NwpCatalogUnreachable`` → ``RetryRequested`` in ``defs/assets.py``), rather than failing
    outright. Any other error still
    fails immediately, and so do the four retried failures for a run initialised more than 36 hours
    ago, which a backfill, or a catch-up tick more than about 25 hours late, can request. Retrying
    is right because Dynamical.org publishes each run as roughly 40 separate commits over about 15
    minutes, so a run can be readable while a variable whose commit has not landed yet still reads
    as empty — that is a run mid-publication, not a broken one.

    Further reading:
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#an-empty-slice-or-a-wholly-missing-variable-is-retried-not-failed-outright>
    — why an unfinished run is retried rather than failed.
    """
    return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))


live_primary_substation_forecasts_job = define_asset_job(
    "live_primary_substation_forecasts_job",
    selection=AssetSelection.assets("live_primary_substation_forecasts"),
    hooks={sentry_capture_failure},
    description=(
        "Forecast the 16 Primaries for 14 days from the day's 00Z ECMWF ENS run with the step-5"
        " neural network, and write the 13 quantiles to power_forecasts at the 12:00 UTC origin."
    ),
)


@schedule(
    job=live_primary_substation_forecasts_job, cron_schedule="0 12 * * *", execution_timezone="UTC"
)
def live_primary_substation_forecasts_schedule(context: ScheduleEvaluationContext) -> RunRequest:
    """Materialise today's ``live_primary_substation_forecasts`` partition daily at 12:00 UTC.

    The forecast's origin is 12:00 UTC, and by then ``ecmwf_ens_schedule`` (10:30 UTC) has normally
    written the day's 00Z run.
    """
    return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))
