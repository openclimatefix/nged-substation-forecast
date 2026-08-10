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
)

power_time_series_and_metadata_schedule = ScheduleDefinition(
    name="power_time_series_and_metadata_schedule",
    job=power_time_series_and_metadata_job,
    cron_schedule="55 * * * *",
)
"""Fires at :55 past every hour — 5 minutes *before* the top of the hour — so this hour's pull
has landed by the time ``live_forecasts_schedule`` ticks at 00/06/12/18 UTC.

``live_forecasts`` declares ``power_time_series_and_metadata`` as a dep, but the two run as
separate jobs on separate schedules and nothing enforces the ordering at runtime. That is
deliberate, not a gap: the offset is an optimisation for freshness, and if it is missed —
because this pull failed, or ran long — ``live_forecasts`` still runs on time against whatever
telemetry is already on disk. Making the ordering a precondition would convert one failed
ingest into a missing forecast, which is the failure mode the whole design exists to avoid.

(The NWP run used is stamped on every forecast row as ``nwp_init_time``, and
``live_forecasts_are_healthy`` reports how many daily NWP runs were missing at forecast time;
telemetry staleness is not yet recorded on the forecast row itself.) Rule: "never make one
production job's run status a precondition for another's" —
<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules>."""

ecmwf_ens_job = define_asset_job(
    "ecmwf_ens_job",
    selection=AssetSelection.assets("ecmwf_ens"),
    hooks={sentry_capture_failure},
)


@schedule(job=ecmwf_ens_job, cron_schedule="30 8 * * *", execution_timezone="UTC")
def ecmwf_ens_schedule(context: ScheduleEvaluationContext) -> RunRequest:
    """Materialise today's ``ecmwf_ens`` partition daily at 08:30 UTC.

    08:30 UTC is a safety margin past the 00Z run's expected publication time (roughly 08:00
    UTC / 9am BST); ``ecmwf_ens_partitions``' ``end_offset=1`` means today's partition key
    already exists by this point. If the run isn't usable yet — absent from the catalog, or
    present with a weather variable still wholesale empty — ``ecmwf_ens`` retries every 30
    minutes for up to 4 hours (``NwpRunNotYetAvailable`` / ``NwpVariableWhollyMissing`` →
    ``RetryRequested`` in ``defs/assets.py``) rather than failing outright; any other error still
    fails immediately.
    Live inference (``live_forecasts``) always uses the freshest run genuinely present
    regardless of this schedule's exact timing.
    """
    return RunRequest(partition_key=context.scheduled_execution_time.strftime("%Y-%m-%d"))


live_forecasts_job = define_asset_job(
    "live_forecasts_job",
    selection=AssetSelection.assets("live_forecasts"),
    hooks={sentry_capture_failure},
)

live_forecasts_schedule = build_schedule_from_partitioned_job(live_forecasts_job)
"""Ticks at 00/06/12/18 UTC, materialising the just-completed window with default run config
(``availability_mode="live"``) — the schedule is always live; replays are manual, launched from
the UI with ``availability_mode="replay"``. This slot fires on the clock regardless of whether
the ingest jobs succeeded; see ``power_time_series_and_metadata_schedule``'s docstring above for
why the two schedules are deliberately not ordered against each other."""
