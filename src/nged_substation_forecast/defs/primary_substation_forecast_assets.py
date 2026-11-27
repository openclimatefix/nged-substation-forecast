"""The daily production forecast for the 16 Primaries, from the step-5 neural network.

`live_primary_substation_forecasts` runs `substation_demand_forecaster` on the day's 00Z ECMWF ENS
run and writes one row per Primary and half-hour to the `power_forecasts` Delta table.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import patito as pt
import polars as pl
from contracts.common import DELIVERY_QUANTILES
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS, PowerForecast
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import if_local_path_then_make_parent_dir
from contracts.weather_schemas import (
    ECMWF_ENS_ENSEMBLE_MEMBERS,
    ECMWF_ENS_LEAD_TIME_HOURS,
    Nwp,
    NwpModelId,
)
from dagster import (
    AssetExecutionContext,
    DagsterExecutionInterruptedError,
    MaterializeResult,
    RetryRequested,
    asset,
)
from delta_store.power_forecasts import write_power_forecasts
from nged_data.storage import scan_cleaned_power
from substation_demand_forecaster import MODEL_NAME
from substation_demand_forecaster.bundle import Bundle, load_bundle, require_servable_bundle
from substation_demand_forecaster.ens_summary import (
    SOURCE_COLUMNS,
    WIND_COMPONENTS,
    summarise_runs,
)
from substation_demand_forecaster.features import (
    build_expected_rows,
    build_input_matrix,
    native_origin,
)
from substation_demand_forecaster.neural_network import forecast

from nged_substation_forecast._sentry import (
    PRIMARY_SUBSTATION_FORECAST_DEGRADED_FINGERPRINT,
    report_asset_degradation,
    report_asset_retry,
    send_primary_substation_forecast_checkin,
)
from nged_substation_forecast.defs._tags import PRODUCTION_LAYER_TAGS
from nged_substation_forecast.defs.assets import (
    _available_nwp_init_times,
    _is_too_old_to_retry,
    ecmwf_ens_partitions,
)

ASSET_NAME: Final[str] = "live_primary_substation_forecasts"

EXPERIMENT_NAME: Final[str] = "primary_substation_demand_forecaster"
"""The `experiment_name` of every row this asset writes, naming the family of forecasts."""

FORECAST_INIT_HOUR_UTC: Final[int] = 12
"""`power_fcst_init_time` is the partition's date at 12:00 UTC: the contract origin."""

FORECAST_HORIZON: Final[timedelta] = timedelta(days=14)
"""The asset keeps the valid times strictly after `power_fcst_init_time` and strictly before
`power_fcst_init_time + FORECAST_HORIZON`: 671 half-hours, from 12:30 UTC to 11:30 UTC 14 days
later."""

HISTORY_DAYS: Final[int] = 13
"""How many daily 00Z runs before the day's run the forecast reads: enough to reach the 12 days
before the 09:00 UTC native origin that the cold-snap column looks back over."""

POWER_HISTORY: Final[timedelta] = timedelta(days=9)
"""How far before the native origin the cleaned-power read reaches; the week lag reaches 8 days
back at most."""

MAX_RETRIES: Final[int] = 6
"""Retries of a missing day's run, `RETRY_DELAY_SECONDS` apart, so until 15:00 UTC.

That is 30 minutes after the last retry of `ecmwf_ens` (8 retries from 10:30 UTC), which can itself
take about 10 minutes to finish."""

RETRY_DELAY_SECONDS: Final[int] = 1800

_ENS_MODEL_ID: Final[str] = NwpModelId.ECMWF_ENS_0_25_degree.name

_RADIATION_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in SOURCE_COLUMNS.values() if column in Nwp.radiation_var_names
)


class _DayRunMissing(Exception):
    """The day's 00Z ENS run is not in the `nwp` table yet."""


def partition_nwp_init_time(partition_key: str) -> datetime:
    """Return the 00Z ENS run of an `ecmwf_ens_partitions` key (`YYYY-MM-DD`)."""
    return datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=UTC)


def history_init_times(nwp_init_time: datetime) -> list[datetime]:
    """Return the 00Z runs of the `HISTORY_DAYS` days before `nwp_init_time`, oldest first."""
    return [nwp_init_time - timedelta(days=days) for days in range(HISTORY_DAYS, 0, -1)]


def _read_native_nwp(
    *, settings: Settings, bundle: Bundle, nwp_init_time: datetime
) -> pl.DataFrame:
    """Return the native ENS rows of the day's run and the history runs, in the Primaries' cells.

    Args:
        settings: The settings naming the `nwp` table.
        bundle: The bundle's Primaries and their H3 cells.
        nwp_init_time: The day's 00Z run.

    Returns:
        The `nwp` table's `h3_index`, `init_time`, `ensemble_member`, `valid_time`, and the
        `SOURCE_COLUMNS` and `WIND_COMPONENTS` values, for runs initialised from `HISTORY_DAYS`
        days before `nwp_init_time` up to it. Runs initialised later, which a backfill can see,
        are left out.
    """
    cells = bundle.primaries["h3_index"].unique().to_list()
    return (
        pl.scan_delta(
            settings.nwp_data_path, storage_options=typeddict_to_dict(settings.storage_options)
        )
        .filter(
            pl.col("nwp_model_id") == _ENS_MODEL_ID,
            pl.col("init_time").is_between(
                nwp_init_time - timedelta(days=HISTORY_DAYS), nwp_init_time
            ),
            pl.col("h3_index").is_in(cells),
        )
        .select(
            "h3_index",
            "init_time",
            "ensemble_member",
            "valid_time",
            *SOURCE_COLUMNS.values(),
            *WIND_COMPONENTS,
        )
        .collect()
    )


def describe_day_run_faults(
    *, native: pl.DataFrame, nwp_init_time: datetime, bundle: Bundle
) -> list[str]:
    """Return one sentence per way the day's run falls short of a whole run in the Primaries' cells.

    The forecast is still made from the members and values present, and a value still missing goes
    through the standardiser as the training mean, so these faults degrade the forecast silently
    unless they are reported. The five cases, in the order they are returned, are a cell with no
    rows, a cell with fewer than 51 members, a lead missing inside the run, a run that stops short
    of 360 hours, and radiation that is null after lead 0.

    Args:
        native: `_read_native_nwp`'s rows; only the day's run is read.
        nwp_init_time: The day's 00Z run.
        bundle: The bundle's Primaries and their H3 cells.

    Returns:
        One sentence per case that occurs; empty for a whole run.
    """
    run = native.filter(pl.col("init_time") == nwp_init_time)
    faults = []
    cells = set(bundle.primaries["h3_index"].to_list())
    missing_cells = sorted(cells - set(run["h3_index"].unique().to_list()))
    if missing_cells:
        faults.append(f"no rows in H3 cell(s) {missing_cells}")
    members = run.group_by("h3_index").agg(n=pl.col("ensemble_member").n_unique())
    short_cells = members.filter(pl.col("n") < len(ECMWF_ENS_ENSEMBLE_MEMBERS)).sort("h3_index")
    if short_cells.height:
        faults.append(
            f"fewer than {len(ECMWF_ENS_ENSEMBLE_MEMBERS)} members in H3 cell(s) "
            f"{short_cells['h3_index'].to_list()} (fewest {short_cells['n'].min()})"
        )
    missing_leads: dict[int, list[int]] = {}
    short: dict[int, int] = {}
    for cell, valid_times in (
        run.group_by("h3_index").agg(pl.col("valid_time").unique()).sort("h3_index").iter_rows()
    ):
        leads = {
            int((valid_time - nwp_init_time) / timedelta(hours=1)) for valid_time in valid_times
        }
        last = max(leads)
        gaps = [lead for lead in ECMWF_ENS_LEAD_TIME_HOURS if lead < last and lead not in leads]
        if gaps:
            missing_leads[cell] = gaps
        if last < ECMWF_ENS_LEAD_TIME_HOURS[-1]:
            short[cell] = last
    if missing_leads:
        faults.append(f"missing lead time(s) in hours, by H3 cell: {missing_leads}")
    if short:
        faults.append(
            f"the run stops short of {ECMWF_ENS_LEAD_TIME_HOURS[-1]} h, by H3 cell: {short}"
        )
    after_lead_0 = run.filter(pl.col("valid_time") > nwp_init_time)
    for column in _RADIATION_COLUMNS:
        n_null = after_lead_0[column].null_count()
        if n_null:
            faults.append(f"{n_null} null {column} value(s) after lead 0")
    return faults


def _read_cleaned_power(
    *, settings: Settings, bundle: Bundle, nwp_init_time: datetime
) -> pl.DataFrame:
    """Return the Primaries' cleaned readings in the `POWER_HISTORY` before the native origin.

    Args:
        settings: The settings naming the `cleaned_power_time_series` table.
        bundle: The bundle's Primaries.
        nwp_init_time: The day's 00Z run.

    Returns:
        `time_series_id` (Int32), `time` (Datetime, UTC), and `power` (Float32).
    """
    origin = native_origin(nwp_init_time)
    return (
        scan_cleaned_power(
            delta_path=settings.cleaned_power_time_series_data_path,
            storage_options=settings.storage_options,
        )
        .filter(
            pl.col("time_series_id").is_in(bundle.primaries["time_series_id"].to_list()),
            pl.col("time") > origin - POWER_HISTORY,
            pl.col("time") <= origin,
        )
        .collect()
    )


def contract_rows(
    *, keys: pl.DataFrame, quantiles: pl.DataFrame, bundle: Bundle, nwp_init_time: datetime
) -> pt.DataFrame[PowerForecast]:
    """Return the `PowerForecast` rows of one forecast: megawatts, at the 12:00 UTC origin.

    Keeps the valid times strictly after `power_fcst_init_time` and strictly before
    `FORECAST_HORIZON` after it, multiplies each fraction of capacity by the Primary's
    `capacity_mw`, and clamps every quantile of an `MVA` Primary at zero, because apparent power
    cannot be negative. The clamp keeps each row's quantiles in non-decreasing order.

    Args:
        keys: `InputMatrix.keys`.
        quantiles: One column per `POWER_FCST_QUANTILE_COLUMNS` entry, as fractions of capacity,
            row for row with `keys`.
        bundle: The bundle's Primaries, capacities, units, and release number.
        nwp_init_time: The day's 00Z run.

    Returns:
        Validated rows, `fold_id` `"live"`.
    """
    init = nwp_init_time + timedelta(hours=FORECAST_INIT_HOUR_UTC)
    is_mva = pl.col("unit") == "MVA"
    megawatts = [
        pl.when(is_mva)
        .then((pl.col(column) * pl.col("capacity_mw")).clip(lower_bound=0.0))
        .otherwise(pl.col(column) * pl.col("capacity_mw"))
        .cast(pl.Float32)
        .alias(column)
        for column in POWER_FCST_QUANTILE_COLUMNS
    ]
    rows = (
        pl.concat(
            [keys.select("primary_index", "time_series_id", "valid_time"), quantiles],
            how="horizontal",
        )
        .join(
            bundle.primaries.select("primary_index", "capacity_mw", "unit"),
            on="primary_index",
            how="left",
            maintain_order="left",
        )
        .filter(pl.col("valid_time") > init, pl.col("valid_time") < init + FORECAST_HORIZON)
        .select(
            "valid_time",
            "time_series_id",
            *megawatts,
            ml_flow_experiment_id=pl.lit(None, dtype=pl.Int32),
            nwp_init_time=pl.lit(nwp_init_time, dtype=pl.Datetime("us", "UTC")),
            power_fcst_model_name=pl.lit(MODEL_NAME),
            experiment_name=pl.lit(EXPERIMENT_NAME),
            power_fcst_model_version=pl.lit(bundle.model_release, dtype=pl.Int16),
            power_fcst_init_time=pl.lit(init, dtype=pl.Datetime("us", "UTC")),
            fold_id=pl.lit("live"),
        )
    )
    return PowerForecast.validate(rows)


def _report_degradation(*, kind: str, message: str, settings: Settings) -> None:
    """Send one synthesised degradation to Sentry, grouped per kind and per environment."""
    report_asset_degradation(
        asset_name=ASSET_NAME,
        exc=RuntimeError(message),
        fingerprint=[
            PRIMARY_SUBSTATION_FORECAST_DEGRADED_FINGERPRINT,
            kind,
            settings.sentry_environment,
        ],
    )


def _report_input_faults(
    *,
    context: AssetExecutionContext,
    settings: Settings,
    native: pl.DataFrame,
    nwp_init_time: datetime,
    bundle: Bundle,
) -> list[datetime]:
    """Report the missing history runs and the day's run faults; return the missing runs.

    Rule 7 of the inherent-stability rules puts this warning path under a catch-all, so a bug in
    the probe degrades the report rather than failing the forecast:
    <https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules>.
    """
    run = nwp_init_time.date().isoformat()
    try:
        present = set(native["init_time"].unique().to_list())
        missing = [init for init in history_init_times(nwp_init_time) if init not in present]
        if missing:
            names = ", ".join(init.date().isoformat() for init in missing)
            message = (
                f"{ASSET_NAME} for the {run} 00Z ENS run is missing {len(missing)} of the"
                f" {HISTORY_DAYS} history run(s) ({names}). The thermal-lag and solar-gain columns"
                " were computed from the history that remains, and the lags' expected-weather rows"
                " fell back to the next-freshest run."
            )
            context.log.warning(message)
            _report_degradation(kind="history_runs_missing", message=message, settings=settings)
        faults = describe_day_run_faults(native=native, nwp_init_time=nwp_init_time, bundle=bundle)
        if faults:
            message = (
                f"{ASSET_NAME} forecast from an incomplete {run} 00Z ENS run: {'; '.join(faults)}."
                " The forecast used what is present: a missing lead is interpolated from its"
                " neighbours, fewer members give the ensemble mean and spread of the members"
                " present, and an input still missing takes its training mean."
            )
            context.log.warning(message)
            _report_degradation(kind="day_run_incomplete", message=message, settings=settings)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        # No fingerprint: this exception was caught, so it carries the stack trace Sentry groups on.
        context.log.exception("Could not check the ENS runs the forecast reads")
        report_asset_degradation(asset_name=ASSET_NAME, exc=exc)
        return []
    return missing


def months_past_refit(*, bundle: Bundle, nwp_init_time: datetime) -> int:
    """Return how many calendar months the day's run falls after the bundle's refit month.

    A real bundle's last offset knot is its refit month, so a positive value means the forecast's
    valid times sit past the last knot, where the offset is held flat. A monthly refit keeps the
    value at 0.
    """
    refit = datetime.strptime(bundle.refit_month, "%Y-%m").replace(tzinfo=UTC)
    return (nwp_init_time.year - refit.year) * 12 + nwp_init_time.month - refit.month


def _report_stale_bundle(*, settings: Settings, bundle: Bundle, nwp_init_time: datetime) -> None:
    months = months_past_refit(bundle=bundle, nwp_init_time=nwp_init_time)
    if months > 0:
        _report_degradation(
            kind="bundle_stale",
            message=(
                f"{ASSET_NAME} forecast the {nwp_init_time.date().isoformat()} 00Z ENS run with the"
                f" {bundle.refit_month} bundle, {months} month(s) after its refit month. The"
                " offset is held at the last knot's value; deploy the current month's bundle."
            ),
            settings=settings,
        )


def _wait_for_day_run(
    *, context: AssetExecutionContext, settings: Settings, nwp_init_time: datetime
) -> bool:
    """Return whether the day's run is in the `nwp` table; request a retry while one is left.

    Args:
        context: The asset's context, for its retry number and its log.
        settings: The settings naming the `nwp` table and the Sentry environment.
        nwp_init_time: The day's 00Z run.

    Returns:
        True when the run is present. False when it is absent and no retry is left, or the run is
        too old to retry; the degradation has then been reported.

    Raises:
        RetryRequested: When the run is absent and a retry is left.
    """
    if nwp_init_time in set(_available_nwp_init_times(settings)):
        return True
    run = nwp_init_time.date().isoformat()
    missing = _DayRunMissing(f"The {run} 00Z ECMWF ENS run is not in the nwp table.")
    too_old = _is_too_old_to_retry(nwp_init_time=nwp_init_time, now=datetime.now(UTC))
    if too_old or context.retry_number >= MAX_RETRIES:
        message = (
            f"{ASSET_NAME} wrote no forecast for the {run} 00Z ENS run, because the run is not in"
            " the nwp table. The service has no forecast for that day."
        )
        context.log.warning(message)
        _report_degradation(kind="day_run_missing", message=message, settings=settings)
        return False
    if context.retry_number == 0:
        report_asset_retry(asset_name=ASSET_NAME, exc=missing)
    context.log.warning(f"{missing} Requesting a retry in {RETRY_DELAY_SECONDS} s.")
    raise RetryRequested(max_retries=MAX_RETRIES, seconds_to_wait=RETRY_DELAY_SECONDS) from missing


@asset(
    tags=PRODUCTION_LAYER_TAGS,
    partitions_def=ecmwf_ens_partitions,
    deps=["ecmwf_ens", "clean_nged_power_data"],
)
def live_primary_substation_forecasts(context: AssetExecutionContext) -> MaterializeResult:
    """Forecast the 16 Primaries' demand for 14 days from the day's 00Z ECMWF ENS run.

    **What it reads and writes.** The partition is a day of ``ecmwf_ens_partitions``. The asset
    reads that day's 00Z ENS run and the 13 daily runs before it from the ``nwp`` Delta table that
    ``ecmwf_ens`` writes, and the Primaries' cleaned readings from the
    ``cleaned_power_time_series`` table that ``clean_nged_power_data`` writes. It loads the step-5
    neural network's bundle. The Primaries and their capacities come from the bundle, never from
    the ``TimeSeriesMetadata`` table. It writes to ``power_forecasts``, under ``experiment_name``
    ``primary_substation_demand_forecaster`` and ``fold_id`` ``"live"``, with
    ``power_fcst_init_time`` at 12:00 UTC on the partition's day. A re-run replaces that
    ``power_fcst_init_time``'s rows.

    **When it runs.** ``live_primary_substation_forecasts_schedule`` fires it at 12:00 UTC, after
    ``ecmwf_ens_schedule`` (10:30 UTC) has normally written the day's run. The two schedules are
    coupled only through the ``nwp`` table on disk, never through run status. On success for
    today's partition, the asset sends a check-in to the Sentry cron monitor
    ``live-primary-substation-forecasts``, which alarms if no check-in arrives by 16:00 UTC.

    **What a degraded run looks like.** Each of these reaches Sentry tagged
    ``degraded_asset=live_primary_substation_forecasts``:

    - The day's run is absent: the asset retries every 30 minutes up to 6 times, with a Sentry
      warning on the first retry. If the run is still absent, the asset reports the degradation
      and finishes without writing, so that day has no forecast. A backfill of a run more than 36
      hours old does not retry.
    - A history run is missing: the thermal-lag and solar-gain columns are computed from the
      history that remains, as research computes them, the lags' expected-weather rows fall back
      to the next-freshest run, and the asset names the missing runs.
    - The day's run is incomplete (fewer members, a missing lead, a short run, a missing cell, or
      null radiation), judged cell by cell: the forecast uses the values present and the asset
      names the faults.
    - No ENS run covers some of the lags' expected-weather rows: those rows read zeros, and the
      asset counts them.
    - The day's run falls in a month after the bundle's refit month: the offset is held at the last
      knot's value, and the asset names the bundle.
    - A day or week lag has no cleaned reading: the neural network reads its missing encoding,
      which it was trained with, and the forecast degrades towards the lag-free forecast. This is
      not reported, because stale telemetry has its own check, ``power_data_is_fresh``.

    A bundle that is missing, has an unknown format version, or was trained on CAMS values the
    service cannot reproduce raises, because that is a deployment fault of ours. A backfill reads
    the runs initialised up to its own day, so it sees the weather the live run saw; its lags can
    differ from the live run's, because late telemetry is now on disk.
    """
    settings = Settings()
    nwp_init_time = partition_nwp_init_time(context.partition_key)
    bundle = load_bundle(Path(settings.primary_substation_bundle_path))
    require_servable_bundle(bundle)
    if not _wait_for_day_run(context=context, settings=settings, nwp_init_time=nwp_init_time):
        return MaterializeResult(metadata={"nwp_init_time": str(nwp_init_time), "n_rows": 0})
    # After the wait, so a run that retried reports the stale bundle once, not on every attempt.
    _report_stale_bundle(settings=settings, bundle=bundle, nwp_init_time=nwp_init_time)

    native = _read_native_nwp(settings=settings, bundle=bundle, nwp_init_time=nwp_init_time)
    missing_runs = _report_input_faults(
        context=context,
        settings=settings,
        native=native,
        nwp_init_time=nwp_init_time,
        bundle=bundle,
    )
    summary = summarise_runs(native.lazy(), bundle.primaries).collect()
    matrix = build_input_matrix(
        bundle=bundle,
        summary=summary,
        cleaned_power=_read_cleaned_power(
            settings=settings, bundle=bundle, nwp_init_time=nwp_init_time
        ),
        nwp_init_time=nwp_init_time,
    )
    expected_rows = build_expected_rows(bundle=bundle, summary=summary, nwp_init_time=nwp_init_time)
    n_uncovered = int(np.isnat(expected_rows.run_init_time).sum())
    if n_uncovered:
        message = (
            f"{ASSET_NAME} for the {nwp_init_time.date().isoformat()} 00Z ENS run has"
            f" {n_uncovered} expected-weather row(s) that no ENS run covers. A lag that reads one"
            " of those rows reads zeros."
        )
        context.log.warning(message)
        _report_degradation(kind="expected_weather_uncovered", message=message, settings=settings)
    quantiles = forecast(
        bundle=bundle,
        matrix=matrix.standardised,
        primary_index=matrix.keys["primary_index"].to_numpy(),
        expected_rows=expected_rows,
        levels=DELIVERY_QUANTILES,
    )
    rows = contract_rows(
        keys=matrix.keys,
        quantiles=pl.DataFrame(
            dict(zip(POWER_FCST_QUANTILE_COLUMNS, quantiles.mean.T, strict=True))
        ),
        bundle=bundle,
        nwp_init_time=nwp_init_time,
    )

    power_fcst_init_time = rows["power_fcst_init_time"][0]
    if_local_path_then_make_parent_dir(settings.power_forecasts_data_path)
    write_power_forecasts(
        rows,
        settings.power_forecasts_data_path,
        replace_partition=(EXPERIMENT_NAME, "live"),
        replace_predicate_extra=f"power_fcst_init_time = '{power_fcst_init_time.isoformat()}'",
        storage_options=settings.storage_options,
    )
    # The check-in says "the live service is alive today", so a backfill of a past day sends none.
    if nwp_init_time.date() == datetime.now(UTC).date():
        send_primary_substation_forecast_checkin(settings)

    return MaterializeResult(
        metadata={
            "nwp_init_time": str(nwp_init_time),
            "power_fcst_init_time": str(power_fcst_init_time),
            "n_rows": rows.height,
            "n_time_series": rows["time_series_id"].n_unique(),
            "model_release": bundle.model_release,
            "research_commit": bundle.research_commit,
            "refit_month": bundle.refit_month,
            "missing_history_runs": ", ".join(init.date().isoformat() for init in missing_runs),
            "n_uncovered_expected_weather_rows": n_uncovered,
        }
    )
