"""Dagster asset checks: the operator's at-a-glance data-health status.

``power_data_is_fresh`` warns when NGED's telemetry feed has stalled — no new
``power_time_series`` data for over ``_POWER_DATA_STALENESS_THRESHOLD``. The hourly
``power_time_series_and_metadata_job`` runs this check every time it materialises the asset, so
Dagster's Checks view is the single "is the data up to date and healthy?" surface: a green tick
when every series is current, a yellow **WARN** naming the late count when the feed has stalled.

The check reads the Delta table's *actual* data recency, not the asset's materialisation
timestamp — the job succeeds hourly even when NGED publishes nothing, so only the on-disk
``time`` reveals whether fresh data really landed. A native materialisation-freshness policy
would miss exactly the failure this check exists to catch.

``evaluate_power_freshness`` is a pure function so it is unit-testable without Dagster or Delta,
and it is the hand-off point for routing per-series staleness to Sentry: the same
``PowerFreshnessResult`` is fed to ``report_power_freshness`` (in ``nged_substation_forecast._sentry``)
rather than recomputed. The two mechanisms stay complementary — the
[Sentry missed-check-in alarm](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
fires on total silence from outside the deployment, while this check (and its Sentry warning) report
per-series staleness from inside Dagster while the daemon is alive.

``live_forecasts_are_healthy`` does the same job for the one asset NGED actually consumes. It
answers two questions the asset's own success status cannot: did this 6-hourly slot really land
valid forecast rows on disk, and how many daily NWP runs were missing when the forecast was made?
Both are read back from disk after the write, so a run that "succeeded" while writing nothing —
or writing null/non-finite forecasts, hindcast rows, or a short population — still shows up.
Missed NWP runs are *counted as runs*, never measured in hours of age: healthy NWP is 12–30 hours
old depending on the slot, so any absolute age threshold would fire on two slots in four every
day. See
[Inherent Stability → Three audiences, three channels](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#three-audiences-three-channels).

Both checks are ``AssetCheckSeverity.WARN`` and ``blocking=False``, and neither can raise: a
warning path that fails would turn fail-open into fail-closed at exactly the wrong moment (rule 7
of [The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).
"""

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import polars as pl
from contracts._uri import ObjectStoreOptions, delta_table_exists, object_exists
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    MetadataValue,
    TableColumn,
    TableRecord,
    TableSchema,
    asset_check,
)
from nged_data.storage import time_series_coverage

from nged_substation_forecast._sentry import report_power_freshness
from nged_substation_forecast.defs.assets import power_time_series_and_metadata
from nged_substation_forecast.defs.production_assets import (
    _available_nwp_init_times,
    live_forecasts,
)

logger = logging.getLogger(__name__)

_LATE_TABLE_SCHEMA: Final[TableSchema] = TableSchema(
    columns=[
        TableColumn("time_series_id", "int"),
        TableColumn("last_seen", "string", description="Most recent data on disk, or 'never'."),
        TableColumn("hours_late", "float", description="Hours since last data (null if never)."),
        TableColumn("status", "string", description="'stale' or 'never'."),
    ]
)
"""Fixed schema for the late-series metadata table (required so an empty table still renders)."""

_LATE_STATUS_ORDER: Final[tuple[str, ...]] = ("never", "stale")
"""Runtime tuple — declared order for the ``status`` column's ``pl.Enum``, which is also the
row order in the late-series table (never-reported series listed before merely-stale ones)."""

_POWER_DATA_STALENESS_THRESHOLD: Final[timedelta] = timedelta(hours=24)
"""A ``time_series_id`` is 'late' if its most recent observation is older than this.

NGED publishes roughly every 6 hours and our pipeline back-fills gaps automatically once the
feed recovers, so 24 hours is comfortably past normal jitter while still catching a genuine
multi-slot stall the same day."""


@dataclass(frozen=True)
class PowerFreshnessResult:
    """Outcome of a power-data freshness evaluation.

    ``late`` lists every late series — never-reported first, then most-stale first — with columns
    ``time_series_id``, ``last_seen`` (null if never reported), ``hours_late`` (null if never
    reported) and ``status`` (``"never"`` or ``"stale"``).
    """

    n_series_total: int
    n_stale: int
    n_never: int
    threshold_hours: float
    late: pl.DataFrame

    @property
    def n_late(self) -> int:
        """Total late series: those that went stale plus those that never reported."""
        return self.n_stale + self.n_never

    @property
    def is_healthy(self) -> bool:
        """True when no series is late."""
        return self.n_late == 0


def evaluate_power_freshness(
    coverage: pl.DataFrame,
    roster_ids: pl.Series | None,
    now: datetime,
    threshold: timedelta,
) -> PowerFreshnessResult:
    """Classify each time series as fresh, stale, or never-reported.

    Pure and deterministic — no Dagster, Delta, or clock access — so it is unit-testable
    directly and reused (not recomputed) by ``report_power_freshness`` for the Sentry warning.

    Args:
        coverage: One row per ``time_series_id`` that has data, carrying its most recent
            observation ``time`` in a ``last_time`` column (a ``TimeSeriesCoverage`` frame; any
            ``first_time`` column is ignored — freshness depends only on the latest observation).
        roster_ids: The full set of expected ``time_series_id``s (from the ``TimeSeriesMetadata``
            roster), used to flag ids that have *never* sent data. ``None`` when no roster is
            available, in which case never-reported ids cannot be detected.
        now: Current time (UTC).
        threshold: A series is stale when ``last_time < now - threshold``.

    Returns:
        A ``PowerFreshnessResult`` summarising the health of the power feed.
    """
    # Strip any Patito model so the frame-building below uses plain Polars semantics.
    coverage = pl.DataFrame._from_pydf(coverage._df)
    last_time_dtype = coverage.schema["last_time"]
    status_dtype = pl.Enum(_LATE_STATUS_ORDER)
    cutoff = now - threshold

    # Stale: has data on disk, but the newest observation predates the cutoff.
    #
    # NOTE: this is deliberately not restricted to `roster_ids`. A series that NGED has
    # decommissioned (dropped from the metadata roster) but that still has old rows on disk will
    # keep being flagged stale — which is what we want for now: we would rather be told about a
    # series that has gone quiet than silently stop watching it. If a permanently-yellow check
    # for a genuinely retired series becomes a nuisance, intersect the stale ids with
    # `roster_ids` here (when a roster is available) so only currently-expected series count.
    stale = coverage.filter(pl.col("last_time") < cutoff).select(
        "time_series_id",
        last_seen=pl.col("last_time"),
        hours_late=(pl.lit(now) - pl.col("last_time")).dt.total_seconds() / 3600.0,
        status=pl.lit("stale", dtype=status_dtype),
    )

    # Never reported: in the roster, but with no rows in the Delta table at all.
    if roster_ids is not None:
        never_ids = roster_ids.filter(~roster_ids.is_in(coverage["time_series_id"].implode()))
    else:
        never_ids = pl.Series("time_series_id", [], dtype=coverage.schema["time_series_id"])
    never = pl.DataFrame({"time_series_id": never_ids}).select(
        "time_series_id",
        last_seen=pl.lit(None, dtype=last_time_dtype),
        hours_late=pl.lit(None, dtype=pl.Float64),
        status=pl.lit("never", dtype=status_dtype),
    )

    # Never-reported first, then most-stale first. `status` is an ordered `pl.Enum` (never before
    # stale by declared order), so the ordering does not rely on the alphabetical accident that
    # "never" < "stale"; never-rows have a null `hours_late` but the status key keeps them ahead.
    late = pl.concat([never, stale]).sort(["status", "hours_late"], descending=[False, True])

    if roster_ids is not None:
        n_series_total = pl.concat([roster_ids, coverage["time_series_id"]]).n_unique()
    else:
        n_series_total = coverage.height

    return PowerFreshnessResult(
        n_series_total=n_series_total,
        n_stale=stale.height,
        n_never=never.height,
        threshold_hours=threshold.total_seconds() / 3600.0,
        late=late,
    )


def _read_roster_ids(
    metadata_path: str, storage_options: ObjectStoreOptions | None
) -> pl.Series | None:
    """Return the expected ``time_series_id``s from the metadata roster, or ``None`` if absent."""
    if not object_exists(metadata_path, storage_options):
        return None
    roster = (
        pl.scan_parquet(metadata_path, storage_options=typeddict_to_dict(storage_options))
        .select("time_series_id")
        .collect()
    )
    return roster["time_series_id"]


def _late_table_metadata(late: pl.DataFrame) -> MetadataValue:
    """Render the late-series frame as a Dagster table for the check's UI metadata."""
    records = [
        TableRecord(
            {
                "time_series_id": row["time_series_id"],
                "last_seen": "never" if row["last_seen"] is None else str(row["last_seen"]),
                "hours_late": None if row["hours_late"] is None else round(row["hours_late"], 1),
                "status": row["status"],
            }
        )
        for row in late.iter_rows(named=True)
    ]
    return MetadataValue.table(records, schema=_LATE_TABLE_SCHEMA)


def _to_asset_check_result(result: PowerFreshnessResult) -> AssetCheckResult:
    """Turn a ``PowerFreshnessResult`` into a WARN-severity Dagster check result."""
    threshold_h = result.threshold_hours
    if result.n_series_total == 0:
        description = "No power data on disk yet."
    elif result.is_healthy:
        description = (
            f"All {result.n_series_total} time series are up to date (within {threshold_h:.0f}h)."
        )
    else:
        description = (
            f"{result.n_late}/{result.n_series_total} time series are late: "
            f"{result.n_stale} stale (>{threshold_h:.0f}h since last data), "
            f"{result.n_never} never reported."
        )
    return AssetCheckResult(
        # A stalled feed is expected to self-heal via back-fill, so warn — never fail the run and
        # block downstream assets. Absent data is not "healthy" either, hence the count guard.
        passed=result.is_healthy and result.n_series_total > 0,
        severity=AssetCheckSeverity.WARN,
        description=description,
        metadata={
            "n_late": result.n_late,
            "n_stale": result.n_stale,
            "n_never_reported": result.n_never,
            "n_series_total": result.n_series_total,
            "threshold_hours": threshold_h,
            "late_time_series": _late_table_metadata(result.late),
        },
    )


@asset_check(
    asset=power_time_series_and_metadata,
    blocking=False,
    description=(
        "Warn if any time series has no fresh power data within the staleness threshold "
        "(stale) or has never reported at all (never)."
    ),
)
def power_data_is_fresh() -> AssetCheckResult:
    """Report how many time series are late on the ``power_time_series`` Delta table.

    Runs automatically alongside every ``power_time_series_and_metadata`` materialisation (hourly
    via ``power_time_series_and_metadata_schedule``), so the check re-evaluates freshness each
    hour regardless of whether new data landed.
    """
    settings = Settings()
    storage_options = settings.storage_options
    coverage = time_series_coverage(settings.power_time_series_data_path, storage_options)
    roster_ids = _read_roster_ids(settings.metadata_path, storage_options)
    result = evaluate_power_freshness(
        coverage=coverage,
        roster_ids=roster_ids,
        now=datetime.now(timezone.utc),
        threshold=_POWER_DATA_STALENESS_THRESHOLD,
    )
    # Forward per-series staleness to Sentry (a no-op unless a DSN is set and some series is late).
    # Best-effort: report_power_freshness never raises, so a telemetry hiccup can't fail this check.
    report_power_freshness(settings, result)
    return _to_asset_check_result(result)


# ---------------------------------------------------------------------------
# live_forecasts_are_healthy: did this slot write valid rows, and is the NWP feed whole?
# ---------------------------------------------------------------------------

_LIVE_FOLD_ID: Final[str] = "live"
"""The ``PowerForecast.fold_id`` sentinel marking a production forecast rather than a CV fold.

Mirrors the sentinel documented on ``contracts.power_schemas.FoldId`` and defaulted by
``BaseForecaster.predict``; it is also a Delta partition column, so filtering on it prunes the
whole CV half of ``power_forecasts`` before any data is read."""

_NWP_RUN_INTERVAL: Final[timedelta] = timedelta(days=1)
"""How often a new NWP run lands: one ECMWF ENS 00Z run per day (``ecmwf_ens_partitions``)."""

_NWP_RUN_EXPECTED_ON_DISK_BY: Final[timedelta] = timedelta(hours=14)
"""How long after its ``init_time`` a daily NWP run must be on disk before it counts as missed.

``ecmwf_ens_schedule`` fires at 08:30 UTC and ``ecmwf_ens`` retries a not-yet-published run every
30 minutes for up to 4 hours, so the latest moment a *healthy* ingest can land the day's 00Z run
is about 12:30 UTC; 14:00 UTC leaves margin for the download and write themselves. The deadline
is deliberately generous because the two errors cost very different amounts: too tight and the
check cries wolf daily (exactly the failure mode of an absolute age threshold), too loose and a
genuinely missed run is reported one 6-hourly slot later than it might have been. At 14 hours the
00:00, 06:00 and 12:00 slots expect yesterday's run and the 18:00 slot expects today's, so every
healthy slot reports zero missed runs and a failed download is reported from the 18:00 slot
onwards."""

_LIVE_SLOT_INTERVAL: Final[timedelta] = timedelta(hours=6)
"""The live forecast cadence (00/06/12/18 UTC), mirroring ``live_forecast_partitions``.

Used only to work out which slot to report on when the check is invoked without a partition key;
a partitioned run takes its slot from the partition itself."""

_MAX_MISSING_SERIES_LISTED: Final[int] = 20
"""Cap on how many missing ``time_series_id``s the description spells out.

Keeps the one-line description readable at V2 scale (~2,500 series) when a whole population is
missing; the true count is always carried by the ``n_time_series_missing`` metadata field, so a
truncated list never makes a large gap look small."""

_UNIX_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=timezone.utc)
"""Origin for ``_floor_to_interval``'s arithmetic. Both cadences we floor to (daily NWP runs at
00Z, 6-hourly forecast slots at 00/06/12/18) are aligned to it."""


@dataclass(frozen=True)
class LiveForecastRows:
    """One live slot's forecast rows on disk, as summarised by ``_read_live_forecast_rows``.

    Every field is an aggregate rather than the rows themselves: one slot is ~1M rows at V1 scale
    and ~86M at V2, so the summary is computed inside Polars and only these scalars come back.
    """

    n_rows: int

    n_nonfinite_power: int
    """Rows whose ``power_fcst`` is null, NaN, or infinite."""

    n_hindcast: int
    """Rows whose ``valid_time`` is at or before ``power_fcst_init_time`` — forbidden by
    ``PowerForecast``'s own constraint, so any such row means a pipeline regression."""

    n_ensemble_members: int
    time_series_ids: tuple[int, ...]
    latest_valid_time: datetime | None

    nwp_init_time: datetime | None
    """The freshest ``nwp_init_time`` stamped on the rows (null for a model using no NWP)."""

    @property
    def n_invalid(self) -> int:
        """Rows that are present but unusable."""
        return self.n_nonfinite_power + self.n_hindcast

    @property
    def n_time_series(self) -> int:
        """How many distinct time series this slot forecast."""
        return len(self.time_series_ids)


@dataclass(frozen=True)
class MissedNwpRuns:
    """How many daily NWP runs were absent when a forecast was made.

    ``n_missed`` is ``None`` — not zero — when the NWP table holds no run at all at or before the
    forecast time, because "how many are missing" has no finite answer there.
    """

    latest_init_time: datetime | None
    expected_latest_init_time: datetime
    n_missed: int | None

    @property
    def is_healthy(self) -> bool:
        """True only when nothing is missing (an unknown count is never healthy)."""
        return self.n_missed == 0


@dataclass(frozen=True)
class LiveForecastHealthResult:
    """Outcome of a live-forecast health evaluation for one 6-hourly slot."""

    power_fcst_init_time: datetime
    rows: LiveForecastRows
    nwp: MissedNwpRuns

    missing_time_series_ids: tuple[int, ...]
    """Series the promoted model was trained on that this slot did not forecast. Empty when the
    trained population could not be read (``n_expected_time_series is None``), which is unknown
    rather than healthy — hence it never *gates* the check."""

    n_expected_time_series: int | None

    @property
    def horizon_hours(self) -> float | None:
        """How far past ``power_fcst_init_time`` the slot's furthest forecast row reaches."""
        if self.rows.latest_valid_time is None:
            return None
        return (self.rows.latest_valid_time - self.power_fcst_init_time).total_seconds() / 3600.0

    @property
    def is_healthy(self) -> bool:
        """True when the slot wrote usable rows for the whole trained population against a
        complete NWP feed."""
        return (
            self.rows.n_rows > 0
            and self.rows.n_invalid == 0
            and not self.missing_time_series_ids
            and self.nwp.is_healthy
        )


def _floor_to_interval(moment: datetime, interval: timedelta) -> datetime:
    """Round ``moment`` down to the previous whole multiple of ``interval`` since the Unix epoch."""
    return _UNIX_EPOCH + (moment - _UNIX_EPOCH) // interval * interval


def count_missed_nwp_runs(
    available_init_times: Iterable[datetime],
    *,
    as_of: datetime,
    run_interval: timedelta = _NWP_RUN_INTERVAL,
    expected_on_disk_by: timedelta = _NWP_RUN_EXPECTED_ON_DISK_BY,
) -> MissedNwpRuns:
    """Count the daily NWP runs missing between the freshest run on disk and the freshest expected.

    Pure and deterministic — no Dagster, Delta or clock access. Counting *runs* rather than hours
    of age is the whole point: we ingest one run a day and forecast four times a day, so healthy
    NWP is anywhere between 12 and 30 hours old, and any absolute age threshold tight enough to
    catch an outage fires on two slots in four every day. This count is zero in every healthy
    slot, whichever slot it is.

    Args:
        available_init_times: The ``init_time``s genuinely present in the NWP Delta table. Runs
            initialised after ``as_of`` are ignored — they could not have been used.
        as_of: The moment to judge availability at, i.e. the slot's ``power_fcst_init_time``.
        run_interval: How often a run is ingested (one per day).
        expected_on_disk_by: How long after its ``init_time`` a run should have landed.

    Returns:
        A ``MissedNwpRuns`` carrying the freshest run on disk, the freshest that ought to exist,
        and the count between them (``None`` if there is no usable run at all).
    """
    expected_latest = _floor_to_interval(as_of - expected_on_disk_by, run_interval)
    usable = [init_time for init_time in available_init_times if init_time <= as_of]
    latest = max(usable, default=None)
    if latest is None:
        return MissedNwpRuns(None, expected_latest, None)
    # Clamped at zero: NWP fresher than the deadline requires is not a fault. That is exactly the
    # healthy 12:00 slot, where the day's run has already landed (08:30) but a deadline generous
    # enough to survive the ingest retry window still only asks for yesterday's.
    n_missed = max(0, (expected_latest - latest) // run_interval)
    return MissedNwpRuns(latest, expected_latest, n_missed)


def evaluate_live_forecast_health(
    rows: LiveForecastRows,
    nwp_init_times: Iterable[datetime],
    *,
    power_fcst_init_time: datetime,
    expected_time_series_ids: Sequence[int] | None,
) -> LiveForecastHealthResult:
    """Judge one live slot: did it write usable rows, and was the NWP feed whole when it ran?

    Pure and deterministic, so it is unit-testable without Dagster or Delta. Deliberately says
    nothing about forecast *skill* — whether the numbers are any good is production monitoring's
    job, not a data-health check's.

    Args:
        rows: The slot's forecast rows as summarised from disk.
        nwp_init_times: Every ``init_time`` present in the NWP Delta table.
        power_fcst_init_time: The slot being judged.
        expected_time_series_ids: The promoted model's trained population, or ``None`` when it
            could not be read — in which case population completeness is simply not assessed.

    Returns:
        A ``LiveForecastHealthResult`` summarising the slot's health.
    """
    if expected_time_series_ids is None:
        missing: tuple[int, ...] = ()
        n_expected = None
    else:
        missing = tuple(sorted(set(expected_time_series_ids) - set(rows.time_series_ids)))
        n_expected = len(set(expected_time_series_ids))
    return LiveForecastHealthResult(
        power_fcst_init_time=power_fcst_init_time,
        rows=rows,
        nwp=count_missed_nwp_runs(nwp_init_times, as_of=power_fcst_init_time),
        missing_time_series_ids=missing,
        n_expected_time_series=n_expected,
    )


_EMPTY_LIVE_FORECAST_ROWS: Final[LiveForecastRows] = LiveForecastRows(
    n_rows=0,
    n_nonfinite_power=0,
    n_hindcast=0,
    n_ensemble_members=0,
    time_series_ids=(),
    latest_valid_time=None,
    nwp_init_time=None,
)
"""What a slot with nothing on disk looks like — also the answer when the table does not exist."""


def _read_live_forecast_rows(
    power_forecasts_path: str,
    storage_options: ObjectStoreOptions | None,
    power_fcst_init_time: datetime,
) -> LiveForecastRows:
    """Summarise the ``power_forecasts`` rows one live slot wrote, without materialising them.

    Returns the empty summary if the Delta table does not exist yet (a brand-new deployment), so
    "no table" and "no rows for this slot" reach the evaluator as the same unhealthy state.

    The scan is pruned to the ``fold_id="live"`` Delta partitions and then to the one
    ``power_fcst_init_time``, and every column is reduced to a scalar inside Polars, so only the
    aggregates cross back into Python. The ``pl.len()`` is safe from the 32-bit row-count
    wraparound documented in ``CLAUDE.md``: it counts one slot's rows (~1M at V1 scale, ~86M at
    V2), not the whole table.
    """
    if not delta_table_exists(power_forecasts_path, storage_options):
        return _EMPTY_LIVE_FORECAST_ROWS

    slot = pl.scan_delta(
        power_forecasts_path, storage_options=typeddict_to_dict(storage_options)
    ).filter(
        pl.col("fold_id") == _LIVE_FOLD_ID,
        pl.col("power_fcst_init_time") == power_fcst_init_time,
    )
    summary = slot.select(
        n_rows=pl.len(),
        # A null `power_fcst` makes the left operand true, so under Kleene logic the row is
        # counted once and `is_finite()`'s own null on that row never leaks into the sum.
        n_nonfinite_power=(
            pl.col("power_fcst").is_null() | ~pl.col("power_fcst").is_finite()
        ).sum(),
        n_hindcast=(pl.col("valid_time") <= pl.lit(power_fcst_init_time)).sum(),
        n_ensemble_members=pl.col("ensemble_member").n_unique(),
        time_series_ids=pl.col("time_series_id").unique().implode(),
        latest_valid_time=pl.max("valid_time"),
        nwp_init_time=pl.max("nwp_init_time"),
    ).collect(engine="streaming")

    row = summary.row(0, named=True)
    return LiveForecastRows(
        n_rows=int(row["n_rows"]),
        n_nonfinite_power=int(row["n_nonfinite_power"] or 0),
        n_hindcast=int(row["n_hindcast"] or 0),
        n_ensemble_members=int(row["n_ensemble_members"]),
        time_series_ids=tuple(sorted(row["time_series_ids"] or ())),
        latest_valid_time=row["latest_valid_time"],
        nwp_init_time=row["nwp_init_time"],
    )


def _nwp_init_times_on_disk(settings: Settings) -> list[datetime]:
    """Every NWP ``init_time`` on disk — an empty list if the table does not exist yet.

    Reuses ``live_forecasts``' own metadata-only partition read, so the check counts runs from
    exactly the same source the asset selects them from.
    """
    if not delta_table_exists(settings.nwp_data_path, settings.storage_options):
        return []
    return _available_nwp_init_times(settings)


def _trained_time_series_ids(production_model_path: str) -> tuple[int, ...] | None:
    """The promoted model's trained population, or ``None`` if it cannot be read.

    ``None`` (rather than an empty tuple) keeps "we don't know which series to expect" distinct
    from "the model was trained on nothing", so an unreadable ``meta.json`` weakens the check
    instead of turning it permanently yellow.
    """
    meta_path = Path(production_model_path) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        ids = json.loads(meta_path.read_text()).get("trained_time_series_ids")
        return None if ids is None else tuple(int(i) for i in ids)
    except OSError, ValueError, TypeError, AttributeError:
        logger.exception(f"Could not read trained time series ids from {meta_path}")
        return None


def _describe_live_forecast_health(result: LiveForecastHealthResult) -> str:
    """One human-readable line: the slot's verdict, and every problem found."""
    slot = result.power_fcst_init_time.isoformat()
    rows = result.rows
    problems: list[str] = []
    if rows.n_rows == 0:
        problems.append("no forecast rows were written")
    if rows.n_nonfinite_power:
        problems.append(f"{rows.n_nonfinite_power} rows have a null/NaN/infinite power_fcst")
    if rows.n_hindcast:
        problems.append(f"{rows.n_hindcast} rows target a valid_time at or before the init time")
    if result.missing_time_series_ids:
        listed = result.missing_time_series_ids[:_MAX_MISSING_SERIES_LISTED]
        suffix = "" if len(listed) == len(result.missing_time_series_ids) else ", …"
        ids = ", ".join(str(i) for i in listed) + suffix
        problems.append(f"{len(result.missing_time_series_ids)} trained series missing ({ids})")
    if result.nwp.n_missed is None:
        problems.append("no NWP run is available at or before this slot")
    elif result.nwp.n_missed:
        problems.append(
            f"{result.nwp.n_missed} daily NWP run(s) missed "
            f"(freshest on disk {result.nwp.latest_init_time}, "
            f"expected {result.nwp.expected_latest_init_time})"
        )
    if not problems:
        return (
            f"{rows.n_rows} valid forecast rows for {slot} across {rows.n_time_series} time "
            f"series and {rows.n_ensemble_members} ensemble members; no NWP runs missed."
        )
    return f"Live forecast for {slot} is degraded: " + "; ".join(problems) + "."


def _live_forecast_check_metadata(result: LiveForecastHealthResult) -> dict[str, Any]:
    """Render the evaluation as the check's Dagster UI metadata."""
    rows = result.rows
    return {
        "power_fcst_init_time": result.power_fcst_init_time.isoformat(),
        "n_rows": rows.n_rows,
        "n_invalid_rows": rows.n_invalid,
        "n_nonfinite_power": rows.n_nonfinite_power,
        "n_hindcast_rows": rows.n_hindcast,
        "n_time_series": rows.n_time_series,
        "n_time_series_expected": (
            result.n_expected_time_series
            if result.n_expected_time_series is not None
            else MetadataValue.text("unknown — no promoted model meta.json")
        ),
        "n_time_series_missing": len(result.missing_time_series_ids),
        "missing_time_series_ids": str(
            list(result.missing_time_series_ids[:_MAX_MISSING_SERIES_LISTED])
        ),
        "n_ensemble_members": rows.n_ensemble_members,
        "forecast_horizon_hours": (
            round(result.horizon_hours, 1)
            if result.horizon_hours is not None
            else MetadataValue.text("no rows")
        ),
        "n_missed_nwp_runs": (
            result.nwp.n_missed
            if result.nwp.n_missed is not None
            else MetadataValue.text("unknown — no NWP run at or before this slot")
        ),
        "nwp_init_time_on_disk": str(result.nwp.latest_init_time),
        "nwp_init_time_expected": result.nwp.expected_latest_init_time.isoformat(),
        "nwp_init_time_on_rows": str(rows.nwp_init_time),
    }


def _to_live_forecast_check_result(result: LiveForecastHealthResult) -> AssetCheckResult:
    """Turn a ``LiveForecastHealthResult`` into a WARN-severity Dagster check result."""
    return AssetCheckResult(
        # WARN, never ERROR, and non-blocking: a degraded slot is still the best forecast we have,
        # and blocking here would contradict the principle that a partition fails only when there
        # is genuinely no useful data.
        passed=result.is_healthy,
        severity=AssetCheckSeverity.WARN,
        description=_describe_live_forecast_health(result),
        metadata=_live_forecast_check_metadata(result),
    )


def _checked_power_fcst_init_time(context: AssetCheckExecutionContext, now: datetime) -> datetime:
    """Which slot's forecast this evaluation is about.

    A partitioned run — the scheduled path, and any manual replay — reports on its own partition,
    whose ``power_fcst_init_time`` is the window's *end* (see ``live_forecasts``' docstring). That
    keeps a replay of an old slot judging the slot it actually rebuilt. Invoked without a
    partition, the check falls back to the most recent slot that has come due.
    """
    if context.has_partition_key:
        return context.partition_time_window.end
    return _floor_to_interval(now, _LIVE_SLOT_INTERVAL)


def _evaluate_live_forecasts(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """Read this slot back off disk and judge it.

    Split out from the check itself so the check's ``except`` wraps everything — the Settings
    load, all three reads, the evaluation and the metadata build — rather than only part of it.
    """
    settings = Settings()
    power_fcst_init_time = _checked_power_fcst_init_time(context, datetime.now(timezone.utc))
    result = evaluate_live_forecast_health(
        _read_live_forecast_rows(
            settings.power_forecasts_data_path, settings.storage_options, power_fcst_init_time
        ),
        _nwp_init_times_on_disk(settings),
        power_fcst_init_time=power_fcst_init_time,
        expected_time_series_ids=_trained_time_series_ids(settings.production_model_path),
    )
    return _to_live_forecast_check_result(result)


@asset_check(
    asset=live_forecasts,
    blocking=False,
    description=(
        "Warn if this 6-hourly slot wrote no forecast rows, wrote unusable ones, skipped part of "
        "the promoted model's trained population, or ran against a gappy NWP feed."
    ),
)
def live_forecasts_are_healthy(context: AssetCheckExecutionContext) -> AssetCheckResult:
    """Report whether one live slot landed valid rows, and how many NWP runs it was missing.

    Runs alongside every ``live_forecasts`` materialisation (6-hourly via
    ``live_forecasts_schedule``), *after* the asset's Delta write, so it reads back what was
    actually persisted rather than what was in memory. This is not a measure of forecast skill —
    only of whether rows exist and hold usable data.

    It covers the slots where the asset *succeeded*: a slot whose asset raised never reaches this
    check (Dagster does not run a check whose asset op failed), and that case is already loud —
    the run fails and ``live_forecasts_job``'s ``sentry_capture_failure`` hook reports it. The gap
    this closes is the quiet one: a run that succeeded while writing nothing usable, or that
    forecast from a days-old NWP run.
    """
    try:
        return _evaluate_live_forecasts(context)
    except Exception as exc:
        # Catch-all is deliberate. A warning path must never be able to fail the thing it warns
        # about: this check is non-blocking, but it runs inside `live_forecasts_job`, whose
        # `sentry_capture_failure` hook would turn a raise here into a failed production run —
        # fail-open silently becoming fail-closed. Logged at ERROR with the traceback (never a
        # silent swallow) and surfaced as an unhealthy check, so a bug in here stays visible.
        logger.exception("Could not evaluate live-forecast health")
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=f"Could not evaluate live-forecast health: {exc!r}",
        )
