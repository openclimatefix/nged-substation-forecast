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

Series we already know are dead — ``_KNOWN_DEAD_TIME_SERIES_IDS`` — are dropped from the check's
*inputs* rather than from its output, so every count describes the series we are still watching, and
the Sentry warning inherits the silencing without knowing it exists. The check keeps naming the
silenced ids every hour, green or yellow, and turns yellow on its own if one of them reports again.

``evaluate_power_freshness`` is a pure function so it is unit-testable without Dagster or Delta,
and it is the hand-off point for routing per-series staleness to Sentry: the same
``PowerFreshnessResult`` is fed to ``report_power_freshness`` (in
``nged_substation_forecast._sentry``) rather than recomputed. The two mechanisms stay
complementary — the
[Sentry missed-check-in alarm](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
fires on total silence from outside the deployment, while this check (and its Sentry warning) report
per-series staleness from inside Dagster while the daemon is alive.

``live_forecasts_are_healthy`` does the same job for the one asset NGED actually consumes. It
answers two questions the asset's own success status cannot: did this 6-hourly slot really land
valid forecast rows on disk, and how many daily NWP runs were missing when the forecast was made?
Both are read back from disk after the write, so a run that "succeeded" while writing nothing —
or writing null/non-finite forecasts, hindcast rows, or a short population — still shows up.
Missed NWP runs are *counted as runs*, never measured in hours of age — healthy NWP is 12–30 hours
old depending on the slot, so any absolute age threshold would fire on two slots in four every day.
The full argument is in
[Inherent Stability → Three audiences, three channels](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#three-audiences-three-channels).

Both checks are ``AssetCheckSeverity.WARN`` and ``blocking=False``, and nothing either one's *body*
does can fail its own step: each sits under a catch-all which logs the traceback, reports the
exception to Sentry (the run no longer fails, so the failure hook no longer fires) and returns an
unhealthy result. Catching ``BaseException`` is what makes that absolute — a Rust panic in any of
the pyo3 extensions these bodies read through arrives as a ``PanicException``, which does not
derive from ``Exception``. Only cancellation is re-raised, which is what we want: a cancelled run
should cancel. What remains outside the guard is Dagster's own machinery: resource init, and
serialising the returned ``AssetCheckResult`` into an event. A warning path that could fail would
turn fail-open into fail-closed at exactly the wrong moment (rule 7 of
[The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).

The two differ in how much they salvage short of that catch-all. In ``live_forecasts_are_healthy``
an absent ``power_forecasts`` or NWP table reads as empty, and an absent *or unreadable*
promoted-model ``meta.json`` degrades to "population unknown", so the rest of the report survives;
a read *error* on either table still costs the whole report. ``power_data_is_fresh`` salvages
nothing below the catch-all on purpose: its only per-read fallback would be "roster unknown", and a
fresh power table would then render an unreadable roster as a green tick.
"""

import json
import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions, delta_table_exists, object_exists
from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSeverity,
    DagsterExecutionInterruptedError,
    MetadataValue,
    TableColumn,
    TableRecord,
    TableSchema,
    asset_check,
)
from nged_data.storage import time_series_coverage

from nged_substation_forecast._sentry import report_check_degradation, report_power_freshness
from nged_substation_forecast.defs.assets import power_time_series_and_metadata
from nged_substation_forecast.defs.production_assets import (
    LIVE_FORECAST_HORIZON,
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

_MAX_LATE_SERIES_IN_TABLE: Final[int] = 50
"""Cap on how many late series the check's metadata table lists.

Unlike the other capped listings this one lands in Dagster's event log, so it is written to durable
storage every hour a stall lasts. Uncapped, a whole-feed stall at V2 scale (~2,500 series) would
serialise to about 355 KB per hourly evaluation; at 50 rows it is about 8 KB.

Matches ``_sentry.MAX_LATE_SERIES_IN_CONTEXT`` (50) rather than the tighter
``_sentry.MAX_LATE_SERIES_IN_MESSAGE`` (20) that bounds the Sentry message body, because a browsable
table and the Sentry event context serve the same drill-down purpose.

The rows follow ``_LATE_STATUS_ORDER``, so the listing is the head of that order rather than the 50
series in most trouble: when never-reported series outnumber the cap, no stale series is listed at
all, however stale it is. ``n_stale`` and ``n_never_reported`` stay exact for the watched
population throughout, and are what the operator should read first. Why the table is capped, and
why at this number:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check>
"""

_LATE_STATUS_ORDER: Final[tuple[str, ...]] = ("never", "stale")
"""Runtime tuple — declared order for the ``status`` column's ``pl.Enum``, which is also the
row order in the late-series table (never-reported series listed before merely-stale ones)."""

_POWER_DATA_STALENESS_THRESHOLD: Final[timedelta] = timedelta(hours=24)
"""A ``time_series_id`` is 'late' if its most recent observation is older than this.

NGED publishes at irregular, several-hours-apart intervals and our pipeline back-fills gaps
automatically once the feed recovers, so 24 hours is comfortably past normal jitter while still
catching a genuine multi-slot stall the same day."""

_KNOWN_DEAD_TIME_SERIES_IDS: Final[tuple[int, ...]] = (33,)
"""Series ``power_data_is_fresh`` stops warning about, because we already know they are dead.

33: the site monitor is broken (reported by James at NGED); no data since 2026-01-26.

Delete an entry to start warning about that series again. The check keeps naming every id listed
here, so the silencing cannot be forgotten, and turns yellow by itself if one of them reports data
again. How to edit the list, and what each description means:
<https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/#degraded-input-data-nwp-feed-down-or-telemetry-stalled>

Why it is a source constant rather than operator-editable state:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#silence-the-series-we-already-know-are-dead>
"""


@dataclass(frozen=True)
class PowerFreshnessResult:
    """Outcome of a power-data freshness evaluation.

    ``late`` lists every late series — never-reported first, then most-stale first — with columns
    ``time_series_id``, ``last_seen`` (null if never reported), ``hours_late`` (null if never
    reported) and ``status`` (``"never"`` or ``"stale"``).

    Every count here describes the series we are *watching*: the silenced ones are excluded from
    ``n_series_total`` as well as from ``late``.
    """

    n_series_total: int
    n_stale: int
    n_never: int
    threshold_hours: float
    late: pl.DataFrame

    silenced_ids: tuple[int, ...] = ()
    """The ids we were asked to ignore, echoed verbatim — including any that match no series at
    all, so a mistyped id shows up as silencing nothing rather than vanishing."""

    resurrected_ids: tuple[int, ...] = ()
    """Silenced ids that have reported since the cutoff, so the silencing no longer fits."""

    @property
    def n_late(self) -> int:
        """Total late series: those that went stale plus those that never reported."""
        return self.n_stale + self.n_never

    @property
    def is_healthy(self) -> bool:
        """True when no *watched* series is late.

        Deliberately says nothing about ``resurrected_ids``: this is the gate
        ``report_power_freshness`` gives to Sentry's *staleness* warning, and a series that has
        started reporting again is not stale. The asset check reacts to a resurrection on its own,
        through ``passed``.
        """
        return self.n_late == 0


def evaluate_power_freshness(
    coverage: pl.DataFrame,
    roster_ids: pl.Series | None,
    now: datetime,
    threshold: timedelta,
    *,
    silenced_ids: Collection[int] = (),
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
        silenced_ids: Series we already know are dead, dropped from both inputs before anything is
            classified, so every count describes the watched population. One that has reported
            since the cutoff comes back as ``resurrected_ids`` instead.

    Returns:
        A ``PowerFreshnessResult`` summarising the health of the power feed.
    """
    # Strip any Patito model so the frame-building below uses plain Polars semantics.
    coverage = pl.DataFrame._from_pydf(coverage._df)
    last_time_dtype = coverage.schema["last_time"]
    status_dtype = pl.Enum(_LATE_STATUS_ORDER)
    cutoff = now - threshold

    # Drop the silenced series from the *inputs*, so `stale`, `never`, `late` and every count below
    # describe what we are watching without any of them needing to know about silencing. A silenced
    # id fresher than the cutoff is reported instead: the fault we silenced has evidently healed,
    # and the complement of `stale`'s `<` is exactly the right test — an id that is in neither
    # frame (a typo, or one NGED has never published) is correctly not a resurrection.
    silenced = list(silenced_ids)
    resurrected_ids = tuple(
        coverage.filter(pl.col("last_time") >= cutoff, pl.col("time_series_id").is_in(silenced))[
            "time_series_id"
        ].sort()
    )
    coverage = coverage.filter(~pl.col("time_series_id").is_in(silenced))
    if roster_ids is not None:
        roster_ids = roster_ids.filter(~roster_ids.is_in(silenced))

    # Stale: has data on disk, but the newest observation predates the cutoff.
    #
    # NOTE: this is deliberately not restricted to `roster_ids`. A series that NGED has
    # decommissioned but that still has old rows on disk will keep being flagged stale — which is
    # what we want for now: we would rather be told about a series that has gone quiet than
    # silently stop watching it. Restricting to `roster_ids` would not silence one anyway:
    # `upsert_metadata` never drops a series, so a retired series stays in the roster for good.
    # Silencing one takes the explicit record of dead ids above, which serves a retired series and
    # a broken sensor alike — the check cannot tell them apart, and does not need to.
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
        silenced_ids=tuple(silenced),
        resurrected_ids=resurrected_ids,
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
    """Render the late-series frame as a Dagster table for the check's UI metadata.

    Renders every row it is given: the caller truncates to ``_MAX_LATE_SERIES_IN_TABLE`` first, so
    that the row count reported as ``n_late_listed`` and the table itself cannot disagree.
    """
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


def _describe_power_freshness(result: PowerFreshnessResult) -> str:
    """One human-readable line: the state of the watched feed, and what is being ignored."""
    threshold_h = result.threshold_hours
    if result.n_series_total == 0:
        summary = "No power data on disk yet."
    elif result.is_healthy:
        summary = (
            f"All {result.n_series_total} watched time series are up to date "
            f"(within {threshold_h:.0f}h)."
        )
    else:
        summary = (
            f"{result.n_late}/{result.n_series_total} watched time series are late: "
            f"{result.n_stale} stale (>{threshold_h:.0f}h since last data), "
            f"{result.n_never} never reported."
        )
    sentences = [summary]
    # Named on every run, healthy or not, so silencing something is never quietly forgotten.
    if result.silenced_ids:
        ids = ", ".join(str(i) for i in result.silenced_ids)
        sentences.append(f"Ignoring {len(result.silenced_ids)} known-dead time series: {ids}.")
    if result.resurrected_ids:
        ids = ", ".join(str(i) for i in result.resurrected_ids)
        sentences.append(
            f"Reporting again, so no longer dead: {ids}. Remove from "
            "_KNOWN_DEAD_TIME_SERIES_IDS in defs/checks.py."
        )
    return " ".join(sentences)


def _to_asset_check_result(result: PowerFreshnessResult) -> AssetCheckResult:
    """Turn a ``PowerFreshnessResult`` into a WARN-severity Dagster check result."""
    # Truncate once, here, so the table and the count that describes it come from one slice.
    listed = result.late.head(_MAX_LATE_SERIES_IN_TABLE)
    return AssetCheckResult(
        # A stalled feed is expected to self-heal via back-fill, so warn — never fail the run and
        # block downstream assets. Absent data is not "healthy" either, hence the count guard.
        # A resurrection is the one yellow nothing else can raise: the series is healthy, so
        # `is_healthy` stays true, and only an edit to the dead list clears it.
        passed=result.is_healthy and result.n_series_total > 0 and not result.resurrected_ids,
        severity=AssetCheckSeverity.WARN,
        description=_describe_power_freshness(result),
        metadata={
            "n_late": result.n_late,
            "n_stale": result.n_stale,
            "n_never_reported": result.n_never,
            "n_series_total": result.n_series_total,
            "threshold_hours": result.threshold_hours,
            "n_late_listed": listed.height,
            "late_time_series": _late_table_metadata(listed),
            "n_silenced": len(result.silenced_ids),
            "silenced_time_series_ids": str(list(result.silenced_ids)),
        },
    )


def _check_power_data_freshness() -> AssetCheckResult:
    """Read the power table's recency off disk and judge it.

    Split out from the check itself so the check's ``except`` wraps the whole body.
    """
    settings = Settings()
    storage_options = settings.storage_options
    coverage = time_series_coverage(settings.power_time_series_data_path, storage_options)
    roster_ids = _read_roster_ids(settings.metadata_path, storage_options)
    result = evaluate_power_freshness(
        coverage=coverage,
        roster_ids=roster_ids,
        now=datetime.now(UTC),
        threshold=_POWER_DATA_STALENESS_THRESHOLD,
        # Read at the call site, not defaulted into the signature, so a test can monkeypatch it.
        silenced_ids=_KNOWN_DEAD_TIME_SERIES_IDS,
    )
    # Forward per-series staleness to Sentry (a no-op unless a DSN is set and some series is late).
    # Best-effort: report_power_freshness never raises, so a telemetry hiccup costs no more than
    # its own event — were it to raise, the caller's catch-all would discard this whole evaluation.
    report_power_freshness(settings, result)
    return _to_asset_check_result(result)


@asset_check(
    asset=power_time_series_and_metadata,
    blocking=False,
    description=(
        "Warn if any watched time series has no fresh power data within the staleness threshold "
        "(stale) or has never reported at all (never), or if a known-dead series has started "
        "reporting again."
    ),
)
def power_data_is_fresh() -> AssetCheckResult:
    """Report how many time series are late on the ``power_time_series`` Delta table.

    Runs automatically alongside every ``power_time_series_and_metadata`` materialisation (hourly
    via ``power_time_series_and_metadata_schedule``), so the check re-evaluates freshness each
    hour regardless of whether new data landed.

    Cannot fail its own step: the whole body is guarded, so an object-store error, a half-written
    ``metadata.parquet`` or a bug in here degrades to an unhealthy result rather than failing the
    hourly production run. (A merely *slow* object store makes a slow check, not a degraded one —
    nothing here imposes a step-level timeout.)
    """
    try:
        return _check_power_data_freshness()
    except BaseException as exc:
        # `BaseException`, not `Exception`: a Rust panic surfaces as a pyo3 `PanicException`, which
        # derives from `BaseException`, and each compiled extension this body reads through (polars,
        # delta-rs, obstore) defines its own class — so there is no one name to catch. Naming what
        # must propagate instead is the only version that stays true as dependencies come and go.
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        # The width has one cost worth knowing when writing tests: pytest's `fail` and `skip` raise
        # from `BaseException` too, so they are swallowed here. Never use one as a "must not be
        # called" sentinel inside this body; assert after the call instead.
        # Rule 7: a non-blocking check that *errors* still fails its run, and this job carries the
        # `sentry_capture_failure` hook, so a raise here would turn fail-open into fail-closed.
        # Sentry is told explicitly because not failing the run means that hook no longer fires.
        logger.exception("Could not evaluate power-data freshness")
        report_check_degradation(check_name="power_data_is_fresh", exc=exc)
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=f"Could not evaluate power-data freshness: {exc!r}",
        )


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

``ecmwf_ens_schedule`` fires at 08:30 UTC and ``ecmwf_ens`` retries an unusable run every 30
minutes, up to 8 times. The delays alone put the last healthy landing at about 12:30 UTC, but one
of the two retryable failures (``NwpVariableWhollyMissing``) is raised *after* the download, so
the worst case pays a download and convert on every attempt too — about 12:40 UTC at the ~1
min/run measured in
<https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/>. That leaves
81 minutes of margin to
14:00 UTC, spread over 9 attempts: the deadline is breached only if download-and-convert
*averages* about 10 minutes across all of them, not if one attempt is slow. So the 645 s download
recorded in ``dynamical_data.ecmwf_ens.download`` would cost only ~10 of those 81 minutes on its
own; it takes a sustained slowdown, not a single bad fetch. The deadline
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
"""Cap on how many missing ``time_series_id``s the description and the metadata spell out.

Keeps the one-line description readable at V2 scale (~2,500 series) when a whole population is
missing. ``n_time_series_missing`` carries the uncapped count."""

_UNIX_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)
"""Origin for ``_floor_to_interval``'s arithmetic. Both cadences we floor to (daily NWP runs at
00Z, 6-hourly forecast slots at 00/06/12/18) are aligned to it."""

_MIN_HORIZON_FRACTION: Final[float] = 0.5
"""Fraction of ``LIVE_FORECAST_HORIZON`` a slot's rows must span before the horizon is called
truncated.

Half is deliberately loose. A *healthy* slot reaches almost the full 14 days — ECMWF ENS covers
about 15 days from its ``init_time`` and the run is 12–30 hours old at forecast time, leaving
roughly 13.75 days — so the only ways to fall below 7 days are a partially-ingested run or NWP
about a week stale, and the latter is already reported as missed runs. A tighter floor would start
competing with the missed-run count for the same fault while risking false alarms; this one catches
the case nothing else sees, which is fresh-but-truncated NWP."""

_MIN_HORIZON_HOURS: Final[float] = (
    LIVE_FORECAST_HORIZON.total_seconds() / 3600.0 * _MIN_HORIZON_FRACTION
)
"""``_MIN_HORIZON_FRACTION`` of the horizon ``live_forecasts`` asks for, in hours."""


@dataclass(frozen=True)
class LiveForecastRows:
    """One live slot's forecast rows on disk, as summarised by ``_read_live_forecast_rows``.

    Every field is an aggregate rather than the rows themselves: one slot is ~1M rows at V1 scale
    and ~86M at V2, so the summary is computed inside Polars and only these scalars come back.
    """

    n_rows: int

    n_invalid: int
    """Rows that are present but unusable — the *union* of the two counts below, so a row that is
    both non-finite and a hindcast is counted once and ``n_invalid`` can never exceed ``n_rows``."""

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
    def horizon_is_truncated(self) -> bool:
        """True when the slot's rows stop well short of the forecast horizon we asked for.

        ``live_forecasts`` drops rows outside the selected NWP run's coverage, so a run that was
        only partly ingested — or one so old that its remaining coverage is nearly used up —
        silently delivers a much shorter forecast than NGED expect, with every row still perfectly
        well-formed. That is structural, not a matter of skill, so it belongs here.
        """
        return self.horizon_hours is not None and self.horizon_hours < _MIN_HORIZON_HOURS

    @property
    def is_healthy(self) -> bool:
        """True when the slot is fully healthy.

        That is, when it wrote usable rows, over the expected horizon, for the whole trained
        population, against a complete NWP feed.
        """
        return (
            self.rows.n_rows > 0
            and self.rows.n_invalid == 0
            and not self.horizon_is_truncated
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

    Pure and deterministic — no Dagster, Delta or clock access. We count missed *runs* rather than
    hours of age because we ingest one ECMWF run per day but forecast four times a day, so healthy
    NWP is anywhere from 12 to 30 hours old. Counting runs reads zero in every healthy slot,
    whichever slot it is; counting hours would not (see the module docstring).

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
    n_invalid=0,
    n_nonfinite_power=0,
    n_hindcast=0,
    n_ensemble_members=0,
    time_series_ids=(),
    latest_valid_time=None,
    nwp_init_time=None,
)
"""What a slot with nothing on disk looks like — also the answer when the table does not exist."""


@dataclass(frozen=True)
class PromotedModelFacts:
    """The two things about the promoted model the check needs, read from its ``meta.json``.

    Both are ``None`` when the corresponding field is absent, so an unreadable or partial
    ``meta.json`` weakens the check rather than turning it permanently yellow.
    """

    experiment_name: str | None
    """Which ``power_forecasts`` experiment partition this model's slots are written to."""

    trained_time_series_ids: tuple[int, ...] | None
    """The population every slot is expected to forecast (the train==predict invariant)."""


_UNKNOWN_PROMOTED_MODEL: Final[PromotedModelFacts] = PromotedModelFacts(None, None)
"""What an absent or unreadable promoted-model ``meta.json`` reduces to."""


def _read_live_forecast_rows(
    power_forecasts_path: str,
    storage_options: ObjectStoreOptions | None,
    power_fcst_init_time: datetime,
    experiment_name: str | None,
) -> LiveForecastRows:
    """Summarise the ``power_forecasts`` rows one live slot wrote, without materialising them.

    Returns the empty summary if the Delta table does not exist yet (a brand-new deployment), so
    "no table" and "no rows for this slot" reach the evaluator as the same unhealthy state.

    ``experiment_name`` scopes the read to the promoted model's own rows. It matters because
    ``write_power_forecasts`` replaces one ``(experiment_name, fold_id)`` partition at a time, so
    promoting a champion from a *different* experiment leaves the outgoing experiment's rows for
    the same slot in place: without this filter the check would aggregate live rows from two
    experiments and report faults sourced entirely from dead ones. ``None`` — the promoted model's
    ``meta.json`` is absent or unreadable, so there is no name to scope to — falls back to reading
    every live experiment, which over-reports rather than under-reports.

    The scan is pruned to the matching ``(experiment_name, fold_id="live")`` Delta partitions and
    then to the one ``power_fcst_init_time``, and every column is reduced to a scalar inside
    Polars, so only the aggregates cross back into Python. The ``pl.len()`` is safe from the
    32-bit row-count wraparound documented in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/#data-handling>:
    it counts one slot's rows (~1M at V1 scale, ~86M at V2), not the whole table.
    """
    if not delta_table_exists(power_forecasts_path, storage_options):
        return _EMPTY_LIVE_FORECAST_ROWS

    slot = pl.scan_delta(
        power_forecasts_path, storage_options=typeddict_to_dict(storage_options)
    ).filter(
        pl.col("fold_id") == _LIVE_FOLD_ID,
        pl.col("power_fcst_init_time") == power_fcst_init_time,
    )
    if experiment_name is not None:
        slot = slot.filter(pl.col("experiment_name") == experiment_name)

    # A null `power_fcst` makes the left operand true, so under Kleene logic the row is counted
    # once and `is_finite()`'s own null on that row never leaks into the sum.
    nonfinite_power = pl.col("power_fcst").is_null() | ~pl.col("power_fcst").is_finite()
    hindcast = pl.col("valid_time") <= pl.lit(power_fcst_init_time)
    # `nwp_init_time` is `allow_missing=True` on `PowerForecast` — a model that uses no NWP (a
    # persistence baseline) never writes the column at all. Selecting it unconditionally would
    # raise `ColumnNotFoundError` into the check's catch-all, costing the whole report — every
    # other field, all of them computable — for one cosmetic one.
    nwp_init_time = (
        pl.max("nwp_init_time")
        if "nwp_init_time" in slot.collect_schema().names()
        else pl.lit(None, dtype=UTC_DATETIME_DTYPE)
    )
    summary = slot.select(
        n_rows=pl.len(),
        # The union, not the sum: a row that is both must not be counted twice.
        n_invalid=(nonfinite_power | hindcast).sum(),
        n_nonfinite_power=nonfinite_power.sum(),
        n_hindcast=hindcast.sum(),
        n_ensemble_members=pl.col("ensemble_member").n_unique(),
        time_series_ids=pl.col("time_series_id").unique().implode(),
        latest_valid_time=pl.max("valid_time"),
        nwp_init_time=nwp_init_time,
    ).collect(engine="streaming")

    row = summary.row(0, named=True)
    return LiveForecastRows(
        n_rows=int(row["n_rows"]),
        n_invalid=int(row["n_invalid"] or 0),
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


def _read_promoted_model_facts(production_model_path: str) -> PromotedModelFacts:
    """Read the promoted model's experiment name and trained population from its ``meta.json``.

    One read for both, because both come from the same file that ``BaseForecaster.save`` wrote.
    Any absence or malformation degrades to ``_UNKNOWN_PROMOTED_MODEL`` rather than raising: the
    model directory is populated out-of-band (by the ``promoted_model`` asset on a laptop, by the
    image build on the box), so the check must cope with it being absent.
    """
    meta_path = Path(production_model_path) / "meta.json"
    if not meta_path.exists():
        return _UNKNOWN_PROMOTED_MODEL
    try:
        meta = json.loads(meta_path.read_text())
        ids = meta.get("trained_time_series_ids")
        if ids is not None and not isinstance(ids, list):
            # A JSON string, int, etc. here is malformed, not merely a different shape: iterating
            # a str with int() would silently mis-parse it (e.g. "12" -> (1, 2)) instead of
            # degrading, which is the "strict about malformed input" rule from CLAUDE.md
            # ("Inherent stability").
            raise TypeError(  # noqa: TRY301 — raised to reach this function's own degrade handler.
                f"trained_time_series_ids must be a list, got {type(ids).__name__}: {ids!r}"
            )
        experiment_name = meta.get("model_params", {}).get("experiment_name")
        if experiment_name is None:
            # `BaseForecaster.save` always writes `model_params.experiment_name`, so its absence
            # means this file is not one we wrote. Degrade both facts together rather than half of
            # them: keeping the trained population while losing the name to scope the read to would
            # let an unfiltered union of two experiments' rows pass for one complete population,
            # and would blame this slot for an outgoing champion's rows.
            return _UNKNOWN_PROMOTED_MODEL
        return PromotedModelFacts(
            experiment_name=experiment_name,
            trained_time_series_ids=None if ids is None else tuple(int(i) for i in ids),
        )
    except OSError, ValueError, TypeError, AttributeError:
        logger.exception(f"Could not read the promoted model's meta.json at {meta_path}")
        return _UNKNOWN_PROMOTED_MODEL


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
    if result.horizon_is_truncated:
        problems.append(
            f"the forecast reaches only {result.horizon_hours:.1f}h ahead, short of the "
            f"{_MIN_HORIZON_HOURS:.0f}h floor"
        )
    if result.missing_time_series_ids:
        listed = result.missing_time_series_ids[:_MAX_MISSING_SERIES_LISTED]
        suffix = "" if len(listed) == len(result.missing_time_series_ids) else ", …"
        ids = ", ".join(str(i) for i in listed) + suffix
        problems.append(f"{len(result.missing_time_series_ids)} trained series missing ({ids})")
    if result.nwp.n_missed is None:
        problems.append("no NWP run is available at or before this slot")
    elif result.nwp.n_missed:
        problems.append(
            f"the NWP feed is {result.nwp.n_missed} daily run(s) behind "
            f"(freshest on disk {result.nwp.latest_init_time}, "
            f"expected {result.nwp.expected_latest_init_time})"
        )
    if not problems:
        return (
            f"{rows.n_rows} valid forecast rows for {slot} across {rows.n_time_series} time "
            f"series and {_plural(rows.n_ensemble_members, 'ensemble member')}; the NWP feed is "
            "up to date."
        )
    return f"Live forecast for {slot} is degraded: " + "; ".join(problems) + "."


def _plural(count: int, noun: str) -> str:
    """``"3 ensemble members"`` / ``"1 ensemble member"`` — regular plurals only."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _live_forecast_check_metadata(result: LiveForecastHealthResult) -> dict[str, Any]:
    """Render the evaluation as the check's Dagster UI metadata.

    Every numeric entry is *omitted* rather than replaced by explanatory text when its value is
    unknown, so each key keeps one type across runs and Dagster can plot it over time. The
    description always spells out what was unknown and why.
    """
    rows = result.rows
    metadata: dict[str, Any] = {
        "power_fcst_init_time": result.power_fcst_init_time.isoformat(),
        "n_rows": rows.n_rows,
        "n_invalid_rows": rows.n_invalid,
        "n_nonfinite_power": rows.n_nonfinite_power,
        "n_hindcast_rows": rows.n_hindcast,
        "n_time_series": rows.n_time_series,
        "n_ensemble_members": rows.n_ensemble_members,
        "nwp_init_time_on_disk": str(result.nwp.latest_init_time),
        "nwp_init_time_expected": result.nwp.expected_latest_init_time.isoformat(),
        "nwp_init_time_on_rows": str(rows.nwp_init_time),
    }
    if result.n_expected_time_series is not None:
        # Gated on the same condition as n_time_series_expected below: an unreadable meta.json
        # makes "how many are missing" as unknown as "how many are expected", so 0 must not be
        # reported here — it would be indistinguishable from a verified-complete population.
        listed_ids = result.missing_time_series_ids[:_MAX_MISSING_SERIES_LISTED]
        metadata["n_time_series_missing"] = len(result.missing_time_series_ids)
        metadata["missing_time_series_ids"] = str(list(listed_ids))
        # How many of them the field above actually spells out, so a capped list is never mistaken
        # for the whole set. Same rule as `n_late_listed` on the freshness check.
        metadata["n_time_series_missing_listed"] = len(listed_ids)
        metadata["n_time_series_expected"] = result.n_expected_time_series
    if result.horizon_hours is not None:
        metadata["forecast_horizon_hours"] = round(result.horizon_hours, 1)
    if result.nwp.n_missed is not None:
        metadata["n_missed_nwp_runs"] = result.nwp.n_missed
    return metadata


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
    power_fcst_init_time = _checked_power_fcst_init_time(context, datetime.now(UTC))
    promoted = _read_promoted_model_facts(settings.production_model_path)
    result = evaluate_live_forecast_health(
        _read_live_forecast_rows(
            settings.power_forecasts_data_path,
            settings.storage_options,
            power_fcst_init_time,
            promoted.experiment_name,
        ),
        _nwp_init_times_on_disk(settings),
        power_fcst_init_time=power_fcst_init_time,
        expected_time_series_ids=promoted.trained_time_series_ids,
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
    except BaseException as exc:
        # The same guard as `power_data_is_fresh`, for the same reason — see the comment there for
        # why it catches `BaseException`, what that costs when writing tests, and rule 7 for why a
        # warning path may never raise.
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        logger.exception("Could not evaluate live-forecast health")
        report_check_degradation(check_name="live_forecasts_are_healthy", exc=exc)
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=f"Could not evaluate live-forecast health: {exc!r}",
        )
