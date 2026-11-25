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

Series we already know have stopped reporting — ``_SILENCED_TIME_SERIES_IDS`` — are dropped from
the check's *inputs* rather than from its output, so every count describes the series we are
still watching, and the Sentry warning inherits the silencing without knowing it exists. The
check keeps naming the silenced ids every hour, green or yellow, and turns yellow on its own if
one of them reports again.

``evaluate_power_freshness`` is a pure function so it is unit-testable without Dagster or Delta,
and it is the hand-off point for routing per-series staleness to Sentry: the same
``PowerFreshnessResult`` is fed to ``report_power_freshness`` (in
``nged_substation_forecast._sentry``) rather than recomputed. This check and its Sentry warning
report per-series staleness from inside Dagster, so they need the Dagster daemon to be alive.

``cleaned_power_keeps_up_with_raw`` warns when the cleaned power table has fallen behind the raw
``power_time_series`` table it is built from.

Both checks are ``AssetCheckSeverity.WARN`` and ``blocking=False``, and nothing either one's
*body* does can fail its own step: each sits under a catch-all which logs the traceback, reports
the exception to Sentry (the run no longer fails, so the failure hook no longer fires) and
returns an unhealthy result. Catching ``BaseException`` is what makes that absolute — a Rust
panic in any of the pyo3 extensions these bodies read through arrives as a ``PanicException``,
which does not derive from ``Exception``. Only cancellation is re-raised, which is what we want:
a cancelled run should cancel. What remains outside the guard is Dagster's own machinery:
resource init, and serialising the returned ``AssetCheckResult`` into an event. A warning path
that could fail would turn fail-open into fail-closed at exactly the wrong moment (rule 7 of [The
rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).

The design reasoning behind ``power_data_is_fresh`` is on the production-deployment page — [Warn
on stale power
data](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check).

``power_data_is_fresh`` salvages nothing below the catch-all on purpose: its only per-read
fallback would be "metadata table unknown", and a fresh power table would then render an unreadable
metadata table as a green tick.
"""

import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import polars as pl
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions, delta_table_exists, object_exists
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    DagsterExecutionInterruptedError,
    MetadataValue,
    TableColumn,
    TableRecord,
    TableSchema,
    asset_check,
)
from delta_store.cleaned_power_time_series import read_cleaning_provenance
from deltalake import DeltaTable
from nged_data.storage import time_series_coverage

from nged_substation_forecast._sentry import report_check_degradation, report_power_freshness
from nged_substation_forecast.defs.assets import power_time_series_and_metadata

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

Unlike the other capped listings this one lands in Dagster's event log, durable storage written
every hour a stall lasts. At V2 scale (~2,500 series) an uncapped table serialises to about 355 KB
per hourly evaluation, against about 8 KB at 50 rows. The cap matches the Sentry event *context*
(also 50) rather than the tighter 20 on the Sentry message body, so the same leading series appear
in both and there is one less threshold to keep in sync.

The rows follow ``_LATE_STATUS_ORDER``, so the listing is the head of that order rather than the 50
series in most trouble: when never-reported series alone fill the cap, no stale series is listed at
all, however stale it is. ``n_stale`` and ``n_never_reported`` stay exact for the watched population
throughout, and are what the operator should read first.

Further reading:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check>
"""

_LATE_STATUS_ORDER: Final[tuple[str, ...]] = ("never", "stale")
"""Runtime tuple — declared order for the ``status`` column's ``pl.Enum``, which is also the
row order in the late-series table (never-reported series listed before merely-stale ones)."""

_POWER_DATA_STALENESS_THRESHOLD: Final[timedelta] = timedelta(hours=24)
"""A ``time_series_id`` is 'late' if its most recent observation is older than this.

NGED publishes at irregular, several-hours-apart intervals and our pipeline back-fills gaps
automatically once the feed recovers, so 24 hours is comfortably past normal jitter while still
catching a genuine multi-slot stall the same day. Fuller reasoning:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check>
"""

_SILENCED_TIME_SERIES_IDS: Final[tuple[int, ...]] = (33,)
"""Series ``power_data_is_fresh`` stops warning about, because we know they are out of service.

33: no readings since 2026-01-26; NGED reported the meter out of service.

Delete an entry to start warning about that series again. How to edit the list, and what each
description means:
<https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/#degraded-input-data-nwp-feed-down-or-telemetry-stalled>

The list is source code rather than operator-editable state because it ships read-only in the
container image, so the check could not edit it even if it wanted to: a check that writes anything
is a warning path that can fail, which rule 7 forbids. The check names every listed id in its own
output each hour, green or yellow, so the silencing stays visible and cannot quietly be forgotten,
and it turns yellow by itself if a silenced series reports data again — the yellow is the prompt
for a human to delete the line. Fuller reasoning:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#silence-a-series-that-has-stopped-reporting>
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
        started reporting again is not stale. The asset check reacts to a resurrection on its
        own, through ``passed``.
        """
        return self.n_late == 0


def evaluate_power_freshness(
    coverage: pl.DataFrame,
    expected_ids: pl.Series | None,
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
        expected_ids: The full set of expected ``time_series_id``s (from the ``TimeSeriesMetadata``
            table), used to flag ids that have *never* sent data. ``None`` when no metadata
            table is available, in which case never-reported ids cannot be detected.
        now: Current time (UTC).
        threshold: A series is stale when ``last_time < now - threshold``.
        silenced_ids: Series we already know have stopped reporting, dropped from both inputs
            before anything is classified, so every count describes the watched population. One
            that has reported since the cutoff comes back as ``resurrected_ids`` instead.

    Returns:
        A ``PowerFreshnessResult`` summarising the health of the power feed.
    """
    # Strip any Patito model so the frame-building below uses plain Polars semantics.
    coverage = pl.DataFrame._from_pydf(coverage._df)
    last_time_dtype = coverage.schema["last_time"]
    status_dtype = pl.Enum(_LATE_STATUS_ORDER)
    cutoff = now - threshold

    # Drop the silenced series from the *inputs* rather than the output, so every count below
    # describes what we are watching without needing to know silencing exists. Fuller reasoning:
    # <https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#silence-a-series-that-has-stopped-reporting>
    # A silenced id fresher than the cutoff is reported instead as resurrected: the fault we
    # silenced has evidently healed, and the complement of `stale`'s `<` is exactly the right
    # test — an id that is in neither frame (a typo, or one NGED has never published) is
    # correctly not a resurrection.
    silenced = list(silenced_ids)
    resurrected_ids = tuple(
        coverage.filter(pl.col("last_time") >= cutoff, pl.col("time_series_id").is_in(silenced))[
            "time_series_id"
        ].sort()
    )
    coverage = coverage.filter(~pl.col("time_series_id").is_in(silenced))
    if expected_ids is not None:
        expected_ids = expected_ids.filter(~expected_ids.is_in(silenced))

    # Stale: has data on disk, but the newest observation predates the cutoff.
    #
    # NOTE: this is deliberately not restricted to `expected_ids`. A series keeps being flagged
    # stale even after NGED decommissions it, as long as old rows remain on disk — which is what we
    # want for now: we would rather be told about a series that has gone quiet than silently stop
    # watching it. Restricting to `expected_ids` would not silence one anyway: `upsert_metadata`
    # never drops a series, so a retired series stays in the metadata table for good. Silencing one
    # takes the explicit record of silenced ids above, which serves a retired series and a broken
    # sensor alike — the check cannot tell them apart, and does not need to.
    stale = coverage.filter(pl.col("last_time") < cutoff).select(
        "time_series_id",
        last_seen=pl.col("last_time"),
        hours_late=(pl.lit(now) - pl.col("last_time")).dt.total_seconds() / 3600.0,
        status=pl.lit("stale", dtype=status_dtype),
    )

    # Never reported: in the metadata table, but with no rows in the Delta table at all.
    if expected_ids is not None:
        never_ids = expected_ids.filter(~expected_ids.is_in(coverage["time_series_id"].implode()))
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

    if expected_ids is not None:
        n_series_total = pl.concat([expected_ids, coverage["time_series_id"]]).n_unique()
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


def _read_expected_ids_and_fault_notes(
    metadata_path: str, storage_options: ObjectStoreOptions | None
) -> tuple[pl.Series | None, dict[int, str]]:
    """Read the expected ``time_series_id``s and NGED's fault notes from the metadata table.

    Returns ``(None, {})`` if the metadata table is absent. The notes are the non-null values of
    the table's ``information`` column, which ``TimeSeriesMetadata`` allows to be missing, so a
    table without the column gives no notes.
    """
    if not object_exists(metadata_path, storage_options):
        return None, {}
    metadata_table = pl.scan_parquet(
        metadata_path, storage_options=typeddict_to_dict(storage_options)
    )
    has_notes = "information" in metadata_table.collect_schema().names()
    columns = ["time_series_id", "information"] if has_notes else ["time_series_id"]
    metadata = metadata_table.select(columns).collect()
    notes = dict(metadata.drop_nulls("information").rows()) if has_notes else {}
    return metadata["time_series_id"], notes


def _late_table_metadata(late: pl.DataFrame) -> MetadataValue:
    """Render the late-series frame as a Dagster table for the check's UI metadata.

    Renders every row it is given: the caller truncates to ``_MAX_LATE_SERIES_IN_TABLE`` first,
    so that the row count reported as ``n_late_listed`` and the table itself cannot disagree.
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


def _describe_power_freshness(
    result: PowerFreshnessResult, fault_notes: Mapping[int, str] | None = None
) -> str:
    """One human-readable line: the state of the watched feed, and what is being ignored.

    ``fault_notes`` maps each ``time_series_id`` to the text of the ``Information`` field in NGED's
    JSON file for that series, for example ``Analogues not working.``. NGED uses the field to note
    a known meter fault or a customer's status. ``upsert_metadata`` stores it in the ``information``
    column of the metadata table, and ``_read_expected_ids_and_fault_notes`` reads it back. Every
    note is named, whether or not the series is silenced.
    """
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
        sentences.append(f"Ignoring {len(result.silenced_ids)} silenced time series: {ids}.")
    if result.resurrected_ids:
        ids = ", ".join(str(i) for i in result.resurrected_ids)
        sentences.append(
            f"Reporting again, so remove {ids} from _SILENCED_TIME_SERIES_IDS in defs/checks.py."
        )
    if fault_notes:
        notes = "; ".join(
            f"{series_id}: {json.dumps(note, ensure_ascii=False)}"
            for series_id, note in sorted(fault_notes.items())
        )
        sentences.append(f"NGED fault notes: {notes}.")
    return " ".join(sentences)


def _to_asset_check_result(
    result: PowerFreshnessResult, fault_notes: Mapping[int, str] | None = None
) -> AssetCheckResult:
    """Turn a ``PowerFreshnessResult`` into a WARN-severity Dagster check result."""
    # Truncate once, here, so the table and the count that describes it come from one slice.
    listed = result.late.head(_MAX_LATE_SERIES_IN_TABLE)
    return AssetCheckResult(
        # A stalled feed is expected to self-heal via back-fill, so warn — never fail the run and
        # block downstream assets. Absent data is not "healthy" either, hence the count guard.
        # A resurrection is the one yellow nothing else can raise: the series is healthy, so
        # `is_healthy` stays true, and only an edit to the silenced list clears it.
        passed=result.is_healthy and result.n_series_total > 0 and not result.resurrected_ids,
        severity=AssetCheckSeverity.WARN,
        description=_describe_power_freshness(result, fault_notes),
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
    expected_ids, fault_notes = _read_expected_ids_and_fault_notes(
        settings.metadata_path, storage_options
    )
    result = evaluate_power_freshness(
        coverage=coverage,
        expected_ids=expected_ids,
        now=datetime.now(UTC),
        threshold=_POWER_DATA_STALENESS_THRESHOLD,
        # Read at the call site, not defaulted into the signature, so a test can monkeypatch it.
        silenced_ids=_SILENCED_TIME_SERIES_IDS,
    )
    # Forward per-series staleness to Sentry (a no-op unless a DSN is set and some series is late).
    # Best-effort: report_power_freshness never raises, so a telemetry hiccup costs no more than
    # its own event — were it to raise, the caller's catch-all would discard this whole evaluation.
    report_power_freshness(settings, result)
    return _to_asset_check_result(result, fault_notes)


@asset_check(
    asset=power_time_series_and_metadata,
    blocking=False,
    description=(
        "Warn if any watched time series has no fresh power data within the staleness threshold "
        "(stale) or has never reported at all (never), or if a silenced series has started "
        "reporting again. The description also names the `Information` note in NGED's JSON file "
        "for each series that has one."
    ),
)
def power_data_is_fresh() -> AssetCheckResult:
    """Report how many time series are late on the ``power_time_series`` Delta table.

    Runs automatically alongside every ``power_time_series_and_metadata`` materialisation (hourly
    via ``power_time_series_and_metadata_schedule``), so the check re-evaluates freshness each
    hour regardless of whether new data landed. It reads the Delta table's *actual* data recency
    off disk, not the asset's materialisation timestamp — the hourly job succeeds even when NGED
    publishes nothing new, so a native Dagster freshness policy, which only sees materialisation
    timestamps, would miss that exact failure. A ``time_series_id`` in
    ``_SILENCED_TIME_SERIES_IDS`` is dropped from every count here, so a sensor already known to
    have stopped reporting cannot keep this check permanently yellow; the check turns yellow
    again on its own if that series reports data. The same per-series staleness this check
    reports also reaches Sentry as a warning, via ``report_power_freshness``.

    Cannot fail its own step: the whole body is guarded, so an object-store error, a half-written
    ``metadata.parquet`` or a bug in here degrades to an unhealthy result rather than failing the
    hourly production run. (A merely *slow* object store makes a slow check, not a degraded one —
    nothing here imposes a step-level timeout.)

    Further reading: [why the check reads Delta directly rather than trusting materialisation,
    and the staleness
    threshold](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check),
    [why silencing a series that has stopped reporting is a source
    constant](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#silence-a-series-that-has-stopped-reporting).
    """
    try:
        return _check_power_data_freshness()
    except BaseException as exc:
        # `BaseException`, not `Exception`, and why a warning-severity check must never raise
        # otherwise (rule 7):
        # <https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check>
        # The same guard swallows pytest's own `fail`/`skip`, which matters when testing this body:
        # <https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#a-production-except-baseexception-swallows-pytests-fail-and-skip-too>
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        # Sentry is told explicitly because not failing the run means the `sentry_capture_failure`
        # hook that would otherwise report it no longer fires.
        logger.exception("Could not evaluate power-data freshness")
        report_check_degradation(check_name="power_data_is_fresh", exc=exc)
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=f"Could not evaluate power-data freshness: {exc!r}",
        )


# ---------------------------------------------------------------------------
# cleaned_power_keeps_up_with_raw: is the cleaned power table built from recent raw data?
# ---------------------------------------------------------------------------

_CLEANING_LAG_WARN_COMMITS: Final[int] = 2
"""How many raw-table commits the cleaned table may trail before the check warns.

The ingest commits a new raw version only when NGED delivers new rows, about every 6 hours. A lag
of 0 or 1 commits is healthy, because the check runs in parallel with ``clean_nged_power_data`` in
the same job and may see the cleaning from the previous run. A lag of 2 means the cleaning has
missed at least one NGED delivery, about 6 to 12 hours. Counting raw commits rather than hours of
reading time counts missed runs, and does not false-alarm after an NGED backlog, when one ingest
catches up several hours of readings at once."""


def _check_cleaned_power_keeps_up() -> AssetCheckResult:
    """Compare the raw table's Delta version with the version the cleaned table was built from.

    Split out from the check itself so the check's ``except`` wraps the whole body.
    """
    settings = Settings()
    storage_options = settings.storage_options
    options = typeddict_to_dict(storage_options) or {}
    raw_path = settings.power_time_series_data_path
    if not delta_table_exists(raw_path, storage_options):
        return AssetCheckResult(
            passed=True, description=f"{raw_path} does not exist yet, so there is nothing to clean."
        )
    raw_table = DeltaTable(raw_path, storage_options=options)
    raw_version = raw_table.version()

    provenance = read_cleaning_provenance(
        settings.cleaned_power_time_series_data_path, storage_options
    )
    if provenance is None:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=(
                "The cleaned power table is absent or unreadable, or its newest write records no"
                " provenance."
                " Materialise `clean_nged_power_data`."
            ),
            metadata={"raw_version": raw_version},
        )
    if provenance.raw_table_id != str(raw_table.metadata().id):
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=(
                "The cleaned power table was built from a different raw power table than the one"
                " on disk now. Materialise `clean_nged_power_data` with `force` set."
            ),
            metadata={"raw_version": raw_version, "cleaned_raw_version": provenance.raw_version},
        )
    lag = raw_version - provenance.raw_version
    passed = lag < _CLEANING_LAG_WARN_COMMITS
    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN,
        description=(
            f"The cleaned power table is {lag} raw commit(s) behind the raw power table."
            + ("" if passed else " Check the recent `clean_nged_power_data` runs.")
        ),
        metadata={
            "raw_version": raw_version,
            "cleaned_raw_version": provenance.raw_version,
            "n_raw_commits_behind": lag,
        },
    )


@asset_check(
    asset=power_time_series_and_metadata,
    blocking=False,
    description=(
        "Warn if the cleaned power table was built from a raw power table 2 or more commits"
        " (about 6 to 12 hours of NGED deliveries) older than the current one, from a different"
        " raw table, or if it is absent."
    ),
)
def cleaned_power_keeps_up_with_raw() -> AssetCheckResult:
    """Report whether ``clean_nged_power_data`` is keeping up with NGED's deliveries.

    Runs alongside every ``power_time_series_and_metadata`` materialisation, in parallel with
    ``clean_nged_power_data`` in ``power_time_series_and_metadata_job``. It compares the raw
    ``power_time_series`` table's current Delta version with the ``raw_version`` recorded in the
    ``cleaned_power_time_series`` table's newest write commit, and warns when the cleaned table is
    2 or more raw commits behind, was built from a different raw table, is absent, or records no
    provenance. It is the only signal for a cleaning asset that silently stopped running, because
    ``power_data_is_fresh`` reads raw coverage and the Sentry failure hook fires only on a run
    that failed. A stale or unreadable ``TimeSeriesMetadata`` table also stops the cleaning, so a
    bad metadata table makes the cleaned power go stale while every reader of the cleaned table
    keeps succeeding.

    Cannot fail its own step: the whole body is guarded, so an unreadable table degrades to a
    warning rather than failing the hourly production run.
    """
    try:
        return _check_cleaned_power_keeps_up()
    except BaseException as exc:
        # The same guard as `power_data_is_fresh`, for the same reasons.
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        logger.exception("Could not compare the cleaned power table with the raw one")
        report_check_degradation(check_name="cleaned_power_keeps_up_with_raw", exc=exc)
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.WARN,
            description=f"Could not compare the cleaned power table with the raw one: {exc!r}",
        )
