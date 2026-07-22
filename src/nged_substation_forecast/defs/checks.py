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
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import polars as pl
from contracts._uri import ObjectStoreOptions, object_exists
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from dagster import (
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
