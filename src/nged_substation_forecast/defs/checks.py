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
and it is the hand-off point for the future Sentry missed-check-in alarm
(`#63 <https://github.com/openclimatefix/nged-substation-forecast/issues/63>`_): that alarm,
when it lands, consumes the same ``PowerFreshnessResult``. The two stay complementary — the
Sentry alarm fires on total silence from outside the deployment, this check reports per-series
staleness from inside Dagster.
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
from nged_data.storage import latest_time_per_time_series_id

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
    max_times: pl.DataFrame,
    roster_ids: pl.Series | None,
    now: datetime,
    threshold: timedelta,
) -> PowerFreshnessResult:
    """Classify each time series as fresh, stale, or never-reported.

    Pure and deterministic — no Dagster, Delta, or clock access — so it is unit-testable
    directly and reusable by the future Sentry alarm.

    Args:
        max_times: One row per ``time_series_id`` that has data, carrying its most recent
            ``time`` in a ``max_time`` column.
        roster_ids: The full set of expected ``time_series_id``s (from the ``TimeSeriesMetadata``
            roster), used to flag ids that have *never* sent data. ``None`` when no roster is
            available, in which case never-reported ids cannot be detected.
        now: Current time (UTC).
        threshold: A series is stale when ``max_time < now - threshold``.

    Returns:
        A ``PowerFreshnessResult`` summarising the health of the power feed.
    """
    # Strip any Patito model so the frame-building below uses plain Polars semantics.
    max_times = pl.DataFrame._from_pydf(max_times._df)
    last_seen_dtype = max_times.schema["max_time"]
    cutoff = now - threshold

    # Stale: has data on disk, but the newest observation predates the cutoff.
    stale = max_times.filter(pl.col("max_time") < cutoff).select(
        "time_series_id",
        last_seen=pl.col("max_time"),
        hours_late=(pl.lit(now) - pl.col("max_time")).dt.total_seconds() / 3600.0,
        status=pl.lit("stale", dtype=pl.String),
    )

    # Never reported: in the roster, but with no rows in the Delta table at all.
    if roster_ids is not None:
        never_ids = roster_ids.filter(~roster_ids.is_in(max_times["time_series_id"].implode()))
    else:
        never_ids = pl.Series("time_series_id", [], dtype=max_times.schema["time_series_id"])
    never = pl.DataFrame({"time_series_id": never_ids}).select(
        "time_series_id",
        last_seen=pl.lit(None, dtype=last_seen_dtype),
        hours_late=pl.lit(None, dtype=pl.Float64),
        status=pl.lit("never", dtype=pl.String),
    )

    # Never-reported first, then most-stale first (status "never" sorts before "stale").
    late = pl.concat([never, stale]).sort(["status", "hours_late"], descending=[False, True])

    if roster_ids is not None:
        n_series_total = pl.concat([roster_ids, max_times["time_series_id"]]).n_unique()
    else:
        n_series_total = max_times.height

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
    roster = pl.read_parquet(metadata_path, storage_options=typeddict_to_dict(storage_options))
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
    max_times = latest_time_per_time_series_id(
        settings.power_time_series_data_path, storage_options
    )
    roster_ids = _read_roster_ids(settings.metadata_path, storage_options)
    result = evaluate_power_freshness(
        max_times=max_times,
        roster_ids=roster_ids,
        now=datetime.now(timezone.utc),
        threshold=_POWER_DATA_STALENESS_THRESHOLD,
    )
    return _to_asset_check_result(result)
