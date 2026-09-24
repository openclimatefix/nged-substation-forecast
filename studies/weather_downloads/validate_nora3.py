"""Validate the combined NORA3 wind parquet that `fetch_nora3.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. It reads
`data/studies/weather/NORA3/NORA3_wind.parquet` and runs the checks in `CHECK_NAMES`: heights, row
count against hours x heights x cells, duplicate keys, month-by-month contiguity of the hourly time
axis, the first and last hour against `FIRST_MONTH` and `LAST_MONTH`, the months against the month
cache and the lineage note, the step across the join between the aggregated dataset and the monthly
files, null and NaN counts, value ranges, and floors that catch double-scaled data.

**The script prints one PASS or FAIL line per check and no count.** Row, cell, and hour counts
reveal the size of the private trial-area box, so the measured numbers go only to `validation.json`
on the private disk next to the parquet. The script exits non-zero when any check fails.

Run it with `uv run python studies/weather_downloads/validate_nora3.py`.
"""

import calendar
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
from paths import WEATHER_DOWNLOADS_DIR

PRODUCT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "NORA3"
KEY_COLUMNS: Final[list[str]] = ["time", "height_m", "y_index", "x_index"]
HEIGHTS_M: Final[set[int]] = {50, 100}
FIRST_MONTH: Final[str] = "2015-01"
LAST_MONTH: Final[str] = "2026-08"
"""The range the extended download is meant to cover: the canonical file must start at the first
hour of `FIRST_MONTH` and end at the last hour of `LAST_MONTH`. Edit `LAST_MONTH` when extending."""

SEAM_LAST_HOUR: Final[datetime] = datetime(2025, 1, 31, 23, tzinfo=UTC).replace(tzinfo=None)
"""The aggregated dataset's last hour. The next hour comes from a monthly file."""

SEAM_WINDOW: Final[tuple[datetime, datetime]] = (
    datetime(2025, 1, 1, tzinfo=UTC).replace(tzinfo=None),
    datetime(2025, 2, 28, 23, tzinfo=UTC).replace(tzinfo=None),
)
"""Hours around the join over which the ordinary hour-to-hour change is measured."""

SEAM_MAX_RATIO: Final[float] = 1.5
"""The step across the join may not exceed this multiple of the 99th percentile of the ordinary
steps in `SEAM_WINDOW`."""

MAX_PLAUSIBLE_SPEED_M_S: Final[float] = 80.0
"""Above any wind speed a 3 km reanalysis serves at 50 m or 100 m over land."""

MIN_MAX_SPEED_M_S: Final[float] = 5.0
MIN_MEAN_SPEED_M_S: Final[float] = 1.0
MIN_DIRECTION_SPAN_DEG: Final[float] = 90.0
"""Floors that catch data scaled twice: no set of months of 50 m or 100 m wind peaks below 5 m/s or
averages below 1 m/s, and directions span far more than 90 degrees."""

MAX_NAN_FRACTION: Final[float] = 0.001
"""A reanalysis over land and sea has no gaps, so more than 0.1% NaN signals a masking fault."""

CHECK_NAMES: Final[tuple[str, ...]] = (
    "heights",
    "row_count",
    "duplicate_keys",
    "time_axis_contiguous_per_month",
    "first_hour",
    "last_hour",
    "seam_continuity",
    "months_match_cache_and_lineage",
    "no_nulls",
    "nan_fraction",
    "value_ranges",
    "value_floors",
)
"""Every check, in report order. A check absent from the failure dictionary passed."""


def _hour_stamp(*, hours: pl.Series, index: int) -> datetime:
    """Return one distinct hour as a timezone-naive `datetime`."""
    return hours[index]


def _time_failures(*, frame: pl.DataFrame, hours: pl.Series) -> dict[str, str]:
    """Check the time axis and the months against the month cache and the lineage note.

    Args:
        frame: The combined NORA3 wind rows.
        hours: The sorted distinct `time` values of `frame`.

    Returns:
        A failed check's name mapped to a detail message that may carry counts.
    """
    failures: dict[str, str] = {}
    per_month = (
        hours.to_frame()
        .group_by(pl.col("time").dt.strftime("%Y-%m").alias("month"))
        .agg(
            n_hours=pl.len(),
            span_hours=(pl.col("time").max() - pl.col("time").min()).dt.total_hours() + 1,
        )
    )
    for month, n_hours, span_hours in per_month.iter_rows():
        expected = calendar.monthrange(int(month[:4]), int(month[5:]))[1] * 24
        if not n_hours == span_hours == expected:
            failures["time_axis_contiguous_per_month"] = (
                f"{month} holds {n_hours} distinct hours spanning {span_hours}, expected {expected}"
            )
    if hours.dt.minute().max() != 0 or hours.dt.second().max() != 0:
        failures["time_axis_contiguous_per_month"] = "time stamps are not on the hour"

    lineage = json.loads((PRODUCT_DIR / "lineage.json").read_text())
    months_lineage = lineage["months_combined"]
    months_cache = sorted(path.stem for path in (PRODUCT_DIR / "_month_cache").glob("*.parquet"))
    months_parquet = sorted(per_month["month"].to_list())
    if not months_parquet == months_cache == months_lineage:
        failures["months_match_cache_and_lineage"] = (
            f"parquet {months_parquet}, cache {months_cache}, lineage {months_lineage}"
        )

    first = _hour_stamp(hours=hours, index=0)
    expected_first = datetime(int(FIRST_MONTH[:4]), int(FIRST_MONTH[5:]), 1, tzinfo=UTC).replace(
        tzinfo=None
    )
    if first != expected_first:
        failures["first_hour"] = f"{first}, expected {expected_first}"
    last_year, last_month = int(LAST_MONTH[:4]), int(LAST_MONTH[5:])
    expected_last = datetime(
        last_year, last_month, calendar.monthrange(last_year, last_month)[1], 23, tzinfo=UTC
    ).replace(tzinfo=None)
    last = _hour_stamp(hours=hours, index=-1)
    if last != expected_last:
        failures["last_hour"] = f"{last}, expected {expected_last}"
    return failures


def _mean_abs_step_by_hour(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return, per hour, the mean absolute wind speed change from the hour before.

    Args:
        frame: The combined NORA3 wind rows, restricted to the hours of interest.

    Returns:
        One row per hour after the first, with `time` and `mean_abs_step`, the mean over cells
        and heights.
    """
    return (
        frame.sort("time")
        .with_columns(
            step=(
                pl.col("wind_speed_m_s")
                - pl.col("wind_speed_m_s").shift(1).over("height_m", "y_index", "x_index")
            ).abs()
        )
        .group_by("time")
        .agg(mean_abs_step=pl.col("step").mean())
        .sort("time")
        .drop_nulls()
    )


def _seam_report(*, frame: pl.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    """Compare the wind speed step across the aggregate-to-monthly-file join with ordinary steps.

    Args:
        frame: The combined NORA3 wind rows.

    Returns:
        The measured steps and a failed check's name mapped to a detail message.
    """
    window = frame.filter(pl.col("time").is_between(*SEAM_WINDOW))
    steps = _mean_abs_step_by_hour(frame=window)
    seam_hour = SEAM_LAST_HOUR + timedelta(hours=1)
    seam_step = float(steps.filter(pl.col("time") == seam_hour)["mean_abs_step"].item())
    ordinary = steps.filter(pl.col("time") != seam_hour)["mean_abs_step"]
    reference = float(ordinary.quantile(0.99))  # ty: ignore[invalid-argument-type]
    measured = {
        "seam_step_m_s": seam_step,
        "ordinary_step_p99_m_s": reference,
        "ordinary_step_median_m_s": float(ordinary.median()),  # ty: ignore[invalid-argument-type]
    }
    failures: dict[str, str] = {}
    if seam_step > SEAM_MAX_RATIO * reference:
        failures["seam_continuity"] = (
            f"step across the join {seam_step:.3f} m/s exceeds {SEAM_MAX_RATIO} x the ordinary "
            f"99th percentile step {reference:.3f} m/s"
        )
    return measured, failures


def _column_failures(*, frame: pl.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    """Measure nulls, NaNs and ranges of the two value columns and check them.

    Args:
        frame: The combined NORA3 wind rows.

    Returns:
        The per-column statistics and a failed check's name mapped to a detail message.
    """
    failures: dict[str, str] = {}
    stats: dict[str, Any] = {}
    for column, high in (
        ("wind_speed_m_s", MAX_PLAUSIBLE_SPEED_M_S),
        ("wind_direction_deg", 360.0),
    ):
        values = frame[column].fill_nan(None)
        n_nan = int(frame[column].is_nan().sum())
        low_value = float(values.min())  # ty: ignore[invalid-argument-type]
        high_value = float(values.max())  # ty: ignore[invalid-argument-type]
        mean_value = float(values.mean())  # ty: ignore[invalid-argument-type]
        stats[column] = {
            "nulls": frame[column].null_count(),
            "nans": n_nan,
            "min": low_value,
            "max": high_value,
            "mean": mean_value,
        }
        if frame[column].null_count():
            failures["no_nulls"] = f"{column} has nulls"
        if n_nan / max(frame.height, 1) > MAX_NAN_FRACTION:
            failures["nan_fraction"] = f"{column} has {n_nan} NaNs, above {MAX_NAN_FRACTION:.1%}"
        if low_value < 0 or high_value > high:
            failures["value_ranges"] = (
                f"{column} range [{low_value}, {high_value}] not in [0, {high}]"
            )
        if column == "wind_speed_m_s" and (
            high_value < MIN_MAX_SPEED_M_S or mean_value < MIN_MEAN_SPEED_M_S
        ):
            failures["value_floors"] = f"{column} max {high_value}, mean {mean_value}: too low"
        if column == "wind_direction_deg" and high_value - low_value < MIN_DIRECTION_SPAN_DEG:
            failures["value_floors"] = f"{column} spans under {MIN_DIRECTION_SPAN_DEG} degrees"
    return stats, failures


def validate(*, frame: pl.DataFrame) -> dict[str, Any]:
    """Run every check on the combined frame.

    Args:
        frame: The combined NORA3 wind rows.

    Returns:
        A report of measured numbers plus a `failures` dictionary of failed check names mapped to
        detail messages, empty when every check passed.
    """
    n_rows = frame.height
    n_cells = frame.select(pl.struct("y_index", "x_index").n_unique()).item()
    hours = frame["time"].unique().sort()
    heights = set(frame["height_m"].unique().to_list())

    failures: dict[str, str] = {}
    if heights != HEIGHTS_M:
        failures["heights"] = f"heights are {sorted(heights)}, expected {sorted(HEIGHTS_M)}"
    expected_rows = hours.len() * len(HEIGHTS_M) * n_cells
    if n_rows != expected_rows:
        failures["row_count"] = f"{n_rows} rows, expected {expected_rows} (hours x heights x cells)"
    n_duplicates = n_rows - frame.select(KEY_COLUMNS).unique().height
    if n_duplicates:
        failures["duplicate_keys"] = f"{n_duplicates} duplicate (time, height, y, x) keys"
    failures |= _time_failures(frame=frame, hours=hours)
    seam, seam_failures = _seam_report(frame=frame)
    failures |= seam_failures
    stats, column_failures = _column_failures(frame=frame)
    failures |= column_failures

    return {
        "rows": n_rows,
        "hours": hours.len(),
        "first_hour": str(hours[0]),
        "last_hour": str(hours[-1]),
        "cells": n_cells,
        "heights_m": sorted(heights),
        "seam": seam,
        "columns": stats,
        "duplicate_keys": n_duplicates,
        "failures": failures,
    }


def main() -> int:
    """Validate `NORA3_wind.parquet`, print one line per check, and write the full report.

    Returns:
        0 when every check passed, 1 otherwise.
    """
    report = validate(frame=pl.read_parquet(PRODUCT_DIR / "NORA3_wind.parquet"))
    for name in CHECK_NAMES:
        print(f"{'FAIL' if name in report['failures'] else 'PASS'}  {name}")
    (PRODUCT_DIR / "validation.json").write_text(json.dumps(report, indent=2, default=str))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
