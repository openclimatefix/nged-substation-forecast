"""Validate the combined NORA3 wind parquet that `fetch_nora3.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>. It reads
`data/studies/weather/NORA3/NORA3_wind.parquet`, checks the row count, null and NaN counts, value
ranges, duplicate keys, the grid's shape, and the hourly time axis, prints one report, and writes
the same report to `validation.json` next to the parquet. The check exits non-zero when any check
fails. The report carries counts and ranges of values only, never a grid index range, because that
range is as private as the trial-area box.

Run it with `uv run python studies/weather_downloads/validate_nora3.py`.
"""

import calendar
import json
import sys
from pathlib import Path
from typing import Any, Final

import polars as pl
from paths import WEATHER_DOWNLOADS_DIR

PRODUCT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "NORA3"
KEY_COLUMNS: Final[list[str]] = ["time", "height_m", "y_index", "x_index"]
HEIGHTS_M: Final[set[int]] = {50, 100}
MAX_PLAUSIBLE_SPEED_M_S: Final[float] = 80.0
"""Above any wind speed a 3 km reanalysis serves at 50 m or 100 m over land."""

MAX_NAN_FRACTION: Final[float] = 0.001
"""A reanalysis over land and sea has no gaps, so more than 0.1% NaN signals a masking fault."""


def validate(*, frame: pl.DataFrame) -> dict[str, Any]:
    """Run every check on the combined frame.

    Args:
        frame: The combined NORA3 wind rows.

    Returns:
        A report of measured numbers plus a `failures` list, empty when every check passed.
    """
    failures: list[str] = []
    n_rows = frame.height
    n_cells = frame.select(pl.struct("y_index", "x_index").n_unique()).item()
    n_hours = frame["time"].n_unique()
    heights = set(frame["height_m"].unique().to_list())

    if heights != HEIGHTS_M:
        failures.append(f"heights are {sorted(heights)}, expected {sorted(HEIGHTS_M)}")
    expected_rows = n_hours * len(HEIGHTS_M) * n_cells
    if n_rows != expected_rows:
        failures.append(f"{n_rows} rows, expected {expected_rows} (hours x heights x cells)")
    n_duplicates = n_rows - frame.select(KEY_COLUMNS).unique().height
    if n_duplicates:
        failures.append(f"{n_duplicates} duplicate (time, height, y, x) keys")

    hours = frame["time"].unique().sort()
    step_hours = hours.diff().drop_nulls().dt.total_hours().unique().to_list()
    if step_hours != [1] and hours.len() > 1:
        failures.append(f"time steps in hours are {step_hours}, expected [1]")
    if hours.dt.minute().max() != 0 or hours.dt.second().max() != 0:
        failures.append("time stamps are not on the hour")

    lineage = json.loads((PRODUCT_DIR / "lineage.json").read_text())
    expected_hours = sum(
        calendar.monthrange(int(label[:4]), int(label[5:]))[1] * 24
        for label in lineage["months_combined"]
    )
    if n_hours != expected_hours:
        failures.append(
            f"{n_hours} hours, but lineage.json's months_combined hold {expected_hours}"
        )
    first_expected = lineage["months_combined"][0] + "-01"
    if str(hours[0])[:10] != first_expected:
        failures.append(f"first hour is on {str(hours[0])[:10]}, expected {first_expected}")

    stats: dict[str, Any] = {}
    for column, high in (
        ("wind_speed_m_s", MAX_PLAUSIBLE_SPEED_M_S),
        ("wind_direction_deg", 360.0),
    ):
        values = frame[column]
        n_null = values.null_count()
        n_nan = int(values.is_nan().sum())
        low_value = float(values.min())  # ty: ignore[invalid-argument-type]
        high_value = float(values.max())  # ty: ignore[invalid-argument-type]
        stats[column] = {"nulls": n_null, "nans": n_nan, "min": low_value, "max": high_value}
        if n_null:
            failures.append(f"{column} has {n_null} nulls")
        if n_nan / max(n_rows, 1) > MAX_NAN_FRACTION:
            failures.append(f"{column} has {n_nan} NaNs, above {MAX_NAN_FRACTION:.1%}")
        if low_value < 0 or high_value > high:
            failures.append(f"{column} range [{low_value}, {high_value}] outside [0, {high}]")

    return {
        "rows": n_rows,
        "hours": n_hours,
        "first_hour": str(hours[0]),
        "last_hour": str(hours[-1]),
        "cells": n_cells,
        "heights_m": sorted(heights),
        "columns": stats,
        "duplicate_keys": n_duplicates,
        "failures": failures,
    }


def main() -> int:
    """Validate `NORA3_wind.parquet`, print and write the report.

    Returns:
        0 when every check passed, 1 otherwise.
    """
    report = validate(frame=pl.read_parquet(PRODUCT_DIR / "NORA3_wind.parquet"))
    text = json.dumps(report, indent=2, default=str)
    print(text)
    (PRODUCT_DIR / "validation.json").write_text(text)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
