"""Coverage and pinball loss in the tails of a quantile forecast, for the table of a `result.md`.

NGED's main use of these forecasts is reducing the curtailment of generators, so the tails of the
forecast distribution matter most. `tail_calibration_table` reports, at each of the levels 0.01,
0.05, 0.95, and 0.99, how often the measured value fell at or below the forecast quantile (its
coverage, whose target is the level itself) and the pinball loss of that quantile. Both come with a
95% interval from resampling whole months, the design `studies.bootstrap` uses.

A prediction set holds the quantile levels its experiment chose, in columns named `q05`, `q95`, and
so on. The stored sets of the demand experiments hold the nine levels 0.05, 0.10, 0.25, 0.40, 0.50,
0.60, 0.75, 0.90, and 0.95, so the 0.01 and 0.99 rows of their tables say "not delivered" rather
than being estimated. The coverage of a quantile that a forecaster derived by widening other
quantiles is still the coverage of that column, so the table needs no knowledge of how the column
was made.
"""

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import polars as pl

from studies.bootstrap import bootstrap_month_totals

TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
"""The quantile levels that the tail-calibration table reports, lowest first."""

ACTUAL_COLUMN: Final[str] = "actual"
"""The column of the measured value, in the same unit as the quantile columns."""

TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "level",
    "n_rows",
    "coverage",
    "coverage_lower_95",
    "coverage_upper_95",
    "gap",
    "pinball",
    "pinball_lower_95",
    "pinball_upper_95",
)
"""The columns of `tail_calibration_table`, in order."""


def level_column(*, level: float) -> str:
    """Return the name of the quantile column for a level, such as `q05` for 0.05.

    Args:
        level: A quantile level between 0 and 1 with at most two decimal places.

    Returns:
        The column name `q` followed by the level in whole percent, zero-padded to two digits.

    Raises:
        ValueError: If the level is outside (0, 1) or does not name a whole percent.
    """
    percent = round(level * 100)
    if not 0 < percent < 100 or abs(level * 100 - percent) > 1e-9:
        msg = f"level must be a whole percent strictly between 0 and 1, got {level}"
        raise ValueError(msg)
    return f"q{percent:02d}"


def _month_labels(*, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's month, taken from the forecast's initialisation time.

    Taking the month from the run keeps a run in one month, as `losses_for_bootstrap` does.
    """
    return frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()


def _mean_with_interval(*, values: np.ndarray, months: np.ndarray) -> tuple[float, float, float]:
    """Return the mean of per-row values with a 95% interval from resampling whole months.

    The rows are totalled by month first, because resampling a few months of totals gives the same
    interval as resampling millions of rows (see `bootstrap_month_totals`). Every row counts alike,
    whichever fitting seed it came from.
    """
    unique_months, month_index = np.unique(months, return_inverse=True)
    sums = np.bincount(month_index, weights=values, minlength=len(unique_months))
    counts = np.bincount(month_index, minlength=len(unique_months)).astype(np.float64)
    lower, upper = bootstrap_month_totals(month_sums=sums[None, :], month_counts=counts)
    return float(values.mean()), lower, upper


def pinball(*, actual: np.ndarray, quantile: np.ndarray, level: float) -> np.ndarray:
    """Return the pinball loss of one quantile forecast at each row.

    Args:
        actual: The measured values, shape (n_rows,).
        quantile: The forecast quantile at `level`, shape (n_rows,), in the unit of `actual`.
        level: The quantile level, between 0 and 1.

    Returns:
        The loss per row, shape (n_rows,): `level` times the shortfall when the forecast is below
        the measurement, and `1 - level` times the excess when the forecast is above.
    """
    error = actual - quantile
    return np.maximum(level * error, (level - 1.0) * error)


def tail_calibration_table(
    *, frame: pl.DataFrame, levels: Sequence[float] = TAIL_LEVELS
) -> pl.DataFrame:
    """Report coverage and pinball loss at each tail level over all the rows of one arm.

    Args:
        frame: One arm's forecast rows, with `power_fcst_init_time`, an `actual` column, and a
            `q<percent>` column per delivered level. Rows with a null `actual` or a null quantile
            at a level are left out of that level's row.
        levels: The levels to report, such as `TAIL_LEVELS`.

    Returns:
        One row per level, in the order of `levels`, with the columns of `TABLE_COLUMNS`:
        `coverage` is the share of rows whose actual value was at or below the quantile, `gap` is
        `coverage - level` (positive when the quantile sits too high), and `pinball` is the mean
        pinball loss in the unit of the quantile columns. The 95% intervals resample whole months.
        A level with no column in `frame` has `n_rows` of 0 and nulls in every other column.
    """
    months_all = _month_labels(frame=frame)
    rows = []
    for level in levels:
        column = level_column(level=level)
        if column not in frame.columns:
            rows.append({"level": level, "n_rows": 0})
            continue
        usable = ~(frame[ACTUAL_COLUMN].is_null() | frame[column].is_null()).to_numpy()
        actual = frame[ACTUAL_COLUMN].to_numpy()[usable].astype(np.float64)
        quantile = frame[column].to_numpy()[usable].astype(np.float64)
        months = months_all[usable]
        covered = (actual <= quantile).astype(np.float64)
        losses = pinball(actual=actual, quantile=quantile, level=level)
        coverage, coverage_lower, coverage_upper = _mean_with_interval(
            values=covered, months=months
        )
        loss, loss_lower, loss_upper = _mean_with_interval(values=losses, months=months)
        rows.append(
            {
                "level": level,
                "n_rows": len(actual),
                "coverage": coverage,
                "coverage_lower_95": coverage_lower,
                "coverage_upper_95": coverage_upper,
                "gap": coverage - level,
                "pinball": loss,
                "pinball_lower_95": loss_lower,
                "pinball_upper_95": loss_upper,
            }
        )
    return pl.DataFrame(
        rows, schema=dict.fromkeys(TABLE_COLUMNS, pl.Float64) | {"n_rows": pl.Int64}
    )


def tail_calibration_markdown(*, tables: Mapping[str, pl.DataFrame], unit: str) -> str:
    """Render the tail-calibration tables of several arms as one markdown table for a `result.md`.

    Args:
        tables: Each arm's name mapped to its `tail_calibration_table`.
        unit: The unit of the pinball loss, such as "points of capacity", written in the header.

    Returns:
        A markdown table with one row per arm and level. A level that the arm did not deliver reads
        "not delivered". Coverage shows its 95% interval, and the gap is in percentage points.
    """
    lines = [
        f"| Arm | Level | Coverage (95% interval) | Gap (points) | Pinball ({unit}) |",
        "|---|---|---|---|---|",
    ]
    for arm, table in tables.items():
        for row in table.iter_rows(named=True):
            if row["n_rows"] == 0:
                lines.append(f"| {arm} | {row['level']:.2f} | not delivered | | |")
                continue
            lines.append(
                f"| {arm} | {row['level']:.2f} "
                f"| {row['coverage']:.3f} ({row['coverage_lower_95']:.3f} to "
                f"{row['coverage_upper_95']:.3f}) "
                f"| {100 * row['gap']:+.1f} "
                f"| {row['pinball']:.4f} ({row['pinball_lower_95']:.4f} to "
                f"{row['pinball_upper_95']:.4f}) |"
            )
    return "\n".join(lines) + "\n"
