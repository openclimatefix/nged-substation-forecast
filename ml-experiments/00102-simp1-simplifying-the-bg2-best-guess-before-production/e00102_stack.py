"""The 14-day error correction and the spread factor of experiment 00102's stack member.

Copied from the helpers `corrections_for`, `with_corrections`, and `stack` of experiment 00048's
`e00048_common.py`, which its `e00048_assemble.py` applies, widened to 11 quantile levels. The
correction lives only in the stack member (ST and its offset twin STO) and in ablation A4, which
adds it to every member.

For each daily run with origin `o`, the correction of one series is the mean of (observed power
minus the median forecast) over the 14 days of half-hours ending at the newest usable observation,
`o - 3.5 h`. Each past half-hour takes the forecast as issued: the median of the latest earlier
daily run, from the refit that served that run. The correction is added to every quantile, then the
spread factor 1.15 widens the quantiles about the median.

Leakage check: a correction reads only half-hours whose valid time is at most `o - 3.5 h`, and
only forecasts from runs that began at or before the half-hour. Both are asserted for every run
(`corrections_for` raises `ValueError`), and `--self-test` demonstrates that changing the observed
power after `o - 3.5 h` leaves the correction unchanged.

    uv run python ml-experiments/00102-*/e00102_stack.py --self-test
"""

import argparse
from datetime import timedelta
from typing import Final

import e00102_common as common
import numpy as np
import polars as pl

CORRECTION_DAYS: Final[int] = 14
KEEP_COLUMNS: Final[tuple[str, ...]] = (*common.KEYS, "seed")


def corrections_for(*, frame: pl.DataFrame, power: np.ndarray, window_days: int) -> pl.DataFrame:
    """Return each run's error correction for one series, with its history count.

    Args:
        frame: One series' quantiles for one seed, one row per run and valid time, with every
            daily run that the correction may read.
        power: The series' observed power over capacity on the dense grid.
        window_days: Length of the correction window in days.

    Returns:
        One row per run: `power_fcst_init_time`, `correction`, `history_count`, and
        `newest_valid_used` (the latest valid time the correction reads, null with no history).

    Raises:
        ValueError: If a past forecast comes from a run that began after its half-hour, or a
            correction reads a half-hour that is later than 3.5 hours before the forecast origin.
    """
    window = window_days * common.STEPS_PER_DAY
    minimum = common.MIN_CORRECTION_DAYS * common.STEPS_PER_DAY
    runs = (
        frame.select("power_fcst_init_time")
        .unique()
        .sort("power_fcst_init_time")["power_fcst_init_time"]
    )
    origins = np.asarray(common.grid_index(times=runs))
    valid_index = np.asarray(common.grid_index(times=frame["valid_time"]))
    run_index = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    next_origin = np.append(origins[1:], np.iinfo(np.int64).max)[
        np.searchsorted(origins, run_index)
    ]
    keep = (valid_index >= run_index) & (valid_index < next_origin)
    median = np.full(power.shape, np.nan, dtype=np.float64)
    source = np.full(power.shape, -1, dtype=np.int64)
    median[valid_index[keep]] = frame[common.LEVEL_COLUMNS[common.MEDIAN]].to_numpy()[keep]
    source[valid_index[keep]] = run_index[keep]
    rows = []
    for origin_time, origin in zip(runs, origins, strict=True):
        newest = origin - common.NEWEST_AGE
        span = np.arange(newest - window + 1, newest + 1)
        used = ~np.isnan(median[span]) & ~np.isnan(power[span])
        if span.max() > newest or (source[span][used] > span[used]).any():
            msg = "a correction reads a half-hour later than origin - 3.5 h, or a later forecast"
            raise ValueError(msg)
        count = int(used.sum())
        residual = power[span][used].astype(np.float64) - median[span][used]
        rows.append(
            {
                "power_fcst_init_time": origin_time,
                "correction": float(residual.mean()) if count >= minimum else 0.0,
                "history_count": count,
                "newest_valid_used": (
                    common.GRID_START + int(span[used].max()) * common.STEP if count else None
                ),
            }
        )
    return pl.DataFrame(rows, schema_overrides={"newest_valid_used": frame["valid_time"].dtype})


def with_corrections(
    *,
    base: pl.DataFrame,
    series: pl.DataFrame,
    arrays: dict[str, common.SeriesArrays],
    window_days: int,
) -> pl.DataFrame:
    """Add a `correction` column to a fit's quantiles, one correction per series, seed, and run.

    Args:
        base: Quantiles keyed by `common.KEYS` and `seed`, the 11 level columns, for every daily
            run the correction may read.
        series: `common.series_table()`.
        arrays: Each site's arrays on the dense grid.
        window_days: Length of the correction window in days.

    Returns:
        `base` with the `correction` column.

    Raises:
        ValueError: If a correction reads a valid time later than 3.5 hours before its run's
            origin.
    """
    sites = dict(zip(series["time_series_id"], series["site"], strict=True))
    pieces = []
    for (series_id, _seed), part in base.group_by("time_series_id", "seed", maintain_order=True):
        table = corrections_for(
            frame=part, power=arrays[sites[series_id]].power, window_days=window_days
        )
        latest = table.filter(
            pl.col("newest_valid_used")
            > pl.col("power_fcst_init_time") - common.TELEMETRY_DELAY - common.STEP
        )
        if latest.height:
            msg = f"series {series_id}: a correction reads a valid time after origin - 3.5 h"
            raise ValueError(msg)
        pieces.append(
            part.join(
                table.select("power_fcst_init_time", "correction"),
                on="power_fcst_init_time",
                how="left",
                validate="m:1",
            )
        )
    return pl.concat(pieces)


def stack(*, frame: pl.DataFrame, correct: bool, spread: bool) -> pl.DataFrame:
    """Apply the correction and the spread factor to quantiles that carry a `correction` column.

    Args:
        frame: Quantiles with the 11 level columns and, if `correct`, a `correction` column.
        correct: Add the correction to every quantile.
        spread: Widen the quantiles about the median by `common.SPREAD_FACTOR`.

    Returns:
        The keys, `seed`, and the 11 sorted quantile columns.
    """
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    if correct:
        quantiles = quantiles + frame["correction"].to_numpy()[:, None]
    if spread:
        median = quantiles[:, [common.MEDIAN]]
        quantiles = median + common.SPREAD_FACTOR * (quantiles - median)
    return frame.select(*KEEP_COLUMNS).with_columns(
        pl.DataFrame(
            np.sort(quantiles, axis=1).astype(np.float32),
            schema=list(common.LEVEL_COLUMNS),
            orient="row",
        )
    )


def corrected_stack(
    *,
    raw: pl.DataFrame,
    series: pl.DataFrame,
    arrays: dict[str, common.SeriesArrays],
    window_days: int = CORRECTION_DAYS,
) -> pl.DataFrame:
    """Return a member's quantiles after the 14-day correction and the spread factor.

    Args:
        raw: The member's raw quantiles for all 365 daily runs (the correction of one run reads the
            forecasts of the runs before it).
        series: `common.series_table()`.
        arrays: Each site's arrays on the dense grid.
        window_days: Length of the correction window in days.

    Returns:
        The keys, `seed`, and the 11 quantile columns.
    """
    corrected = with_corrections(base=raw, series=series, arrays=arrays, window_days=window_days)
    return stack(frame=corrected, correct=True, spread=True)


def _synthetic_check() -> None:
    """Check the correction on a constant forecast error, and that it ignores later observations.

    A series whose median forecast is 0.50 and whose observed power is 0.60 everywhere has a
    correction of 0.10 once three days of forecasts exist, and 0.0 before. Replacing the observed
    power after `origin - 3.5 h` by large values leaves the correction of that origin unchanged.
    """
    runs = [common.GRID_START + timedelta(days=100 + d, hours=9) for d in range(20)]
    rows = []
    for run in runs:
        for step in range(1, 48 * 3 + 1):
            valid = run + step * common.STEP
            rows.append(
                {
                    "time_series_id": 1,
                    "power_fcst_init_time": run,
                    "valid_time": valid,
                    "seed": 0,
                    **dict.fromkeys(common.LEVEL_COLUMNS, 0.5),
                }
            )
    frame = pl.DataFrame(rows).cast({"seed": pl.Int8})
    power = np.full(common.grid_index(times=common.GRID_END) + 1, 0.6, dtype=np.float32)
    table = corrections_for(frame=frame, power=power, window_days=CORRECTION_DAYS)
    first, last = table["correction"][0], table["correction"][-1]
    assert first == 0.0, first
    assert abs(last - 0.1) < 1e-6, last
    probe = runs[10]
    cut = int(common.grid_index(times=probe)) - common.NEWEST_AGE
    future = power.copy()
    future[cut + 1 :] = 50.0
    again = corrections_for(frame=frame, power=future, window_days=CORRECTION_DAYS)
    here = table.filter(pl.col("power_fcst_init_time") == probe)["correction"][0]
    there = again.filter(pl.col("power_fcst_init_time") == probe)["correction"][0]
    assert here == there, (here, there)
    changed = future.copy()
    changed[cut - 5] = 50.0
    moved = corrections_for(frame=frame, power=changed, window_days=CORRECTION_DAYS)
    assert moved.filter(pl.col("power_fcst_init_time") == probe)["correction"][0] != here
    print(
        f"synthetic check passed: first correction {first:.3f}, last {last:.3f}; "
        "observations after origin - 3.5 h change nothing, one before it does"
    )


def main() -> None:
    """Run the self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    _synthetic_check()


if __name__ == "__main__":
    main()
