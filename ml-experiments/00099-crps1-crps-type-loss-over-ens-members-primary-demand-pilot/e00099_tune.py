"""Tune the constant Hessian of the custom objectives on a held-out stretch of the training period.

Nothing from the validation year (2025-07-01 onward) is read. Each arm is fitted at a refit on
2025-01-01 on the rows of runs whose date is a multiple of three days after 2024-04-01, as in the
main fits, and scored on the 6-member rows of the runs one day later, from 2025-01-02 to 2025-03-31,
which the fit has not seen. The held-out rows are pooled over their 6 members with the arm's masses.

    uv run python ml-experiments/00099-*/e00099_tune.py [--hessians 10 20 40]
"""

import argparse
import time
from datetime import UTC, datetime
from typing import Final

import e00099_common as common
import e00099_fit as fit
import e00099_objective as objective
import numpy as np
import polars as pl

REFIT: Final[datetime] = datetime(2025, 1, 1, tzinfo=UTC)
HELD_OUT_FIRST: Final[datetime] = datetime(2025, 1, 2, 9, tzinfo=UTC)
HELD_OUT_LAST: Final[datetime] = datetime(2025, 3, 31, 9, tzinfo=UTC)
SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D5")


def held_out_runs(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return the held-out runs' 6-member rows with derived columns and lags."""
    rows = (
        pl.scan_parquet(fit.MEMBERS_TRAIN_DIR / "*.parquet")
        .filter(
            (pl.col("time_series_id") == row["time_series_id"])
            & (pl.col("power_fcst_init_time") >= HELD_OUT_FIRST)
            & (pl.col("power_fcst_init_time") <= HELD_OUT_LAST)
        )
        .collect()
    )
    run_day = (
        pl.col("power_fcst_init_time") - pl.lit(common.origin_of(run=fit.FIRST_TRAIN_RUN))
    ).dt.total_days()
    rows = rows.filter(run_day % fit.RUN_STRIDE == 1)
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return fit.add_lags(frame=rows, power=arrays.power).sort(
        "power_fcst_init_time", "valid_time", "ensemble_member"
    )


def main() -> None:
    """Fit the arms at each Hessian and print the held-out quantile score and tail coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hessians", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    parser.add_argument("--arms", nargs="+", default=["p2", "p3", "c1"])
    args = parser.parse_args()
    series = common.primary_series()
    results: dict[tuple[str, float], list[tuple[float, float, float, float, float]]] = {}
    for site in SITES:
        row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
        arrays = common.load_series_arrays(series=series.filter(pl.col("site") == site))[site]
        known = fit.known_rows(rows=fit.training_rows(row=row, arrays=arrays), refit=REFIT)
        held = held_out_runs(row=row, arrays=arrays)
        index = np.asarray(common.grid_index(times=held["valid_time"]))
        actual = common.take(values=arrays.power, index=index)
        keep = np.isfinite(actual)
        held, actual = held.filter(pl.Series(keep)), actual[keep]
        matrix = held.select(fit.input_columns()).to_numpy().astype(np.float32)
        cases = held.height // common.MEMBERS_PER_RUN
        case_actual = actual.reshape(cases, common.MEMBERS_PER_RUN)[:, 0]
        for name in args.arms:
            for hessian in args.hessians if fit.ARMS[name].loss != "pinball" else [0.0]:
                fit.HESSIAN = hessian
                arm = fit.ARMS[name]
                started = time.monotonic()
                booster, start = fit.fit_arm(
                    arm=arm,
                    rows=fit.arm_rows(known=known, arm=arm),
                    row=row,
                    refit=REFIT,
                    rounds=fit.ROUNDS,
                    seed=0,
                )
                atoms = fit.predict_atoms(booster=booster, matrix=matrix, start=start)
                pooled = fit.mixture(
                    atoms=atoms.reshape(cases, common.MEMBERS_PER_RUN, objective.N_LEVELS),
                    masses=arm.masses,
                )
                score = common.pinball_quantile_score(quantiles=pooled, actual=case_actual) * 100
                below = (case_actual[:, None] <= pooled).mean(axis=0)
                results.setdefault((name, hessian), []).append(
                    (score.mean(), below[0], below[1], below[9], below[10])
                )
                print(
                    f"{site} {name} H={hessian:g}: score {score.mean():.3f}, "
                    f"coverage q01 {below[0]:.3f} q05 {below[1]:.3f} q95 {below[9]:.3f} "
                    f"q99 {below[10]:.3f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
    print("\nmean over the series")
    for (name, hessian), values in results.items():
        mean = np.mean(values, axis=0)
        print(
            f"{name} H={hessian:g}: score {mean[0]:.3f}, coverage q01 {mean[1]:.3f} "
            f"q05 {mean[2]:.3f} q95 {mean[3]:.3f} q99 {mean[4]:.3f}"
        )


if __name__ == "__main__":
    main()
