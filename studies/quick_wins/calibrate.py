"""Idea B1 (post hoc): rescale each forecast's ensemble spread using past outcomes.

For every sampled run, each series' members are moved towards or away from their ensemble mean by
a factor k, `member' = mean + k * (member - mean)`, so the ensemble mean (and so the MAE) is
unchanged. The factor comes from a grid and is the one with the lowest mean CRPS for that series on
earlier runs, using only outcomes whose valid time is at or before the current run's initialisation
time, so nothing from the scored run or later is used. Factors below 1 are allowed. The first
`WARM_UP_RUNS` runs have little history, so they are left out of both the calibrated and the
uncalibrated scores that the contrast compares.

    uv run python studies/quick_wins/calibrate.py --arms production_cleaned [--fold ...] [--seed 0]
"""

import argparse
from datetime import datetime
from typing import Final

import arms as arm_definitions
import fit_arms
import month_bootstrap
import polars as pl
import quick_wins_inputs as inputs
from contracts.settings import Settings
from per_run_scores import normalised_by_capacity, per_run_scores
from report_tables import markdown_table
from studies.power import scan_power

SPREAD_FACTORS: Final[tuple[float, ...]] = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
"""The grid of factors a series' spread is multiplied by."""

WARM_UP_RUNS: Final[int] = 8
"""Weekly runs (about two months) left out while history builds up."""

MIN_HISTORY_ROWS: Final[int] = 200
"""A series with fewer past scored rows than this keeps its spread (factor 1)."""

NOMINAL_COVERAGE: Final[float] = 0.8
"""Share of outcomes an ensemble's 10% to 90% band should hold."""


def _scaled(*, forecasts: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Move every member away from its ensemble mean by `factor`."""
    mean = pl.col("power_fcst").mean().over("time_series_id", "valid_time")
    return forecasts.with_columns(power_fcst=mean + factor * (pl.col("power_fcst") - mean))


def _scored_with_band(*, forecasts: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    """Score a run and add whether the outcome fell inside the members' 10% to 90% band."""
    scores = per_run_scores(forecasts=forecasts, actuals=actuals)
    band = forecasts.group_by("time_series_id", "valid_time").agg(
        low=pl.col("power_fcst").quantile(0.1), high=pl.col("power_fcst").quantile(0.9)
    )
    return scores.join(band, on=["time_series_id", "valid_time"]).with_columns(
        inside=(pl.col("actual") >= pl.col("low")) & (pl.col("actual") <= pl.col("high"))
    )


def _of_type(*, scores: pl.DataFrame, types: pl.DataFrame, series_type: str | None) -> pl.DataFrame:
    """Keep the rows of one series type, or all rows when `series_type` is None."""
    rows = scores.join(types, on="time_series_id")
    return rows if series_type is None else rows.filter(pl.col("time_series_type") == series_type)


def calibrate_arm(
    *, forecast_dir_files: list, actuals: pl.DataFrame, capacity: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the raw scores, the calibrated scores, and the chosen factors for runs after warm-up.

    Args:
        forecast_dir_files: The run files of one arm, in initialisation order.
        actuals: Observed power.
        capacity: Effective capacity per series.

    Returns:
        The raw scores, the calibrated scores, and each run's chosen factor per series.
    """
    per_factor: list[dict[float, pl.DataFrame]] = []
    init_times: list[datetime] = []
    for path in forecast_dir_files:
        forecasts = pl.read_parquet(path)
        init_times.append(forecasts["power_fcst_init_time"][0])
        per_factor.append(
            {
                k: _scored_with_band(
                    forecasts=_scaled(forecasts=forecasts, factor=k), actuals=actuals
                )
                for k in SPREAD_FACTORS
            }
        )
    raw_parts, calibrated_parts, chosen_parts = [], [], []
    for index in range(WARM_UP_RUNS, len(init_times)):
        cutoff = init_times[index]
        history = pl.concat(
            [
                table.filter(pl.col("valid_time") <= cutoff).select(
                    "time_series_id", "crps", pl.lit(k).alias("factor")
                )
                for earlier in per_factor[:index]
                for k, table in earlier.items()
            ]
        )
        by_series = history.group_by("time_series_id", "factor").agg(
            mean_crps=pl.col("crps").mean(), rows=pl.len()
        )
        best = (
            by_series.filter(pl.col("rows") >= MIN_HISTORY_ROWS)
            .sort("mean_crps")
            .group_by("time_series_id", maintain_order=True)
            .first()
            .select("time_series_id", "factor")
        )
        series = per_factor[index][1.0]["time_series_id"].unique()
        chosen = (
            pl.DataFrame({"time_series_id": series})
            .join(best, on="time_series_id", how="left")
            .with_columns(pl.col("factor").fill_null(1.0))
        )
        calibrated = pl.concat(
            [
                per_factor[index][k].join(
                    chosen.filter(pl.col("factor") == k).select("time_series_id"),
                    on="time_series_id",
                )
                for k in SPREAD_FACTORS
            ]
        )
        raw_parts.append(per_factor[index][1.0])
        calibrated_parts.append(calibrated)
        chosen_parts.append(chosen.with_columns(power_fcst_init_time=pl.lit(cutoff)))
    return (
        normalised_by_capacity(scores=pl.concat(raw_parts), capacity=capacity),
        normalised_by_capacity(scores=pl.concat(calibrated_parts), capacity=capacity),
        pl.concat(chosen_parts),
    )


def main() -> None:
    """Calibrate the named arms and write the contrast of calibrated against uncalibrated CRPS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = Settings()
    window = inputs.window_for(fold_id=args.fold)
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    types = pl.DataFrame(
        inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    ).select("time_series_id", pl.col("time_series_type").cast(pl.String))
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    actuals = (
        scan_power()
        .filter(pl.col("time") >= window.val_start, pl.col("time") <= window.val_end)
        .collect()
    )
    out_root = inputs.OUTPUT_ROOT / "calibration" / args.fold
    out_root.mkdir(parents=True, exist_ok=True)
    lines = ["# Spread calibration (B1, post hoc)", ""]
    rows = []
    for name in args.arms:
        arm = arm_definitions.ALL_ARMS[name]
        files = sorted(
            (fit_arms.arm_dir(fold_id=args.fold, arm=arm, seed=args.seed) / "forecasts").glob(
                "init=*.parquet"
            )
        )
        raw, calibrated, chosen = calibrate_arm(
            forecast_dir_files=files, actuals=actuals, capacity=capacity
        )
        target = out_root / f"{name}_seed{args.seed}"
        target.mkdir(exist_ok=True)
        raw.write_parquet(target / "raw.parquet")
        calibrated.write_parquet(target / "calibrated.parquet")
        chosen.write_parquet(target / "factors.parquet")
        groups = [
            ("all types", None),
            *[(t, t) for t in sorted(types["time_series_type"].unique())],
        ]
        for label, series_type in groups:
            raw_rows = _of_type(scores=raw, types=types, series_type=series_type)
            calibrated_rows = _of_type(scores=calibrated, types=types, series_type=series_type)
            raw_table = month_bootstrap.monthly_table(scores=raw_rows, metric="crps_norm")
            calibrated_table = month_bootstrap.monthly_table(
                scores=calibrated_rows, metric="crps_norm"
            )
            diff = month_bootstrap.difference_interval(arm=calibrated_table, reference=raw_table)
            rows.append(
                {
                    "arm": name,
                    "group": label,
                    "raw_crps": month_bootstrap.level_interval(table=raw_table).estimate,
                    "calibrated_crps": month_bootstrap.level_interval(
                        table=calibrated_table
                    ).estimate,
                    "difference": diff.estimate,
                    "lower_95": diff.lower_95,
                    "upper_95": diff.upper_95,
                    "months_better": diff.months_better,
                    "n_months": diff.n_months,
                    "raw_band_coverage": raw_rows["inside"].mean(),
                    "calibrated_band_coverage": calibrated_rows["inside"].mean(),
                }
            )
    result = pl.DataFrame(rows)
    result.write_parquet(out_root / f"calibration_seed{args.seed}.parquet")
    lines += [
        (
            f"Nominal 10% to 90% band coverage is {NOMINAL_COVERAGE:.0%}. Runs after the first "
            f"{WARM_UP_RUNS} are scored. Differences are calibrated minus uncalibrated "
            "normalised CRPS."
        ),
        "",
        markdown_table(frame=result),
    ]
    (out_root / f"report_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
