"""Compare the arms on the cases where observed power is in its top decile (exploratory).

A case is in the top decile when its observed power over effective capacity exceeds the 90th
percentile of the same series' cleaned power from 2019-09 to before 2025-06-17 (the stage 1a
training window). For each technology and lead band the script reports each arm's mean absolute
error on those cases, the slope of observed on predicted power on those cases, and the paired
differences of the two ERA5-trained arms against the ENS-trained arm that has the same averaging
(`s1_memberavg` minus `ens_trained`, `s1_control` minus `ens_trained_control`). Intervals resample
whole validation months with replacement (2,000 resamples, stream seed `20261013`), the same draw
for every arm; months with no top-decile case drop out of a draw. A mean is the mean over series of
each series' mean, over the series with at least one case in the draw. Output:
`data/studies/per_study/stage1_pilot/results/tail.md` and `tail.parquet`.

Selecting on the observed value truncates the outcome, which pulls a slope below 1 for every arm, so
read each slope against `era5_ceiling` and use the paired differences, not the absolute slope.

    uv run python studies/stage1_pilot/stage1_tail.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
import stage1_stats as stats
from stage1_fit_stage1a import observed_training_power

SCORE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "scores"
RESULT_DIR: Final[Path] = inputs.RESULTS_DIR
BANDS: Final[tuple[str, ...]] = ("all", *(name for name, _, _ in inputs.LEAD_BANDS))
ARMS: Final[tuple[str, ...]] = (
    "c0",
    "climatology",
    "ens_trained_control",
    "ens_trained",
    "s1_control",
    "s1_memberavg",
    "era5_ceiling",
)
PAIRS: Final[dict[str, tuple[str, str]]] = {
    "s1_memberavg minus ens_trained": ("s1_memberavg", "ens_trained"),
    "s1_control minus ens_trained_control": ("s1_control", "ens_trained_control"),
}


def top_decile_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return every scored row of every arm with its band, technology, and top-decile flag."""
    threshold = (
        observed_training_power(sites=sites)
        .group_by("time_series_id")
        .agg(threshold=pl.col("target").quantile(0.9))
    )
    rows = pl.concat([pl.read_parquet(SCORE_DIR / f"rows_seed{s}.parquet") for s in inputs.SEEDS])
    band = pl.lit("none")
    for name, low, high in reversed(inputs.LEAD_BANDS):
        band = (
            pl.when((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    return (
        rows.filter(pl.col("arm").is_in(ARMS))
        .join(sites.select("time_series_id", "technology"), on="time_series_id")
        .join(threshold, on="time_series_id")
        .with_columns(band=band, top=pl.col("actual_norm") > pl.col("threshold"))
    )


def month_arrays(
    *, part: pl.DataFrame, series: list[int], months: list[str]
) -> dict[str, np.ndarray]:
    """Return per-seed, per-series, per-month sums of error, x, y, xy, xx, and the case count."""
    seeds = sorted(part["seed"].unique().to_list())
    shape = (len(seeds), len(series), len(months))
    out = {name: np.zeros(shape) for name in ("err", "x", "y", "xy", "xx", "n")}
    grouped = (
        part.with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("seed", "time_series_id", "month")
        .agg(
            err=pl.col("abs_error").sum(),
            x=pl.col("point_norm").sum(),
            y=pl.col("actual_norm").sum(),
            xy=(pl.col("point_norm") * pl.col("actual_norm")).sum(),
            xx=(pl.col("point_norm") ** 2).sum(),
            n=pl.len(),
        )
    )
    for row in grouped.iter_rows(named=True):
        index = (
            seeds.index(row["seed"]),
            series.index(row["time_series_id"]),
            months.index(row["month"]),
        )
        for name, values in out.items():
            values[index] = row[name]
    return out


def resample_mean_error(*, table: dict[str, np.ndarray], draws: np.ndarray) -> np.ndarray:
    """Return the mean over series of each series' mean error, per draw (mean over seeds)."""
    err = table["err"].sum(axis=0)[:, draws].sum(axis=1)
    n = table["n"].sum(axis=0)[:, draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_series = np.where(n > 0, err / n, np.nan)
    return np.nanmean(per_series, axis=0)


def resample_slope(*, table: dict[str, np.ndarray], draws: np.ndarray) -> np.ndarray:
    """Return the pooled OLS slope of y on x per draw, averaged over seeds."""
    sums = {k: table[k].sum(axis=1)[:, draws].sum(axis=1) for k in ("x", "y", "xy", "xx", "n")}
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (sums["n"] * sums["xy"] - sums["x"] * sums["y"]) / (
            sums["n"] * sums["xx"] - sums["x"] ** 2
        )
    return np.nanmean(slope, axis=0)


def summarise(*, point: float, values: np.ndarray) -> tuple[float, float, float]:
    """Return the full-sample value with the 2.5th and 97.5th percentile of the resampled values."""
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(point), float(low), float(high)


def main() -> None:
    """Compute the top-decile tables and write the report."""
    sites = inputs.site_table()
    rows = top_decile_rows(sites=sites)
    rng = np.random.default_rng(stats.BOOTSTRAP_SEED)
    out_rows = []
    report = [
        "# Top-decile results (exploratory)",
        "",
        (
            "Cases whose observed power exceeds the series' 90th percentile of cleaned power in "
            "the training window. Points of capacity; 95% intervals resample whole validation "
            "months. "
            "Selecting on the observed value truncates the outcome, so absolute slopes sit below 1 "
            "for every arm; compare them with `era5_ceiling`."
        ),
        "",
    ]
    for technology in ("solar", "wind"):
        scoped = rows.filter((pl.col("technology") == technology) & pl.col("top"))
        series = sorted(scoped["time_series_id"].unique().to_list())
        months = sorted(scoped["valid_time"].dt.strftime("%Y-%m").unique().to_list())
        draws = rng.integers(0, len(months), size=(stats.N_RESAMPLES, len(months)))
        everything = np.arange(len(months))
        report += [f"## {technology}", ""]
        for band in BANDS:
            part = scoped if band == "all" else scoped.filter(pl.col("band") == band)
            tables = {
                arm: month_arrays(
                    part=part.filter(pl.col("arm") == arm), series=series, months=months
                )
                for arm in ARMS
            }
            counts = int(tables["ens_trained"]["n"].sum() // len(inputs.SEEDS))
            report += [f"### {technology}, {band} ({counts:,} top-decile cases)", ""]
            lines = ["| arm | MAE | slope |", "| --- | --- | --- |"]
            for arm in ARMS:
                mae = summarise(
                    point=resample_mean_error(table=tables[arm], draws=everything).item(),
                    values=np.array(
                        [resample_mean_error(table=tables[arm], draws=d) for d in draws]
                    ),
                )
                slope = summarise(
                    point=resample_slope(table=tables[arm], draws=everything).item(),
                    values=np.array([resample_slope(table=tables[arm], draws=d) for d in draws]),
                )
                lines.append(
                    f"| {arm} | {mae[0]:.2f} ({mae[1]:.2f} to {mae[2]:.2f}) | "
                    f"{slope[0]:.3f} ({slope[1]:.3f} to {slope[2]:.3f}) |"
                )
                out_rows.append(
                    {
                        "technology": technology,
                        "band": band,
                        "arm": arm,
                        "mae": mae[0],
                        "mae_lower": mae[1],
                        "mae_upper": mae[2],
                        "slope": slope[0],
                        "slope_lower": slope[1],
                        "slope_upper": slope[2],
                        "cases": counts,
                    }
                )
            report += [*lines, ""]
            lines = [
                "| paired difference | MAE difference | months with a lower error |",
                "| --- | --- | --- |",
            ]
            for name, (treatment, reference) in PAIRS.items():
                diff = np.array(
                    [
                        resample_mean_error(table=tables[treatment], draws=d)
                        - resample_mean_error(table=tables[reference], draws=d)
                        for d in draws
                    ]
                )
                centre, low, high = summarise(
                    point=(
                        resample_mean_error(table=tables[treatment], draws=everything)
                        - resample_mean_error(table=tables[reference], draws=everything)
                    ).item(),
                    values=diff,
                )
                per_month = tables[treatment]["err"].sum(axis=(0, 1)) - tables[reference][
                    "err"
                ].sum(axis=(0, 1))
                present = tables[treatment]["n"].sum(axis=(0, 1)) > 0
                lines.append(
                    f"| {name} | {centre:+.2f} ({low:+.2f} to {high:+.2f}) | "
                    f"{int((per_month[present] < 0).sum())} of {int(present.sum())} |"
                )
                out_rows.append(
                    {
                        "technology": technology,
                        "band": band,
                        "arm": name,
                        "mae": centre,
                        "mae_lower": low,
                        "mae_upper": high,
                        "slope": float("nan"),
                        "slope_lower": float("nan"),
                        "slope_upper": float("nan"),
                        "cases": counts,
                    }
                )
            report += [*lines, ""]
    pl.DataFrame(out_rows).write_parquet(RESULT_DIR / "tail.parquet")
    (RESULT_DIR / "tail.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
