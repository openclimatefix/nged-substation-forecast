"""Rungs 1 to 3 for NGED battery A, with the public batteries' code paths.

- Rung 1: mean output by half-hour of day and within-day price third, and the price-lag alignment.
- Rung 2: the price-only held-out prediction (planned contrast B1) from `battery_rung2.out_of_fold`,
  with 95% intervals that resample whole calendar months.
- Rung 3: the implied state of charge from `battery_rung3.fit_efficiency`, the drift test, and the
  per-fold fits.

Output is a fraction of the series' own 99th percentile absolute output, so the errors in "% of
p99" match the public batteries' and no megawatt value appears. Saves tables with no series ID
under the study's data folder (`nged_battery_a_*.parquet`) and writes
`report_nged_battery_a_rungs.md`, which includes the comparison with the four public batteries.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/nged_battery_a_rungs.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR, TERCILE_LABELS
from battery_rung1 import alignment_table
from battery_rung2 import ARMS, CONTRASTS, month_resampled_interval, out_of_fold
from battery_rung3 import (
    BOUND_FRACTION,
    DAYS_IN_WINDOW,
    DRIFT_ETAS,
    ETA_MAX,
    ETA_MIN,
    SHORT_WINDOW_HALF_HOURS,
    cell_energy_path,
    fit_efficiency,
    smallest_capacity,
)
from studies.nged_battery_a import ALIAS, battery_a_frame

PERCENT: Final[float] = 100.0


def rung1(*, frame: pl.DataFrame) -> list[str]:
    """Return the rung 1 lines and save the heat map table.

    Args:
        frame: The window frame from `battery_a_frame`.

    Returns:
        Report lines.
    """
    frame.group_by("tod", "tercile").agg(
        mean_output=pl.col("output_mw").mean(), n=pl.len()
    ).write_parquet(OUTPUT_DIR / "nged_battery_a_tod_tercile_means.parquet")
    by_tercile = (
        frame.group_by("tercile")
        .agg(
            mean=pl.col("output_mw").mean(),
            charging_share=(pl.col("output_mw") < -0.05).mean(),
            discharging_share=(pl.col("output_mw") > 0.05).mean(),
        )
        .sort("tercile")
    )
    by_hour = frame.group_by("tod").agg(mean=pl.col("output_mw").mean()).sort("tod")
    peak = by_hour.sort("mean")
    return [
        "### Rung 1: output against price",
        "",
        f"- Rows: {frame.height}; mean output {frame['output_mw'].mean():.3f} of p99.",
        "- Mean output (fraction of p99) by within-day price third: "
        + "; ".join(
            f"{TERCILE_LABELS[row['tercile']]}: {row['mean']:.3f} "
            f"(charging {row['charging_share']:.0%}, discharging {row['discharging_share']:.0%} "
            "of half-hours, beyond 0.05 of p99)"
            for row in by_tercile.iter_rows(named=True)
        ),
        "- Correlation of output with the day-ahead price, price shifted by k half-hours "
        "(positive k = older price): "
        + ", ".join(f"k={k}: {c:.3f}" for k, c in alignment_table(frame=frame)),
        (
            f"- Half-hour of the UTC day with the lowest mean output: {peak['tod'][0] / 2:.1f} h "
            f"({peak['mean'][0]:.3f}); with the highest: {peak['tod'][-1] / 2:.1f} h "
            f"({peak['mean'][-1]:.3f})."
        ),
        "",
    ]


def rung2(*, frame: pl.DataFrame) -> tuple[list[str], dict[str, tuple[float, float, float]]]:
    """Return the rung 2 lines and the B1 contrast.

    Args:
        frame: The window frame from `battery_a_frame`.

    Returns:
        Report lines, and the planned contrast `price_rank` minus `time_of_day` as (mean, low, high)
        in percent of p99.
    """
    predicted = out_of_fold(frame=frame)
    predicted = predicted.with_columns(
        **{arm: pl.col(arm) for arm in ARMS},
    ).with_columns(
        **{f"error_{arm}": (pl.col("output_mw") - pl.col(arm)).abs() * PERCENT for arm in ARMS}
    )
    predicted = predicted.with_columns(
        **{
            f"difference_{treatment}_{reference}": pl.col(f"error_{treatment}")
            - pl.col(f"error_{reference}")
            for treatment, reference, _ in CONTRASTS
        }
    )
    monthly = (
        predicted.group_by("month")
        .agg(
            pl.len().alias("n"),
            *[
                pl.col(c).sum().alias(f"{c}_sum")
                for c in predicted.columns
                if c.startswith(("error_", "difference_"))
            ],
        )
        .sort("month")
    )
    lines = [
        "### Rung 2: price-only held-out prediction",
        "",
        (
            f"Four contiguous three-month folds over {monthly.height} calendar months, errors in "
            "% of p99, 95% intervals resampling whole calendar months (2,000 resamples)."
        ),
        "",
    ]
    sse_mean = float(((predicted["output_mw"] - predicted["output_mw"].mean()) ** 2).sum())
    for arm in ARMS:
        mean, low, high = month_resampled_interval(monthly=monthly, column=f"error_{arm}")
        sse = float(((predicted["output_mw"] - predicted[arm]) ** 2).sum())
        lines.append(
            f"- {arm}: {mean:.2f} [{low:.2f}, {high:.2f}] % of p99; "
            f"out-of-fold R2 (vs the year mean) {1 - sse / sse_mean:.3f}"
        )
    b1 = {}
    for treatment, reference, kind in CONTRASTS:
        column = f"difference_{treatment}_{reference}"
        mean, low, high = month_resampled_interval(monthly=monthly, column=column)
        better = int((monthly[f"{column}_sum"] < 0).sum())
        lines.append(
            f"- {treatment} minus {reference} ({kind}): {mean:.2f} [{low:.2f}, {high:.2f}] "
            f"% of p99; negative in {better} of {monthly.height} months"
        )
        b1[f"{treatment}_{reference}"] = (mean, low, high)
    lines.append("")
    return lines, b1


def rung3(*, frame: pl.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Return the rung 3 lines and the numbers the comparison table uses.

    Capacity is in units of p99-power-hours, so `capacity` is also "hours at p99 power".

    Args:
        frame: The window frame from `battery_a_frame`.

    Returns:
        Report lines, and the fitted efficiency, hours of energy, cycles per day, and time at a
        bound.
    """
    output = frame["output_mwh"].to_numpy()
    eta, capacity, _, _ = fit_efficiency(output_mwh=output)
    _, path = smallest_capacity(output_mwh=output, eta=eta)
    discharged = float(output[output > 0].sum())
    charged = float(-output[output < 0].sum())
    at_bound = float(
        ((path <= BOUND_FRACTION * capacity) | (path >= (1 - BOUND_FRACTION) * capacity)).mean()
    )
    cycles = discharged / eta / capacity / DAYS_IN_WINDOW
    drift = ", ".join(
        f"eta {e}: {smallest_capacity(output_mwh=output, eta=e)[0]:.1f} h" for e in DRIFT_ETAS
    )
    cell = cell_energy_path(output_mwh=output, eta=eta)
    n_weeks = len(cell) // SHORT_WINDOW_HALF_HOURS
    weekly = cell[: n_weeks * SHORT_WINDOW_HALF_HOURS].reshape(n_weeks, SHORT_WINDOW_HALF_HOURS)
    week_ranges = weekly.max(axis=1) - weekly.min(axis=1)
    fold_lines = []
    fold_cycles = []
    for fold in range(4):
        mask = (frame["fold"] == fold).to_numpy()
        fold_output = output[mask]
        fold_eta, fold_capacity, _, _ = fit_efficiency(output_mwh=fold_output)
        fold_days = mask.sum() / 48
        fold_cycles.append(
            float(fold_output[fold_output > 0].sum()) / fold_eta / fold_capacity / fold_days
        )
        fold_lines.append(
            f"fold {fold}: eta {fold_eta:.3f}, {fold_capacity:.1f} h, "
            f"{fold_cycles[-1]:.2f} cycles per day"
        )
    pl.DataFrame(
        {"time": frame["time"], "output_mw": frame["output_mw"], "soc_fraction": path / capacity}
    ).write_parquet(OUTPUT_DIR / "nged_battery_a_soc_path.parquet")
    lines = [
        "### Rung 3: implied state of charge",
        "",
        (
            f"One-way efficiency searched on [{ETA_MIN}, {ETA_MAX}]. Energy is in hours at the "
            "series' p99 power. The path sums only the rows present, so a missing half-hour adds "
            "nothing to the state of charge."
        ),
        "",
        (
            f"- Fitted one-way efficiency {eta:.3f} (round trip {eta**2:.3f}); at the search "
            f"bound: {eta <= ETA_MIN + 1e-9 or eta >= ETA_MAX - 1e-9}."
        ),
        f"- Fitted energy capacity: {capacity:.2f} hours at p99 power.",
        (
            "- Efficiency from the energy balance, sqrt(exported / imported): "
            f"{np.sqrt(discharged / charged):.3f}."
        ),
        f"- Fraction of the window at a bound: {at_bound:.1%}.",
        f"- Implied cycles per day (whole-year fit): {cycles:.2f}.",
        f"- Capacity if the efficiency were fixed (drift test): {drift}.",
        (
            f"- Post hoc: range of the fitted-eta path within each of {n_weeks} whole weeks "
            f"(hours at p99 power): median {np.median(week_ranges):.2f}, 95th percentile "
            f"{np.percentile(week_ranges, 95):.2f}, maximum {week_ranges.max():.2f}."
        ),
        "- Each fold fitted alone: " + "; ".join(fold_lines),
        "",
    ]
    return lines, {
        "eta": eta,
        "hours": capacity,
        "cycles": cycles,
        "at_bound": at_bound,
        "fold_cycles": float(np.median(fold_cycles)),
    }


def public_comparison(
    *, frame: pl.DataFrame, own: dict[str, float], b1: dict[str, tuple[float, float, float]]
) -> list[str]:
    """Return the table comparing NGED battery A with the four public batteries.

    Args:
        frame: The window frame from `battery_a_frame`.
        own: The fitted efficiency, hours of energy, cycles per day, and time at a bound.
        b1: The contrast `price_rank` minus `time_of_day` as (mean, low, high).

    Returns:
        Report lines.
    """
    paths = pl.read_parquet(OUTPUT_DIR / "rung3_soc_paths.parquet")
    intervals = pl.read_parquet(OUTPUT_DIR / "rung2_intervals.parquet")
    means = pl.read_parquet(OUTPUT_DIR / "rung1_tod_tercile_means.parquet")
    rows = []
    for bmu_id in BATTERIES:
        sub = paths.filter(pl.col("bmu_id") == bmu_id)
        p99 = float(np.quantile(sub["output_mw"].abs().to_numpy(), 0.99))
        eta = float(sub["eta"][0])
        capacity = float(sub["capacity_mwh"][0])
        output_mwh = sub["output_mw"].to_numpy() / 2
        cycles = float(output_mwh[output_mwh > 0].sum()) / eta / capacity / DAYS_IN_WINDOW
        grid = means.filter(pl.col("bmu_id") == bmu_id)
        by_tercile = (
            grid.group_by("tercile")
            .agg(
                total=(pl.col("mean_output_mw") * pl.col("n")).sum(),
                n=pl.col("n").sum(),
            )
            .with_columns(mean=pl.col("total") / pl.col("n") / p99)
            .sort("tercile")
        )
        b1_row = intervals.filter(
            (pl.col("bmu_id") == bmu_id)
            & (pl.col("quantity") == "price_rank minus time_of_day (B1, planned)")
        ).row(0, named=True)
        rows.append(
            (
                NAMES[bmu_id],
                eta,
                capacity / p99,
                cycles,
                by_tercile["mean"].to_list(),
                (b1_row["mean"], b1_row["lower_95"], b1_row["upper_95"]),
            )
        )
    own_terciles = (
        frame.group_by("tercile").agg(mean=pl.col("output_mw").mean()).sort("tercile")["mean"]
    )
    rows.append(
        (
            f"{ALIAS}, whole-year fit",
            own["eta"],
            own["hours"],
            own["cycles"],
            own_terciles.to_list(),
            b1["price_rank_time_of_day"],
        )
    )
    lines = [
        "### Comparison with the four public batteries",
        "",
        (
            "| Battery | One-way efficiency | Hours of energy at p99 power | Cycles per day "
            "| Mean output / p99, cheap, middle, dear third | B1: price rank minus time of day, "
            "% of p99 |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for name, eta, hours, cycles, terciles, (mean, low, high) in rows:
        tercile_text = (
            ", ".join(f"{v:.3f}" for v in terciles) if terciles is not None else "see rung 1"
        )
        lines.append(
            f"| {name} | {eta:.3f} | {hours:.2f} | {cycles:.2f} | {tercile_text} "
            f"| {mean:.2f} [{low:.2f}, {high:.2f}] |"
        )
    lines.append("")
    return lines


def main() -> None:
    """Run rungs 1 to 3 and write the report fragment."""
    frame, _ = battery_a_frame()
    lines = [f"## {ALIAS}: rungs 1 to 3", ""]
    lines += rung1(frame=frame)
    lines2, b1 = rung2(frame=frame)
    lines3, own = rung3(frame=frame)
    lines += lines2 + lines3 + public_comparison(frame=frame, own=own, b1=b1)
    (OUTPUT_DIR / "report_nged_battery_a_rungs.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
