"""Rung 5 summary: intervals, post hoc contrasts B5 and B6, and the report fragment.

Reads rung 4's `rung4_fits.parquet` (arms A0, A1, A3, and A4) and rung 5's `rung5_fits.parquet`
and writes `rung5_intervals.parquet` and `report_rung5.md`. Nothing is refitted. Intervals resample
whole aggregates, as in `battery_rung4_summary.py`, and have the same limits.

Run: `uv run python studies/battery_pv_separation/battery_rung5_summary.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR
from battery_rung4 import ARM_LABELS as RUNG4_ARM_LABELS
from battery_rung4_summary import (
    SOLAR_SHARES,
    _battery_table,
    _share_table,
    contrast_rows,
    level_rows,
)
from battery_rung5 import ARM_LABELS as RUNG5_ARM_LABELS
from battery_synthetic import AGGREGATE_P99_MW

SKIES: Final[tuple[str, ...]] = ("regional", "gb_mean")
MAIN_ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A5", "A6", "A3", "A4", "A5c", "A6c")
EXPLORATORY_ARMS: Final[tuple[str, ...]] = ("A5_1h", "A5_4h", "A6_1h", "A6_4h")
ALL_ARMS: Final[tuple[str, ...]] = (*MAIN_ARMS, *EXPLORATORY_ARMS)
ARM_LABELS: Final[dict[str, str]] = {**RUNG4_ARM_LABELS, **RUNG5_ARM_LABELS, "A0": "Solar only"}
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("A6", "A1", "post hoc B5"),
    ("A6", "A6c", "post hoc B6, price-specific part"),
    ("A5", "A1", "exploratory"),
    ("A5", "A5c", "exploratory"),
    ("A6", "A0", "exploratory"),
    ("A5", "A0", "exploratory"),
    ("A6", "A3", "exploratory"),
    ("A6c", "A0", "exploratory"),
    ("A5c", "A0", "exploratory"),
    ("A1", "A4", "exploratory"),
    ("A5_1h", "A5", "exploratory"),
    ("A5_4h", "A5", "exploratory"),
    ("A6_1h", "A6", "exploratory"),
    ("A6_4h", "A6", "exploratory"),
)
"""The treatment arm, the reference arm, and the kind of contrast."""
MAIN_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = CONTRASTS[:10]
FALSE_ALARM_ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A4", "A5", "A6", "A5c", "A6c")
SUMMER_WEEK_START: Final[str] = "2026-06-22"
SERIES_ARMS: Final[tuple[str, ...]] = ("A1", "A3", "A5", "A6")
ARM_TABLE: Final[tuple[str, ...]] = (
    "| Arm | Meaning | Regressor columns |",
    "|---|---|---|",
    (
        "| A5 | Rank-rule schedule, 2 h | the schedule in per-unit power: -1 in the 4 cheapest "
        "half-hours of each UTC day, +1 in the 4 dearest, idle when the dearest block is not after "
        "the cheapest or the spread is below the round-trip efficiency of 0.85 |"
    ),
    (
        "| A6 | Linear-programme schedule, 2 h | the schedule in per-unit power from a daily "
        "price-taker linear programme: one-way efficiency 0.92, state of charge 5% to 95%, at most "
        "1 cycle a day, day ends at its start state of charge |"
    ),
    "| A5c, A6c | Controls | A5 and A6 built from the day-ahead prices of 7 days earlier |",
    (
        "| A5_1h, A5_4h, A6_1h, A6_4h | Exploratory | A5 and A6 for 1-hour and 4-hour batteries "
        "(2 and 8 half-hours; energy 1 and 4 hours at full power) |"
    ),
    (
        "| A0, A1, A3, A4 | From rung 4 | solar only; the two price columns; the true battery; "
        "the two price columns from 7 days earlier |"
    ),
)


def false_alarm_lines(*, fits: pl.DataFrame) -> list[str]:
    """Return the report lines on detection against the no-solar threshold (post hoc contrast B6).

    Args:
        fits: Rung 4's and rung 5's fits together.

    Returns:
        Markdown lines.
    """
    lines = [
        "### Post hoc contrast B6: false alarms among the no-solar aggregates",
        "",
        (
            "The statistic is the share of the variance of the four-hour changes, after the "
            "fitted regressor's contribution is removed, that the fitted plant explains. A0's "
            "threshold is the largest value of that statistic among A0's 16 no-solar aggregates "
            "(battery only, solar share 0%) of the sky, so A0 has 0 false alarms by "
            "construction. A false alarm is a no-solar aggregate of another arm whose value is "
            "strictly above A0's threshold. Rung 4's A1 and A4 are shown for comparison."
        ),
        "",
    ]
    for sky in SKIES:
        frame = fits.filter(pl.col("sky") == sky)
        zero = frame.filter(pl.col("share") == 0)
        a0_threshold = float(zero.filter(pl.col("arm") == "A0")["variance_explained"].max())  # ty: ignore[invalid-argument-type]
        lines += [
            f"#### Sky: {sky}",
            "",
            f"A0's threshold: {a0_threshold:.4f}.",
            "",
            (
                "| Arm | Max no-solar variance explained | No-solar above A0's threshold | "
                "With solar above A0's threshold: 10% | 25% | 50% |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for arm in (*FALSE_ALARM_ARMS, *EXPLORATORY_ARMS):
            arm_zero = zero.filter(pl.col("arm") == arm)
            cells = []
            for share in SOLAR_SHARES:
                sub = frame.filter((pl.col("arm") == arm) & (pl.col("share") == share))
                cells.append(
                    f"{int((sub['variance_explained'] > a0_threshold).sum())} of {sub.height}"
                )
            alarms = int((arm_zero["variance_explained"] > a0_threshold).sum())
            lines.append(
                f"| {arm} | {float(arm_zero['variance_explained'].max()):.4f} "  # ty: ignore[invalid-argument-type]
                f"| {alarms} of {arm_zero.height} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    return lines


def coefficient_lines(*, fits: pl.DataFrame) -> list[str]:
    """Return the report lines on the fitted schedule coefficient.

    Args:
        fits: Rung 4's and rung 5's fits together.

    Returns:
        Markdown lines.
    """
    lines = [
        "### Fitted schedule coefficient",
        "",
        (
            "The coefficient is the battery's fitted power in megawatts per unit of schedule. "
            "The ratio divides it by the battery's true 99th-percentile output in the aggregate "
            "(100 MW times the battery's share), so 1 means the fit found the battery's size. "
            "Regional sky; aggregates with solar (shares 10%, 25%, and 50%) and without."
        ),
        "",
        "| Arm | Mean ratio with solar | Range with solar | Mean ratio no solar | Range no solar |",
        "|---|---|---|---|---|",
    ]
    regional = fits.filter(pl.col("sky") == "regional").with_columns(
        ratio=pl.col("coefficient_1") / ((1.0 - pl.col("share")) * AGGREGATE_P99_MW)
    )
    for arm in ("A5", "A6", "A5c", "A6c", *EXPLORATORY_ARMS):
        solar = regional.filter((pl.col("arm") == arm) & (pl.col("share") > 0))["ratio"]
        none = regional.filter((pl.col("arm") == arm) & (pl.col("share") == 0))["ratio"]
        lines.append(
            f"| {arm} | {solar.mean():.2f} | {solar.min():.2f} to {solar.max():.2f} "
            f"| {none.mean():.2f} | {none.min():.2f} to {none.max():.2f} |"
        )
    lines.append("")
    return lines


def contribution_lines(*, series: pl.DataFrame) -> list[str]:
    """Return the report lines on how well each arm's fitted battery follows the true battery.

    Args:
        series: Rung 4's and rung 5's fitted series of the drawn aggregate.

    Returns:
        Markdown lines.
    """
    lines = [
        "### Fitted battery contribution against the true battery",
        "",
        (
            "Pearson correlation and mean absolute error (% of the battery's 99th-percentile "
            "output) between the arm's fitted battery contribution (coefficient times schedule) "
            "and the true battery, for the aggregate of Burwell solar (25%) and the Lakeside "
            "battery (75%), regional sky."
        ),
        "",
        (
            "| Arm | Correlation, year | Correlation, summer week | Error, year (%) | "
            "Error, summer week (%) |"
        ),
        "|---|---|---|---|---|",
    ]
    week = pl.col("time") >= pl.lit(SUMMER_WEEK_START).str.to_datetime()
    week &= pl.col("time") < pl.lit(SUMMER_WEEK_START).str.to_datetime() + pl.duration(days=7)
    for arm in SERIES_ARMS:
        sub = series.filter(pl.col("arm") == arm)
        cells = []
        for part in (sub, sub.filter(week)):
            truth = part["battery_truth_mw"].to_numpy()
            fitted = part["regressor_contribution_mw"].to_numpy()
            finite = np.isfinite(truth)
            cells.append(float(np.corrcoef(truth[finite], fitted[finite])[0, 1]))
        errors = []
        scale = float(np.nanquantile(np.abs(sub["battery_truth_mw"].to_numpy()), 0.99))
        for part in (sub, sub.filter(week)):
            truth = part["battery_truth_mw"].to_numpy()
            fitted = part["regressor_contribution_mw"].to_numpy()
            errors.append(float(np.nanmean(np.abs(truth - fitted)) / scale * 100.0))
        lines.append(
            f"| {arm} | {cells[0]:.2f} | {cells[1]:.2f} | {errors[0]:.1f} | {errors[1]:.1f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    """Write the intervals and the report fragment."""
    fits = pl.concat(
        [
            pl.read_parquet(OUTPUT_DIR / "rung4_fits.parquet").filter(
                pl.col("arm").is_in(["A0", "A1", "A3", "A4"])
            ),
            pl.read_parquet(OUTPUT_DIR / "rung5_fits.parquet"),
        ],
        how="diagonal_relaxed",
    )
    series = pl.concat(
        [
            pl.read_parquet(OUTPUT_DIR / "rung4_series.parquet").filter(
                pl.col("arm").is_in(["A1", "A3"])
            ),
            pl.read_parquet(OUTPUT_DIR / "rung5_series.parquet"),
        ],
        how="diagonal_relaxed",
    )
    levels = pl.DataFrame(level_rows(fits=fits, arms=ALL_ARMS))
    contrasts = pl.DataFrame(contrast_rows(fits=fits, arms=ALL_ARMS, contrasts=CONTRASTS))
    pl.concat([levels, contrasts]).write_parquet(OUTPUT_DIR / "rung5_intervals.parquet")
    expected = len(BATTERIES) * 4 * 4 * len(SKIES) * len(ALL_ARMS)
    arm_rows = [(f"{arm}: {ARM_LABELS[arm]}", arm) for arm in ALL_ARMS]
    contrast_labels = [
        (f"{treatment} minus {reference} ({kind})", f"{treatment} minus {reference}")
        for treatment, reference, kind in CONTRASTS
    ]
    lines = [
        "## Rung 5: does one structured battery schedule beat unconstrained price regressors?",
        "",
        (
            "The aggregates, skies, fit, and error measure are rung 4's. Each new arm adds one "
            "regressor column, a battery schedule in per-unit power built from the day-ahead "
            "price, with one free signed coefficient. Rung 4's A0, A1, A3, and A4 are read from "
            "rung 4's fits. The contrasts B5 and B6 were written before this rung ran, but after "
            "rung 4's results and outside the plan, so they are post hoc: B5 is A6 minus "
            "A1 in solar series error (% of solar p99, 16 aggregates, regional sky), and B6 is "
            "the count of no-solar aggregates above A0's threshold for A5 and A6, and A6 minus "
            "A6c (the part of A6's gain that is specific to the right week's prices). Every "
            "other number is exploratory. Negative differences mean the first arm recovers "
            "solar better."
        ),
        "",
        "### Arms and their regressor columns",
        "",
        *ARM_TABLE,
        "",
        f"Fits expected: {expected}; fits present: {fits.height}.",
        "",
        "### Solar series error by arm and share (% of solar p99; pooled over 16 aggregates)",
        "",
    ]
    for sky in SKIES:
        lines += [f"#### Sky: {sky}", ""]
        lines += _share_table(
            table=levels, sky=sky, metric="nmae_of_solar_p99", rows=arm_rows, digits=1
        )
    lines += ["### Contrasts (points of solar p99)", ""]
    for sky in SKIES:
        lines += [f"#### Sky: {sky}, pooled over 16 aggregates", ""]
        lines += _share_table(
            table=contrasts, sky=sky, metric="nmae_of_solar_p99", rows=contrast_labels, digits=1
        )
    differences = [f"{t} minus {r}" for t, r, _ in MAIN_CONTRASTS]
    lines += [
        "### Contrasts for each battery (regional sky, points of solar p99, all three shares)",
        "",
        "Intervals resample the 4 solar sets.",
        "",
    ]
    for chunk in (differences[:5], differences[5:]):
        lines += _battery_table(table=contrasts, quantities=chunk, metric="nmae_of_solar_p99")
    lines += [
        "### Solar error for each battery (regional sky, % of solar p99, all three shares)",
        "",
        "Intervals resample the 4 solar sets.",
        "",
    ]
    for chunk in (list(MAIN_ARMS[:4]), list(MAIN_ARMS[4:]), list(EXPLORATORY_ARMS)):
        lines += _battery_table(table=levels, quantities=chunk, metric="nmae_of_solar_p99")
    lines += [
        "### Energy ratio, AC capacity ratio, and correlation (pooled over 16 aggregates)",
        "",
        (
            "Energy ratio is the recovered solar energy over the true solar energy. The AC "
            "capacity ratio is the fitted AC capacity over the direct fit to the solar half, "
            "scaled to the aggregate's solar half. Both are 1 for a perfect recovery."
        ),
        "",
    ]
    for sky in SKIES:
        for metric in ("energy_ratio", "ac_ratio", "correlation"):
            lines += [f"#### Sky: {sky}, {metric}", ""]
            lines += _share_table(
                table=levels,
                sky=sky,
                metric=metric,
                rows=[(f"{arm}", arm) for arm in ALL_ARMS],
                digits=2,
            )
    lines += false_alarm_lines(fits=fits)
    lines += coefficient_lines(fits=fits)
    lines += contribution_lines(series=series)
    lines += [f"Batteries: {', '.join(f'{NAMES[b]} ({b})' for b in BATTERIES)}.", ""]
    text = "\n".join(lines)
    (OUTPUT_DIR / "report_rung5.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
