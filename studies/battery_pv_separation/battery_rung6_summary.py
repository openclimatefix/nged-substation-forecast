"""Rung 6 summary: intervals, post hoc contrasts B7 and B8, false alarms, and the report fragment.

Reads `rung6_fits.parquet` and rung 4's `rung4_fits.parquet` (for A0, A1, and A3) and writes
`rung6_intervals.parquet` and `report_rung6.md`. Nothing is refitted. Intervals resample whole
aggregates, as in `battery_rung4_summary`: the 16 aggregates (4 batteries by 4 solar sets) for a
pooled interval, or the 4 solar sets for one battery's interval.

Run: `uv run python studies/battery_pv_separation/battery_rung6_summary.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR
from battery_rung4 import SKIES
from battery_rung4_summary import SOLAR_SHARES, _battery_table, _fmt, _share_table
from battery_rung4_summary import contrast_rows as _contrast_rows
from battery_rung4_summary import level_rows as _level_rows
from battery_rung6 import (
    ARM_LABELS,
    ARMS,
    FALSE_ALARM_SHARE,
    SMOOTHNESS_ARMS,
    SMOOTHNESS_SKY,
)
from battery_synthetic import AGGREGATE_P99_MW, SOLAR_SET_LABELS

RUNG4_ARMS: Final[tuple[str, ...]] = ("A0", "A1", "A3")
REPORT_ARMS: Final[tuple[str, ...]] = (*RUNG4_ARMS, *ARMS)
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("A8", "A0", "post hoc B7"),
    ("A9", "A0", "post hoc B8"),
    ("A8", "A1", "exploratory"),
    ("A8", "A3", "exploratory"),
    ("A8h", "A8", "exploratory"),
    ("A8d", "A8", "exploratory"),
    ("A9", "A8", "exploratory"),
    ("A0b", "A0", "exploratory"),
    ("A8", "A0b", "exploratory"),
    ("A9", "A0b", "exploratory"),
    ("A3b", "A3", "exploratory"),
    ("A8", "A3b", "exploratory"),
    ("A1b", "A0b", "exploratory"),
    ("A1b", "A1", "exploratory"),
    ("A8", "A1b", "exploratory"),
)
"""The treatment arm, the reference arm, and the kind of contrast."""
BATTERY_ARMS: Final[tuple[str, ...]] = ("A8", "A8h", "A8d", "A9")
BATTERY_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("battery_mae_pct_of_p99", "battery output error (% of its p99)"),
    ("battery_correlation", "battery output correlation"),
    ("soc_correlation", "state-of-charge correlation (mean over windows)"),
)
ARM_TABLE: Final[tuple[str, ...]] = (
    "| Arm | Meaning | Battery limits |",
    "|---|---|---|",
    "| A0 | Rung 4's solar-only physical fit | none |",
    "| A1 | Rung 4's price level and rank regressors | none |",
    "| A3 | Rung 4's oracle: the true battery as a regressor | none |",
    (
        "| A8 | Joint model; power is the true battery's 99th-percentile output, energy is rung "
        "3's fitted capacity (both scaled to the aggregate) | P, E |"
    ),
    "| A8h | A8 with half the energy capacity | P, E/2 |",
    "| A8d | A8 with double the energy capacity | P, 2E |",
    "| A9 | Joint model with a 2-hour battery at the true power | P, 2 h times P |",
    "| A0b | The four fleet curves fitted to the aggregate with no battery | 0, 0 |",
    (
        "| A1b | The four fleet curves fitted to the aggregate with no battery, plus the two price "
        "regressors of A1 as free signed regressors | 0, 0 |"
    ),
    "| A3b | The four fleet curves fitted to the aggregate minus the true battery | 0, 0 |",
)


def _combined(*, rung4: pl.DataFrame, rung6: pl.DataFrame) -> pl.DataFrame:
    """Return the fits of rung 4's A0, A1, and A3 and of rung 6's arms in one frame.

    Args:
        rung4: `rung4_fits.parquet`.
        rung6: `rung6_fits.parquet`.

    Returns:
        One row per aggregate, sky, and arm, without the smoothness-penalty arms.
    """
    return pl.concat(
        [
            rung4.filter(pl.col("arm").is_in(RUNG4_ARMS)),
            rung6.filter(pl.col("arm").is_in(ARMS)),
        ],
        how="diagonal",
    )


FALSE_ALARM_ARMS: Final[tuple[str, ...]] = (
    "A0",
    "A1",
    "A0b",
    "A1b",
    "A8",
    "A8h",
    "A8d",
    "A9",
    "A3b",
)
"""The arms the false-alarm tables list. `A0` and `A1` are rung 4's, scored by rule 1 only."""
RULE_2_ARMS: Final[tuple[str, ...]] = tuple(a for a in FALSE_ALARM_ARMS if a not in ("A0", "A1"))
STRICT_FALSE_ALARM_SHARE: Final[float] = 0.01
"""The stricter capacity threshold of rule 2, as a share of the aggregate's 99th percentile."""


def _above(*, values: pl.Series, limit: float) -> int:
    """Count the values strictly above a limit; NaN is not above it."""
    array = values.to_numpy()
    return int((array > limit).sum())


def _counts_by_share(*, frame: pl.DataFrame, arm: str, column: str, limit: float) -> list[str]:
    """Return "k of n" for each solar share: how many of its aggregates have `column` above `limit`.

    Args:
        frame: The combined fits of one sky.
        arm: The arm.
        column: The fitted column compared with the limit.
        limit: The limit.

    Returns:
        One string for each of `SOLAR_SHARES`.
    """
    cells = []
    for share in SOLAR_SHARES:
        sub = frame.filter((pl.col("arm") == arm) & (pl.col("share") == share))
        cells.append(f"{_above(values=sub[column], limit=limit)} of {sub.height}")
    return cells


def _rule_table(
    *, frame: pl.DataFrame, arms: tuple[str, ...], column: str, limit: float, unit: str, digits: int
) -> list[str]:
    """Return one false-alarm table: the statistic at 0% and the counts above the limit.

    Args:
        frame: The combined fits of one sky.
        arms: The arms to list.
        column: The statistic compared with the limit.
        limit: The limit.
        unit: The statistic's unit, for the header; empty for none.
        digits: The decimal places of the statistic.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            f"| Arm | Mean at 0%{unit} | Largest at 0%{unit} | False alarms at 0% | "
            "Detected at 10% | 25% | 50% |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        zero = frame.filter((pl.col("arm") == arm) & (pl.col("share") == 0))
        values = zero[column].to_numpy()
        if np.isnan(values).all():
            mean, largest = "none", "none"
        else:
            mean, largest = f"{np.nanmean(values):.{digits}f}", f"{np.nanmax(values):.{digits}f}"
        lines.append(
            f"| {arm} | {mean} | {largest} | "
            f"{_above(values=zero[column], limit=limit)} of {zero.height} | "
            + " | ".join(_counts_by_share(frame=frame, arm=arm, column=column, limit=limit))
            + " |"
        )
    return lines


def false_alarm_lines(*, fits: pl.DataFrame) -> list[str]:
    """Return the report lines on false alarms and detection under both rules.

    Rule 1 is rung 4's: the share of the variance of the four-hour changes that the fitted solar
    explains, after the fitted battery or regressors are removed, counted against the largest
    value over A0's no-solar aggregates. Rule 2 is rung 6's: the fitted solar capacity counted
    against a fixed share of the aggregate's 99th percentile.

    Args:
        fits: The combined fits.

    Returns:
        Markdown lines.
    """
    two_mw = FALSE_ALARM_SHARE * AGGREGATE_P99_MW
    one_mw = STRICT_FALSE_ALARM_SHARE * AGGREGATE_P99_MW
    lines = [
        "### Post hoc capacity rule for B4, for the joint model: false alarms in the no-solar sums",
        "",
        (
            "The no-solar aggregates are the 16 batteries-only aggregates (share 0%). Two rules "
            "count a detection, and every arm is scored by both so that the arms are comparable."
        ),
        "",
        (
            "**Rule 1** (rung 4's) uses the share of the variance of the four-hour changes that "
            "the fitted solar explains once the arm's other fitted parts (the battery, or the "
            "price regressors) are removed. The threshold is the largest value of that share "
            "over A0's 16 no-solar aggregates, on the same sky, so A0 has 0 false alarms by "
            "construction. An aggregate is detected when its value is strictly above the "
            "threshold. A fit of exactly 0 MW of solar has nothing to explain, so its value is "
            "listed as none and the aggregate is not counted as detected, although 0 is above "
            "A0's negative threshold. **Rule 2** counts an aggregate as detected when the "
            "fitted solar "
            f"capacity is above {FALSE_ALARM_SHARE:.0%} of its 99th percentile "
            f"({two_mw:.0f} MW), and also above {STRICT_FALSE_ALARM_SHARE:.0%} "
            f"({one_mw:.0f} MW). Rung 4's A0 and A1 fit a plant whose capacity has a floor "
            "above both thresholds, so rule 2 lists only the fleet-curve arms."
        ),
        "",
    ]
    for sky in SKIES:
        frame = fits.filter(pl.col("sky") == sky)
        a0_zero = frame.filter((pl.col("arm") == "A0") & (pl.col("share") == 0))
        a0_threshold = float(np.nanmax(a0_zero["variance_explained"].to_numpy()))
        explained = frame.with_columns(
            variance_explained=pl.when(pl.col("fitted_ac_mw") > 0).then(
                pl.col("variance_explained")
            )
        )
        lines += [
            f"#### Sky: {sky}, rule 1 (A0's largest no-solar value is {a0_threshold:.4f})",
            "",
            *_rule_table(
                frame=explained,
                arms=FALSE_ALARM_ARMS,
                column="variance_explained",
                limit=a0_threshold,
                unit="",
                digits=4,
            ),
            "",
            f"#### Sky: {sky}, rule 2 (fitted solar above {two_mw:.0f} MW)",
            "",
            *_rule_table(
                frame=frame,
                arms=RULE_2_ARMS,
                column="fitted_ac_mw",
                limit=two_mw,
                unit=" (MW)",
                digits=2,
            ),
            "",
            f"#### Sky: {sky}, rule 2, stricter (fitted solar above {one_mw:.0f} MW)",
            "",
            *_rule_table(
                frame=frame,
                arms=RULE_2_ARMS,
                column="fitted_ac_mw",
                limit=one_mw,
                unit=" (MW)",
                digits=2,
            ),
            "",
        ]
    return lines


def battery_recovery_lines(*, fits: pl.DataFrame) -> list[str]:
    """Return the report lines on how well the joint model recovers the battery and its charge.

    Args:
        fits: The rung 6 fits.

    Returns:
        Markdown lines.
    """
    lines = [
        "### The recovered battery against the true battery",
        "",
        (
            "Aggregates with solar (shares 10%, 25%, and 50%), regional sky: mean and range over "
            "the 48 aggregates. The battery output error is the mean absolute difference between "
            "the fitted and the true battery output, as a percentage of the true battery's 99th "
            "percentile. The state-of-charge correlation compares the fitted path with rung 3's "
            "path of the true battery, window by window (each window's starting charge is free), "
            "and averages over windows."
        ),
        "",
        "| Quantity | " + " | ".join(BATTERY_ARMS) + " |",
        "|---|" + "---|" * len(BATTERY_ARMS),
    ]
    regional = fits.filter((pl.col("sky") == "regional") & (pl.col("share") > 0))
    for column, label in BATTERY_METRICS:
        cells = []
        for arm in BATTERY_ARMS:
            values = regional.filter(pl.col("arm") == arm)[column]
            cells.append(f"{values.mean():.2f} ({values.min():.2f} to {values.max():.2f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "By battery (mean over the 12 aggregates of each battery), A8:", ""]
    lines += [
        "| Battery | Output error (% of p99) | Output correlation | State-of-charge correlation |",
        "|---|---|---|---|",
    ]
    for battery in BATTERIES:
        sub = regional.filter((pl.col("arm") == "A8") & (pl.col("battery") == battery))
        lines.append(
            f"| {NAMES[battery]} | {sub['battery_mae_pct_of_p99'].mean():.2f} | "
            f"{sub['battery_correlation'].mean():.3f} | {sub['soc_correlation'].mean():.2f} |"
        )
    return [*lines, ""]


def smoothness_lines(*, rung6: pl.DataFrame) -> list[str]:
    """Return the report lines on the penalty for changes in the battery's output.

    Args:
        rung6: `rung6_fits.parquet`.

    Returns:
        Markdown lines.
    """
    lines = [
        "### A penalty on changes of the battery's output (regional sky)",
        "",
        (
            "Each A8 variant adds a cost per megawatt of half-hour-to-half-hour change in the "
            "battery's output, against 1 per megawatt of residual. Means over the 48 aggregates "
            "with solar; the false-alarm column counts the 16 no-solar aggregates."
        ),
        "",
        (
            "| Arm | Penalty | Solar error (% of p99) | Energy ratio | AC capacity ratio | "
            "Battery output error (% of p99) | Fitted solar at 0%: mean (MW) | "
            "False alarms at 0% |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    threshold = FALSE_ALARM_SHARE * AGGREGATE_P99_MW
    regional = rung6.filter(pl.col("sky") == SMOOTHNESS_SKY)
    for arm, penalty in {"A8": 0.0, **SMOOTHNESS_ARMS}.items():
        solar = regional.filter((pl.col("arm") == arm) & (pl.col("share") > 0))
        zero = regional.filter((pl.col("arm") == arm) & (pl.col("share") == 0))
        error_pct = solar.select(pl.col("nmae_of_solar_p99").mean()).item() * 100
        lines.append(
            f"| {arm} | {penalty} | {error_pct:.1f} | "
            f"{solar['energy_ratio'].mean():.2f} | {solar['ac_ratio'].mean():.2f} | "
            f"{solar['battery_mae_pct_of_p99'].mean():.2f} | {zero['fitted_ac_mw'].mean():.2f} | "
            f"{int((zero['fitted_ac_mw'] > threshold).sum())} of {zero.height} |"
        )
    return [*lines, ""]


LIKE_FOR_LIKE: Final[tuple[str, ...]] = (
    "A8 minus A3b",
    "A8 minus A1b",
    "A1b minus A0b",
    "A1b minus A1",
    "A8 minus A1",
)
"""The contrasts the like-for-like table lists. `A8`, `A1b`, `A0b`, and `A3b` share the joint
model's solar model, and `A1` has rung 4's physical plant."""
RUNG7_LIMIT_BMU: Final[str] = "2__ATGPL000"
RUNG7_LIMIT_POWER_MW: Final[float] = 50.0
AT_LIMIT_TOLERANCE_MW: Final[float] = 1e-6


def like_for_like_lines(*, contrasts: pl.DataFrame) -> list[str]:
    """Return the report lines comparing the joint model with price regressors, like for like.

    Args:
        contrasts: The contrast rows of `rung6_intervals.parquet`.

    Returns:
        Markdown lines.
    """
    lines = [
        "### Like for like: the joint model against the same solar model with other regressors",
        "",
        (
            "A8, A1b, A0b, and A3b share one solar model (the four fleet curves fitted to "
            "half-hourly levels), so their differences isolate the battery model. A1 uses rung "
            "4's physical plant fitted to four-hour changes, so differences from A1 mix the "
            "regressors with the solar model. Points of solar p99, pooled over 16 aggregates and "
            "all three shares; negative means the first arm recovers solar better."
        ),
        "",
        "| Contrast | Regional sky | GB-mean sky |",
        "|---|---|---|",
    ]
    for quantity in LIKE_FOR_LIKE:
        cells = []
        for sky in SKIES:
            row = contrasts.filter(
                (pl.col("sky") == sky)
                & (pl.col("group") == "pooled")
                & (pl.col("share") == 0.0)
                & (pl.col("quantity") == quantity)
            ).row(0, named=True)
            cells.append(_fmt(row=row))
        lines.append(f"| {quantity} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def capacity_bias_lines(*, levels: pl.DataFrame, rung6: pl.DataFrame) -> list[str]:
    """Return the report table on the joint model's tendency to overstate solar capacity.

    Args:
        levels: The level rows of `rung6_intervals.parquet`.
        rung6: `rung6_fits.parquet`.

    Returns:
        Markdown lines.
    """
    lines = [
        "### Rung 6 capacity bias",
        "",
        (
            "The joint model's fitted AC capacity runs above the direct fit to the solar half "
            "(a ratio above 1), and rung 7 inherits that. Ratios are pooled over the 16 "
            "aggregates and all three solar shares, with the 95% interval from resampling the "
            "aggregates."
        ),
        "",
        "| Quantity | Regional sky | GB-mean sky |",
        "|---|---|---|",
    ]
    for arm in ("A8", "A9"):
        for metric, label in (("ac_ratio", "AC capacity ratio"), ("energy_ratio", "energy ratio")):
            cells = []
            for sky in SKIES:
                row = levels.filter(
                    (pl.col("sky") == sky)
                    & (pl.col("group") == "pooled")
                    & (pl.col("share") == 0.0)
                    & (pl.col("metric") == metric)
                    & (pl.col("quantity") == arm)
                ).row(0, named=True)
                cells.append(_fmt(row=row, digits=2))
            lines.append(f"| {arm} {label} | " + " | ".join(cells) + " |")
    nan_cells = []
    for sky in SKIES:
        sub = rung6.filter(
            (pl.col("arm") == "A0b") & (pl.col("sky") == sky) & (pl.col("share") == 0.1)
        )
        nan_cells.append(f"{int(sub['correlation'].is_nan().sum())} of {sub.height}")
    lines.append(
        "| A0b: aggregates at the 10% share with no defined correlation (the fit gave 0 MW of "
        "solar) | " + " | ".join(nan_cells) + " |"
    )
    small_cells = []
    for sky in SKIES:
        sub = rung6.filter(
            (pl.col("arm") == "A0b") & (pl.col("sky") == sky) & (pl.col("share") == 0.1)
        )
        small_cells.append(
            f"{int((sub['fitted_ac_mw'] < FALSE_ALARM_SHARE * AGGREGATE_P99_MW).sum())} of "
            f"{sub.height}"
        )
    lines.append(
        "| A0b: aggregates at the 10% share with a fitted capacity under the 2 MW detection "
        "threshold | " + " | ".join(small_cells) + " |"
    )
    lines += _rung7_capacity_rows()
    return [*lines, ""]


def _rung7_capacity_rows() -> list[str]:
    """Return the rung 7 rows of the capacity-bias table, or none when rung 7 has not run."""
    fits_path = OUTPUT_DIR / "rung7_fits.parquet"
    series_path = OUTPUT_DIR / "rung7_series.parquet"
    if not (fits_path.exists() and series_path.exists()):
        return []
    fits = pl.read_parquet(fits_path).filter(pl.col("source") == "real")
    series = pl.read_parquet(series_path).filter(
        (pl.col("power_mw") == RUNG7_LIMIT_POWER_MW) & pl.col("aggregate_mw").is_not_nan()
    )
    at_limit = (
        series["fitted_battery_mw"].abs() >= RUNG7_LIMIT_POWER_MW - AT_LIMIT_TOLERANCE_MW
    ).mean()
    first = (
        fits.filter(pl.col("power_mw") == 0.0)
        .sort("fitted_ac_mw", descending=True)
        .row(0, named=True)
    )
    at_200 = fits.filter((pl.col("bmu") == first["bmu"]) & (pl.col("power_mw") == 200.0))
    totals = fits.group_by("power_mw").agg(pl.col("fitted_ac_mw").sum()).sort("fitted_ac_mw")
    biggest = totals.row(-1, named=True)
    return [
        "",
        "The same bias in rung 7, where the BMUs have no known solar capacity:",
        "",
        "| Quantity | Value |",
        "|---|---|",
        (
            f"| Share of {RUNG7_LIMIT_BMU}'s half-hours with the fitted battery at its "
            f"+-{RUNG7_LIMIT_POWER_MW:.0f} MW limit, P = {RUNG7_LIMIT_POWER_MW:.0f} MW | "
            f"{float(at_limit):.1%} |"  # ty: ignore[invalid-argument-type]
        ),
        (
            f"| The largest BMU's ({first['bmu']}) fitted capacity at P = 0 and at "
            f"P = 200 MW | {first['fitted_ac_mw']:.1f} MW to {at_200['fitted_ac_mw'][0]:.1f} MW |"
        ),
        (
            f"| The largest total fitted solar capacity of the 25 BMUs over the sweep of P "
            f"(at P = {biggest['power_mw']:g} MW) | {biggest['fitted_ac_mw']:.1f} MW |"
        ),
    ]


def main() -> None:
    """Write the intervals and the report fragment."""
    rung4 = pl.read_parquet(OUTPUT_DIR / "rung4_fits.parquet")
    rung6 = pl.read_parquet(OUTPUT_DIR / "rung6_fits.parquet")
    fits = _combined(rung4=rung4, rung6=rung6)
    levels = pl.DataFrame(_level_rows(fits=fits, arms=REPORT_ARMS))
    contrasts = pl.DataFrame(_contrast_rows(fits=fits, arms=REPORT_ARMS, contrasts=CONTRASTS))
    pl.concat([levels, contrasts]).write_parquet(OUTPUT_DIR / "rung6_intervals.parquet")
    arm_rows = [(f"{arm}: {ARM_LABELS.get(arm, arm)}", arm) for arm in REPORT_ARMS]
    contrast_labels = [
        (f"{treatment} minus {reference} ({kind})", f"{treatment} minus {reference}")
        for treatment, reference, kind in CONTRASTS
    ]
    expected = 4 * 4 * 4 * len(SKIES) * len(ARMS) + 4 * 4 * 4 * len(SMOOTHNESS_ARMS)
    lines = [
        "## Rung 6: a joint solar and state-of-charge battery model",
        "",
        (
            "The aggregates are rung 4's: 4 solar halves by 4 batteries by solar shares of 0%, "
            "10%, 25%, and 50% of a 100 MW 99th-percentile aggregate. The joint model writes the "
            "aggregate as solar plus a battery. Solar is a non-negative weighted sum of the four "
            "fleet curves (east, south, west, and tracker). The battery is a free charge and "
            "discharge schedule within a power limit, an energy capacity, and a one-way "
            "efficiency of 0.92, and the fit minimises the absolute residual by linear "
            "programming, in windows of about 4 weeks with a free starting charge in each and "
            "solar weights shared by all windows. A cost of 0.1 per megawatt on charging and "
            "discharging stops the battery dumping energy by charging and discharging at once. "
            "Solar error is the mean absolute error of the recovered solar over daylight "
            "half-hours as a percentage of the true solar half's 99th percentile."
        ),
        "",
        "### Arms",
        "",
        *ARM_TABLE,
        "",
        f"Fits expected: {expected}; fits present: {rung6.height}.",
        "",
        "### Solar series error by arm and share (% of solar p99; pooled over 16 aggregates)",
        "",
    ]
    for sky in SKIES:
        lines += [f"#### Sky: {sky}", ""]
        lines += _share_table(
            table=levels, sky=sky, metric="nmae_of_solar_p99", rows=arm_rows, digits=1
        )
    lines += [
        "### Post hoc contrasts B7 and B8 and the other differences (points of solar p99)",
        "",
        (
            "Negative means the first arm recovers solar better. B7 is A8 minus A0 and B8 is A9 "
            "minus A0. A0 is rung 4's physical plant fit and A0b is the joint model's own solar "
            "model with no battery, so A8 minus A0b isolates what the battery model adds."
        ),
        "",
    ]
    for sky in SKIES:
        lines += [f"#### Sky: {sky}, pooled over 16 aggregates", ""]
        lines += _share_table(
            table=contrasts, sky=sky, metric="nmae_of_solar_p99", rows=contrast_labels, digits=1
        )
    differences = [f"{t} minus {r}" for t, r, _ in CONTRASTS[:4]]
    lines += [
        "### Contrasts for each battery (regional sky, points of solar p99, all three shares)",
        "",
        "Intervals resample the 4 solar sets.",
        "",
        *_battery_table(table=contrasts, quantities=differences, metric="nmae_of_solar_p99"),
        "### Solar error for each battery (regional sky, % of solar p99, all three shares)",
        "",
        "Intervals resample the 4 solar sets.",
        "",
        *_battery_table(table=levels, quantities=list(REPORT_ARMS), metric="nmae_of_solar_p99"),
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
                rows=[(arm, arm) for arm in REPORT_ARMS],
                digits=2,
            )
    lines += false_alarm_lines(fits=fits)
    lines += like_for_like_lines(contrasts=contrasts)
    lines += capacity_bias_lines(levels=levels, rung6=rung6)
    lines += battery_recovery_lines(fits=rung6.filter(pl.col("arm").is_in(ARMS)))
    lines += smoothness_lines(rung6=rung6)
    lines += [
        "### Solar sets",
        "",
        *[f"- {label}" for label in SOLAR_SET_LABELS.values()],
    ]
    (OUTPUT_DIR / "report_rung6.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
