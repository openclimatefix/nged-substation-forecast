"""Summarise rung 7c, run 2: write `report_rung7c_known_answer.md`.

Reads `rung7c_known_answer_fits.parquet`. Nothing is refitted. Intervals resample the 48
aggregates (4 solar halves by 4 batteries by 3 demand-like BMUs) of one demand kind and one solar
size, as in `battery_rung4_summary`.

Run: `uv run python studies/battery_pv_separation/battery_rung7c_known_answer_summary.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung4_summary import PERCENT, interval
from battery_rung7c_known_answer import DEMAND_BMUS, DEMAND_KINDS, POWERS_MW, SOLAR_P99_MW

TRUE_LABEL: Final[str] = "True"
"""The assumed-power label of the fit given the battery half's true power."""
FALSE_ALARM_LIMITS_MW: Final[tuple[float, float]] = (1.0, 2.0)
"""The thresholds of rule 2: 1% and 2% of an aggregate's 100 MW 99th percentile."""


def with_assumed_power(*, fits: pl.DataFrame) -> pl.DataFrame:
    """Return the fits with one row per sweep power and one `True` row per aggregate.

    Args:
        fits: `rung7c_known_answer_fits.parquet`.

    Returns:
        The fits with the column `assumed`, the assumed power as text or `True`. A fit at a sweep
        power that is also the true power appears under both labels.
    """
    sweep = fits.filter(pl.col("power_mw").is_in(POWERS_MW)).with_columns(
        assumed=pl.col("power_mw").map_elements(lambda p: f"{p:g}", return_dtype=pl.String)
    )
    true = fits.filter(pl.col("power_mw") == pl.col("true_power_mw")).with_columns(
        assumed=pl.lit(TRUE_LABEL)
    )
    return pl.concat([sweep, true])


def assumed_order() -> list[str]:
    """Return the assumed-power labels in display order."""
    return [f"{p:g}" for p in POWERS_MW] + [TRUE_LABEL]


def _fmt(*, values: np.ndarray, digits: int) -> str:
    mean, low, high = interval(values=values)
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _table(*, frame: pl.DataFrame, column: str, scale: float, digits: int) -> list[str]:
    """Return a table with a row per demand kind and solar size and a column per assumed power."""
    order = assumed_order()
    lines = ["| Demand-like half | Solar p99 (MW) | " + " | ".join(order) + " |"]
    lines.append("|---|---|" + "---|" * len(order))
    for kind in DEMAND_KINDS:
        for solar_p99 in SOLAR_P99_MW[1:]:
            cells = []
            for label in order:
                values = frame.filter(
                    (pl.col("demand_kind") == kind)
                    & (pl.col("solar_p99_mw") == solar_p99)
                    & (pl.col("assumed") == label)
                )[column].drop_nulls()
                cells.append(_fmt(values=values.to_numpy() * scale, digits=digits))
            lines.append(f"| {kind} | {solar_p99:g} | " + " | ".join(cells) + " |")
    return lines


def _difference_table(*, fits: pl.DataFrame) -> list[str]:
    """Return the difference separation's capacity ratios, which do not depend on the battery."""
    once = fits.filter(pl.col("power_mw") == 0.0).filter(pl.col("solar_p99_mw") > 0)
    lines = [
        (
            "| Demand-like half | Solar p99 (MW) | Capacity over the fleet-curve fit to the solar "
            "half alone | Capacity over the direct physical fit |"
        ),
        "|---|---|---|---|",
    ]
    for kind in DEMAND_KINDS:
        for solar_p99 in SOLAR_P99_MW[1:]:
            part = once.filter(
                (pl.col("demand_kind") == kind) & (pl.col("solar_p99_mw") == solar_p99)
            )
            fleet = (part["difference_ac_mw"] / part["fleet_alone_ac_mw"]).to_numpy()
            physical = (part["difference_ac_mw"] / part["reference_ac_mw"]).to_numpy()
            lines.append(
                f"| {kind} | {solar_p99:g} | {_fmt(values=fleet, digits=2)} | "
                f"{_fmt(values=physical, digits=2)} |"
            )
    return lines


def _false_alarm_table(*, fits: pl.DataFrame) -> list[str]:
    order = assumed_order()
    zero = with_assumed_power(fits=fits).filter(pl.col("solar_p99_mw") == 0)
    lines = ["| Demand-like half | Limit (MW) | " + " | ".join(order) + " |"]
    lines.append("|---|---|" + "---|" * len(order))
    for kind in DEMAND_KINDS:
        for limit in FALSE_ALARM_LIMITS_MW:
            cells = []
            for label in order:
                values = zero.filter(
                    (pl.col("demand_kind") == kind) & (pl.col("assumed") == label)
                )["fitted_ac_mw"]
                cells.append(f"{int((values > limit).sum())} of {len(values)}")
            lines.append(f"| {kind} | {limit:g} | " + " | ".join(cells) + " |")
    return lines


def _per_series_table(*, frame: pl.DataFrame, solar_p99: float) -> list[str]:
    order = assumed_order()
    lines = ["| Demand-like half | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for bmu in DEMAND_BMUS:
        for kind in DEMAND_KINDS:
            means = (
                frame.filter(
                    (pl.col("demand_bmu") == bmu)
                    & (pl.col("demand_kind") == kind)
                    & (pl.col("solar_p99_mw") == solar_p99)
                )
                .group_by("assumed")
                .agg(pl.col("ratio_fleet").mean())
            )
            lookup = dict(zip(means["assumed"], means["ratio_fleet"], strict=True))
            lines.append(
                f"| {bmu} {kind} | " + " | ".join(f"{lookup[label]:.2f}" for label in order) + " |"
            )
    return lines


def main() -> None:
    """Write the report fragment."""
    fits = pl.read_parquet(OUTPUT_DIR / "rung7c_known_answer_fits.parquet")
    scored = with_assumed_power(fits=fits).filter(pl.col("solar_p99_mw") > 0)
    scored = scored.with_columns(ratio_fleet=pl.col("fitted_ac_mw") / pl.col("fleet_alone_ac_mw"))
    lines = [
        "## Rung 7c: known-answer test",
        "",
        (
            "Rung 7b's model (four fleet curves, a seasonal calendar baseline, and a battery on "
            "half-hourly levels) has never been fitted to a sum with a known answer that "
            "contains a demand-like part. Each sum here is a solar half, a battery half, and a "
            "demand-like half. The solar half is one of rung 4's four solar sets with a 99th "
            "percentile of 25 or 50 MW. The battery half is one of rung 4's four batteries with "
            "a 99th percentile of 100 MW minus the solar half's. The demand-like half is the "
            f"real output of one of {', '.join(DEMAND_BMUS)} (the three aggregate BMUs with the "
            "largest 99th-percentile absolute output among the 16 with no cloud signal), or that "
            "BMU's calendar replica, scaled to a 99th-percentile absolute output of 50 MW. "
            "Aggregates with a 0 MW solar half test for false alarms. The model is fitted with "
            "the regional sky of the solar set, a battery of the assumed power with 2 hours of "
            "energy, and a one-way efficiency of 0.92, to levels in windows of about 4 weeks. "
            "The true battery power is 75 MW for a 25 MW solar half, 50 MW for a 50 MW solar "
            "half, and 100 MW for no solar. Intervals resample the 48 aggregates of a row."
        ),
        "",
        (
            "The capacity ratios use two references. The fleet-curve reference is the same model "
            "(with the calendar baseline and no battery) fitted to the solar half alone, scaled "
            "to the aggregate's solar half: it is what the model fits with no battery or "
            "demand-like half to confuse it. The physical reference is rung 6's direct fit of a "
            "free-orientation plant to the solar half."
        ),
        "",
        "### Fitted solar capacity over the fleet-curve fit to the solar half alone",
        "",
        *_table(frame=scored, column="ratio_fleet", scale=1.0, digits=2),
        "",
        "### Fitted solar capacity over the direct physical fit to the solar half",
        "",
        *_table(frame=scored, column="ac_ratio", scale=1.0, digits=2),
        "",
        "### Solar series error (% of the solar half's p99)",
        "",
        *_table(frame=scored, column="nmae_of_solar_p99", scale=PERCENT, digits=1),
        "",
        "### Recovered solar energy over true solar energy",
        "",
        *_table(frame=scored, column="energy_ratio", scale=1.0, digits=2),
        "",
        "### The difference separation, for reference",
        "",
        (
            "The solar study's difference separation (fleet curves fitted to four-hour changes, "
            "then the baseline) has no battery, so one ratio per row applies at every assumed "
            "power."
        ),
        "",
        *_difference_table(fits=fits),
        "",
        "### Capacity ratio for each demand-like half (fleet-curve reference), 25 MW solar half",
        "",
        *_per_series_table(frame=scored, solar_p99=25.0),
        "",
        "### Capacity ratio for each demand-like half (fleet-curve reference), 50 MW solar half",
        "",
        *_per_series_table(frame=scored, solar_p99=50.0),
        "",
        "### False alarms with no solar half (fitted solar above the limit)",
        "",
        *_false_alarm_table(fits=fits),
    ]
    (OUTPUT_DIR / "report_rung7c_known_answer.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
