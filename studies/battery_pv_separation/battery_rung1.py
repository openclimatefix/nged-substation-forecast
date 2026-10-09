"""Rung 1: one real battery, no model.

Prints the numbers behind Figure 2 into `report_rung1.md` under the study's data folder. Saves two
tables: the mean output by half-hour of day and within-day price third for each battery, which
`nged_battery_a_rungs.py` compares against, and the check that the output and price clocks line up.

Run: `OMP_NUM_THREADS=2 uv run python studies/battery_pv_separation/battery_rung1.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR, TERCILE_LABELS, battery_frame

LAGS: Final[range] = range(-3, 4)
"""Half-hour shifts of the price against the output in the alignment check."""


def alignment_table(*, frame: pl.DataFrame) -> list[tuple[int, float]]:
    """Return the correlation of output with the day-ahead price at several shifts.

    Args:
        frame: A battery frame from `battery_frame`.

    Returns:
        Pairs of shift in half-hours and correlation. A positive shift pairs the output with the
        price that many half-hours earlier.
    """
    output = frame["output_mw"].to_numpy()
    price = frame["day_ahead_gbp_per_mwh"].to_numpy()
    n = len(output)
    return [
        (
            lag,
            float(
                np.corrcoef(
                    output[max(0, lag) : n + min(0, lag)], price[max(0, -lag) : n - max(0, lag)]
                )[0, 1]
            ),
        )
        for lag in LAGS
    ]


def main() -> None:
    """Write the rung 1 tables and report fragment."""
    lines = ["## Rung 1: one real battery, no model", ""]
    tables = []
    for bmu_id in BATTERIES:
        frame = battery_frame(bmu_id=bmu_id)
        grid = (
            frame.group_by("tod", "tercile")
            .agg(mean_output_mw=pl.col("output_mw").mean(), n=pl.len())
            .with_columns(bmu_id=pl.lit(bmu_id))
        )
        tables.append(grid)
        by_tercile = (
            frame.group_by("tercile")
            .agg(
                mean_mw=pl.col("output_mw").mean(),
                charging_share=(pl.col("output_mw") < -1.0).mean(),
                discharging_share=(pl.col("output_mw") > 1.0).mean(),
            )
            .sort("tercile")
        )
        by_price_level = (
            frame.with_columns(
                level=(pl.col("day_ahead_gbp_per_mwh").rank("average") / frame.height * 3)
                .ceil()
                .cast(pl.Int64)
                .clip(1, 3)
            )
            .group_by("level")
            .agg(mean_mw=pl.col("output_mw").mean())
            .sort("level")
        )
        lines += [f"### {NAMES[bmu_id]} ({bmu_id})", ""]
        lines += [
            (
                f"- Rows: {frame.height}; mean output {frame['output_mw'].mean():.2f} MW; "
                f"exact zeros {(frame['output_mw'] == 0).mean():.1%}."
            ),
            "- Mean output (MW) by within-day price third: "
            + "; ".join(
                f"{TERCILE_LABELS[row['tercile']]}: {row['mean_mw']:.1f} "
                f"(charging {row['charging_share']:.0%}, "
                f"discharging {row['discharging_share']:.0%} "
                f"of half-hours, beyond 1 MW)"
                for row in by_tercile.iter_rows(named=True)
            ),
            "- Mean output (MW) by whole-year day-ahead price third (low, mid, high): "
            + ", ".join(f"{v:.1f}" for v in by_price_level["mean_mw"]),
            "- Correlation of output with the day-ahead price, price shifted by k half-hours "
            "(positive k = older price): "
            + ", ".join(f"k={k}: {c:.3f}" for k, c in alignment_table(frame=frame)),
            "",
        ]
    pl.concat(tables).write_parquet(OUTPUT_DIR / "rung1_tod_tercile_means.parquet")
    (OUTPUT_DIR / "report_rung1.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
