"""Collect the battery study's data checks and rung reports into `report.md`.

Run after `battery_primer.py`, `battery_rung1.py`, `battery_rung2.py`, `battery_rung3.py`,
`battery_rung4_summary.py`, `battery_rung5_summary.py`, `battery_rung6_summary.py`,
`battery_rung7_summary.py`, `battery_rung7b_summary.py`,
`battery_rung6_wrong_power_summary.py`, `battery_rung7c_costs_summary.py`, and
`battery_rung7c_known_answer_summary.py`:
`uv run python studies/battery_pv_separation/battery_report.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import (
    BATTERIES,
    EXPECTED_ROWS,
    NAMES,
    OUTPUT_DIR,
    battery_frame,
    battery_output,
    market_frame,
)

MIN_ZERO_RUN: Final[int] = 48
"""A run of exact-zero half-hours at least this long (one day) is listed."""
REGISTER_COLUMNS: Final[tuple[str, ...]] = (
    "elexon_bmu_id",
    "candidate_class",
    "fuel_type",
    "hint_reason",
    "generation_capacity_mw",
    "demand_capacity_mw",
)


def zero_runs(*, output_mw: np.ndarray, times: list) -> list[tuple[str, int]]:
    """Find runs of exact-zero output of at least `MIN_ZERO_RUN` half-hours.

    Args:
        output_mw: The output.
        times: The matching period starts.

    Returns:
        The start date and length in half-hours of each run.
    """
    runs, start = [], None
    for index, value in enumerate([*output_mw, 1.0]):
        if value == 0.0 and start is None:
            start = index
        elif value != 0.0 and start is not None:
            if index - start >= MIN_ZERO_RUN:
                runs.append((str(times[start].date()), index - start))
            start = None
    return runs


def main() -> None:
    """Write `report.md`."""
    market = market_frame()
    register = pl.read_csv(OUTPUT_DIR / "bmu_list.csv").filter(
        pl.col("elexon_bmu_id").is_in(BATTERIES)
    )
    nulls = dict(zip(market.columns, market.null_count().row(0), strict=True))
    lines = [
        "# Battery study: primer and rungs 1 to 7c",
        "",
        "## Data checks",
        "",
        (
            f"- Price rows on the half-hourly grid: {market.height} (expected {EXPECTED_ROWS}); "
            f"nulls per column: {nulls}."
        ),
        (
            "- B1610 is stamped at the half-hour end and was moved back 30 minutes to the period "
            "start before joining. Rung 1 reports the correlation at shifts of -3 to +3 "
            "half-hours; "
            "it peaks at shift 0 for all four batteries."
        ),
        "",
    ]
    for bmu_id in BATTERIES:
        output = battery_output(bmu_id=bmu_id)
        joined = battery_frame(bmu_id=bmu_id)
        runs = zero_runs(output_mw=output["output_mw"].to_numpy(), times=output["time"].to_list())
        row = (
            register.filter(pl.col("elexon_bmu_id") == bmu_id)
            .select(REGISTER_COLUMNS)
            .row(0, named=True)
        )
        lines += [
            f"### {NAMES[bmu_id]} ({bmu_id})",
            "",
            (
                f"- B1610 rows {output.height} (expected {EXPECTED_ROWS}); rows after joining "
                f"to day-ahead prices {joined.height}."
            ),
            "- Runs of at least one day of exact zeros: "
            + (", ".join(f"{d} ({n / 48:.1f} days)" for d, n in runs) or "none"),
            f"- Register entry: {row}",
            "",
        ]
    names = (
        "primer",
        *(f"rung{n}" for n in range(1, 8)),
        "rung7b",
        "rung6_wrong_power",
        "rung7c_costs",
        "rung7c_known_answer",
    )
    fragments = [(OUTPUT_DIR / f"report_{name}.md").read_text() for name in names]
    text = "\n".join(lines) + "\n" + "\n\n".join(fragments)
    (OUTPUT_DIR / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
