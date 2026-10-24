"""Experiment 00341, step 2b: choose the penalty after the stop rule fired, and commit it.

The pre-registered rule (at most one change point per farm-year in 2021-09 to 2024-06 at c = 2) was
not met at any `c` in the grid, because several change points of the unmodified series survive the
largest penalty and so are probably real changes. This script counts as a false alarm only the
change points that disappear at the largest penalty (`c = 16`, `h = 12`) and picks the smallest
`c` (and `h`), separately for each arm group (A, B, C, D), at which that excess is at most one
per farm-year for every primary farm and block length. It writes `work/penalty.json`, which the
grid stage reads, before any real-series result exists.

    uv run --no-sync python ml-experiments/00341-*/e00341_penalty.py
"""

import json

import e00341_common as common
import e00341_inject as inj
import polars as pl


def main() -> None:
    """Write the chosen penalty and the full trade-off table."""
    frame = pl.read_parquet(common.WORK_DIR / "calibrate_rows.parquet")
    years = (common.CAL_END_DAY - common.CAL_START_DAY + 1) / 365.25
    out: dict = {"rule": "excess over the largest penalty; see script docstring"}
    for detector, grid, name in (("cal_pelt", inj.C_GRID, "c"), ("cal_cusum", inj.H_GRID, "h")):
        sub = frame.filter(
            (pl.col("detector") == detector)
            & (pl.col("cell") == "null")
            & pl.col("arm").is_in(inj.FA_PRIMARY_ARMS)
            & pl.col("block").is_in(inj.FA_PRIMARY_BLOCKS)
        )
        floor = sub.filter(pl.col("setting") == grid[-1]).select(
            "site", "arm", "block", floor=pl.col("value")
        )
        per_arm: dict = {}
        tables: dict = {}
        for group, arms in (
            ("A", ("A_p999@2", "A_p999@0")),
            ("B", ("B",)),
            ("C", ("C",)),
            ("D", ("D",)),
        ):
            chosen = grid[-1]
            table: dict = {}
            for setting in grid:
                joined = sub.filter(
                    (pl.col("setting") == setting) & pl.col("arm").is_in(arms)
                ).join(floor, on=["site", "arm", "block"])
                excess = (
                    float((joined["value"] - joined["floor"]).clip(lower_bound=0).to_numpy().max())
                    / years
                )
                table[str(setting)] = float(excess)
            for setting in grid:
                if table[str(setting)] <= 1.0:
                    chosen = setting
                    break
            per_arm[group] = chosen
            tables[group] = table
        out[name] = per_arm
        out[f"{name}_excess_per_farm_year"] = tables
    (common.WORK_DIR / "penalty.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
