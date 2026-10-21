"""Print the gate tables of experiment 129 from the stage C tables.

Reads `work/kat_<arm>.parquet`, `work/coverage_<arm>.parquet`, and the semi-synthetic tables
(`work/semi_c_*.parquet`), applies the gates fixed in the plan, and writes `work/gates.json`. Every
number the write-up quotes is printed here.

    uv run --no-sync python ml-experiments/00129-*/e00129_report.py
"""

import json
from typing import Final

import e00129_common as common
import numpy as np
import polars as pl

RECOVERY_BAND: Final[tuple[float, float]] = (0.75, 1.25)
RECOVERY_PRIMARIES_NEEDED: Final[int] = 4
COVERAGE_CELLS_NEEDED: Final[int] = 21
PLACEBO_SHARE_LIMIT: Final[float] = 0.20
"""A post-pilot choice: the placebo difference of fits at most this share of the real-sky one."""
DRIFT_LIMIT_PER_5_YEARS: Final[float] = 0.05
ARM_NAMES: Final[tuple[str, ...]] = ("arm2", "arm3", "arm4@0.1")


def recovery_gate(*, arm_name: str) -> pl.DataFrame:
    """Return, per Primary, the divided recovery of both synthetic plants and whether it passes."""
    table = pl.read_parquet(common.WORK_DIR / f"kat_{arm_name}.parquet").filter(
        pl.col("fraction") == 1.0, ~pl.col("cloud_load")
    )
    per_source = table.group_by("site", "source", "plant").agg(recovery=pl.col("recovery").median())
    divisor = (
        per_source.filter(pl.col("plant") == "matched")
        .group_by("site")
        .agg(divisor=pl.col("recovery").median())
    )
    low, high = RECOVERY_BAND
    return (
        per_source.filter(pl.col("plant") != "matched")
        .join(divisor, on="site")
        .with_columns(divided=pl.col("recovery") / pl.col("divisor"))
        .group_by("site")
        .agg(
            divisor=pl.col("divisor").first(),
            east_west=pl.col("divided").filter(pl.col("plant") == "east_west").first(),
            south_west=pl.col("divided").filter(pl.col("plant") == "south_west").first(),
            passes=pl.col("divided").is_between(low, high).all(),
        )
        .sort("site")
    )


def coverage_gate(*, arm_name: str) -> pl.DataFrame:
    """Return the coverage cells of an arm."""
    return pl.read_parquet(common.WORK_DIR / f"coverage_{arm_name}.parquet")


def semi_table() -> pl.DataFrame:
    """Return every semi-synthetic row, with the arm named as in the stage C tables."""
    frames = []
    for path in sorted(common.WORK_DIR.glob("semi_c_*.parquet")):
        frame = pl.read_parquet(path)
        if "arm4" in path.name:
            share = path.stem.rsplit("_", 1)[-1]
            frame = frame.with_columns(arm=pl.lit(f"arm4@{share}"))
        frames.append(frame)
    return pl.concat(frames)


def placebo_and_drift(*, semi: pl.DataFrame) -> pl.DataFrame:
    """Return, per arm, the placebo share of the real-sky recovery and the drift per 5 years."""
    by_sky = semi.group_by("arm", "site", "sky").agg(
        recovery=pl.col("recovery_mean").mean(), drift=pl.col("drift_5y").mean()
    )
    real = by_sky.filter(pl.col("sky") == "real").select(
        "arm", "site", real_recovery="recovery", drift="drift"
    )
    placebo = by_sky.filter(pl.col("sky") == "placebo").select(
        "arm", "site", placebo_recovery="recovery"
    )
    return (
        real.join(placebo, on=["arm", "site"])
        .with_columns(placebo_share=pl.col("placebo_recovery") / pl.col("real_recovery"))
        .sort("arm", "site")
    )


def main() -> None:
    """Print the gate tables and write `work/gates.json`."""
    summary: dict[str, object] = {}
    with pl.Config(tbl_rows=80, tbl_cols=14, float_precision=3):
        semi = placebo_and_drift(semi=semi_table())
        print("PLACEBO AND DRIFT (semi-synthetic, mean over draws)")
        print(semi)
        for arm_name in ARM_NAMES:
            recovery = recovery_gate(arm_name=arm_name)
            cells = coverage_gate(arm_name=arm_name)
            arm_semi = semi.filter(pl.col("arm") == arm_name)
            print(f"\n=== {arm_name} ===")
            print("recovery after the divisor, 100%, no load")
            print(recovery)
            print("coverage cells by plant and load")
            print(
                cells.group_by("plant", "cloud_load")
                .agg(
                    covered=pl.col("covered").sum(), n=pl.len(), width=pl.col("width_share").mean()
                )
                .sort("plant", "cloud_load")
            )
            summary[arm_name] = {
                "recovery_primaries_passing": int(recovery["passes"].sum()),
                "coverage_cells_covered": int(cells["covered"].sum()),
                "coverage_cells": cells.height,
                "placebo_share_median": float(np.median(arm_semi["placebo_share"].to_numpy())),
                "placebo_share_max": float(np.max(arm_semi["placebo_share"].to_numpy())),
                "drift_5y_mean": float(np.mean(arm_semi["drift"].to_numpy())),
                "drift_5y_max": float(np.max(arm_semi["drift"].to_numpy())),
            }
    summary["limits"] = {
        "recovery_primaries_needed": RECOVERY_PRIMARIES_NEEDED,
        "coverage_cells_needed": COVERAGE_CELLS_NEEDED,
        "placebo_share_limit": PLACEBO_SHARE_LIMIT,
        "drift_limit_per_5_years": DRIFT_LIMIT_PER_5_YEARS,
    }
    (common.WORK_DIR / "gates.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
