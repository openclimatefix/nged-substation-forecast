"""Print the checks of experiment 349: wiring gate, sunniest third by Primary, capacity ratios.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00351-*/e00351_checks.py
"""

import e00351_common as common
import e00351_run_common as run_common
import e00351_score as score
import numpy as np
import polars as pl
from studies.experiment_store import losses_for_bootstrap
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SOURCE_139 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00139-differentiable-physics-pv-as-of-capacity-in-gam-k"
    / "work"
    / "dp_preds"
)


def wiring_gate() -> None:
    """Compare arm A with experiment 139's stored pin forecasts at every Primary."""
    keys = ["power_fcst_init_time", "valid_time"]
    for site in run_common.SCORED_SITES:
        ours = pl.read_parquet(score.forecasts.out_path(arm="pin", period="scored", site=site))
        theirs = pl.read_parquet(SOURCE_139 / f"pin_scored_{site}.parquet")
        joined = ours.join(theirs, on=keys, suffix="_139")
        gap = max(
            float(np.max(np.abs((joined[q] - joined[f"{q}_139"]).to_numpy())))
            for q in ("q01", "q50", "q99")
        )
        print(f"gate {site}: {ours.height} rows, {joined.height} matched, largest gap {gap}")


def sunny_by_primary() -> None:
    """Print P minus A on the sunniest third by Primary and leaving one Primary out."""
    frames = {
        arm: score.load_arm(arm=arm, period="scored", sites=run_common.SCORED_SITES)
        for arm in ("pin", "p")
    }
    edges = score.tertile_edges(frame=frames["pin"])
    frames = {
        arm: score.add_slices(frame=score.with_losses(frame=f), edges=edges)
        for arm, f in frames.items()
    }
    stacked = losses_for_bootstrap(arm_losses={arm: f.drop("arm") for arm, f in frames.items()})
    sunny = stacked.filter(pl.col("third") == "sunny", pl.col("lead_hours") <= 168.0)
    score.N_RESAMPLES = 5000
    rows = []
    for site in run_common.SCORED_SITES:
        for name, part in (
            (f"{site} only", sunny.filter(pl.col("label") == site)),
            (f"leave out {site}", sunny.filter(pl.col("label") != site)),
        ):
            result = score.contrast(losses=part, treatment="p", reference="pin")
            rows.append({"scope": name, **result})
    table = pl.DataFrame(rows)
    common.write_parquet_atomic(frame=table, path=common.WORK_DIR / "sunny_by_primary.parquet")
    with pl.Config(float_precision=3, tbl_rows=20):
        print(table)


def applied_capacity(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return, per Primary and origin, the capacity applied (the latest data year's estimate)."""
    data = table.filter(pl.col("training_rows") >= common.MIN_YEAR_ROWS)
    latest = data.group_by("site", "origin").agg(pl.col("season_year").max().alias("season_year"))
    return data.join(latest, on=["site", "origin", "season_year"]).select(
        "site", "origin", "estimate_points", "raw_points", "pinned_points"
    )


def capacity_ratios() -> None:
    """Compare arm P's capacity with experiment 349's table and the pin, per Primary and month."""
    p = applied_capacity(
        table=pl.concat(
            [
                pl.read_parquet(f)
                for f in sorted((common.WORK_DIR / "asof_physical").glob("asof_*.parquet"))
            ]
        )
    )
    b = applied_capacity(
        table=pl.concat(
            [pl.read_parquet(f) for f in sorted(common.TABLE_349_DIR.glob("asof_*.parquet"))]
        )
    )
    both = p.join(b, on=["site", "origin"], suffix="_b").with_columns(
        p_over_pin=pl.col("estimate_points") / pl.col("pinned_points"),
        b_over_pin=pl.col("estimate_points_b") / pl.col("pinned_points"),
        p_over_b=pl.col("estimate_points") / pl.col("estimate_points_b"),
        r_over_pin=pl.col("raw_points") / pl.col("pinned_points"),
    )
    common.write_parquet_atomic(frame=both, path=common.WORK_DIR / "capacity_comparison.parquet")
    with pl.Config(float_precision=3, tbl_rows=80, tbl_cols=12):
        print(
            both.filter(pl.col("origin").dt.strftime("%Y-%m").is_in(["2025-07", "2026-06"]))
            .select(
                "site",
                "origin",
                "pinned_points",
                "estimate_points_b",
                "estimate_points",
                "raw_points",
                "p_over_pin",
                "b_over_pin",
                "p_over_b",
            )
            .sort("site", "origin")
        )
        summary = (
            both.group_by("site")
            .agg(
                n=pl.len(),
                p_vs_pin_gt10=((pl.col("p_over_pin") - 1).abs() > 0.1).sum(),
                b_vs_pin_gt10=((pl.col("b_over_pin") - 1).abs() > 0.1).sum(),
                p_vs_b_gt10=((pl.col("p_over_b") - 1).abs() > 0.1).sum(),
                mean_p_over_pin=pl.col("p_over_pin").mean(),
                mean_b_over_pin=pl.col("b_over_pin").mean(),
                mean_r_over_pin=pl.col("r_over_pin").mean(),
            )
            .sort("site")
        )
        print(summary)


if __name__ == "__main__":
    import sys

    if "--capacity-only" in sys.argv:
        capacity_ratios()
    else:
        wiring_gate()
        sunny_by_primary()
        capacity_ratios()
