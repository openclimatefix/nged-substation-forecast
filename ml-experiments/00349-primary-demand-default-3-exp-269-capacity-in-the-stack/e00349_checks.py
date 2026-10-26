"""Print the checks of experiment 349: wiring gate, sunniest third by Primary, capacity ratios.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00349-*/e00349_checks.py
"""

import e00349_common as common
import e00349_run_common as run_common
import e00349_score as score
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
    """Print B minus A on the sunniest third by Primary and leaving one Primary out."""
    frames = {
        arm: score.load_arm(arm=arm, period="scored", sites=run_common.SCORED_SITES)
        for arm in ("pin", "b")
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
            result = score.contrast(losses=part, treatment="b", reference="pin")
            rows.append({"scope": name, **result})
    table = pl.DataFrame(rows)
    common.write_parquet_atomic(frame=table, path=common.WORK_DIR / "sunny_by_primary.parquet")
    with pl.Config(float_precision=3, tbl_rows=20):
        print(table)


def capacity_ratios() -> None:
    """Print the estimate over the pin for the latest season-year at selected origins."""
    table = pl.concat(
        [
            pl.read_parquet(path)
            for path in sorted((common.WORK_DIR / "asof").glob("asof_*.parquet"))
        ]
    ).filter(pl.col("season_year") == 2025)
    first = table.group_by("site").agg(pl.col("origin").min().alias("first_origin"))
    print(first.sort("site"))
    with pl.Config(float_precision=3, tbl_rows=60):
        print(
            table.with_columns(ratio=pl.col("estimate_points") / pl.col("pinned_points"))
            .filter(pl.col("origin").dt.strftime("%Y-%m").is_in(["2024-07", "2025-07", "2026-06"]))
            .select("site", "origin", "estimate_points", "pinned_points", "ratio")
            .sort("site", "origin")
        )


if __name__ == "__main__":
    wiring_gate()
    sunny_by_primary()
    capacity_ratios()
