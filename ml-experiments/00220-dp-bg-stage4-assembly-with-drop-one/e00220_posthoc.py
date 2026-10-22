"""Post hoc: split H-own minus A+H by network seed and by substation (explains prediction 2).

Prints, for each seed of experiment 201's arm 2, the quantile-score gap between the network's own
quantiles and A plus that seed's median correction, over all five decision substations, at D10, and
without D10; also each seed's arm 2 minus arm D at the six substations (gate 5).

    uv run --no-sync python ml-experiments/00220-*/e00220_posthoc.py
"""

import numpy as np
import polars as pl

import e00220_common as common  # isort: skip
import e00220_score as score  # isort: skip


def main() -> None:
    """Print the per-seed gaps."""
    rows = pl.read_parquet(common.ARM_DIR / "rows.parquet")
    site = rows["site"].to_numpy()
    five = rows["site"].is_in(list(common.DECISION_SITES)).to_numpy()
    qa = np.load(common.ARM_DIR / "a.npy").astype(np.float64)
    a = score.row_metrics(rows=rows, quantiles=qa)["qs"]
    d = score.row_metrics(rows=rows, quantiles=score.read_arm_d(rows=rows))["qs"]
    for seed in range(common.N_SEEDS):
        frame = pl.read_parquet(common.E201_DIR / f"arm2_seed{seed}.parquet")
        joined = rows.select(common.KEYS).join(
            frame.select([*common.KEYS, "delta", *common.QUANTILES]), on=common.KEYS, how="left"
        )
        own_q = joined.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
        delta = joined["delta"].to_numpy().astype(np.float64)[:, None]
        own = score.row_metrics(rows=rows, quantiles=own_q)["qs"]
        a_h = score.row_metrics(rows=rows, quantiles=np.sort(qa + delta, axis=1))["qs"]
        print(
            f"seed {seed}: own-A+H {(own - a_h)[five].mean():+.4f}; "
            f"A+H-A {(a_h - a)[five].mean():+.4f}; "
            f"D10 {(own - a_h)[five & (site == 'D10')].mean():+.4f}; "
            f"without D10 {(own - a_h)[five & (site != 'D10')].mean():+.4f}; "
            f"arm 2 minus D, six sites {(own - d).mean():+.4f}"
        )


if __name__ == "__main__":
    main()
