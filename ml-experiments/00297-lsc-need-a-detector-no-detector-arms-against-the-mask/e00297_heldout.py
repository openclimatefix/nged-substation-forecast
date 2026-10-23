"""Post hoc: re-score the band quantile head with held-out worlds.

Pass 2 fits the head on rows of one month parity from every world and scores the other parity of
the same worlds. This script fits the head only on even-seed worlds (rows of the other parity) and
scores only odd-seed worlds, so no scored world, injected shift, or seed was seen in fitting.
The injector and the real record are shared, so a leak through them is not removed.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00297_heldout.py D5
"""

from __future__ import annotations

import sys
import time
from typing import Final

import e00297_fed as fed
import e00297_pass2 as p2
import e00297_score as score
import numpy as np
import polars as pl
import xgboost as xgb
from e00297_common import WORK_DIR

OUT: Final = WORK_DIR / "scores3"


def main(site: str) -> None:
    """Fit on even-seed worlds, score odd-seed worlds."""
    started = time.time()
    data = fed.get_site(site)
    frames = []
    for path in sorted(score.FEATURES.glob(f"{site}_*.parquet")):
        world = int(path.stem.split("_")[1])
        k0 = (
            pl.read_parquet(score.OUT / path.name)
            .filter(pl.col("arm") == "K0")
            .select("row", "event", "onset_month", "actual_nra", "actual_obs")
        )
        frames.append(
            pl.read_parquet(path)
            .join(k0, on="row")
            .with_columns(world=pl.lit(world, dtype=pl.Int32))
        )
    frame = pl.concat(frames)
    ordinary = (frame["event"] < 0).to_numpy()
    fold_month = np.where(ordinary, frame["month"].to_numpy(), frame["onset_month"].to_numpy())
    parity = (fold_month % 2).astype(np.int64)
    weight = np.where(ordinary, p2.ORDINARY_WEIGHT, 1.0)
    k0_mean = frame["k0_mean"].to_numpy().astype(np.float64)
    residual = frame["actual_obs"].to_numpy().astype(np.float64) - k0_mean
    even = (frame["world"].to_numpy() % 2) == 0
    edges = score.edges_of(data=data)
    OUT.mkdir(parents=True, exist_ok=True)
    q_all = np.full((len(frame), 11), np.nan)
    for p in (0, 1):
        train = even & (parity == 1 - p)
        head = p2.fit_head(
            frame=frame.filter(pl.Series(train)), residual=residual[train], weight=weight[train]
        )
        test = np.flatnonzero(~even & (parity == p))
        q_all[test] = k0_mean[test][:, None] + head.predict(
            xgb.DMatrix(p2.head_features(frame[test]))
        )
    for world in sorted(set(frame["world"].to_list())):
        if world % 2 == 0:
            continue
        sel = (frame["world"] == world).to_numpy()
        sub = frame.filter(pl.Series(sel))
        rows = sub["row"].to_numpy().astype(np.int64)
        delta = np.zeros(len(data.cache.end_times))
        delta[data.position[rows]] = sub["actual_obs"].to_numpy() - sub["actual_nra"].to_numpy()
        arms = {"BAND_HEAD_HELDOUT": (p2.sorted_quantiles(q_all[sel]), k0_mean[sel])}
        scored = score.score_arms(arms=arms, rows=rows, data=data, delta_points=delta, edges=edges)
        meta = pl.read_parquet(score.OUT / f"{site}_{world}.parquet").filter(pl.col("arm") == "K0")
        meta = meta.drop(
            [
                c
                for c in meta.columns
                if c.startswith(("qs_", "ae_", "tw", "q0", "q5", "q9")) or c == "arm"
            ]
        )
        tmp = OUT / f"{site}_{world}.tmp"
        scored.join(meta, on="row").with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(
            tmp
        )
        tmp.rename(OUT / f"{site}_{world}.parquet")
    print(f"{site} done, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
