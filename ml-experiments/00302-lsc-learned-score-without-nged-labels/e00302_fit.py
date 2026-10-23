# ruff: noqa: E501
"""Train the re-ranker of L1 (labelled negatives) and L2 (label-free negatives) on odd months, and score.

Per fold and arm, writes `real_scores_<arm>_f<k>.parquet` (one score per real candidate) and
`inj_scores_<arm>_f<k>.parquet` (one score per candidate of the held-out-pair and fitted-pair
evaluation worlds, with the world and event ids), under the experiment's private folder. L3 uses
L2's model, so it has no model of its own.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=4 uv run --no-sync python e00302_fit.py FOLD [FOLD ...]
"""

from __future__ import annotations

import json
import sys
import time
from typing import Final

import numpy as np
import polars as pl
from e00302_common import E195_DIR, WORK_DIR
from e00302_data import KINDS
from e00302_models import NetScore, load_worlds, usable

ARMS: Final[dict[str, str]] = {"L1": "label", "L2": "label_free"}
"""Arm to the training-label column: L1 drops explained, D8-drop, and block-day candidates."""
KEEP: Final[tuple[str, ...]] = (
    "series",
    "stamp",
    "source",
    "explained",
    "odd",
    "d8_drop",
    "end_est",
    "n_opposite",
    "opp_ratio",
)


def score_frame(*, net: NetScore, frame: tuple[pl.DataFrame, np.ndarray]) -> np.ndarray:
    """Re-ranker score of every candidate; -inf for pair-source candidates, as in experiment 195."""
    table = frame[0]
    is_x = table["source"].to_numpy() == "x"
    out = np.full(table.height, -np.inf)
    if is_x.any():
        out[is_x] = net((table.filter(pl.Series(is_x)), frame[1][is_x]))
    return out


def main(*, fold: int) -> None:
    """Fit L1 and L2 on one fold and write their scores."""
    started = time.time()
    train_full = load_worlds(fold=fold, kind="train", count=KINDS["train"][0])
    valid_full = load_worlds(fold=fold, kind="valid", count=KINDS["valid"][0])
    real_table = pl.read_parquet(E195_DIR / "real_candidates.parquet")
    real_windows = np.load(E195_DIR / "real_windows.npy")
    inj = {
        k: load_worlds(fold=fold, kind=k, count=KINDS[k][0]) for k in ("eval_held", "eval_fitted")
    }
    summary = {}
    for arm, column in ARMS.items():
        train = usable(frame=train_full, sources=("x",), label_column=column)
        valid = usable(frame=valid_full, sources=("x",), label_column=column)
        net = NetScore(train=train, valid=valid, kind="rerank")
        real = score_frame(net=net, frame=(real_table, real_windows))
        real_table.select(*KEEP).with_columns(score=pl.Series(real)).write_parquet(
            WORK_DIR / f"real_scores_{arm}_f{fold}.parquet"
        )
        pieces = []
        for kind, frame in inj.items():
            score = score_frame(net=net, frame=frame)
            pieces.append(
                frame[0]
                .select(*KEEP, "world", "event_id", "label", "label_free")
                .with_columns(score=pl.Series(score), kind=pl.lit(kind))
            )
        pl.concat(pieces).write_parquet(WORK_DIR / f"inj_scores_{arm}_f{fold}.parquet")
        summary[arm] = {
            "train_rows": train[0].height,
            "train_positives": int((train[0]["label"] == 1).sum()),
            "train_negatives": int((train[0]["label"] == 0).sum()),
            "valid_rows": valid[0].height,
            "history": net.history,
            "n_parameters": net.n_parameters,
        }
        print(f"fold {fold} {arm}: {summary[arm]} at {time.time() - started:.0f} s", flush=True)
    (WORK_DIR / f"fit_summary_f{fold}.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )


if __name__ == "__main__":
    for f in sys.argv[1:] or ["0", "1", "2"]:
        main(fold=int(f))
