# ruff: noqa: E501
"""Loading a fold's data and fitting the logistic regression and the re-ranker, with optional extra rows."""

from __future__ import annotations

import os
from typing import Final

import numpy as np
import polars as pl
from e00249_common import E195_DIR, WORK_DIR
from e00249_data import KINDS
from e00249_models import LogisticScore, NetScore, load_worlds, usable
from e00249_ronc import FEATURE_NAMES

Frame = tuple[pl.DataFrame, np.ndarray]
THREADS: Final[int] = int(os.environ.get("E249_THREADS", "4"))
KEEP: Final[tuple[str, ...]] = (*FEATURE_NAMES, "source", "label")


def real_candidates() -> Frame:
    """Experiment 195's cached candidates of the uninjected record, with their raw windows."""
    return pl.read_parquet(E195_DIR / "real_candidates.parquet"), np.load(
        E195_DIR / "real_windows.npy"
    )


def fold_frames(*, fold: int) -> tuple[Frame, Frame, Frame]:
    """Injected training worlds, validation worlds (even months, usable rows), and the full validation frame."""
    train = load_worlds(fold=fold, kind="train", count=KINDS["train"][0])
    valid_full = load_worlds(fold=fold, kind="valid", count=KINDS["valid"][0])
    return train, usable(frame=valid_full, sources=("x", "p")), valid_full


def _stack(
    *, base: Frame, extra: Frame | None, weight: np.ndarray | None
) -> tuple[Frame, np.ndarray | None]:
    """Append `extra` (rows with `label`) to `base`; weights are 1 for base rows."""
    if extra is None:
        return base, None
    table = pl.concat([base[0].select(KEEP), extra[0].select(KEEP)])
    windows = np.concatenate([base[1], extra[1].astype(base[1].dtype)])
    w = np.concatenate([np.ones(base[0].height), weight]) if weight is not None else None
    return (table, windows), w


def fit_arms(
    *,
    fold: int,
    extra: Frame | None = None,
    weight: np.ndarray | None = None,
) -> tuple[LogisticScore, NetScore]:
    """Fit the logistic regression (all candidates) and the re-ranker (RO-NC candidates) on a fold.

    `extra` holds additional labelled real rows with per-row `weight`; the injected positives and
    negatives have weight 1. The validation worlds are injected only, so validation never sees a
    pseudo-label.
    """
    train_full, valid, _ = fold_frames(fold=fold)
    feat = usable(frame=train_full, sources=("x", "p"))
    x_only = usable(frame=train_full, sources=("x",))
    valid_x = usable(frame=(valid[0], valid[1]), sources=("x",))
    feat_all, w_feat = _stack(base=feat, extra=extra, weight=weight)
    if extra is not None:
        is_x = extra[0]["source"].to_numpy() == "x"
        x_extra: Frame | None = (extra[0].filter(pl.Series(is_x)), extra[1][is_x])
        w_x = weight[is_x] if weight is not None else None
    else:
        x_extra, w_x = None, None
    x_all, w_rr = _stack(base=x_only, extra=x_extra, weight=w_x)
    lr = LogisticScore(train=feat_all, weight=w_feat)
    rr = NetScore(train=x_all, valid=valid_x, kind="rerank", weight=w_rr, threads=THREADS)
    return lr, rr


def reranker_scores(*, rr: NetScore, frame: Frame) -> np.ndarray:
    """Re-ranker score for RO-NC candidates; -inf for the pair-alarm-only candidates it never scores."""
    table = frame[0]
    is_x = table["source"].to_numpy() == "x"
    out = np.full(table.height, -np.inf)
    if is_x.any():
        out[is_x] = rr((table.filter(pl.Series(is_x)), frame[1][is_x]))
    return out


assert WORK_DIR
