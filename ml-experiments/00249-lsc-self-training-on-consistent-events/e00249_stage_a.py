# ruff: noqa: E501
"""Stage A: fit round 1, apply the label-free rule, and print the pseudo-positive counts.

Run before any real transfer is scored:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00249-*/e00249_stage_a.py FOLD [FOLD ...]

Writes `round1_f<k>.pkl` (both fitted models), `round1_scores_f<k>.npz` (their scores on every real
candidate), and `pseudo_f<k>.parquet` (every real candidate with its rule flags).
"""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import polars as pl
from e00249_common import WORK_DIR, pair_folds
from e00249_fit import fit_arms, real_candidates, reranker_scores
from e00249_pseudo import consistent, hard_negatives, holdout_mask


def main(*, fold: int) -> None:
    """Fit round 1 on a fold and write its scores and the rule's flags."""
    started = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    table, windows = real_candidates()
    lr, rr = fit_arms(fold=fold)
    s_lr = lr((table, windows))
    s_rr = reranker_scores(rr=rr, frame=(table, windows))
    with (WORK_DIR / f"round1_f{fold}.pkl").open("wb") as handle:
        pickle.dump({"lr": lr, "rr": rr}, handle)
    np.savez(WORK_DIR / f"round1_scores_f{fold}.npz", lr=s_lr, rr=s_rr)
    strict = consistent(table=table, windows=windows, gate=s_lr, tier="strict")
    rule = consistent(table=table, windows=windows, gate=s_lr, tier="loose")
    folds = pair_folds()
    event_out = holdout_mask(table=table, fold=fold, folds=folds, level="event")
    series_out = holdout_mask(table=table, fold=fold, folds=folds, level="series")
    negatives = hard_negatives(table=table, rule=rule)
    out = table.select("series", "stamp", "source", "odd", "step_mw24", "holiday").with_columns(
        rule=pl.Series(rule),
        strict=pl.Series(strict),
        event_out=pl.Series(event_out),
        series_out=pl.Series(series_out),
        hard_negative=pl.Series(negatives),
        gate=pl.Series(s_lr),
    )
    out.write_parquet(WORK_DIR / f"pseudo_f{fold}.parquet")
    kept = rule & ~event_out
    print(
        f"fold {fold}: {int(strict.sum())} pass the strict tier; {int(rule.sum())} pass the loose rule; {int(kept.sum())} after the event hold-out "
        f"({int((rule & ~series_out).sum())} after the series hold-out); {int(negatives.sum())} hard "
        f"negatives; {int((kept & table['odd'].to_numpy()).sum())} pseudo-positives in odd months; "
        f"{int((kept & (table['source'].to_numpy() == 'x')).sum())} of source x; "
        f"done in {time.time() - started:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    for f in sys.argv[1:] or ["0", "1", "2"]:
        main(fold=int(f))
