# ruff: noqa: E501, PLR0915
"""Stage B: fit round 2 (and round 3) on the pseudo-positives, choose alarms, and score every arm.

    E249_THREADS=2 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00249-*/e00249_stage_b.py FOLD

Per fold it writes `real_f<k>.parquet` (one row per labelled transfer and arm), `alarms_f<k>.parquet`
(every alarm of every arm on the real record), `injected_f<k>.parquet` (one row per injected event and
arm), `used_f<k>.parquet` (the pseudo-positives each variant trained on), and `summary_f<k>.json`.
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import polars as pl
from e00249_common import WORK_DIR, pair_folds, real_transfers
from e00249_data import KINDS, WORLD_DIR
from e00249_fit import Frame, fit_arms, real_candidates, reranker_scores
from e00249_models import load_worlds
from e00249_pseudo import consistent, hard_negatives
from e00249_ronc import pair_scores, ronc_scores
from e00249_score import BUDGET, clusters, flags, injected_hits, real_hits, threshold

HEAVY_SHARE = 0.25
"""The post hoc heavy arm gives the pseudo-positives this share of the injected positives' weight."""


def extra_rows(
    *,
    table: pl.DataFrame,
    windows: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    weight: float | np.ndarray,
) -> tuple[Frame, np.ndarray]:
    """Real rows as labelled extra training rows, with per-row weights (negatives weigh 1)."""
    keep = positive | negative
    label = np.where(positive, 1, 0)[keep]
    w = np.where(positive, weight, 1.0)[keep]
    return (table.filter(pl.Series(keep)).with_columns(label=pl.Series(label)), windows[keep]), w


def main(*, fold: int) -> None:
    """Fit every variant on a fold and score it."""
    started = time.time()
    table, windows = real_candidates()
    real = (table, windows)
    folds = pair_folds()
    stage_a = pl.read_parquet(WORK_DIR / f"pseudo_f{fold}.parquet")
    with (WORK_DIR / f"round1_f{fold}.pkl").open("rb") as handle:
        round1 = pickle.load(handle)
    rule = stage_a["rule"].to_numpy()
    strict = stage_a["strict"].to_numpy()
    negative = stage_a["hard_negative"].to_numpy()
    event_out = stage_a["event_out"].to_numpy()
    series_out = stage_a["series_out"].to_numpy()
    odd = table["odd"].to_numpy()
    n_injected_pos = int(
        (load_worlds(fold=fold, kind="train", count=KINDS["train"][0])[0]["label"] == 1).sum()
    )

    models: dict[str, tuple] = {"1": (round1["lr"], round1["rr"])}
    used: list[pl.DataFrame] = []

    def variant(*, name: str, positive: np.ndarray, neg: np.ndarray, weight: float | None) -> None:
        w = (
            weight
            if weight is not None
            else HEAVY_SHARE * n_injected_pos / max(int(positive.sum()), 1)
        )
        extra, wv = extra_rows(
            table=table, windows=windows, positive=positive, negative=neg, weight=w
        )
        models[name] = fit_arms(fold=fold, extra=extra, weight=wv)
        used.append(
            table.filter(pl.Series(positive))
            .select("series", "stamp", "source", "odd", "step_mw24")
            .with_columns(variant=pl.lit(name), weight=pl.lit(float(w)), fold=pl.lit(fold))
        )
        print(
            f"fold {fold}: variant {name}: {int(positive.sum())} pseudo-positives at weight {w:.3f}, "
            f"{int(neg.sum())} hard negatives; {time.time() - started:.0f} s",
            flush=True,
        )

    keep = rule & ~event_out
    variant(name="2 w0.25", positive=keep, neg=negative, weight=0.25)
    variant(name="2 w0.5", positive=keep, neg=negative, weight=0.5)
    variant(name="2 w0.25 even months", positive=keep & ~odd, neg=negative & ~odd, weight=0.25)
    variant(name="2 w0.25 series hold-out", positive=rule & ~series_out, neg=negative, weight=0.25)
    variant(name="2 w0.25 strict tier", positive=strict & ~event_out, neg=negative, weight=0.25)
    variant(name="2 heavy", positive=keep, neg=negative, weight=None)
    # Round 3: gate on the round 2 logistic regression and recompute the rule.
    gate2 = models["2 w0.25"][0](real)
    rule3 = consistent(table=table, windows=windows, gate=gate2, tier="loose")
    neg3 = hard_negatives(table=table, rule=rule3)
    variant(name="3 w0.25", positive=rule3 & ~event_out, neg=neg3, weight=0.25)

    def scores_for(frame: Frame) -> dict[str, np.ndarray]:
        t = frame[0]
        out = {"RO-NC": ronc_scores(table=t), "Pair alarm": pair_scores(table=t)}
        for name, (lr, rr) in models.items():
            out[f"F{name}"] = lr(frame)
            out[f"R{name}"] = reranker_scores(rr=rr, frame=frame)
        return out

    s_real = scores_for(real)
    thr = {a: threshold(score=s, table=table) for a, s in s_real.items()}
    transfers = real_transfers().filter(
        pl.col("pair").is_in([p for p, f in folds.items() if f == fold])
    )
    real_rows, alarm_rows, summary = [], [], {}
    for arm, s in s_real.items():
        fl = flags(score=s, thr=thr[arm])
        real_rows.append(
            real_hits(table=table, flagged=fl, events=transfers).with_columns(
                arm=pl.lit(arm), fold=pl.lit(fold)
            )
        )
        alarm_rows.append(
            table.filter(pl.Series(fl))
            .select(
                "series",
                "stamp",
                "source",
                "explained",
                "odd",
                "n_opposite",
                "opp_ratio",
                "end_est",
            )
            .with_columns(arm=pl.lit(arm), fold=pl.lit(fold))
        )
        summary[arm] = {"alarms": int(fl.sum()), "clusters": clusters(table=table, flagged=fl)}
        print(
            f"fold {fold}: {arm}: {summary[arm]}, found {int(real_rows[-1]['found'].sum())} of {real_rows[-1].height}",
            flush=True,
        )
    injected_rows = []
    for kind in ("eval_held", "eval_fitted"):
        frame = load_worlds(fold=fold, kind=kind, count=KINDS[kind][0])
        events = pl.concat(
            [
                pl.read_parquet(WORLD_DIR / f"f{fold}_{kind}_{i}_events.parquet").with_columns(
                    world=pl.lit(i)
                )
                for i in range(KINDS[kind][0])
            ]
        )
        for arm, s in scores_for(frame).items():
            fl = flags(score=s, thr=thr[arm])
            for i in range(KINDS[kind][0]):
                mask = (frame[0]["world"] == i).to_numpy()
                ev = events.filter(pl.col("world") == i)
                injected_rows.append(
                    injected_hits(
                        table=frame[0].filter(pl.Series(mask)), flagged=fl[mask], events=ev
                    )
                    .join(
                        ev.select("event_id", "hours", "depth_z", "shape", "n_receivers"),
                        on="event_id",
                    )
                    .with_columns(
                        arm=pl.lit(arm), kind=pl.lit(kind), fold=pl.lit(fold), world=pl.lit(i)
                    )
                )
    pl.concat(real_rows).write_parquet(WORK_DIR / f"real_f{fold}.parquet")
    pl.concat(alarm_rows).write_parquet(WORK_DIR / f"alarms_f{fold}.parquet")
    pl.concat(injected_rows).write_parquet(WORK_DIR / f"injected_f{fold}.parquet")
    pl.concat(used).write_parquet(WORK_DIR / f"used_f{fold}.parquet")
    (WORK_DIR / f"summary_f{fold}.json").write_text(
        json.dumps(
            {"fold": fold, "thresholds": thr, "arms": summary, "budget": BUDGET},
            indent=2,
            default=float,
        )
    )
    print(f"fold {fold} done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["0", "1", "2"]:
        main(fold=int(f))
