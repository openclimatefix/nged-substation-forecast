# ruff: noqa: E501, PLR0915
"""Train every arm on one fold, pick the alarms at 80 unexplained odd-month alarms, and score them.

    OMP_NUM_THREADS=4 uv run --no-sync python ml-experiments/00195-*/e00195_run.py FOLD [FOLD ...]

Per fold the script writes, under the experiment's private folder: `alarms_f<k>.parquet` (every alarm
of every arm on the real record), `real_f<k>.parquet` (one row per labelled transfer of the fold and
arm), `injected_f<k>.parquet` (one row per injected event and arm), and `summary_f<k>.json`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Final

import numpy as np
import polars as pl
from e00195_common import WORK_DIR as BASE_DIR
from e00195_common import pair_folds, real_transfers
from e00195_data import KINDS, Base
from e00195_models import (
    LogisticScore,
    NetScore,
    StackScore,
    TreeScore,
    load_worlds,
    usable,
)
from e00195_ronc import pair_scores, ronc_scores

TAG: Final[str] = os.environ.get("E195_TAG", "")
WORK_DIR = BASE_DIR / TAG if TAG else BASE_DIR
WORLDS_DIR = BASE_DIR / "worlds"
BUDGET: Final[int] = 80
ONSET_TOLERANCE: Final[int] = 4
END_MINIMUM: Final[int] = 12
END_FRACTION: Final[float] = 0.10
ALPHAS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
Frame = tuple[pl.DataFrame, np.ndarray]


def real_candidates(*, base: Base) -> Frame:
    """Candidates of the uninjected record, cached."""
    path = BASE_DIR / "real_candidates.parquet"
    if path.exists():
        return pl.read_parquet(path), np.load(BASE_DIR / "real_windows.npy")
    _, cands = base.clean()
    cands.table.write_parquet(path)
    np.save(BASE_DIR / "real_windows.npy", cands.windows.astype(np.float16))
    return cands.table, cands.windows.astype(np.float16)


def threshold(*, score: np.ndarray, table: pl.DataFrame, budget: int) -> float:
    """Score above which exactly `budget` unexplained odd-month candidates remain."""
    pool = score[table["odd"].to_numpy() & ~table["explained"].to_numpy() & np.isfinite(score)]
    ordered = np.sort(pool)[::-1]
    return float(ordered[budget]) if len(ordered) > budget else -np.inf


def flags(*, score: np.ndarray, thr: float) -> np.ndarray:
    """Candidates whose score is above `thr`."""
    return np.isfinite(score) & (score > thr)


def clusters(*, table: pl.DataFrame, flagged: np.ndarray) -> int:
    """Unexplained odd-month flagged candidates, merging those within 2 hours on one series."""
    rows = table.filter(
        pl.Series(flagged & table["odd"].to_numpy() & ~table["explained"].to_numpy())
    ).sort("series", "stamp")
    count, last = 0, (None, -99)
    for series, stamp in rows.select("series", "stamp").iter_rows():
        if series != last[0] or stamp - last[1] > ONSET_TOLERANCE:
            count += 1
        last = (series, stamp)
    return count


def end_tolerance(*, onset: int, end: int) -> float:
    """The larger of 6 hours and 10% of the duration, in half-hours."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def real_hits(*, table: pl.DataFrame, flagged: np.ndarray, events: pl.DataFrame) -> pl.DataFrame:
    """For each labelled transfer: onset found, found (start and end), and the onset error in hours."""
    alarms = table.filter(pl.Series(flagged))
    by_series = {s: g for (s,), g in alarms.group_by("series")}
    rows = []
    for e in events.iter_rows(named=True):
        onset_found, found, errors = False, False, []
        for s in e["series"]:
            g = by_series.get(s)
            if g is None:
                continue
            for stamp, end_est in zip(g["stamp"].to_list(), g["end_est"].to_list(), strict=True):
                gap = abs(stamp - e["onset"])
                if gap <= 48:
                    errors.append(gap)
                if gap <= ONSET_TOLERANCE:
                    onset_found = True
                    if abs(end_est - e["end"]) <= end_tolerance(onset=e["onset"], end=e["end"]):
                        found = True
        rows.append(
            {
                "card_id": e["card_id"],
                "onset_found": onset_found,
                "found": found,
                "onset_error_h": min(errors) / 2 if errors else None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "card_id": pl.String,
            "onset_found": pl.Boolean,
            "found": pl.Boolean,
            "onset_error_h": pl.Float64,
        },
    )


def injected_hits(
    *, table: pl.DataFrame, flagged: np.ndarray, events: pl.DataFrame
) -> pl.DataFrame:
    """For each injected event: found and onset found, from the flagged candidates that mark it."""
    marks = table.filter(pl.Series(flagged & (table["event_id"].to_numpy() >= 0)))
    by_event = {k: g for (k,), g in marks.group_by("event_id")}
    rows = []
    for e in events.iter_rows(named=True):
        g = by_event.get(e["event_id"])
        onset_found = g is not None
        found = bool(
            onset_found
            and any(
                abs(x - e["end"]) <= end_tolerance(onset=e["onset"], end=e["end"])
                for x in g["end_est"].to_list()
            )
        )
        rows.append({"event_id": e["event_id"], "onset_found": onset_found, "found": found})
    return pl.DataFrame(
        rows, schema={"event_id": pl.Int64, "onset_found": pl.Boolean, "found": pl.Boolean}
    )


def split_budget(
    *, sa: np.ndarray, sb: np.ndarray, table: pl.DataFrame, alpha: float
) -> tuple[float, float]:
    """Thresholds of two scores whose union leaves the most alarms not above `BUDGET` unexplained clusters."""
    best = (
        threshold(score=sa, table=table, budget=int(alpha * 20)),
        threshold(score=sb, table=table, budget=20 - int(alpha * 20)),
    )
    for total in range(20, 240, 2):
        ba = round(alpha * total)
        thr = (
            threshold(score=sa, table=table, budget=ba),
            threshold(score=sb, table=table, budget=total - ba),
        )
        if (
            clusters(table=table, flagged=flags(score=sa, thr=thr[0]) | flags(score=sb, thr=thr[1]))
            > BUDGET
        ):
            break
        best = thr
    return best


def intersect_thresholds(
    *, sa: np.ndarray, sb: np.ndarray, table: pl.DataFrame
) -> tuple[float, float]:
    """Equal-budget thresholds whose intersection leaves the most alarms not above `BUDGET` clusters."""
    best = (-np.inf, -np.inf)
    for total in range(BUDGET, 800, 10):
        thr = (
            threshold(score=sa, table=table, budget=total),
            threshold(score=sb, table=table, budget=total),
        )
        fa, fb = flags(score=sa, thr=thr[0]), flags(score=sb, thr=thr[1])
        both = _near(table=table, a=fa, b=fb)
        if clusters(table=table, flagged=both) > BUDGET:
            break
        best = thr
    return best


def _near(*, table: pl.DataFrame, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flagged candidates of `a` that have a flagged candidate of `b` within 2 hours on the same series."""
    series, stamp = table["series"].to_numpy(), table["stamp"].to_numpy()
    out = np.zeros(len(a), dtype=bool)
    idx_b: dict[int, np.ndarray] = {}
    for s in np.unique(series):
        idx_b[int(s)] = np.sort(stamp[b & (series == s)])
    for i in np.flatnonzero(a):
        arr = idx_b.get(int(series[i]), np.array([], dtype=int))
        if arr.size:
            k = np.searchsorted(arr, stamp[i])
            near = [abs(arr[j] - stamp[i]) for j in (k - 1, k) if 0 <= j < arr.size]
            out[i] = bool(near) and min(near) <= ONSET_TOLERANCE
    return out


def main(*, fold: int) -> None:
    """Train, select alarms on the real record, and score on the fold's held-out events and worlds."""
    started = time.time()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    base = Base()
    real = real_candidates(base=base)
    table = real[0]
    train = {
        "feat": usable(
            frame=load_worlds(fold=fold, kind="train", count=KINDS["train"][0]), sources=("x", "p")
        ),
        "x": usable(
            frame=load_worlds(fold=fold, kind="train", count=KINDS["train"][0]), sources=("x",)
        ),
    }
    valid_full = load_worlds(fold=fold, kind="valid", count=KINDS["valid"][0])
    valid = {
        "feat": usable(frame=valid_full, sources=("x", "p")),
        "x": usable(frame=valid_full, sources=("x",)),
    }
    print(
        f"fold {fold}: data loaded in {time.time() - started:.0f} s; train {train['feat'][0].height} rows, {int((train['feat'][0]['label'] == 1).sum())} positives",
        flush=True,
    )
    lr = LogisticScore(train=train["feat"])
    tree = TreeScore(train=train["feat"], valid=valid["feat"])
    nn_f = NetScore(train=train["feat"], valid=valid["feat"], kind="features")
    nn_r = NetScore(train=train["x"], valid=valid["x"], kind="rerank")
    stack = StackScore(net=nn_f, valid=valid["feat"])
    print(
        f"fold {fold}: models trained in {time.time() - started:.0f} s; tree config {tree.config}; nn {nn_f.history}; rerank {nn_r.history}",
        flush=True,
    )

    def scores_for(frame: Frame) -> dict[str, np.ndarray]:
        t = frame[0]
        is_x = t["source"].to_numpy() == "x"
        out = {
            "RO-NC": ronc_scores(table=t),
            "Pair alarm": pair_scores(table=t),
            "F logistic": lr(frame),
            "F tree": tree(frame),
            "F network": nn_f(frame),
            "C stack": stack(frame),
        }
        rr = np.full(t.height, -np.inf)
        if is_x.any():
            rr[is_x] = nn_r((t.filter(pl.Series(is_x)), frame[1][is_x]))
        out["R re-ranker"] = rr
        return out

    s_real = scores_for(real)
    thr = {a: threshold(score=s, table=table, budget=BUDGET) for a, s in s_real.items()}
    # Union and intersection arms: their thresholds are chosen on the real record's alarm count only.
    s_valid = scores_for(valid_full)
    pairs = {
        "U RO-NC + pair (rule)": ("RO-NC", "Pair alarm"),
        "U F network + RO-NC": ("F network", "RO-NC"),
        "U F network + pair": ("F network", "Pair alarm"),
        "U R re-ranker + pair": ("R re-ranker", "Pair alarm"),
    }
    union_thr: dict[str, tuple[float, float]] = {}
    valid_events = pl.concat(
        [
            pl.read_parquet(p).with_columns(world=pl.lit(i))
            for i in range(KINDS["valid"][0])
            for p in [WORLDS_DIR / f"f{fold}_valid_{i}_events.parquet"]
        ]
    )
    vt = valid_full[0]
    for name, (a, b) in pairs.items():
        best_alpha, best_recall = 0.5, -1.0
        for alpha in ALPHAS:
            ta, tb = split_budget(sa=s_real[a], sb=s_real[b], table=table, alpha=alpha)
            fl = flags(score=s_valid[a], thr=ta) | flags(score=s_valid[b], thr=tb)
            recall = _valid_recall(table=vt, flagged=fl, events=valid_events)
            if recall > best_recall:
                best_alpha, best_recall, union_thr[name] = alpha, recall, (ta, tb)
        print(f"fold {fold}: {name} alpha {best_alpha} valid recall {best_recall:.3f}", flush=True)
    inter_pairs = {
        "I F network & RO-NC": ("F network", "RO-NC"),
        "I F network & pair": ("F network", "Pair alarm"),
    }
    inter_thr = {
        n: intersect_thresholds(sa=s_real[a], sb=s_real[b], table=table)
        for n, (a, b) in inter_pairs.items()
    }

    def arm_flags(
        *, scores: dict[str, np.ndarray], frame_table: pl.DataFrame
    ) -> dict[str, np.ndarray]:
        out = {a: flags(score=scores[a], thr=thr[a]) for a in scores}
        for name, (a, b) in pairs.items():
            out[name] = flags(score=scores[a], thr=union_thr[name][0]) | flags(
                score=scores[b], thr=union_thr[name][1]
            )
        for name, (a, b) in inter_pairs.items():
            fa = flags(score=scores[a], thr=inter_thr[name][0])
            fb = flags(score=scores[b], thr=inter_thr[name][1])
            out[name] = _near(table=frame_table, a=fa, b=fb)
        return out

    real_flags = arm_flags(scores=s_real, frame_table=table)
    folds = pair_folds()
    transfers = real_transfers().filter(
        pl.col("pair").is_in([p for p, f in folds.items() if f == fold])
    )
    real_rows, alarm_rows = [], []
    for arm, fl in real_flags.items():
        hits = real_hits(table=table, flagged=fl, events=transfers).with_columns(
            arm=pl.lit(arm), fold=pl.lit(fold)
        )
        real_rows.append(hits)
        al = (
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
        alarm_rows.append(al)
        print(
            f"fold {fold}: {arm}: {int(fl.sum())} alarms, {clusters(table=table, flagged=fl)} unexplained odd clusters, found {int(hits['found'].sum())} of {hits.height}",
            flush=True,
        )
    injected_rows = []
    for kind in ("eval_held", "eval_fitted"):
        frame = load_worlds(fold=fold, kind=kind, count=KINDS[kind][0])
        events = pl.concat(
            [
                pl.read_parquet(WORLDS_DIR / f"f{fold}_{kind}_{i}_events.parquet").with_columns(
                    world=pl.lit(i)
                )
                for i in range(KINDS[kind][0])
            ]
        )
        scores = scores_for(frame)
        fl_all = arm_flags(scores=scores, frame_table=frame[0])
        for arm, fl in fl_all.items():
            for i in range(KINDS[kind][0]):
                mask = (frame[0]["world"] == i).to_numpy()
                sub = frame[0].filter(pl.Series(mask))
                ev = events.filter(pl.col("world") == i)
                injected_rows.append(
                    injected_hits(table=sub, flagged=fl[mask], events=ev)
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
    (WORK_DIR / f"summary_f{fold}.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "tree_config": tree.config,
                "n_parameters": {"F network": nn_f.n_parameters, "R re-ranker": nn_r.n_parameters},
                "histories": {"F network": nn_f.history, "R re-ranker": nn_r.history},
                "stack_coefficients": stack.coef,
                "thresholds": thr,
                "union_thresholds": union_thr,
                "intersection_thresholds": inter_thr,
                "train_rows": train["feat"][0].height,
                "train_positives": int((train["feat"][0]["label"] == 1).sum()),
            },
            indent=2,
            default=float,
        )
    )
    print(f"fold {fold} done in {time.time() - started:.0f} s", flush=True)


def _valid_recall(*, table: pl.DataFrame, flagged: np.ndarray, events: pl.DataFrame) -> float:
    """Share of validation-world events found, over all validation worlds."""
    total, found = 0, 0
    for w in events["world"].unique().to_list():
        mask = (table["world"] == w).to_numpy()
        hits = injected_hits(
            table=table.filter(pl.Series(mask)),
            flagged=flagged[mask],
            events=events.filter(pl.col("world") == w),
        )
        total += hits.height
        found += int(hits["found"].sum())
    return found / max(total, 1)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["0", "1", "2"]:
        main(fold=int(f))
