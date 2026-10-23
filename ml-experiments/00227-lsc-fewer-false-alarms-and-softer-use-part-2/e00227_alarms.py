# ruff: noqa: E501
"""Alarm sources at several thresholds, as interval tables for the masked-forecast harness.

A source id names a source and a threshold: `LR_b<budget>` (experiment 195's re-ranker, alarms
known 192 half-hours after the step), `LR148_b<budget>` (the same alarms, end known by experiment
148's rule alone), `RO_b<budget>` (RO-NC, known 96 half-hours after the step), `PG_g<gate>` (the
pair alarm of experiment 156 kept where its balance score is at least gate/100), and `UR_h<budget>`
(the re-ranker at budget 80 kept only where a pair alarm also fires or its score passes the
threshold of budget `h`). A budget is the number of unexplained odd-month candidates left above
the threshold (experiment 195's way of setting a threshold); a learned source has one threshold per
fold, because its score comes from a model trained without that fold's Primary pairs.
"""

from __future__ import annotations

from typing import Final

import e00227_events as ev
import numpy as np
import polars as pl
from e00227_common import E195_DIR, WORK_DIR
from e00227_s_common import pair_folds, real_transfers

N_FOLDS: Final[int] = 3

LAG_R: Final[int] = 192
LAG_RONC: Final[int] = 96
END_LAG_UNMATCHED: Final[int] = 96
PAIR_MATCH: Final[int] = 48
"""A learned alarm is also raised by the pair alarm when a pair alarm on the same series has its
step within 24 hours (48 half-hours) of the learned alarm's step."""
BUDGETS: Final[tuple[int, ...]] = (5, 10, 20, 40, 80, 160, 320)
BASE_BUDGET: Final[int] = 80
"""Experiment 195's threshold: 80 unexplained odd-month alarms."""
GATES: Final[tuple[int, ...]] = (40, 55, 70, 80, 90)
UR_BUDGETS: Final[tuple[int, ...]] = (5, 10, 20, 40)
SCORE_ARM: Final[dict[str, str]] = {"LR": "R re-ranker", "LR148": "R re-ranker", "RO": "RO-NC"}


def source_ids() -> list[str]:
    """Every source and threshold of part (a), in a fixed order."""
    return [
        *[f"LR_b{b}" for b in BUDGETS],
        *[f"LR148_b{b}" for b in BUDGETS],
        *[f"RO_b{b}" for b in BUDGETS],
        *[f"PG_g{g}" for g in GATES],
        *[f"UR_h{h}" for h in UR_BUDGETS],
    ]


def is_foldless(source: str) -> bool:
    """Sources with one table for every fold: the pair alarm and RO-NC."""
    return source.startswith(("PG_", "RO_"))


def _scores(*, fold: int) -> pl.DataFrame:
    return pl.read_parquet(WORK_DIR / f"scores_f{fold}.parquet")


def threshold(*, scores: pl.DataFrame, arm: str, budget: int) -> float:
    """Score above which `budget` unexplained odd-month candidates remain (experiment 195's rule)."""
    s = scores.filter((pl.col("arm") == arm) & pl.col("odd") & ~pl.col("explained"))["score"]
    pool = s.to_numpy()
    pool = np.sort(pool[np.isfinite(pool)])[::-1]
    return float(pool[budget]) if len(pool) > budget else -np.inf


def _live() -> pl.DataFrame:
    return (
        pl.read_parquet(ev.INTERVALS_RONC)
        .filter(pl.col("setting") == "R0_M48")
        .group_by("series", "onset")
        .agg(live_known=pl.col("known_from").min())
        .rename({"onset": "stamp"})
    )


def _alarm_frame(*, fold: int, arm: str, thr: float) -> pl.DataFrame:
    """Candidates whose score is above `thr`, with step megawatts, ratios, and the live end stamp."""
    cands = pl.read_parquet(E195_DIR / "real_candidates.parquet").select(
        "series", "stamp", "source", "sign", "step_mw24", "opp_ratio", "n_opposite"
    )
    scores = _scores(fold=fold).filter(pl.col("arm") == arm)
    flagged = scores.filter(pl.col("score").is_finite() & (pl.col("score") > thr))
    return flagged.join(cands, on=["series", "stamp", "source"], how="left", validate="1:1").join(
        _live(), on=["series", "stamp"], how="left"
    )


def _interval_table(*, alarms: pl.DataFrame, lag: int, end_rule_148: bool) -> pl.DataFrame:
    """Alarm frame to the interval table `detected_spans` reads, merging alarms within 24 hours."""
    table = alarms.select(
        "series",
        pl.col("series").alias("alarm_series"),
        pl.lit("alarm").alias("role"),
        pl.lit("R0_M48").alias("setting"),
        onset=pl.col("stamp"),
        end=pl.col("end_est"),
        step_mw=pl.col("sign") * pl.col("step_mw24"),
        onset_known=pl.col("stamp") + lag,
        known_from=pl.col("end_est")
        if end_rule_148
        else pl.max_horizontal(
            pl.col("live_known").fill_null(pl.col("end_est") + END_LAG_UNMATCHED),
            pl.col("end_est"),
        ),
    ).drop_nulls("step_mw")
    return ev.union_intervals(ronc=table, pair=table.clear())


def _also_pair(*, alarms: pl.DataFrame, pair: pl.DataFrame) -> pl.Series:
    keys = pair.select("series", pair_onset=pl.col("onset")).unique()
    hit = (
        alarms.select("series", "stamp")
        .unique()
        .join(keys, on="series")
        .filter((pl.col("pair_onset") - pl.col("stamp")).abs() <= PAIR_MATCH)
        .select("series", "stamp")
        .unique()
        .with_columns(also_pair=pl.lit(True))
    )
    return alarms.join(hit, on=["series", "stamp"], how="left")["also_pair"].fill_null(False)


def tables(*, source: str) -> dict[int, pl.DataFrame]:
    """Each fold's interval table for one source id (a foldless source repeats its one table)."""
    kind, value = source.split("_")
    number = int(value[1:])
    pair_tables = ev.alarm_tables()
    if kind == "PG":
        gated = pair_tables["NEW"].filter(pl.col("balance_score") >= number / 100.0)
        return dict.fromkeys(range(N_FOLDS), gated)
    out = {}
    for fold in range(N_FOLDS):
        scores = _scores(fold=fold)
        if kind == "RO":
            thr = threshold(scores=scores, arm="RO-NC", budget=number)
            alarms = _alarm_frame(fold=0, arm="RO-NC", thr=thr)
            out[fold] = _interval_table(alarms=alarms, lag=LAG_RONC, end_rule_148=False)
        elif kind in ("LR", "LR148"):
            thr = threshold(scores=scores, arm="R re-ranker", budget=number)
            alarms = _alarm_frame(fold=fold, arm="R re-ranker", thr=thr)
            out[fold] = _interval_table(alarms=alarms, lag=LAG_R, end_rule_148=kind == "LR148")
        elif kind == "UR":
            base = threshold(scores=scores, arm="R re-ranker", budget=BASE_BUDGET)
            high = threshold(scores=scores, arm="R re-ranker", budget=number)
            alarms = _alarm_frame(fold=fold, arm="R re-ranker", thr=base)
            keep = _also_pair(alarms=alarms, pair=pair_tables["NEW"]) | (
                pl.Series(alarms["score"]) > high
            )
            out[fold] = _interval_table(alarms=alarms.filter(keep), lag=LAG_R, end_rule_148=False)
        else:
            raise ValueError(source)
    return out


def card_folds() -> dict[str, int]:
    """The fold that holds out each labelled load shift's Primary pair, by card id."""
    folds = pair_folds()
    table = real_transfers()
    return {r["card_id"]: folds[r["pair"]] for r in table.iter_rows(named=True)}
