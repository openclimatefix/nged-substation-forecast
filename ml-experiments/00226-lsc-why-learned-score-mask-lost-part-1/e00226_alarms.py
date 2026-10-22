# ruff: noqa: E501
"""Experiment 195's learned-score alarms as interval tables for the masked-forecast harness.

An alarm of experiment 195 (`alarms_f<k>.parquet`) holds a step stamp and an estimated end. This
module turns each fold's alarms of one arm into the interval table that `e00226_nra.detected_spans`
reads: the onset is the alarm's step stamp, the alarm is known `lag` half-hours after the step (the
reach of the look-ahead features the arm reads), and the end is known at the later of experiment
147's stored live rule (when it raised the same alarm) and `END_LAG_UNMATCHED` after the estimated
end (a lab shortcut for alarms experiment 147 did not raise).
"""

from __future__ import annotations

from typing import Final

import e00226_events as ev
import numpy as np
import polars as pl
from e00226_common import E195_DIR, WORK_DIR
from studies.sources import ML_EXPERIMENTS_DATA_DIR

N_FOLDS: Final[int] = 3
END_LAG_UNMATCHED: Final[int] = 96
"""Half-hours (2 days) after the estimated end at which an unmatched alarm's end is known; the
median gap between experiment 147's end and its live `known_from`."""
REAL_RESULTS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00130-ro-nc-detector-on-the-gam-k-residual"
    / "B"
    / "real_results.parquet"
)
"""Experiment 130's labelled real transfers, whose `series` field names the Primary pair."""
ARMS: Final[dict[str, tuple[str, int]]] = {
    "LEARNED_R": ("R re-ranker", 192),
    "LEARNED_F": ("F logistic", 96),
    "RONC_195": ("RO-NC", 96),
    "LEARNED_R96": ("R re-ranker", 96),
    "U1": ("R re-ranker", 192),
    "U2": ("R re-ranker", 192),
    "U3": ("R re-ranker", 192),
    "U4": ("R re-ranker", 192),
    "L_BOTH": ("R re-ranker", 192),
    "L_ONLY": ("R re-ranker", 192),
    "L_ONLY_NOBAL": ("R re-ranker", 192),
}
"""Arm name -> (experiment 195 arm name, half-hours from the step to the alarm being known).
LEARNED_R96 is post hoc: the re-ranker as if its 2 extra days of look-ahead were available.

U4 is post hoc (pair-alarm onset and learned-score end). The seven arms below LEARNED_R96 are this experiment's, all built from the re-ranker's alarms (see
`learned_tables`): U1, U2, and U3 are the combined masks; L_BOTH, L_ONLY, and L_ONLY_NOBAL split the
re-ranker's alarms for the harm decomposition."""
BALANCE_GATE: Final[float] = 0.7
"""Experiment 166's gate, applied to the candidate's folded opposite-step ratio (U2)."""
PAIR_MATCH: Final[int] = 48
"""A learned alarm is also raised by the pair alarm when a pair alarm on the same series has its
step within 24 hours (48 half-hours) of the learned alarm's step."""
BALANCED_LOW: Final[float] = 0.4
BALANCED_HIGH: Final[float] = 2.5
"""Experiment 195's balanced-opposite band: an opposite step of 0.4 to 2.5 times the own step."""
END_RULE_148: Final[int] = 0
"""U3 adds nothing to the estimated end; `e00226_nra.END_LAG` (60 half-hours) alone sets the end."""


def real_transfers() -> pl.DataFrame:
    """The labelled real transfers: `card_id`, `hours`, and `pair` (the Primary pair key)."""
    return (
        pl.read_parquet(REAL_RESULTS)
        .filter(pl.col("setting") == "XL")
        .select("card_id", "hours", pair=pl.col("series"))
    )


def pair_folds() -> dict[str, int]:
    """Experiment 195's assignment of each Primary pair to one of three folds."""
    table = real_transfers()
    weight = (
        table.group_by("pair")
        .agg(load=(pl.col("hours") > 24).sum() + 0.1 * (pl.col("hours") <= 24).sum())
        .sort(["load", "pair"], descending=[True, False])
    )
    totals = [0.0] * N_FOLDS
    out: dict[str, int] = {}
    for pair, load in weight.iter_rows():
        fold = int(np.argmin(totals))
        out[pair] = fold
        totals[fold] += load
    return out


def card_folds() -> dict[str, int]:
    """The fold that holds out each labelled transfer's Primary pair, by card id."""
    folds = pair_folds()
    return {r["card_id"]: folds[r["pair"]] for r in real_transfers().iter_rows(named=True)}


def _learned_tables(*, name: str, lag: int) -> dict[int, pl.DataFrame]:
    """Each fold's interval table for one of experiment 195's arms, with the alarm's ratios."""
    candidates = pl.read_parquet(E195_DIR / "real_candidates.parquet").select(
        "series", "stamp", "source", "sign", "step_mw24"
    )
    live = (
        pl.read_parquet(
            WORK_DIR.parent / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
        )
        .filter(pl.col("setting") == "R0_M48")
        .group_by("series", "onset")
        .agg(live_known=pl.col("known_from").min())
        .rename({"onset": "stamp"})
    )
    out = {}
    for fold in range(N_FOLDS):
        alarms = (
            pl.read_parquet(E195_DIR / f"alarms_f{fold}.parquet")
            .filter(pl.col("arm") == name)
            .unique(["series", "stamp", "source"])
            .join(candidates, on=["series", "stamp", "source"], how="left", validate="1:1")
            .join(live, on=["series", "stamp"], how="left")
        )
        table = alarms.select(
            "series",
            pl.col("series").alias("alarm_series"),
            pl.lit("alarm").alias("role"),
            pl.lit("R0_M48").alias("setting"),
            onset=pl.col("stamp"),
            end=pl.col("end_est"),
            step_mw=pl.col("sign") * pl.col("step_mw24"),
            opp_ratio=pl.col("opp_ratio"),
            n_opposite=pl.col("n_opposite"),
            onset_known=pl.col("stamp") + lag,
            known_from=pl.max_horizontal(
                pl.col("live_known").fill_null(pl.col("end_est") + END_LAG_UNMATCHED),
                pl.col("end_est"),
            ),
        ).drop_nulls("step_mw")
        out[fold] = ev.union_intervals(ronc=table, pair=table.clear())
    return out


def folded_ratio(*, ratio: pl.Expr) -> pl.Expr:
    """The balance score of an opposite-step ratio: the smaller of the ratio and its inverse."""
    return pl.when(ratio > 0).then(pl.min_horizontal(ratio, 1.0 / ratio)).otherwise(0.0)


def _also_pair(*, table: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    """`table` with `also_pair`: a pair alarm on the series has its step near the alarm's step."""
    keys = pair.select("series", pair_onset=pl.col("onset")).unique()
    hit = (
        table.select("series", "onset")
        .unique()
        .join(keys, on="series")
        .filter((pl.col("pair_onset") - pl.col("onset")).abs() <= PAIR_MATCH)
        .select("series", "onset")
        .unique()
        .with_columns(also_pair=pl.lit(True))
    )
    return table.join(hit, on=["series", "onset"], how="left").with_columns(
        pl.col("also_pair").fill_null(False)
    )


def _pair_onset_learned_end(*, learned: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    """Pair-alarm intervals whose end is the matching learned alarm's end; learned-only alarms kept.

    A learned alarm matches a pair interval on the same series when the steps are within
    `PAIR_MATCH` half-hours. The result takes the pair alarm's onset, onset-known time, and megawatts
    and the learned alarm's end and end-known time, so the mask opens as early as the pair alarm
    and closes where the learned score says the load shift ends. Pair intervals with no match are
    kept as they are.
    """
    cols = [c for c in learned.columns if c in pair.columns]
    pair_rows = pair.select(*cols).with_row_index("pair_id")
    ends = (
        pair_rows.select("pair_id", "series", pair_onset=pl.col("onset"))
        .join(
            learned.select(
                "series",
                learned_onset=pl.col("onset"),
                learned_end=pl.col("end"),
                learned_known=pl.col("known_from"),
            ),
            on="series",
        )
        .filter((pl.col("pair_onset") - pl.col("learned_onset")).abs() <= PAIR_MATCH)
        .sort("pair_id", "learned_onset")
        .group_by("pair_id")
        .agg(end_learned=pl.col("learned_end").max(), known_learned=pl.col("learned_known").max())
    )
    merged = pair_rows.join(ends, on="pair_id", how="left").with_columns(
        end=pl.coalesce("end_learned", "end"),
        known_from=pl.coalesce("known_learned", "known_from"),
    )
    keys = (
        pair.select("series", pair_onset=pl.col("onset"))
        .join(learned.select("series", learned_onset=pl.col("onset")), on="series")
        .filter((pl.col("pair_onset") - pl.col("learned_onset")).abs() <= PAIR_MATCH)
        .select("series", onset=pl.col("learned_onset"))
        .unique()
        .with_columns(has_pair=pl.lit(True))
    )
    only = (
        learned.join(keys, on=["series", "onset"], how="left")
        .filter(pl.col("has_pair").is_null())
        .drop("has_pair")
    )
    return pl.concat(
        [merged.drop("pair_id", "end_learned", "known_learned"), only.select(*cols)],
        how="diagonal_relaxed",
    )


def learned_tables(*, arm: str) -> dict[int, pl.DataFrame]:
    """Each fold's interval table for one of `ARMS`, in the layout of experiment 147's intervals.

    U1: the pair alarm's intervals and the re-ranker's, merged. U2: the re-ranker's alarms whose
    folded opposite-step ratio is at least `BALANCE_GATE`. U3: the re-ranker's alarms with the end
    known by experiment 148's rule alone. L_BOTH, L_ONLY: the re-ranker's alarms that a pair alarm
    also raises, and the rest. L_ONLY_NOBAL: L_ONLY without the balanced-opposite candidates.
    """
    name, lag = ARMS[arm]
    base = _learned_tables(name=name, lag=lag)
    if arm not in ("U1", "U2", "U3", "U4", "L_BOTH", "L_ONLY", "L_ONLY_NOBAL"):
        return {fold: table.drop("opp_ratio", "n_opposite") for fold, table in base.items()}
    pair = ev.alarm_tables()["NEW"]
    out = {}
    for fold, table in base.items():
        if arm == "U1":
            out[fold] = ev.union_intervals(
                ronc=table.drop("opp_ratio", "n_opposite"),
                pair=pair.select(*[c for c in table.columns if c in pair.columns]),
            )
            continue
        if arm == "U4":
            out[fold] = _pair_onset_learned_end(
                learned=table.drop("opp_ratio", "n_opposite"), pair=pair
            )
            continue
        if arm == "U2":
            kept = table.filter(folded_ratio(ratio=pl.col("opp_ratio")) >= BALANCE_GATE)
        elif arm == "U3":
            kept = table.with_columns(known_from=pl.col("end") + END_RULE_148)
        else:
            marked = _also_pair(table=table, pair=pair)
            balanced = (
                (pl.col("n_opposite") >= 1)
                & (pl.col("opp_ratio") >= BALANCED_LOW)
                & (pl.col("opp_ratio") <= BALANCED_HIGH)
            )
            kept = {
                "L_BOTH": marked.filter(pl.col("also_pair")),
                "L_ONLY": marked.filter(~pl.col("also_pair")),
                "L_ONLY_NOBAL": marked.filter(~pl.col("also_pair") & ~balanced),
            }[arm].drop("also_pair")
        out[fold] = kept.drop("opp_ratio", "n_opposite")
    return out
