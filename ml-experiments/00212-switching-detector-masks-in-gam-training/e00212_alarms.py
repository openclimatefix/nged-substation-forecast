"""Experiment 195's learned-score alarms as interval tables for the masked-forecast harness.

An alarm of experiment 195 (`alarms_f<k>.parquet`) holds a step stamp and an estimated end. This
module turns each fold's alarms of one arm into the interval table that `e00212_nra.detected_spans`
reads: the onset is the alarm's step stamp, the alarm is known `lag` half-hours after the step (the
reach of the look-ahead features the arm reads), and the end is known at the later of experiment
147's stored live rule (when it raised the same alarm) and `END_LAG_UNMATCHED` after the estimated
end (a lab shortcut for alarms experiment 147 did not raise).
"""

from __future__ import annotations

from typing import Final

import e00212_events as ev
import numpy as np
import polars as pl
from e00212_common import E195_DIR, WORK_DIR
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
}
"""Arm name -> (experiment 195 arm name, half-hours from the step to the alarm being known).
LEARNED_R96 is post hoc: the re-ranker as if its 2 extra days of look-ahead were available."""


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


def learned_tables(*, arm: str) -> dict[int, pl.DataFrame]:
    """Each fold's interval table for one of `ARMS`, in the layout of experiment 147's intervals."""
    name, lag = ARMS[arm]
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
            onset_known=pl.col("stamp") + lag,
            known_from=pl.max_horizontal(
                pl.col("live_known").fill_null(pl.col("end_est") + END_LAG_UNMATCHED),
                pl.col("end_est"),
            ),
        ).drop_nulls("step_mw")
        out[fold] = ev.union_intervals(ronc=table, pair=table.clear())
    return out
