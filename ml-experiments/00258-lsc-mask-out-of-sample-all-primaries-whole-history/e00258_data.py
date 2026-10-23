# ruff: noqa: E501
"""Alarm tables of experiment 227's frozen sources, the logged closures, and the labelled load shifts.

This module copies the loaders of experiments 195 and 227 so that experiment 258 imports only its own
folder. Nothing here writes outside the experiment's private folder, and nothing is tuned: a source
id such as `LR_b20` names the same threshold as in experiment 227.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK_DIR: Final[Path] = DATA / "00258-lsc-mask-out-of-sample-all-primaries-whole-history"
"""Private tables; never committed."""
E195_DIR: Final[Path] = DATA / "00195-switching-neural-plus-detector-features"
E227_DIR: Final[Path] = DATA / "00227-lsc-fewer-false-alarms-and-softer-use-part-2"
E156_DIR: Final[Path] = DATA / "00156-switching-faster-onset-alarm-from-opposite-steps"
E171_DIR: Final[Path] = DATA / "00171-switching-gated-mask-no-harm-pooled-months"
INTERVALS_RONC: Final[Path] = (
    DATA / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
REAL_RESULTS: Final[Path] = (
    DATA / "00130-ro-nc-detector-on-the-gam-k-residual" / "B" / "real_results.parquet"
)
LOGGED_PATH: Final[Path] = DATA / "00037-switching-first-look" / "intervals.parquet"
D8_DROPS: Final[Path] = (
    DATA / "00159-switching-d8-multi-day-drops-recurrence-and-report" / "drops.parquet"
)
E174_DAYS: Final[Path] = (
    DATA
    / "00174-dp-unlabelled-worst-days-weather-shared-or-single-site"
    / "work"
    / "worst_days_split.parquet"
)
E175_SHAPES: Final[Path] = DATA / "00175-dp-unlabelled-day-shapes" / "work" / "day_shapes.parquet"
BG2_PREDICTIONS: Final[Path] = (
    DATA
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
    / "p_s6_bg2"
    / "predictions.parquet"
)

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = int((GRID_END - GRID_START).total_seconds() // 1800)
DAY: Final[int] = 48
WEEK: Final[int] = 7 * DAY
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
ADJACENCY: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}
SITE_LABEL: Final[dict[int, str]] = {
    26: "D8",
    7: "D9",
    10: "D10",
    8: "D11",
    16: "D14",
    17: "D5",
    18: "D16",
}
"""The anonymised labels of the Primary substations with a published label; the others stay numbered."""
N_FOLDS: Final[int] = 3
LAG_R: Final[int] = 192
LAG_RONC: Final[int] = 96
END_LAG_UNMATCHED: Final[int] = 96
MERGE: Final[int] = 48
PAIR_MATCH: Final[int] = 48
ONSET_TOLERANCE: Final[int] = 4
END_MINIMUM: Final[int] = 12
END_FRACTION: Final[float] = 0.10
BLOCK_TOLERANCE: Final[int] = 6
E175_GRID_FIRST: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)
BALANCE_FLOOR: Final[float] = 0.4
SOURCES: Final[tuple[str, ...]] = ("LR_b20", "LR_b80", "RO_b80", "PG_g70")
"""The frozen sources: experiment 227's headline learned source, its high-budget twin, RO-NC, and the gated pair alarm."""


SERIES_TABLE: Final[Path] = (
    DATA.parent / "studies" / "per_study" / "stage1_pilot" / "demand" / "inputs" / "series.parquet"
)
"""The stage 1 pilot's series table, which carries every series' anonymised site label."""


@lru_cache(maxsize=1)
def site_labels() -> dict[int, str]:
    """The anonymised site label (such as D8) of every series in the series table."""
    table = pl.read_parquet(SERIES_TABLE)
    return {int(r["time_series_id"]): str(r["site"]) for r in table.iter_rows(named=True)}


def label_of(*, series: int) -> str:
    """The anonymised label of a Primary substation, from the series table."""
    return site_labels().get(series, f"series {series}")


def end_tolerance(*, onset: int, end: int) -> float:
    """The larger of 6 hours and 10% of the duration, in half-hours (experiment 195's end rule)."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def stamp_year_month() -> np.ndarray:
    """A year-and-month label (such as 202203) for every grid stamp, for resampling whole months."""
    index = pl.datetime_range(
        GRID_START, GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )
    return (index.dt.year() * 100 + index.dt.month()).to_numpy()


def labelled_loads() -> pl.DataFrame:
    """The 35 labelled real load shifts lasting over 1 day: card_id, onset, end, hours, pair, series."""
    return (
        pl.read_parquet(REAL_RESULTS)
        .filter((pl.col("setting") == "XL") & (pl.col("hours") > 24))
        .select("card_id", "onset", "end", "hours", pair=pl.col("series"))
        .with_columns(series=pl.col("pair").str.split(",").list.eval(pl.element().cast(pl.Int64)))
    )


def pair_folds() -> dict[str, int]:
    """Assign each Primary pair of the 66 labelled real transfers to one of three folds (experiment 227's rule)."""
    table = (
        pl.read_parquet(REAL_RESULTS)
        .filter(pl.col("setting") == "XL")
        .select("hours", pair=pl.col("series"))
    )
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


def series_folds() -> dict[int, list[int]]:
    """For each series, the folds that hold out a labelled Primary pair containing it (all three if none)."""
    out: dict[int, set[int]] = {s: set() for s in SERIES_IDS}
    for pair, fold in pair_folds().items():
        for s in (int(x) for x in pair.split(",")):
            if s in out:
                out[s].add(fold)
    return {s: sorted(f) if f else list(range(N_FOLDS)) for s, f in out.items()}


def logged_closures() -> pl.DataFrame:
    """Logged closures of 2 hours to 60 days (experiment 37's log), with their start and end stamps and sides."""
    return pl.read_parquet(LOGGED_PATH).filter((pl.col("hours") >= 2) & (pl.col("hours") <= 1440))


def closure_edges() -> dict[int, np.ndarray]:
    """For each series, the sorted start and end stamps of logged closures on the series or a connected neighbour."""
    logged = logged_closures()
    per_series: dict[int, list[int]] = {s: [] for s in SERIES_IDS}
    for row in logged.iter_rows(named=True):
        sides = {int(x) for x in (row["side_a"], row["side_b"]) if x is not None and x.isdigit()}
        for s in SERIES_IDS:
            if s in sides or sides & set(ADJACENCY[s]):
                per_series[s] += [int(row["first_stamp"]), int(row["last_stamp_exclusive"])]
    return {s: np.sort(np.array(v, dtype=np.int64)) for s, v in per_series.items()}


# --- alarm tables (copies of experiment 227's e00227_alarms and e00227_events) ---


def union_intervals(*, ronc: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    """Intervals from either alarm; on one series, those starting within 24 hours merge."""
    both = pl.concat([ronc, pair], how="diagonal_relaxed").sort(["series", "onset"])
    out: list[dict] = []
    for row in both.iter_rows(named=True):
        last = next(
            (
                o
                for o in reversed(out)
                if o["series"] == row["series"] and row["onset"] - o["onset"] <= MERGE
            ),
            None,
        )
        if last is None:
            out.append(dict(row))
            continue
        if row["onset_known"] < last["onset_known"]:
            last["onset_known"], last["onset"] = row["onset_known"], row["onset"]
            last["step_mw"] = row["step_mw"]
        last["end"] = max(last["end"], row["end"])
        last["known_from"] = max(last["known_from"], row["known_from"])
    return pl.DataFrame(out, schema=both.schema)


def _threshold(*, scores: pl.DataFrame, arm: str, budget: int) -> float:
    """Score above which `budget` unexplained odd-month candidates remain (experiment 195's rule)."""
    s = scores.filter((pl.col("arm") == arm) & pl.col("odd") & ~pl.col("explained"))[
        "score"
    ].to_numpy()
    pool = np.sort(s[np.isfinite(s)])[::-1]
    return float(pool[budget]) if len(pool) > budget else -np.inf


def _live() -> pl.DataFrame:
    return (
        pl.read_parquet(INTERVALS_RONC)
        .filter(pl.col("setting") == "R0_M48")
        .group_by("series", "onset")
        .agg(live_known=pl.col("known_from").min())
        .rename({"onset": "stamp"})
    )


def _alarm_frame(*, fold: int, arm: str, thr: float) -> pl.DataFrame:
    cands = pl.read_parquet(E195_DIR / "real_candidates.parquet").select(
        "series", "stamp", "source", "sign", "step_mw24", "opp_ratio", "n_opposite"
    )
    scores = pl.read_parquet(E227_DIR / f"scores_f{fold}.parquet").filter(pl.col("arm") == arm)
    flagged = scores.filter(pl.col("score").is_finite() & (pl.col("score") > thr))
    return flagged.join(cands, on=["series", "stamp", "source"], how="left", validate="1:1").join(
        _live(), on=["series", "stamp"], how="left"
    )


def _interval_table(*, alarms: pl.DataFrame, lag: int) -> pl.DataFrame:
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
            pl.col("live_known").fill_null(pl.col("end_est") + END_LAG_UNMATCHED), pl.col("end_est")
        ),
        opp_ratio=pl.col("opp_ratio"),
        n_opposite=pl.col("n_opposite"),
    ).drop_nulls("step_mw")
    return union_intervals(ronc=table, pair=table.clear())


def _pair_table() -> pl.DataFrame:
    window = int(json.loads((E156_DIR / "detect" / "headline.json").read_text())["window"])
    pair = pl.read_parquet(E156_DIR / "detect" / f"real_W{window}_pair.parquet")
    balance = pl.read_parquet(E171_DIR / "balance.parquet").select(
        alarm_series=pl.col("series"), onset=pl.col("stamp"), balance_score=pl.col("balance_score")
    )
    return pair.join(balance, on=["alarm_series", "onset"], how="left").with_columns(
        pl.col("balance_score").fill_null(0.0)
    )


def alarm_tables(*, source: str) -> dict[int, pl.DataFrame]:
    """Each fold's interval table for one source id (a foldless source repeats its table).

    Columns include `series`, `onset` (the step stamp), `end`, `opp_ratio` and `n_opposite`
    (learned and RO-NC sources), and `balance_score` (the pair alarm).
    """
    kind, value = source.split("_")
    number = int(value[1:])
    if kind == "PG":
        pair = _pair_table().filter(
            (pl.col("balance_score") >= number / 100.0) & (pl.col("role") == "alarm")
        )
        return dict.fromkeys(range(N_FOLDS), pair)
    out = {}
    for fold in range(N_FOLDS):
        scores = pl.read_parquet(E227_DIR / f"scores_f{fold}.parquet")
        if kind == "RO":
            thr = _threshold(scores=scores, arm="RO-NC", budget=number)
            out[fold] = _interval_table(
                alarms=_alarm_frame(fold=0, arm="RO-NC", thr=thr), lag=LAG_RONC
            )
        elif kind == "LR":
            thr = _threshold(scores=scores, arm="R re-ranker", budget=number)
            out[fold] = _interval_table(
                alarms=_alarm_frame(fold=fold, arm="R re-ranker", thr=thr), lag=LAG_R
            )
        else:
            raise ValueError(source)
    return out


# --- side tables for the classes ---


def d8_drop_edges() -> np.ndarray:
    """Stamps of the first and last half-hour of each of D8's multi-day drops (experiment 159)."""
    drops = pl.read_parquet(D8_DROPS)
    return np.sort(
        np.concatenate([drops["first_stamp"].to_numpy(), drops["last_stamp"].to_numpy()])
    )


def block_day_ranges() -> dict[int, list[tuple[int, int]]]:
    """The 17 unlabelled block days of experiment 175, as (first, last) stamps per series."""
    unlabelled = pl.read_parquet(E174_DAYS).filter(~pl.col("labelled")).select("site", "local_date")
    days = (
        pl.read_parquet(E175_SHAPES)
        .join(unlabelled, on=["site", "local_date"])
        .filter(pl.col("shape") == "a_block")
    )
    pairs = pl.read_parquet(BG2_PREDICTIONS, columns=["site", "time_series_id"]).unique()
    series = {r["site"]: int(r["time_series_id"]) for r in pairs.iter_rows(named=True)}
    start = (
        pl.col("local_date")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("Europe/London")
        .dt.convert_time_zone("UTC")
    )
    days = days.with_columns(
        series=pl.col("site").replace_strict(series, return_dtype=pl.Int64),
        first=((start.dt.epoch("s") - int(GRID_START.timestamp())) // 1800).cast(pl.Int64),
    )
    out: dict[int, list[tuple[int, int]]] = {}
    for r in days.iter_rows(named=True):
        out.setdefault(int(r["series"]), []).append((int(r["first"]), int(r["first"]) + DAY - 1))
    return out


def calendar_flags() -> np.ndarray:
    """True on every stamp whose local date is an England bank holiday or lies within a day of a UK clock change."""
    import holidays

    index = pl.datetime_range(
        GRID_START, GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )
    local = index.dt.convert_time_zone("Europe/London")
    dates = local.dt.date().to_numpy()
    cal = holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028))
    days = {np.datetime64(d) for d in cal}
    offset = local.dt.dst_offset().dt.total_seconds().to_numpy()
    change = np.flatnonzero(np.diff(offset) != 0) + 1
    flags = np.array([np.datetime64(d) in days for d in dates], dtype=bool)
    for c in change:
        flags[max(c - DAY, 0) : c + DAY] = True
    return flags
