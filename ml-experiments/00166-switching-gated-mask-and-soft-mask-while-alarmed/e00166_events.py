"""The logged real transfers at the six decision substations, their alarms, and the row sets.

A logged transfer contributes one event for each decision substation whose series it involves. An
event carries its logged onset and end, and the stamps at which the pair alarm and RO-NC (the
refit-offset neighbour-count detector) would each have been known, all as positions on the
substation's grid.
"""

from __future__ import annotations

import json
from typing import Final

import e00166_fed as fed
import e00166_nra as nra
import numpy as np
import polars as pl
from e00166_common import E156_DIR, GRID_START, LABELLED_PATH, WORK_DIR

INTERVALS_RONC: Final = (
    WORK_DIR.parent / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
ONSET_LAG_RONC: Final[int] = 96
"""Half-hours from a step to the stamp its RO-NC alarm is known (experiment 156)."""
MERGE: Final[int] = 48
"""Alarms on one series that start within 24 hours merge into one interval."""
MATCH_BEFORE: Final[int] = 48
"""An alarm matches a logged transfer when its step lies from 24 hours before the logged onset to
the logged end (experiment 156's rule for a closure that explains an alarm)."""
AFTER_STAMPS: Final[int] = 144
"""72 hours of target time after the logged end."""
BALANCE_GATE: Final[float] = 0.7
"""Pre-registered before any forecast score was read: the smallest balance score at which the
pair alarm masks (the two opposite steps agree within a factor of 1 / 0.7, about 1.43)."""
BALANCE_FLOOR: Final[float] = 0.4
"""The smallest balance score the pair alarm can have (experiment 156 keeps ratios 0.4 to 2.5)."""
SOFT_FACTORS: Final[tuple[float, ...]] = (10.0, 100.0, 1000.0)
"""Pre-registered observation-noise variance factors of the soft mask, at full reliability."""
POST_HOC_SOFT_FACTORS: Final[tuple[float, ...]] = (1e4, 1e5, 1e6)
"""Stronger inflation factors added after the pre-registered three did almost nothing (post hoc)."""
SCORED_FIRST: Final = GRID_START.replace(year=2025, month=7, day=1)


def stamp(when: object) -> int:
    """The detector stamp of a UTC time."""
    return int((when - GRID_START).total_seconds() // 1800)  # ty: ignore[unsupported-operator]


def with_ronc_onset_known(*, intervals: pl.DataFrame) -> pl.DataFrame:
    """RO-NC intervals with the stamp their onset is known: the step plus `ONSET_LAG_RONC`."""
    return intervals.with_columns(onset_known=pl.col("onset") + ONSET_LAG_RONC)


def union_intervals(*, ronc: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    """Intervals from either alarm; on one series, those starting within 24 hours merge.

    A merged interval keeps the earlier `onset_known` and `onset`, and the later `end` and
    `known_from`, so the union is known as early as either alarm and ends when both have ended.
    """
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


def with_balance(*, intervals: pl.DataFrame, balance: pl.DataFrame) -> pl.DataFrame:
    """Pair-alarm intervals with the balance score of the alarm that opened each interval.

    An interval's alarm is the pair alarm at (`alarm_series`, `onset`). An interval with no stored
    alarm (a merged interval whose first onset differs) gets balance score 0, so it never passes
    the gate. `reliability_weight` runs from 0 at the lowest possible score to 1 at a perfect
    balance.
    """
    keyed = balance.select(
        alarm_series=pl.col("series"), onset=pl.col("stamp"), balance_score=pl.col("balance_score")
    )
    return (
        intervals.join(keyed, on=["alarm_series", "onset"], how="left")
        .with_columns(pl.col("balance_score").fill_null(0.0))
        .with_columns(
            reliability_weight=(
                (pl.col("balance_score") - BALANCE_FLOOR) / (1.0 - BALANCE_FLOOR)
            ).clip(0.0, 1.0)
        )
    )


def alarm_tables() -> dict[str, pl.DataFrame]:
    """The alarm interval tables: RO-NC, the pair alarm, its gated subset, and the union."""
    window = int(json.loads((E156_DIR / "detect" / "headline.json").read_text())["window"])
    ronc = with_ronc_onset_known(
        intervals=pl.read_parquet(INTERVALS_RONC).filter(pl.col("setting") == "R0_M48")
    )
    pair = pl.read_parquet(E156_DIR / "detect" / f"real_W{window}_pair.parquet")
    scored = with_balance(intervals=pair, balance=pl.read_parquet(WORK_DIR / "balance.parquet"))
    return {
        "RONC": ronc,
        "NEW": scored,
        "GATED": scored.filter(pl.col("balance_score") >= BALANCE_GATE),
        "UNION": union_intervals(ronc=ronc, pair=pair),
    }


def logged_transfers() -> pl.DataFrame:
    """Labelled real transfers with their onset and end stamps and the series involved."""
    frame = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    rows = []
    for row in frame.iter_rows(named=True):
        receivers = [int(r["series"]) for r in json.loads(row["receivers"])]
        donor = int(row["donor"]) if row["donor"] not in (None, "") else None
        series = {s for s in (donor, row["side_a"], row["side_b"], *receivers) if s is not None}
        rows.append(
            {
                "card": row["card_id"],
                "onset": stamp(row["onset_utc"]),
                "end": stamp(row["end_utc"]) + 1,
                "series": sorted(int(s) for s in series),
            }
        )
    return pl.DataFrame(rows)


def first_known(*, table: pl.DataFrame, series: int, onset: int, end: int) -> int | None:
    """The earliest `onset_known` of an alarm on `series` whose step matches the logged span."""
    hits = table.filter(
        (pl.col("series") == series)
        & (pl.col("onset") >= onset - MATCH_BEFORE)
        & (pl.col("onset") < end)
    )
    return None if hits.is_empty() else int(hits["onset_known"].min())  # ty: ignore[invalid-argument-type]


def site_events(*, site: str, tables: dict[str, pl.DataFrame], offset: int) -> pl.DataFrame:
    """The events of one substation, with grid positions (stamp plus `offset`).

    Columns: `card`, `site`, `onset`, `end`, `known_new`, `known_ronc`, `known_union`,
    `known_gated` (positions, -1 where the alarm never matches), and `alarm`, the earliest known
    position or `end` when none.
    """
    series = fed.SERIES_OF_SITE[site]
    rows = []
    for row in (
        logged_transfers().filter(pl.col("series").list.contains(series)).iter_rows(named=True)
    ):
        known = {
            name: first_known(table=tables[name], series=series, onset=row["onset"], end=row["end"])
            for name in ("NEW", "RONC", "UNION", "GATED")
        }
        found = [v for v in known.values() if v is not None]
        alarm = min(min(found) + offset, row["end"] + offset) if found else row["end"] + offset
        rows.append(
            {
                "card": row["card"],
                "site": site,
                "onset": row["onset"] + offset,
                "end": row["end"] + offset,
                "known_new": -1 if known["NEW"] is None else known["NEW"] + offset,
                "known_ronc": -1 if known["RONC"] is None else known["RONC"] + offset,
                "known_union": -1 if known["UNION"] is None else known["UNION"] + offset,
                "known_gated": -1 if known["GATED"] is None else known["GATED"] + offset,
                "alarm": alarm,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "card": pl.String,
            "site": pl.String,
            "onset": pl.Int64,
            "end": pl.Int64,
            "known_new": pl.Int64,
            "known_ronc": pl.Int64,
            "known_union": pl.Int64,
            "known_gated": pl.Int64,
            "alarm": pl.Int64,
        },
    )


def oracle_spans(*, events: pl.DataFrame) -> list[nra.Span]:
    """Spans for the oracle mask: the logged interval, known from its onset, end known at once."""
    return [
        nra.Span(r["onset"], r["end"], r["onset"], r["end"], 0.0)
        for r in events.iter_rows(named=True)
    ]


def row_sets(*, data: object, events: pl.DataFrame) -> dict[str, np.ndarray]:
    """Row indices of the BEFORE, INSIDE, and AFTER sets, one array per set and event.

    Returns a dict keyed `"<set>|<card>"`. Rows need a reading, a run of the scored year or of the
    replicate months, and a forecast origin at or after the logged onset. Target time decides the
    set.
    """
    d = data  # type: ignore[assignment]
    origin_pos = np.searchsorted(d.cache.end_times, d.run_origins)[d.run_of_row]  # ty: ignore[unresolved-attribute]
    eligible = np.isin(d.run_period, ["scored", "replicate"])[d.run_of_row]  # ty: ignore[unresolved-attribute]
    ok = np.isfinite(d.actual) & eligible  # ty: ignore[unresolved-attribute]
    target = d.position  # ty: ignore[unresolved-attribute]
    out: dict[str, np.ndarray] = {}
    for r in events.iter_rows(named=True):
        base = ok & (origin_pos >= r["onset"])
        windows = {
            "BEFORE": (r["onset"], r["alarm"]),
            "INSIDE": (r["alarm"], r["end"]),
            "AFTER": (r["end"], r["end"] + AFTER_STAMPS),
        }
        for name, (lo, hi) in windows.items():
            rows = np.flatnonzero(base & (target >= lo) & (target < hi))
            if len(rows):
                out[f"{name}|{r['card']}"] = rows
    return out
