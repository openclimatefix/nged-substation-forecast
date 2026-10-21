"""The opposite-step pair alarm, its intervals, and the detection scoring against injected events.

An alarm fires at a peak of one node's step statistic (at least `k` noise units, the largest within
`window // 2` half-hours either side) when a connected metered node stepped the opposite way by at
least `k / 2` noise units within one half-hour, the two megawatt steps balance, and fewer than two
neighbours stepped the same way. The peak is known `window // 2 + window` half-hours after the
step: the peak test needs `window // 2` more half-hours and the step statistic needs `window`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00156_d_common import ADJACENCY, TUNING_MONTHS, near_log_masks, stamp_months
from e00156_d_detectors import BALANCE_RANGE, DetectorInputs, local_peaks
from e00156_d_endrule import filter_alarms, live_end_intervals
from e00156_d_inject import Event
from e00156_d_intervals import build_intervals
from e00156_d_score import ONSET_TOLERANCE

WINDOWS: Final[tuple[int, ...]] = (4, 8, 12)
"""2, 4, and 6 hours in half-hours."""
K_GRID: Final[tuple[float, ...]] = tuple(2.5 + 0.25 * i for i in range(23))
"""Thresholds 2.5 to 8.0 noise units in steps of 0.25."""
FALSE_ALARM_TARGET: Final[int] = 80
"""RO-NC's real alarms in odd months that no logged closure explains."""
VETO_MIN_NEIGHBOURS: Final[int] = 2
ONSET_LAG_RONC: Final[int] = 96
"""Half-hours from a step to the stamp its RO-NC alarm is known (statistic plus peak test)."""
CAP_DAYS: Final[int] = 14
MERGE: Final[int] = 48
EXPLAIN_PAD: Final[int] = 48
"""An alarm within 24 hours of a logged closure on its series is explained by that closure."""


def known_lag(*, window: int) -> int:
    """Half-hours from the step to the stamp the pair alarm is known."""
    return window // 2 + window


def pair_alarms(*, inputs: DetectorInputs, k: float) -> pl.DataFrame:
    """One row per pair alarm: series, stamp (the step), step_mw, z, partners, and known."""
    window = inputs.window
    half = window // 2
    rows: list[dict] = []
    for series, shift in inputs.shift.items():
        z = shift / inputs.scale[series]
        neighbours = [n for n in ADJACENCY.get(series, ()) if n in inputs.shift]
        for stamp in local_peaks(z=z, threshold=k, window=half):
            own = shift[stamp]
            sign = np.sign(own)
            partners: dict[int, float] = {}
            same = 0
            for n in neighbours:
                segment = inputs.shift[n][max(stamp - 1, 0) : stamp + 2] / inputs.scale[n]
                if not np.isfinite(segment).any():
                    continue
                if (-sign * segment).max() >= k / 2:
                    j = int(np.nanargmax(-sign * segment))
                    partners[n] = float(inputs.shift[n][max(stamp - 1, 0) + j])
                elif (sign * segment).max() >= k / 2:
                    same += 1
            if not partners or same >= min(VETO_MIN_NEIGHBOURS, len(neighbours)):
                continue
            total = float(sum(partners.values()))
            ratio = -total / own
            if not BALANCE_RANGE[0] <= ratio <= BALANCE_RANGE[1]:
                continue
            rows.append(
                {
                    "series": series,
                    "stamp": int(stamp),
                    "step_mw": float(own),
                    "z": float(z[stamp]),
                    "partners": ";".join(f"{n}:{v / total:.2f}" for n, v in partners.items()),
                    "known": int(stamp) + known_lag(window=window),
                }
            )
    schema = {
        "series": pl.Int64,
        "stamp": pl.Int64,
        "step_mw": pl.Float64,
        "z": pl.Float64,
        "partners": pl.String,
        "known": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def unexplained_odd(*, table: pl.DataFrame, logged: dict[int, np.ndarray] | None = None) -> int:
    """Alarms in odd months with no logged closure within 24 hours on their series."""
    if table.is_empty():
        return 0
    logged = logged or near_log_masks(pad_stamps=EXPLAIN_PAD)
    stamps = table["stamp"].to_numpy()
    odd = np.isin(stamp_months()[stamps], list(TUNING_MONTHS))
    explained = np.array([bool(logged[s][t]) for s, t in zip(table["series"], stamps, strict=True)])
    return int((odd & ~explained).sum())


def intervals_of(
    *,
    table: pl.DataFrame,
    relative: dict[int, np.ndarray],
    fast24: DetectorInputs,
    world: int,
) -> pl.DataFrame:
    """Offline-end intervals with the 24-hour merge (`R0_M48`) for a pair-alarm table.

    The end rule and the merge are those of RO-NC's intervals, so the onset alarm is the only
    difference. `onset_known` is the stamp from which the onset is known to a forecast. A partner
    node gets a copy of its alarm's interval (the partner's step was seen with the alarm), because
    the tracker's own test of the partner's next 24 hours is not available when the alarm is
    known.
    """
    schema = {
        "world": pl.Int64,
        "setting": pl.String,
        "series": pl.Int64,
        "role": pl.String,
        "alarm_series": pl.Int64,
        "onset": pl.Int64,
        "end": pl.Int64,
        "known_from": pl.Int64,
        "step_mw": pl.Float64,
        "onset_known": pl.Int64,
    }
    if table.is_empty():
        return pl.DataFrame(schema=schema)
    base = table.select("series", "stamp", "step_mw", "z").with_columns(partners=pl.lit(""))
    tracked = build_intervals(alarm_table=base, relative=relative, window=48, detector="D0")
    live = live_end_intervals(intervals=tracked, inputs=fast24, rule="R0", cap_days=CAP_DAYS)
    merged = filter_alarms(table=base, live=live, tolerance=MERGE)
    rebuilt = build_intervals(alarm_table=merged, relative=relative, window=48, detector="D0")
    final = live_end_intervals(intervals=rebuilt, inputs=fast24, rule="R0", cap_days=CAP_DAYS)
    steps = {(r["series"], r["stamp"]): r for r in table.iter_rows(named=True)}
    rows = []
    for iv in final:
        alarm = steps.get((iv.alarm_series, iv.onset))
        known = iv.onset if alarm is None else alarm["known"]
        step = float("nan") if alarm is None else alarm["step_mw"]
        series_hit = [(iv.series, "alarm")]
        if alarm is not None and iv.role == "alarm":
            partners = [int(p.split(":")[0]) for p in alarm["partners"].split(";") if p]
            series_hit += [(p, "partner") for p in partners if p in relative]
        for series, role in series_hit:
            rows.append(
                {
                    "world": world,
                    "setting": "R0_M48",
                    "series": series,
                    "role": role,
                    "alarm_series": iv.alarm_series,
                    "onset": iv.onset,
                    "end": iv.end,
                    "known_from": iv.known_from,
                    "step_mw": step,
                    "onset_known": known,
                }
            )
    return pl.DataFrame(rows, schema=schema)


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


def score_events(
    *, events: list[Event], moved_mw: dict[int, float], intervals: pl.DataFrame, family: str
) -> pl.DataFrame:
    """For each event: whether an interval starts at its onset, and when that onset is known."""
    rows = []
    for e in events:
        hits = intervals.filter(
            pl.col("series").is_in(list(e.series))
            & ((pl.col("onset") - e.onset).abs() <= ONSET_TOLERANCE)
        )
        rows.append(
            {
                "family": family,
                "event_id": e.event_id,
                "onset": e.onset,
                "found": hits.height > 0,
                "lag_hours": (hits["onset_known"].min() - e.onset) / 2 if hits.height else None,  # ty: ignore[unsupported-operator]
                "moved_mw": moved_mw[e.event_id],
                "covered": len(e.receivers) > 0,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"lag_hours": pl.Float64})


def moved_mw_of(*, events: list[Event], delta: dict[int, np.ndarray]) -> dict[int, float]:
    """Mean megawatts an event moves: the positive deltas summed over series, averaged over time."""
    out = {}
    for e in events:
        total = sum(np.clip(delta[s][e.onset : e.end], 0, None) for s in delta)
        out[e.event_id] = float(np.nanmean(total))
    return out
