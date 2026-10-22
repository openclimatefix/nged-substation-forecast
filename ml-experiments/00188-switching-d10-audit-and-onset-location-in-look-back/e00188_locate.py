"""Locate a transfer's onset in the look-back, and build the alarm tables with located onsets.

At the first forecast origin that can see an alarm, the onset is searched for among the stamps
within 24 hours of the detector's own step stamp, using only readings up to that origin. The
statistic at a candidate stamp `t` is the mean residual over the 12 half-hours from `t` minus the
mean over the 12 half-hours before `t`; the after-window is cut to the readings available, with at
least 4. The pair alarm combines its own series' step with its partners' opposite steps; RO-NC
uses its own series alone. The located onset is the stamp with the largest statistic of the sign
of the alarm's step.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

HALF: Final[int] = 12
"""Half-hours (6 hours) in each side of the step window."""
MIN_AFTER: Final[int] = 4
SEARCH: Final[int] = 48
"""Half-hours (24 hours) either side of the detector's own step stamp that are searched."""
ORIGIN_PHASE: Final[int] = 10
"""A forecast run's newest usable stamp, modulo 12. Runs start at 09:00 UTC and see readings that
ended at least 3.5 hours before, so the newest stamp covers 05:00 to 05:30 (stamp 10 of the day),
and 6-hourly origins repeat that every 12 stamps."""
MATCH: Final[int] = 48
"""Stamps within which an alarm's step stamp counts as at an injected transfer's onset."""


def first_origin(*, known: int) -> int:
    """The newest usable stamp of the first 6-hourly origin that can see an alarm known at `known`."""
    return known + (ORIGIN_PHASE - known) % HALF


def _step(*, series: np.ndarray, stamp: int, last: int) -> float:
    """Mean after `stamp` (up to `last`) minus mean before, NaN when readings are too few."""
    after = series[stamp : min(stamp + HALF, last + 1)]
    before = series[max(stamp - HALF, 0) : stamp]
    if np.isfinite(after).sum() < MIN_AFTER or np.isfinite(before).sum() < MIN_AFTER:
        return float("nan")
    return float(np.nanmean(after) - np.nanmean(before))


def locate(
    *,
    residual: dict[int, np.ndarray],
    series: int,
    partners: list[int],
    stamp: int,
    sign: float,
    last: int,
) -> int:
    """The located onset stamp, or the detector's own stamp when no candidate has readings."""
    best, best_value = stamp, -np.inf
    for t in range(max(stamp - SEARCH, HALF), min(stamp + SEARCH, last - MIN_AFTER + 1) + 1):
        own = _step(series=residual[series], stamp=t, last=last)
        if not np.isfinite(own):
            continue
        others = [_step(series=residual[p], stamp=t, last=last) for p in partners if p in residual]
        value = sign * (own - float(np.nansum(others)))
        if value > best_value:
            best, best_value = t, value
    return best


def alarm_rows(
    *, table: pl.DataFrame, residual: dict[int, np.ndarray], family: str
) -> pl.DataFrame:
    """One row per distinct alarm of an interval table, with its first origin and located onset.

    The alarm's own series is `alarm_series` for the pair alarm (whose interval table also holds a
    copy on each partner, `role == "partner"`) and `series` for RO-NC. The columns are `family`,
    `alarm_series`, `onset` (the detector's step stamp), `known`, `end`, `known_from`, `step_mw`,
    `first_origin`, `located`, and `partners`.
    """
    rows = []
    if "alarm_series" not in table.columns or family == "RONC":
        table = table.with_columns(alarm_series=pl.col("series"), role=pl.lit("alarm"))
    heads = table.filter(pl.col("role") == "alarm").unique(["alarm_series", "onset"], keep="first")
    copies = table.filter(pl.col("role") == "partner")
    for r in heads.iter_rows(named=True):
        known = int(r["onset_known"]) if r.get("onset_known") is not None else r["onset"] + 96
        c1 = first_origin(known=known)
        partners = (
            copies.filter(
                (pl.col("alarm_series") == r["alarm_series"]) & (pl.col("onset") == r["onset"])
            )["series"].to_list()
            if family == "PAIR"
            else []
        )
        step = r["step_mw"]
        sign = 1.0 if not np.isfinite(step) or step >= 0 else -1.0
        rows.append(
            {
                "family": family,
                "alarm_series": int(r["alarm_series"]),
                "onset": int(r["onset"]),
                "known": known,
                "end": int(r["end"]),
                "known_from": int(r["known_from"]),
                "step_mw": float(step),
                "first_origin": c1,
                "located": locate(
                    residual=residual,
                    series=int(r["alarm_series"]),
                    partners=[int(p) for p in partners],
                    stamp=int(r["onset"]),
                    sign=sign,
                    last=c1,
                ),
                "partners": ",".join(str(int(p)) for p in partners),
            }
        )
    return pl.DataFrame(rows)
