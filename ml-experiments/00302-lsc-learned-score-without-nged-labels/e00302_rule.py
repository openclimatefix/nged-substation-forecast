# ruff: noqa: E501
"""Thresholds, alarm counts, and matching of learned-score alarms on the real record.

Alarms are merged: flagged candidates on one series whose consecutive stamps are at most 24 hours
apart form one alarm (experiment 227's counting). Thresholds:

- L0 and L1 (labelled): leave `BUDGET` unexplained candidates of the months the model did not train
  on above the threshold (experiment 195's rule).
- L2 (global rate): the same number of merged alarms over the whole record as L0 raises in the fold.
- L3 (per-Primary rate): each Primary's merged alarm count is the fold's total times the Primary's
  share of predicted rates, with the rate predicted from the share of misfit days (experiment 217)
  by a straight line fitted on the other Primaries.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00302_common import ML_EXPERIMENTS_DATA_DIR

E217_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"
MERGE: Final[int] = 48
"""Half-hours: flagged candidates on one series within 24 hours merge into one alarm."""
BUDGET: Final[int] = 20
ONSET_TOLERANCE: Final[int] = 4
END_MINIMUM: Final[int] = 12
END_FRACTION: Final[float] = 0.10
SHARE_FLOOR: Final[float] = 0.01
RATE_FLOOR: Final[float] = 0.1


def merged_count(*, series: np.ndarray, stamp: np.ndarray) -> int:
    """Number of merged alarms among flagged candidates."""
    if len(stamp) == 0:
        return 0
    order = np.lexsort((stamp, series))
    s, t = series[order], stamp[order]
    new = (s[1:] != s[:-1]) | ((t[1:] - t[:-1]) > MERGE)
    return int(new.sum() + 1)


def budget_threshold(*, table: pl.DataFrame, held_out_odd: bool, budget: int = BUDGET) -> float:
    """Score above which exactly `budget` unexplained candidates of the held-out months remain."""
    pool_mask = (
        (table["odd"].to_numpy() == held_out_odd)
        & ~table["explained"].to_numpy()
        & np.isfinite(table["score"].to_numpy())
    )
    pool = np.sort(table["score"].to_numpy()[pool_mask])[::-1]
    return float(pool[budget]) if len(pool) > budget else -np.inf


def flagged_by(*, table: pl.DataFrame, thresholds: dict[int, float]) -> np.ndarray:
    """Candidates whose score is above their series' threshold."""
    thr = np.array([thresholds[int(s)] for s in table["series"].to_numpy()])
    score = table["score"].to_numpy()
    return np.isfinite(score) & (score > thr)


def threshold_for_count(
    *, score: np.ndarray, series: np.ndarray, stamp: np.ndarray, target: float
) -> float:
    """Score threshold whose flagged candidates form the merged alarm count closest to `target`."""
    ok = np.isfinite(score)
    order = np.argsort(-score[ok])
    s_sorted, t_sorted, v_sorted = series[ok][order], stamp[ok][order], score[ok][order]

    def count(k: int) -> int:
        return merged_count(series=s_sorted[:k], stamp=t_sorted[:k])

    lo, hi = 0, len(v_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(count(lo - 1) - target) < abs(count(lo) - target):
        best = lo - 1
    if best == 0:
        return float(v_sorted[0]) if len(v_sorted) else np.inf
    # Flag the top `best` candidates: the threshold is just below the `best`-th score.
    return float(np.nextafter(v_sorted[best - 1], -np.inf))


def primary_table() -> pl.DataFrame:
    """Per Primary: years of residual, share of misfit days, and RO-NC's alarm rate (experiment 217)."""
    stats = pl.read_parquet(E217_DIR / "stats.parquet").select(
        "series", "site", "years", "misfit_share"
    )
    rates = (
        pl.read_parquet(E217_DIR / "rates.parquet")
        .filter((pl.col("detector") == "RO-NC/current") & (pl.col("group") == "all"))
        .select("series", "rate")
    )
    return stats.join(rates, on="series").sort("series")


def predicted_rates(*, primaries: pl.DataFrame) -> dict[int, float]:
    """Alarm rate per Primary-year predicted for each Primary by a line fitted without that Primary.

    log2(rate) = a + b * log2(share of misfit days + 0.01), fitted by least squares on the others.
    """
    x = np.log2(primaries["misfit_share"].to_numpy() + SHARE_FLOOR)
    y = np.log2(np.maximum(primaries["rate"].to_numpy(), RATE_FLOOR))
    series = primaries["series"].to_list()
    out = {}
    for i, s in enumerate(series):
        keep = np.arange(len(series)) != i
        b, a = np.polyfit(x[keep], y[keep], 1)
        out[int(s)] = float(2 ** (a + b * x[i]))
    return out


def per_primary_thresholds(
    *, table: pl.DataFrame, total: float, primaries: pl.DataFrame
) -> dict[int, float]:
    """One threshold per series so that the merged alarm counts follow the predicted rates and sum to `total`."""
    rate = predicted_rates(primaries=primaries)
    years = dict(zip(primaries["series"].to_list(), primaries["years"].to_list(), strict=True))
    weight = {s: rate[s] * years[s] for s in rate}
    norm = sum(weight.values())
    out: dict[int, float] = {}
    series_all = table["series"].to_numpy()
    for s in sorted({int(v) for v in series_all}):
        if s not in weight:
            out[s] = np.inf
            continue
        m = series_all == s
        out[s] = threshold_for_count(
            score=table["score"].to_numpy()[m],
            series=series_all[m],
            stamp=table["stamp"].to_numpy()[m],
            target=total * weight[s] / norm,
        )
    return out


def end_tolerance(*, onset: int, end: int) -> float:
    """The larger of 6 hours and 10% of the duration, in half-hours."""
    return max(float(END_MINIMUM), END_FRACTION * (end - onset))


def real_hits(*, table: pl.DataFrame, flagged: np.ndarray, events: pl.DataFrame) -> pl.DataFrame:
    """For each labelled load shift: onset found, found (start and end), and the onset error in hours."""
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
    """For each injected load shift: found and onset found, from the flagged candidates that mark it."""
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
