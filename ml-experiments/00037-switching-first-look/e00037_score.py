"""Score the detectors against the oracle-confirmed transfers: budget, recall, placebo, intervals.

Reads the logs-derived tables for scoring only. Thresholds and scales are fixed on odd (tuning)
months and every reported recall is on even (validation) months.
"""

from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00037_common import TUNING_MONTHS
from e00037_detect import DetectorInputs, DetectorType, alarms, prepare
from e00037_grid import COVERAGE_END_STAMP, GRID_LENGTH, GRID_START, stamp_months

HIT_TOLERANCE: Final[int] = 4
NEAR_LOG_STAMPS: Final[int] = 24
BUDGET_PER_SERIES_MONTH: Final[float] = 0.5
STAMPS_PER_MONTH: Final[float] = 1461.0
THRESHOLD_GRID: Final[dict[DetectorType, tuple[float, ...]]] = {
    "D0": (4.0, 5.0, 6.0, 7.0, 8.0),
    "D1": (2.5, 3.0, 3.5, 4.0, 5.0, 6.0),
}
FINE_THRESHOLD_GRID: Final[dict[DetectorType, tuple[float, ...]]] = {
    "D0": tuple(float(h) for h in np.arange(4.0, 8.01, 0.25)),
    "D1": tuple(float(h) for h in np.arange(2.5, 6.01, 0.25)),
}
"""A finer grid, added after the first run to compare the detectors at closer alarm rates."""
WINDOWS: Final[tuple[int, ...]] = (8, 48)
N_RESAMPLES: Final[int] = 10_000
N_PLACEBO_SHIFTS: Final[int] = 40


def near_log_masks(*, intervals: pl.DataFrame, series_ids: list[int]) -> dict[int, np.ndarray]:
    """True within 12 hours of any logged action (close or open) on a sheet touching the series."""
    masks = {sid: np.zeros(GRID_LENGTH, dtype=bool) for sid in series_ids}
    for row in intervals.iter_rows(named=True):
        for label in (row["side_a"], row["side_b"]):
            if label is None or label == "unknown" or int(label) not in masks:
                continue
            for stamp in (row["first_stamp"], row["last_stamp_exclusive"]):
                masks[int(label)][
                    max(stamp - NEAR_LOG_STAMPS, 0) : min(stamp + NEAR_LOG_STAMPS + 1, GRID_LENGTH)
                ] = True
    return masks


def operation_members(*, intervals: pl.DataFrame) -> dict[int, frozenset[int]]:
    """Map an operation id to every series on a sheet of any member closure."""
    members: dict[int, set[int]] = {}
    for row in intervals.iter_rows(named=True):
        bucket = members.setdefault(row["operation"], set())
        for label in (row["side_a"], row["side_b"]):
            if label is not None and label != "unknown":
                bucket.add(int(label))
    return {k: frozenset(v) for k, v in members.items()}


def hits(
    *,
    events: pl.DataFrame,
    alarm_table: pl.DataFrame,
    members: dict[int, frozenset[int]],
    shifts: dict[int, int] | None = None,
) -> pl.DataFrame:
    """For each event, whether any member-series alarm lies within +/-4 stamps; add the delay."""
    by_series = {
        sid: np.sort(group["stamp"].to_numpy())
        for (sid,), group in alarm_table.group_by(["series"])
    }
    out_hit, out_delay = [], []
    for index, row in enumerate(events.iter_rows(named=True)):
        first = row["first_stamp"] + (shifts[index] if shifts else 0)
        best = None
        for sid in members[row["operation"]]:
            stamps = by_series.get(sid)
            if stamps is None:
                continue
            gap = stamps - first
            close = gap[np.abs(gap) <= HIT_TOLERANCE]
            if len(close) and (best is None or abs(close[np.argmin(np.abs(close))]) < abs(best)):
                best = int(close[np.argmin(np.abs(close))])
        out_hit.append(best is not None)
        out_delay.append(best)
    return events.with_columns(hit=pl.Series(out_hit), delay=pl.Series(out_delay, dtype=pl.Int64))


def series_month_cells(*, residuals: dict[int, np.ndarray], split: str) -> pl.DataFrame:
    """Valid stamps per (series, month) in `split`, inside log coverage, as series-months."""
    months = np.array(stamp_months())
    in_split = np.isin(months, list(TUNING_MONTHS)) == (split == "tuning")
    covered = np.arange(GRID_LENGTH) < COVERAGE_END_STAMP
    rows = []
    month_index = _month_index()
    for sid, values in residuals.items():
        valid = np.isfinite(values) & in_split & covered
        counts = np.bincount(month_index[valid], minlength=month_index.max() + 1)
        for m, count in enumerate(counts):
            if count:
                rows.append(
                    {"series": sid, "month_index": m, "series_months": count / STAMPS_PER_MONTH}
                )
    return pl.DataFrame(rows)


def _month_index() -> np.ndarray:
    """Index of the calendar month (0 = 2022-01) of each stamp."""
    return np.array(
        [
            (t := GRID_START + timedelta(minutes=30 * i)).year * 12
            + t.month
            - (GRID_START.year * 12 + 1)
            for i in range(GRID_LENGTH)
        ]
    )


def unexplained_counts(
    *, alarm_table: pl.DataFrame, near: dict[int, np.ndarray], split: str
) -> pl.DataFrame:
    """Count alarms more than 12 hours from any logged action, per (series, month), in `split`."""
    months = np.array(stamp_months())
    month_index = _month_index()
    rows = []
    for row in alarm_table.iter_rows(named=True):
        stamp = row["stamp"]
        in_split = (months[stamp] in TUNING_MONTHS) == (split == "tuning")
        if stamp < COVERAGE_END_STAMP and in_split and not near[row["series"]][stamp]:
            rows.append({"series": row["series"], "month_index": int(month_index[stamp])})
    if not rows:
        return pl.DataFrame(schema={"series": pl.Int64, "month_index": pl.Int64, "n": pl.UInt32})
    return pl.DataFrame(rows).group_by("series", "month_index").len(name="n")


def alarm_rate(*, unexplained: pl.DataFrame, cells: pl.DataFrame) -> float:
    """Unexplained alarms per series-month."""
    return float(unexplained["n"].sum() / cells["series_months"].sum())  # ty: ignore[unsupported-operator]


def weighted_recall(*, hit: np.ndarray, weight: np.ndarray) -> float:
    """Sum of weights of detected events divided by the sum of all weights."""
    return float((hit * weight).sum() / weight.sum()) if weight.sum() > 0 else float("nan")


def bootstrap_recall(
    *, hit_a: np.ndarray, hit_b: np.ndarray | None, weight: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """95% intervals for weighted recall of `hit_a`, and of `hit_a - hit_b` if given."""
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(weight), size=(N_RESAMPLES, len(weight)))
    w = weight[index]
    denominators = w.sum(axis=1)
    recall_a = (hit_a[index] * w).sum(axis=1) / denominators
    out = {"lo": float(np.percentile(recall_a, 2.5)), "hi": float(np.percentile(recall_a, 97.5))}
    if hit_b is not None:
        diff = recall_a - (hit_b[index] * w).sum(axis=1) / denominators
        out |= {
            "diff_lo": float(np.percentile(diff, 2.5)),
            "diff_hi": float(np.percentile(diff, 97.5)),
        }
    return out


def placebo_shifts(*, n_events: int, seed: int) -> dict[int, int]:
    """A random +/- 7 to 28 day shift (in stamps, same time of day) for each event."""
    rng = np.random.default_rng(seed)
    return {i: int(rng.choice([-1, 1]) * rng.integers(7, 29) * 48) for i in range(n_events)}


def run_grid(
    *, residuals: dict[int, np.ndarray], grid: dict[DetectorType, tuple[float, ...]]
) -> dict[tuple[DetectorType, int, float], pl.DataFrame]:
    """Return the alarm table for every detector, window, and threshold in the grid."""
    out = {}
    for window in WINDOWS:
        inputs: DetectorInputs = prepare(residuals=residuals, window=window)
        for detector, thresholds in grid.items():
            for threshold in thresholds:
                out[(detector, window, threshold)] = alarms(
                    inputs=inputs, detector=detector, threshold=threshold
                )
    return out
