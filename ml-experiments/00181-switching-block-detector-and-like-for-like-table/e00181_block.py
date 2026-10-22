"""Part 1, step 2: a rectangular block detector on RO-NC's residual, tuned on odd months.

A block is a step up, a flat level, and a step down. The statistic at stamp `t` and length `L` is
the smaller of the two edge contrasts of the mean over `[t, t+L)` against the `F` half-hours before
`t` and the `F` half-hours from `t+L`, in noise units. Nothing is fitted except the alarm
threshold, which is set on odd-numbered months only.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00181-*/e00181_block.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00181_blocks17 import block_days
from e00181_common import (
    ADJACENCY,
    GRID_LENGTH,
    ML_EXPERIMENTS_DATA_DIR,
    TUNING_MONTHS,
    WORK_DIR,
    near_log_masks,
    stamp_months,
)
from e00181_detectors import local_peaks
from e00181_variants import load_both

LENGTHS: Final[tuple[int, ...]] = (12, 24, 48, 96, 240)
"""Block lengths in half-hours: 6 hours, 12 hours, 1 day, 2 days, and 5 days."""
FLANK_MAX: Final[int] = 48
REAL_RESULTS: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00130-ro-nc-detector-on-the-gam-k-residual"
    / "B"
    / "real_results.parquet"
)
"""The 66 labelled real transfers of experiment 41, with onset and end stamps and series."""
PRESENT: Final[float] = 0.75
BUDGET: Final[int] = 80
"""Unexplained odd-month alarms that RO-NC's threshold was tuned to."""
SUPPORT_CAP: Final[float] = 3.0
PEAK_WINDOW: Final[int] = 48
ONSET_TOLERANCE: Final[int] = 6
"""Stamps (3 hours) within which an alarm counts as at the block's onset."""
LOWEST_THRESHOLD: Final[float] = 2.0
LOG_PAD: Final[int] = 48
MAD_TO_STD: Final[float] = 1.4826


def _window_mean(
    *, cumulative: np.ndarray, counts: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and present-share of the half-open windows `[lo, hi)`, clipped to the grid."""
    lo, hi = np.clip(lo, 0, GRID_LENGTH), np.clip(hi, 0, GRID_LENGTH)
    n = counts[hi] - counts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cumulative[hi] - cumulative[lo]) / n
    return mean, n


def edge_contrasts(*, values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return, at every stamp, the block's mean minus the mean before and minus the mean after."""
    flank = min(length, FLANK_MAX)
    cumulative = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    counts = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(GRID_LENGTH)
    inside, n_in = _window_mean(cumulative=cumulative, counts=counts, lo=t, hi=t + length)
    before, n_before = _window_mean(cumulative=cumulative, counts=counts, lo=t - flank, hi=t)
    after, n_after = _window_mean(
        cumulative=cumulative, counts=counts, lo=t + length, hi=t + length + flank
    )
    ok = (n_in >= PRESENT * length) & (n_before >= PRESENT * flank) & (n_after >= PRESENT * flank)
    up, down = inside - before, inside - after
    up[~ok], down[~ok] = np.nan, np.nan
    return up, down


def block_score(*, values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the signed block score and the single-edge contrast used to scale it.

    The score is the smaller edge contrast when both edges point the same way, else 0.
    """
    up, down = edge_contrasts(values=values, length=length)
    same_up, same_down = (up > 0) & (down > 0), (up < 0) & (down < 0)
    score = np.where(same_up, np.minimum(up, down), np.where(same_down, np.maximum(up, down), 0.0))
    score[~np.isfinite(up) | ~np.isfinite(down)] = np.nan
    return score, up


def odd_scale(*, contrast: np.ndarray) -> float:
    """The robust scale of a contrast over odd-numbered months."""
    months = stamp_months()
    chosen = contrast[np.isin(months, list(TUNING_MONTHS)) & np.isfinite(contrast)]
    return float(MAD_TO_STD * np.median(np.abs(chosen - np.median(chosen))))


LOOKBACK_DAYS: Final[int] = 7
FLOOR_QUANTILE: Final[float] = 0.10


def local_sigma(*, values: np.ndarray) -> np.ndarray:
    """Return, at every stamp, the median of the previous 7 days' daily spreads (1.4826 MAD).

    The spread of a day is the scaled median absolute deviation of its 48 readings about their
    median, needing 24 readings. The floor is the 10th percentile of the series' daily spreads.
    """
    days = GRID_LENGTH // 48
    daily = np.full(days, np.nan)
    for d in range(days):
        chunk = values[d * 48 : (d + 1) * 48]
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size >= 24:
            daily[d] = MAD_TO_STD * np.median(np.abs(chunk - np.median(chunk)))
    floor = np.nanquantile(daily, FLOOR_QUANTILE)
    sigma_day = np.full(days, np.nan)
    for d in range(days):
        recent = daily[max(d - LOOKBACK_DAYS, 0) : d]
        if np.isfinite(recent).any():
            sigma_day[d] = max(np.nanmedian(recent), floor)
    return (
        np.repeat(sigma_day, 48)[:GRID_LENGTH]
        if days * 48 >= GRID_LENGTH
        else np.concatenate([np.repeat(sigma_day, 48), np.full(GRID_LENGTH - days * 48, np.nan)])
    )


def series_scores(
    *, values: np.ndarray, scales: dict[int, float] | None = None, local: bool = False
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Return the signed score in noise units for every length, and the scales used.

    With `local`, each score is first divided by the previous 7 days' spread of the series, and the
    odd-month scale then converts the ratio to noise units.
    """
    sigma = local_sigma(values=values) if local else None
    out, used = {}, {}
    for length in LENGTHS:
        score, contrast = block_score(values=values, length=length)
        if sigma is not None:
            score, contrast = score / sigma, contrast / sigma
        scale = odd_scale(contrast=contrast) if scales is None else scales[length]
        out[length], used[length] = score / scale, scale
    return out, used


def best_z(*, scores: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the signed score of largest magnitude over the lengths, and that length."""
    stack = np.stack([np.nan_to_num(scores[length]) for length in LENGTHS])
    pick = np.argmax(np.abs(stack), axis=0)
    z = np.take_along_axis(stack, pick[None], axis=0)[0]
    return z, np.array(LENGTHS)[pick]


def support(
    *, scores_by_series: dict[int, dict[int, np.ndarray]], series: int
) -> dict[int, np.ndarray]:
    """Return per length the best opposite-signed neighbour score, as a non-negative array."""
    out = {}
    for length in LENGTHS:
        own = np.nan_to_num(scores_by_series[series][length])
        best = np.zeros(GRID_LENGTH)
        for n in ADJACENCY.get(series, ()):
            if n not in scores_by_series:
                continue
            opposite = -np.sign(own) * np.nan_to_num(scores_by_series[n][length])
            best = np.maximum(best, opposite)
        out[length] = np.clip(best, 0.0, SUPPORT_CAP)
    return out


def peaks_table(*, z: np.ndarray, series: int) -> pl.DataFrame:
    """All local peaks of |z| above the lowest threshold."""
    stamps = local_peaks(z=z, threshold=LOWEST_THRESHOLD, window=PEAK_WINDOW)
    return pl.DataFrame(
        {"series": [series] * len(stamps), "stamp": stamps, "z": np.abs(z[stamps])},
        schema={"series": pl.Int64, "stamp": pl.Int64, "z": pl.Float64},
    )


def explained_masks() -> dict[int, np.ndarray]:
    """True where a stamp is within 24 hours of a logged closure or inside a labelled transfer."""
    mask = near_log_masks(pad_stamps=LOG_PAD)
    labelled = (
        pl.read_parquet(REAL_RESULTS)
        .filter(pl.col("setting") == "XL")
        .select("onset", "end", "series")
    )
    for row in labelled.iter_rows(named=True):
        for s in (int(x) for x in str(row["series"]).split(",")):
            if s in mask:
                mask[s][max(row["onset"], 0) : min(row["end"] + 1, GRID_LENGTH)] = True
    return mask


def main() -> None:
    """Calibrate on odd months, count the 17 block days, and write the alarm tables."""
    panel, _ = load_both()
    residual = {s: v for s, v in panel["residual"].items() if np.isfinite(v).any()}
    scores_by_series, scales = {}, {}
    for s, values in residual.items():
        scores_by_series[s], scales[s] = series_scores(values=values)
    local_scores = {s: series_scores(values=v, local=True)[0] for s, v in residual.items()}
    explained = explained_masks()
    months = stamp_months()
    odd = np.isin(months, list(TUNING_MONTHS))
    days = block_days()
    summary: dict = {}
    for arm in ("S0", "S1", "S2"):
        tables = []
        for s in residual:
            z, _ = best_z(scores=local_scores[s] if arm == "S2" else scores_by_series[s])
            if arm == "S1":
                sup = support(scores_by_series=scores_by_series, series=s)
                per_length = {
                    length: np.abs(np.nan_to_num(scores_by_series[s][length])) + sup[length]
                    for length in LENGTHS
                }
                total = np.stack([per_length[length] for length in LENGTHS]).max(axis=0)
                z = np.sign(z) * total
            table = peaks_table(z=z, series=s)
            tables.append(table)
        peaks = pl.concat(tables).with_columns(
            odd=pl.Series(odd[pl.concat(tables)["stamp"].to_numpy()]),
            explained=pl.Series(
                [
                    bool(explained[s][t])
                    for s, t in pl.concat(tables).select("series", "stamp").iter_rows()
                ]
            ),
        )
        peaks.write_parquet(WORK_DIR / f"block_candidates_{arm}.parquet")
        unexplained_odd = peaks.filter(pl.col("odd"), ~pl.col("explained"))
        thresholds = np.sort(unexplained_odd["z"].to_numpy())[::-1]
        threshold = (
            float(thresholds[BUDGET]) + 1e-9 if len(thresholds) > BUDGET else LOWEST_THRESHOLD
        )
        kept = peaks.filter(pl.col("z") > threshold)
        found = []
        for day in days.iter_rows(named=True):
            near = kept.filter(
                pl.col("series") == day["series"],
                (pl.col("stamp") - day["edge_stamp"]).abs() <= ONSET_TOLERANCE,
            )
            day_near = kept.filter(
                pl.col("series") == day["series"],
                pl.col("stamp").is_between(
                    day["day_first_stamp"] - ONSET_TOLERANCE,
                    day["day_last_stamp"] + ONSET_TOLERANCE,
                ),
            )
            found.append((near.height > 0, day_near.height > 0))
        day_table = days.select("site", "series", "local_date", "edge_stamp").with_columns(
            found_edge=pl.Series([f[0] for f in found]),
            found_day=pl.Series([f[1] for f in found]),
            month=pl.Series([int(months[t]) for t in days["edge_stamp"]]),
        )
        day_table = day_table.with_columns(even_month=~pl.col("month").is_in(list(TUNING_MONTHS)))
        n_even_unexplained = kept.filter(~pl.col("odd"), ~pl.col("explained")).height
        summary[arm] = {
            "threshold": threshold,
            "unexplained_odd": kept.filter(pl.col("odd"), ~pl.col("explained")).height,
            "unexplained_even": n_even_unexplained,
            "found_edge": int(day_table["found_edge"].sum()),
            "found_day": int(day_table["found_day"].sum()),
            "found_day_even_months": int(day_table.filter(pl.col("even_month"))["found_day"].sum()),
            "found_edge_even_months": int(
                day_table.filter(pl.col("even_month"))["found_edge"].sum()
            ),
            "n_even_months": int(day_table["even_month"].sum()),
            "n_alarms_all": kept.height,
        }
        day_table.write_parquet(WORK_DIR / f"block_days_found_{arm}.parquet")
        kept.write_parquet(WORK_DIR / f"block_alarms_{arm}.parquet")
        print(arm, json.dumps(summary[arm]))
        print(day_table)
    (WORK_DIR / "block_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
