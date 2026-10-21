r"""Noise and step-retention of the three residuals, before any detector is scored.

For each series, the odd-month robust scale of the 24-hour step statistic (in MW) of arm A's
residual (lagless XGBoost, experiment 82's panel), arm B's (frozen convex model) and arm C's (filter
innovations); and a positive control that adds a constant 3-day step of 4 noise units of arm A to
the residual stream and reports the share of the step that the peak 24-hour statistic keeps.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00130-*/e00130_diagnose.py
"""

from __future__ import annotations

import pickle

import numpy as np
from e00130_common import GRID_LENGTH, SERIES_IDS, load_panel, stamp_months
from e00130_detectors import level_shift, robust_scale
from e00130_gam_panel import GAM_DIR, hour_band, innovations, load_hyper, site_of_series
from studies.dp_assimilation import FilterHyperparameters

WINDOW = 48
STEP_DAYS = 3
ONSETS_PER_SERIES = 30
SEED = 130


def retained_share(
    *, series: int, frozen: dict, hyper: FilterHyperparameters, band: np.ndarray, unit_mw: float
) -> float:
    """Median over onsets of the peak statistic of arm C divided by the injected step."""
    capacity = site_of_series()[series][1]
    residual = frozen["residual"][series]
    rng = np.random.default_rng(SEED + series)
    even = ~np.isin(stamp_months(), [1, 3, 5, 7, 9, 11])
    candidates = np.flatnonzero(even & np.isfinite(residual))
    candidates = candidates[(candidates > 4 * WINDOW) & (candidates < GRID_LENGTH - 8 * WINDOW)]
    shares = []
    step = 4.0 * unit_mw
    for onset in rng.choice(candidates, size=ONSETS_PER_SERIES, replace=False):
        end = onset + STEP_DAYS * 48
        lo, hi = onset - 10 * 48, end + 4 * 48
        window = np.arange(GRID_LENGTH)
        stream = np.full(GRID_LENGTH, np.nan)
        stream[lo:hi] = residual[lo:hi]
        stream[onset:end] += step
        innov = (
            innovations(residual=stream / capacity * 100.0, band=band, hyper=hyper)
            * capacity
            / 100.0
        )
        shift = level_shift(values=innov, window=WINDOW)
        near = slice(onset - 4, onset + 4)
        shares.append(float(np.nanmax(shift[near])) / step)
        del window
    return float(np.median(shares))


def table() -> list[dict[str, float]]:
    """One row per series: odd-month step scales of the three residuals and C's retained share."""
    a_panel = load_panel()
    band = hour_band()
    hyper = load_hyper()
    with (GAM_DIR / "panel_1.pkl").open("rb") as handle:
        stored = pickle.load(handle)
    rows = []
    for series in SERIES_IDS:
        if series not in hyper:
            continue
        a = robust_scale(shift=level_shift(values=a_panel["residual"][series], window=WINDOW))
        b = robust_scale(
            shift=level_shift(values=stored["frozen"]["residual"][series], window=WINDOW)
        )
        c = robust_scale(shift=level_shift(values=stored["innovation"][series], window=WINDOW))
        share = retained_share(
            series=series, frozen=stored["frozen"], hyper=hyper[series], band=band, unit_mw=a
        )
        rows.append(
            {"series": series, "scale_a": a, "scale_b": b, "scale_c": c, "retained_share": share}
        )
    return rows


def main() -> None:
    """Print the table."""
    print("series site  scaleA scaleB scaleC  B/A  C/A  C-retained-share-of-step")
    for row in table():
        series = int(row["series"])
        a, b, c = row["scale_a"], row["scale_b"], row["scale_c"]
        print(
            f"{series:3d} {site_of_series()[series][0]:4s} {a:.3f} {b:.3f} {c:.3f} "
            f"{b / a:.2f} {c / a:.2f} {row['retained_share']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
