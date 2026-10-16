"""Residual diagnostics per arm-seed: spread, autocorrelation, shares, and the median's error.

Computed from the arms' fitted predictions and written to the output folder as `diag_*.parquet`, for
the report and the charts. Nothing here reads an alarm.

    uv run python ml-experiments/00047-*/e00047_diagnostics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00047_common import FITS_17_DIR, GRID_END, GRID_LENGTH, SERIES_IDS, WORK_DIR, near_log_masks
from e00047_detect import (
    ArmPanel,
    Arrays,
    bank_holiday_distance_all,
    capacities,
    load_arm_panel,
    real_events,
)
from e00047_fit import FITS_DIR
from e00047_run import ALL_NAMES

BLOCK: Final[int] = 48
MIN_DAY_PRESENT: Final[float] = 0.75


def autocorrelation(*, values: np.ndarray, lag: int) -> float:
    """Return the Pearson correlation between `values[t]` and `values[t - lag]`, pairs present."""
    a, b = values[lag:], values[:-lag]
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def daily_means(*, values: np.ndarray) -> np.ndarray:
    """Return the mean of each 24-hour block of stamps, NaN where under 75% is present."""
    blocks = values[: len(values) // BLOCK * BLOCK].reshape(-1, BLOCK)
    present = np.isfinite(blocks)
    totals = np.where(present, blocks, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = totals / present.sum(axis=1)
    means[present.mean(axis=1) < MIN_DAY_PRESENT] = np.nan
    return means


def spread_and_autocorrelation(*, panel: ArmPanel, capacity: dict[int, float]) -> pl.DataFrame:
    """One row per series: residual MAD in MW and capacity points, and three autocorrelations."""
    rows = []
    for series in SERIES_IDS:
        residual = panel.residual[series]
        finite = residual[np.isfinite(residual)]
        mad = float(np.median(np.abs(finite - np.median(finite))))
        rows.append(
            {
                "name": panel.name,
                "series": series,
                "mad_mw": mad,
                "mad_points_of_capacity": 100 * mad / capacity[series],
                "lag_1": autocorrelation(values=residual, lag=1),
                "lag_48": autocorrelation(values=residual, lag=BLOCK),
                "daily_lag_1": autocorrelation(values=daily_means(values=residual), lag=1),
            }
        )
    return pl.DataFrame(rows)


def squared_shares(*, panel: ArmPanel, power: Arrays) -> pl.DataFrame:
    """Share of the squared residual inside logged closures and labelled visible transfers.

    Each region's share of the half-hours is returned beside it, so a share of squared residual
    above the share of half-hours says the residual is larger inside the region.
    """
    logged = near_log_masks(pad_stamps=0)
    real = real_events(power=power)
    labelled = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for row in real.iter_rows(named=True):
        for series in (int(s) for s in row["series"].split(",")):
            if series in labelled:
                labelled[series][row["onset"] : row["end"]] = True
    regions = {"logged closures": logged, "labelled visible transfers": labelled}
    total_sq, total_n = 0.0, 0
    inside_sq = dict.fromkeys(regions, 0.0)
    inside_n = dict.fromkeys(regions, 0)
    for series in SERIES_IDS:
        residual = panel.residual[series]
        finite = np.isfinite(residual)
        square = np.where(finite, residual, 0.0) ** 2
        total_sq += float(square.sum())
        total_n += int(finite.sum())
        for name, masks in regions.items():
            inside_sq[name] += float(square[masks[series]].sum())
            inside_n[name] += int((finite & masks[series]).sum())
    return pl.DataFrame(
        [
            {
                "name": panel.name,
                "region": name,
                "share_of_squared_residual": inside_sq[name] / total_sq,
                "share_of_half_hours": inside_n[name] / total_n,
            }
            for name in regions
        ]
    )


def _arm_fits(*, name: str, fits_dir: Path) -> pl.DataFrame:
    """Return `time_series_id`, `valid_time`, `target`, `prediction` for every series of an arm."""
    frames = []
    for series in SERIES_IDS:
        if name == "H0.5_s0":
            frame = pl.read_parquet(FITS_17_DIR / f"{series}.parquet").select(
                "time_series_id", "valid_time", "target", prediction="H0.5"
            )
        else:
            frame = pl.read_parquet(fits_dir / name / f"{series}.parquet").select(
                "time_series_id", "valid_time", "target", "prediction"
            )
        frames.append(frame)
    return pl.concat(frames).filter(pl.col("valid_time") < GRID_END)


def median_error(*, name: str, fits_dir: Path) -> pl.DataFrame:
    """Mean absolute error of the median, in points of capacity, on all rows and holiday rows.

    The holiday rows are the local dates of a bank holiday in England (Christmas included), and
    those dates with the day either side.
    """
    frame = _arm_fits(name=name, fits_dir=fits_dir).with_columns(
        error=((pl.col("prediction") - pl.col("target")).abs() * 100.0).cast(pl.Float64),
        date=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date(),
    )
    distance = bank_holiday_distance_all()
    frame = frame.join(distance, on="date", how="left", validate="m:1")
    scopes = {
        "all rows": pl.lit(True),
        "bank holidays": pl.col("days_from_holiday") == 0,
        "bank holidays and the day either side": pl.col("days_from_holiday").abs() <= 1,
    }
    return pl.DataFrame(
        [
            {
                "name": name,
                "scope": scope,
                "mae_points": float(frame.filter(keep)["error"].mean()),  # ty: ignore[invalid-argument-type]
                "rows": frame.filter(keep).height,
            }
            for scope, keep in scopes.items()
        ]
    )


def main() -> None:
    """Write the four diagnostics tables for every arm-seed whose fits exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    parser.add_argument("--fits", type=Path, default=FITS_DIR)
    parser.add_argument("--names", nargs="*", default=list(ALL_NAMES))
    parser.add_argument("--fallback", action="store_true", help="smoke test: fill missing months")
    args = parser.parse_args()
    capacity = capacities()
    spread, shares, errors = [], [], []
    for name in args.names:
        panel = load_arm_panel(name=name, fits_dir=args.fits, fallback=args.fallback)
        spread.append(spread_and_autocorrelation(panel=panel, capacity=capacity))
        shares.append(squared_shares(panel=panel, power=panel.power))
        if not args.fallback:
            errors.append(median_error(name=name, fits_dir=args.fits))
        print("done", name, flush=True)
    pl.concat(spread).write_parquet(args.out / "diag_spread.parquet")
    pl.concat(shares).write_parquet(args.out / "diag_shares.parquet")
    if errors:
        pl.concat(errors).write_parquet(args.out / "diag_mae.parquet")


if __name__ == "__main__":
    main()
