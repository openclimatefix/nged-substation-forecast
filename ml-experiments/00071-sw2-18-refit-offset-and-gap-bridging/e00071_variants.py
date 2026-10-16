"""The two mechanical fixes: two residual models refitted half a month apart, and gap bridging.

Arm X computes the 24-hour step statistic from each of two residuals (refits on the 1st and on the
16th) and, at each stamp, takes the statistic of the residual whose model did not refit within 24
hours of that stamp. No statistic is computed across a refit, so no alarm is suppressed and no
return step is blocked. Arm G replaces the missing statistic in the middle of a gap of a day or
longer by the difference between the means of the 24 hours either side of the gap. Nothing here
reads a switching log.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import polars as pl
from e00071_common import GRID_LENGTH, GRID_START, SERIES_IDS, load_panel
from e00071_detectors import (
    MIN_PRESENT,
    DetectorInputs,
    level_shift,
    refit_stamps,
    robust_scale,
)
from e00071_fit_offset import FEATURES_PATH, FITS_DIR
from e00071_intervals import relative_residual

Variant = Literal["PF14", "X", "G", "XG", "L"]
VARIANTS: Final[tuple[Variant, ...]] = ("PF14", "X", "G", "XG", "L")
"""L is the control: the stored residual with the refit suppression lifted and no splice."""
WINDOW: Final[int] = 48
BLOCK: Final[int] = 48
"""Half-hours in a day."""
MIN_GAP: Final[int] = 48
"""Shortest run of missing half-hours that is bridged: a day."""
Arrays = dict[int, np.ndarray]


def load_offset_panel(*, panel: dict) -> dict[str, Arrays]:
    """Return `expected` and `residual` of the model refitted on the 16th, aligned to the grid.

    Capacity scaling is the one of the stored panel (median effective capacity of the series), and
    the observed power must equal the stored panel's wherever both exist.
    """
    capacity = (
        pl.scan_parquet(FEATURES_PATH)
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median())
        .collect()
    )
    out: dict[str, Arrays] = {"expected": {}, "residual": {}}
    for series in SERIES_IDS:
        path = FITS_DIR / f"{series}.parquet"
        if not path.exists():
            continue
        cap = capacity.filter(pl.col("time_series_id") == series)["effective_capacity_mw"].item()
        frame = pl.read_parquet(path).with_columns(
            stamp=(pl.col("valid_time").dt.epoch("s") - int(GRID_START.timestamp())) // 1800
        )
        frame = frame.filter(pl.col("stamp").is_between(0, GRID_LENGTH - 1))
        stamps = frame["stamp"].to_numpy()
        expected = np.full(GRID_LENGTH, np.nan)
        residual = np.full(GRID_LENGTH, np.nan)
        expected[stamps] = frame["H0_5"].to_numpy() * cap
        observed = frame["target"].to_numpy() * cap
        residual[stamps] = observed - expected[stamps]
        mismatch = np.nanmax(np.abs(observed - panel["power"][series][stamps]))
        if mismatch > 1e-3:
            raise ValueError(
                f"series {series}: observed power differs from the panel by {mismatch}"
            )
        out["expected"][series], out["residual"][series] = expected, residual
    return out


def second_refit_stamps() -> np.ndarray:
    """The stamps at 00:00 UTC on the 16th of each month, when the second model refits."""
    stamps = refit_stamps() + 15 * BLOCK
    return stamps[stamps < GRID_LENGTH]


def refit_zone() -> np.ndarray:
    """True within 24 hours of a refit on the 1st, where the first model's statistic is unusable."""
    zone = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        zone[max(stamp - WINDOW + 1, 0) : stamp + WINDOW] = True
    return zone


def bridge_gaps(*, values: np.ndarray, shift: np.ndarray, refits: np.ndarray) -> np.ndarray:
    """Fill the statistic at the middle of every gap of `MIN_GAP` half-hours or more.

    The filled value is the mean of the 24 hours after the gap minus the mean of the 24 hours
    before it, and only where each of the two flanks holds `MIN_PRESENT` of its stamps. A gap at the
    start or end of the series has one flank and is left alone, and so is a gap with a refit of the
    model (`refits`) between the start of the before-flank and the end of the after-flank. A fresh
    array is returned.
    """
    out = shift.copy()
    missing = ~np.isfinite(values)
    edges = np.diff(np.concatenate([[0], missing.astype(int), [0]]))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if stop - start < MIN_GAP or start < WINDOW or stop + WINDOW > GRID_LENGTH:
            continue
        if ((refits > start - WINDOW) & (refits <= stop + WINDOW)).any():
            continue
        before, after = values[start - WINDOW : start], values[stop : stop + WINDOW]
        if np.isfinite(before).mean() < MIN_PRESENT or np.isfinite(after).mean() < MIN_PRESENT:
            continue
        middle = (start + stop) // 2
        if np.isfinite(out[middle]):
            continue
        out[middle] = np.nanmean(after) - np.nanmean(before)
    return out


def combine_shift(*, first: np.ndarray, second: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """The first model's statistic outside `zone` and the second model's inside it.

    Where the second model has no statistic inside the zone (its history starts later), the result
    is missing, which raises no alarm there.
    """
    return np.where(zone, second, first)


def shifts(*, variant: Variant, residual: Arrays, offset: Arrays) -> Arrays:
    """The step statistic of every series for one variant, from the (possibly injected) residual."""
    zone = refit_zone()
    first_refits, second_refits = refit_stamps(), second_refit_stamps()
    out: Arrays = {}
    for series, values in residual.items():
        first = level_shift(values=values, window=WINDOW)
        if variant in ("G", "XG"):
            first = bridge_gaps(values=values, shift=first, refits=first_refits)
        if variant in ("X", "XG"):
            second_values = offset.get(series, np.full(GRID_LENGTH, np.nan))
            second = level_shift(values=second_values, window=WINDOW)
            if variant == "XG":
                second = bridge_gaps(values=second_values, shift=second, refits=second_refits)
            first = combine_shift(first=first, second=second, zone=zone)
        out[series] = first
    return out


def scales_for(*, variant: Variant, residual: Arrays, offset: Arrays) -> dict[int, float]:
    """The odd-month noise scale of each series' statistic, from the uninjected residuals.

    The scale is taken before bridging, so a bridged value cannot change the unit it is read in.
    """
    base = "X" if variant in ("X", "XG") else "PF14"
    return {
        series: robust_scale(shift=shift)
        for series, shift in shifts(variant=base, residual=residual, offset=offset).items()
    }


def variant_inputs(
    *, variant: Variant, residual: Arrays, offset: Arrays, scale: dict[int, float]
) -> DetectorInputs:
    """Detector inputs for one variant; X and XG compute no statistic across a refit."""
    return DetectorInputs(
        window=WINDOW,
        shift=shifts(variant=variant, residual=residual, offset=offset),
        scale=scale,
        suppress_refits=variant in ("PF14", "G"),
    )


def tracked_relative(
    *, variant: Variant, panel: dict, offset_panel: dict, residual: Arrays, offset: Arrays
) -> Arrays:
    """The relative residual that the level tracker reads.

    For X and XG it is the second model's inside the 24 hours around a refit of the first model,
    so the tracker's pre-level and blocks do not straddle the first model's refit jump.
    """
    zone = refit_zone()
    out: Arrays = {}
    for series, values in residual.items():
        first = relative_residual(residual=values, expected=panel["expected"][series])
        if variant in ("X", "XG") and series in offset:
            second = relative_residual(
                residual=offset[series], expected=offset_panel["expected"][series]
            )
            first = np.where(zone & np.isfinite(second), second, first)
        out[series] = first
    return out


def load_both() -> tuple[dict, dict]:
    """Return the stored panel and the offset panel."""
    panel = load_panel()
    return panel, load_offset_panel(panel=panel)


def check_variants() -> None:
    """Hand-built arrays: bridging and combining behave as the plan says."""
    values = np.zeros(GRID_LENGTH)
    values[5000 : 5000 + 100] = np.nan
    values[5100:] += 3.0
    shift = level_shift(values=values, window=WINDOW)
    none = np.array([], dtype=int)
    bridged = bridge_gaps(values=values, shift=shift, refits=none)
    middle = 5050
    if not np.isclose(bridged[middle], 3.0) or np.isfinite(shift[middle]):
        raise SystemExit(f"STOP: bridge gave {bridged[middle]} at the middle of the gap")
    short = np.zeros(GRID_LENGTH)
    short[5000 : 5000 + 47] = np.nan
    short[5047:] += 3.0
    if not np.array_equal(
        bridge_gaps(values=short, shift=level_shift(values=short, window=WINDOW), refits=none),
        level_shift(values=short, window=WINDOW),
        equal_nan=True,
    ):
        raise SystemExit("STOP: a gap shorter than a day was bridged")
    if np.isfinite(bridge_gaps(values=values, shift=shift, refits=np.array([5040]))[middle]):
        raise SystemExit("STOP: a gap with a refit inside it was bridged")
    zone = np.zeros(GRID_LENGTH, dtype=bool)
    zone[10:20] = True
    first, second = np.full(GRID_LENGTH, 1.0), np.full(GRID_LENGTH, 2.0)
    combined = combine_shift(first=first, second=second, zone=zone)
    if combined[9] != 1.0 or combined[10] != 2.0 or combined[19] != 2.0 or combined[20] != 1.0:
        raise SystemExit("STOP: combine_shift picks the wrong model at the zone edges")
    print("variants checked on hand-built arrays")
