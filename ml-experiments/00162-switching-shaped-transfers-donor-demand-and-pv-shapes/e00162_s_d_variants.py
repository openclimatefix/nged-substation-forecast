"""The refit-offset step statistic: two residual models refitted half a month apart.

Arm X computes the 24-hour step statistic from each of two residuals (refits on the 1st and on the
16th) and, at each stamp, takes the statistic of the residual whose model did not refit within 24
hours of that stamp. No statistic is computed across a refit, so no alarm is suppressed and no
return step is blocked. Nothing here reads a switching log.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from e00162_s_d_common import (
    GAM_DIR,
    GRID_LENGTH,
    GRID_START,
    ML_EXPERIMENTS_DATA_DIR,
    SERIES_IDS,
    load_panel,
)
from e00162_s_d_detectors import (
    DetectorInputs,
    level_shift,
    refit_stamps,
    robust_scale,
)
from e00162_s_d_intervals import relative_residual

FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
"""Experiment 6's ERA5 feature table, which the second residual model was fitted on."""
FITS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00071-sw2-18-refit-offset-and-gap-bridging" / "offset_fits"
)
"""Experiment 71's fits of the model refitted on the 16th; this experiment re-uses them."""
Variant = Literal["PF14", "X"]
VARIANTS: Final[tuple[Variant, ...]] = ("PF14", "X")
WINDOW: Final[int] = 48
BLOCK: Final[int] = 48
"""Half-hours in a day."""
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
        if not np.isfinite(mismatch) or mismatch > 1e-3:
            raise ValueError(
                f"series {series}: observed power differs from the panel by {mismatch}"
            )
        out["expected"][series], out["residual"][series] = expected, residual
    return out


def second_refit_stamps() -> np.ndarray:
    """The stamps at 00:00 UTC on the 16th of each month, when the second model refits."""
    stamps = refit_stamps() + 15 * BLOCK
    return stamps[stamps < GRID_LENGTH]


def refit_zone(*, width: int = WINDOW) -> np.ndarray:
    """True within `width` stamps of a refit on the 1st, where the first model's statistic fails.

    `width` is the span of the step statistic's two windows: 24 hours for the 24-hour statistic.
    """
    zone = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        zone[max(stamp - width + 1, 0) : stamp + width] = True
    return zone


def combine_shift(*, first: np.ndarray, second: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """The first model's statistic outside `zone` and the second model's inside it.

    Where the second model has no statistic inside the zone (its history starts later), the result
    is missing, which raises no alarm there.
    """
    return np.where(zone, second, first)


def shifts(
    *,
    variant: Variant,
    residual: Arrays,
    offset: Arrays,
    window: int = WINDOW,
    gap: int = 0,
) -> Arrays:
    """The step statistic of every series for one variant, from the (possibly injected) residual.

    `window` and `gap` are those of `level_shift`; the refit zone widens to cover both windows.
    """
    zone = refit_zone(width=window + gap)
    out: Arrays = {}
    for series, values in residual.items():
        first = level_shift(values=values, window=window, gap=gap)
        if variant == "X":
            second_values = offset.get(series, np.full(GRID_LENGTH, np.nan))
            second = level_shift(values=second_values, window=window, gap=gap)
            first = combine_shift(first=first, second=second, zone=zone)
        out[series] = first
    return out


def scales_for(
    *, variant: Variant, residual: Arrays, offset: Arrays, window: int = WINDOW, gap: int = 0
) -> dict[int, float]:
    """The odd-month noise scale of each series' statistic, from the uninjected residuals.

    The scale is the odd-month robust scale of the variant's own statistic.
    """
    return {
        series: robust_scale(shift=shift)
        for series, shift in shifts(
            variant=variant, residual=residual, offset=offset, window=window, gap=gap
        ).items()
    }


def variant_inputs(
    *,
    variant: Variant,
    residual: Arrays,
    offset: Arrays,
    scale: dict[int, float],
    window: int = WINDOW,
    gap: int = 0,
) -> DetectorInputs:
    """Detector inputs for one variant; X computes no statistic across a refit."""
    return DetectorInputs(
        window=window,
        shift=shifts(variant=variant, residual=residual, offset=offset, window=window, gap=gap),
        scale=scale,
        suppress_refits=variant == "PF14",
    )


def tracked_relative(
    *, variant: Variant, panel: dict, offset_panel: dict, residual: Arrays, offset: Arrays
) -> Arrays:
    """The relative residual that the level tracker reads.

    For X it is the second model's inside the 24 hours around a refit of the first model,
    so the tracker's pre-level and blocks do not straddle the first model's refit jump.
    """
    zone = refit_zone()
    out: Arrays = {}
    for series, values in residual.items():
        first = relative_residual(residual=values, expected=panel["expected"][series])
        if variant == "X" and series in offset:
            second = relative_residual(
                residual=offset[series], expected=offset_panel["expected"][series]
            )
            first = np.where(zone & np.isfinite(second), second, first)
        out[series] = first
    return out


SOURCE: Final[str] = os.environ.get("E147_SOURCE", "B")
"""Which residual the detectors read: `A16` (experiment 82's lagless XGBoost, all 16 series, the
sanity arm), `A15` (the same without series 12), or `B` (frozen convex model)."""
BLANK_SERIES: Final[int] = 12
"""The generator case with no convex-model fit; blanked in every arm but `A16`."""
SOURCES: Final[tuple[str, ...]] = ("A16", "A15", "B")


def _blank(*, arrays: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {s: np.full(GRID_LENGTH, np.nan) if s == BLANK_SERIES else a for s, a in arrays.items()}


def _gam_arrays() -> tuple[dict, dict]:
    """The stored frozen and innovation panels of the 1st-of-month and 16th-of-month models."""
    first = pickle.loads((GAM_DIR / "panel_1.pkl").read_bytes())
    second = pickle.loads((GAM_DIR / "panel_16.pkl").read_bytes())
    return first, second


def load_both(*, source: str | None = None) -> tuple[dict, dict]:
    """Return the panel and the offset panel that the chosen source's detectors read.

    `power` is always experiment 82's, so that events are drawn and injected identically in every
    source.
    """
    source = SOURCE if source is None else source
    panel = load_panel()
    offset_panel = load_offset_panel(panel=panel)
    if source == "A16":
        return panel, offset_panel
    if source == "A15":
        for part in (panel, offset_panel):
            part["residual"] = _blank(arrays=part["residual"])
        return panel, offset_panel
    first, second = _gam_arrays()
    out = []
    for stored in (first, second):
        frozen = stored["frozen"]
        expected = _blank(arrays=frozen["expected"])
        residual = _blank(arrays=frozen["residual"])
        part: dict = {"power": panel["power"], "expected": expected, "residual": residual}
        out.append(part)
    return out[0], out[1]


def check_variants() -> None:
    """Hand-built arrays: combining picks the right model at the zone edges."""
    zone = np.zeros(GRID_LENGTH, dtype=bool)
    zone[10:20] = True
    first, second = np.full(GRID_LENGTH, 1.0), np.full(GRID_LENGTH, 2.0)
    combined = combine_shift(first=first, second=second, zone=zone)
    if combined[9] != 1.0 or combined[10] != 2.0 or combined[19] != 2.0 or combined[20] != 1.0:
        raise SystemExit("STOP: combine_shift picks the wrong model at the zone edges")
    print("variants checked on hand-built arrays")
