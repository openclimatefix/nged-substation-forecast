"""Detector v2: experiment 196's detector with night evidence, an edge-shape test, and a price gate.

Every feature is a function from the frozen detector's posterior to a new posterior, so that the
scores of experiment 196 (`score`, `false_alarms`) apply unchanged: a half-hour that a feature
removes gets a point mass on zero units.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00199_common import GRID_START, Q_FLOOR, times
from e00199_hmm import Noise, Params, Posterior, unit_means
from e00199_inject import Injection
from e00199_pipeline import Series, run_loop
from e00199_scoring import flagged
from studies.solar import zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FIT_196_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate"
    / "fits_fixed.pkl"
)  # fmt: skip
PRICES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00191-dp-price-sensitive-assets-in-the-history" / "work"
    / "prices.parquet"
)  # fmt: skip
SITE_LATLON: Final[dict[str, tuple[float, float]]] = {
    "D3": (52.976, -0.026),  # Sleaford Road Boston, approximate; night changes little over 0.1 deg
    "D5": (52.921421, -0.05731), "D8": (52.96159, -0.01362), "D10": (53.16853, 0.10461),
    "D11": (53.218079, -0.11218), "D14": (52.969059, -0.01441), "D16": (52.915459, -0.18896),
}  # fmt: skip
NIGHT_ZENITH_DEG: Final[float] = 93.0
"""Night is a solar elevation below -3 degrees."""
SHARP_SHARE: Final[float] = 0.6
"""An edge is sharp when one half-hour carries at least this share of the step."""
MIN_STEP_UNITS: Final[float] = 0.5
SPREAD_GATE_QUANTILE: Final[float] = 0.2
PRICE_FIRST: Final[str] = "2023-06-01"
PRICE_LAST: Final[str] = "2026-06-30"


class _Remap(pickle.Unpickler):
    """Load experiment 196's pickles under this folder's module names."""

    def find_class(self, module: str, name: str) -> type:
        return super().find_class(module.replace("e00196", "e00199"), name)


def load_frozen_fit() -> tuple[Noise, Params]:
    """Experiment 196's D8 noise and wide-prior detector parameters."""
    import io

    obj = _Remap(io.BytesIO(FIT_196_PATH.read_bytes())).load()
    return obj["noise"], obj["fits"]["wide"]["params"]


def night_mask(*, label: str, n: int) -> np.ndarray:
    """True where the sun is more than 3 degrees below the horizon at the half-hour's midpoint."""
    lat, lon = SITE_LATLON[label]
    mid = times(n=n).dt.offset_by("15m")
    return zenith(stamps=mid, latitude=lat, longitude=lon) > NIGHT_ZENITH_DEG


def spread_array(*, n: int) -> tuple[np.ndarray, float]:
    """Spark spread (pounds per MWh) on the grid, NaN without a price, and the gate threshold."""
    table = pl.read_parquet(PRICES_PATH).select("start", "spread")
    out = np.full(n, np.nan)
    idx = ((table["start"].dt.epoch("s").to_numpy() - int(GRID_START.timestamp())) // 1800).astype(
        int
    )
    keep = (idx >= 0) & (idx < n)
    out[idx[keep]] = table["spread"].to_numpy()[keep]
    window = table.filter(
        pl.col("start").is_between(
            pl.lit(PRICE_FIRST).str.to_datetime(time_zone="UTC"),
            pl.lit(PRICE_LAST).str.to_datetime(time_zone="UTC"),
        )
    )["spread"].drop_nulls()
    return out, float(window.quantile(SPREAD_GATE_QUANTILE) or 0.0)


def injection_spread(*, inj: Injection, spread: np.ndarray, first: int) -> np.ndarray:
    """Spread at the time G2 recorded each injected half-hour (K1 in place, K2 from its source)."""
    out = spread.copy()
    for w in np.unique(inj.window_id[inj.window_id >= 1]):
        idx = np.flatnonzero(inj.window_id == w)
        out[idx] = spread[first : first + len(idx)]
    return out


def mask_to_night(*, s: Series, night: np.ndarray) -> Series:
    """Same site with the reading of every non-night half-hour missing."""
    y = np.where(night, s.y, np.nan)
    return s.with_reading(y)


def detect(
    *, s: Series, noise: Noise, params: Params, night: np.ndarray | None
) -> tuple[Posterior, np.ndarray]:
    """Run the frozen loop; with `night`, only night half-hours carry evidence (feature N2).

    Returns the posterior and the zero-unit mean-removed reading (the edge test reads it).
    """
    use = s if night is None else mask_to_night(s=s, night=night)
    offset, post = run_loop(s=use, noise=noise, params=params, nan_to_null=night is not None)
    base = s.y - unit_means(load=s.load, g=0.0, q=Q_FLOOR)[:, 0] - offset
    return post, base


def spells(*, flag: np.ndarray, join_gap: int = 1) -> list[tuple[int, int]]:
    """Maximal runs of True, joined across gaps of up to `join_gap` half-hours (inclusive ends)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flag.astype(int), [0]])))
    runs = [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= join_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def edge_is_sharp(*, x: np.ndarray, boundary: int, unit: float, share: float = SHARP_SHARE) -> bool:
    """True when the edge between half-hours `boundary` and `boundary + 1` is one sharp jump.

    The step is the mean of the three half-hours after the edge (skipping the one next to it) minus
    the mean of the three before; it must be at least half a unit, and the largest one-half-hour
    change within one half-hour of the edge must carry `share` of it. An edge beside missing data
    cannot be tested and passes.
    """
    lo, hi = boundary - 3, boundary + 5
    if lo < 0 or hi >= len(x):
        return True
    before, after = x[boundary - 2 : boundary + 1], x[boundary + 2 : boundary + 5]
    near = x[boundary - 1 : boundary + 3]
    if not (np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(near).all()):
        return True
    step = after.mean() - before.mean()
    if abs(step) < MIN_STEP_UNITS * unit:
        return False
    biggest = np.abs(np.diff(near)).max()
    return bool(biggest >= share * abs(step))


@dataclass(frozen=True)
class Gates:
    """Which features v2 applies on top of the frozen detector's flags."""

    shape: bool = False
    price: bool = False
    share: float = SHARP_SHARE


def apply_gates(
    *,
    post: Posterior,
    x: np.ndarray,
    unit: float,
    spread: np.ndarray,
    gate_threshold: float,
    gates: Gates,
) -> np.ndarray:
    """Posterior with the half-hours of rejected spells set to a point mass on zero units."""
    smoothed = post.smoothed.copy()
    if not (gates.shape or gates.price):
        return smoothed
    flag = flagged(smoothed=smoothed)
    for a, b in spells(flag=flag):
        keep = True
        if gates.shape:
            keep &= edge_is_sharp(x=x, boundary=a - 1, unit=unit, share=gates.share) and (
                edge_is_sharp(x=x, boundary=b, unit=unit, share=gates.share)
            )
        if keep and gates.price:
            seg = spread[a : b + 1]
            seg = seg[np.isfinite(seg)]
            keep = not (seg.size > 0 and seg.mean() <= gate_threshold)
        if not keep:
            smoothed[a : b + 1] = 0.0
            smoothed[a : b + 1, 0] = 1.0
    return smoothed
