"""W0: D0 on a residual with its half-hour-to-half-hour memory filtered out.

A first-order autoregressive coefficient is fitted per series and half-hour of day (48 slots, in
UTC) on odd-month residuals, and the whitened residual is the residual minus the coefficient times
the previous half-hour's residual. Everything after the filter is D0's.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from e00062_common import BLOCK, GRID_LENGTH, TUNING_MONTHS, stamp_months
from e00062_detectors import DetectorInputs, level_shift, robust_scale

MAX_COEFFICIENT: Final[float] = 0.98
"""Coefficients are clipped to 0 to this value, so the filter never amplifies or inverts."""


def fit_coefficients(*, residuals: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Per series, 48 coefficients: the odd-month least-squares slope of r(t) on r(t - 1).

    The slope goes through the origin and uses pairs where both residuals are present. Slot `h`
    holds the pairs whose later stamp is the `h`-th half-hour of the UTC day. A slot with no
    pairs, or with a zero denominator, gets coefficient 0.
    """
    odd = np.isin(stamp_months(), list(TUNING_MONTHS))
    slot = np.arange(GRID_LENGTH) % BLOCK
    out: dict[int, np.ndarray] = {}
    for series, values in residuals.items():
        later, earlier = values[1:], values[:-1]
        use = odd[1:] & np.isfinite(later) & np.isfinite(earlier)
        coefficients = np.zeros(BLOCK)
        for h in range(BLOCK):
            pick = use & (slot[1:] == h)
            denominator = float(np.sum(earlier[pick] ** 2))
            if denominator > 0:
                coefficients[h] = np.clip(
                    float(np.sum(later[pick] * earlier[pick])) / denominator, 0.0, MAX_COEFFICIENT
                )
        out[series] = coefficients
    return out


def whiten(*, residual: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """The residual minus the slot's coefficient times the previous half-hour's residual.

    The first stamp and every stamp whose residual or predecessor is missing are NaN.
    """
    slot = np.arange(GRID_LENGTH) % BLOCK
    out = np.full(GRID_LENGTH, np.nan)
    out[1:] = residual[1:] - coefficients[slot[1:]] * residual[:-1]
    return out


def prepare_whitened(
    *,
    residuals: dict[int, np.ndarray],
    coefficients: dict[int, np.ndarray],
    window: int,
    scale: dict[int, float] | None = None,
) -> DetectorInputs:
    """W0's inputs: the 24-hour (or other window) step of every whitened series and its scale.

    A given `scale` keeps the uninjected noise scale, as `prepare` does for D0.
    """
    shift = {
        s: level_shift(values=whiten(residual=r, coefficients=coefficients[s]), window=window)
        for s, r in residuals.items()
    }
    return DetectorInputs(
        window=window,
        shift=shift,
        scale=scale or {s: robust_scale(shift=d) for s, d in shift.items()},
    )
