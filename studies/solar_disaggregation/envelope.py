"""The census's envelope fit of a BMU's solar AC capacity, as the baseline for the physical fit.

The census study (`studies/solar_bmu_census/solar_estimate.py`) fits `min(r * a * c(t), a)` to the
99th percentile of output within each of nine bands of a shape `c(t)`. A study script may not import
another study's script, so this module repeats the fit's few lines. The shape `c(t)` is the cosine
of the solar zenith (scaled so its peak over the judged half-hours is 1) or CAMS irradiance divided
by the highest clear-sky irradiance of any hour.
"""

from itertools import pairwise
from typing import Final

import numpy as np

BAND_EDGES: Final[tuple[float, ...]] = tuple(round(0.1 * k, 1) for k in range(1, 11))
"""Edges of the nine bands of `c(t)`, from 0.1 to 1.0."""
MIN_HALF_HOURS_PER_BAND: Final[int] = 20
ENVELOPE_QUANTILE: Final[float] = 0.99
DC_AC_RATIO: Final[float] = 1.4


def envelope_ac_capacity_mw(
    *, shape: np.ndarray, output_mw: np.ndarray, dc_ac_ratio: float = DC_AC_RATIO
) -> float | None:
    """Fit the AC capacity `a` of `min(r * a * c, a)` to the upper envelope of output.

    Args:
        shape: `c(t)` at each half-hour, scaled so that its peak is 1.
        output_mw: The output at the same half-hours, in megawatts.
        dc_ac_ratio: The DC:AC ratio `r`.

    Returns:
        The fitted `a` in megawatts, or None when no band holds enough half-hours.
    """
    fitted_shape: list[float] = []
    envelope: list[float] = []
    for low, high in pairwise(BAND_EDGES):
        in_band = (shape >= low) & ((shape < high) | (high == BAND_EDGES[-1]))
        if in_band.sum() < MIN_HALF_HOURS_PER_BAND:
            continue
        fitted_shape.append(
            min(dc_ac_ratio * float(np.quantile(shape[in_band], ENVELOPE_QUANTILE)), 1.0)
        )
        envelope.append(float(np.quantile(output_mw[in_band], ENVELOPE_QUANTILE)))
    if not fitted_shape:
        return None
    shape_array, envelope_array = np.array(fitted_shape), np.array(envelope)
    fitted = float(shape_array @ envelope_array / (shape_array @ shape_array))
    return fitted if fitted > 0 else None
