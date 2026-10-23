"""The signed history of a child Primary that reports apparent power, for arm (d).

The reading of such a child is the magnitude `|S|` of its flow, so a child with reverse flow reads
positive at midday when its flow is negative. The meter fit of experiment 152 (the convex demand
model of experiment 110 fitted through the meter) predicts the signed real power `P` and the
reading. Arm (d) turns the reading into a signed active power: multiply by the power factor and set
the sign negative on a half-hour where the fitted `P` lies below `-z * sigma`, with `z` the normal
quantile of the chosen probability and `sigma` the standard deviation of the fit's residuals
`reading - fitted reading` over the 12 months before the fit's cut.

**No sign uses a fit trained on a later reading.** A half-hour at time `t` takes its sign from the
monthly meter fit with the latest cut at or before `t`, and a half-hour before the first cut takes
its sign from the first fit, which is only ever used by fits of a cut at or after that first cut.
"""

import pickle
from datetime import datetime
from statistics import NormalDist

import e00223_common as common
import numpy as np
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_meter import MeterFit, predict_meter

WINDOW_DAYS: int = 365


def load_meter_fits(*, child: str) -> list[MeterFit]:
    """Return experiment 152's monthly meter fits of a child, ascending by cut."""
    return pickle.loads((common.E152_FITS_DIR / f"{child}_meter.pkl").read_bytes())


def _cut_row(*, cache: FeatureCache, cut: datetime) -> int:
    return int(np.searchsorted(cache.end_times, naive_utc(moment=cut), side="right")) - 1


def fitted_flow(*, cache: FeatureCache, child: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the fitted signed flow and the residual standard deviation, per grid row.

    Rows with no usable weather input are `NaN`.
    """
    fits = load_meter_fits(child=child)
    features = cache.sites[child]
    n = len(cache.end_times)
    flow = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    cut_rows = [_cut_row(cache=cache, cut=fit.fit.cut) for fit in fits]
    starts = [0, *cut_rows[1:]]
    ends = [*cut_rows[1:], n - 1]
    for fit, start, end, cut_row in zip(fits, starts, ends, cut_rows, strict=True):
        rows = np.arange(start, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
        )
        rows = rows[usable]
        if not len(rows):
            continue
        result = predict_meter(cache=cache, meter=fit, index=rows)
        flow[rows] = result.real_power
        window = np.arange(max(cut_row - WINDOW_DAYS * 48, 0), cut_row + 1)
        window = window[
            np.isfinite(features.y[window])
            & np.isfinite(features.smoothed_temperature_c[window]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[window])
            & np.isfinite(features.temperature_c[window])
        ]
        residual = (
            features.y[window] - predict_meter(cache=cache, meter=fit, index=window).apparent_power
        )
        sigma[rows] = float(np.std(residual)) if len(window) else np.nan
    return flow, sigma


def sign_of(*, cache: FeatureCache, child: str, probability: float) -> np.ndarray:
    """Return +1 or -1 per grid row: -1 where the fitted flow is reverse at the probability."""
    flow, sigma = fitted_flow(cache=cache, child=child)
    z = NormalDist().inv_cdf(probability)
    reverse = np.isfinite(flow) & np.isfinite(sigma) & (flow < -z * sigma)
    return np.where(reverse, -1.0, 1.0)


def child_series(*, cache: FeatureCache, child: str, variant: str) -> np.ndarray:
    """Return a child's reading in points of its own effective capacity, as a variant sees it.

    `naive` is the stored reading. `fold` and `fold50` convert a child that reports apparent power
    to a signed active power; a child that reports real power is unchanged.
    """
    y = cache.sites[child].y
    if variant == "naive" or child not in common.ALL_MVA:
        return y
    sign = sign_of(cache=cache, child=child, probability=common.SIGN_PROBABILITY[variant])
    return common.POWER_FACTOR * sign * y
