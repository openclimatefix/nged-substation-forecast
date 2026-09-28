"""Turning a forecast's running totals into average rates per forecast step.

ECMWF's precipitation and radiation fields are totals accumulated since the start of the forecast.
The live pipeline receives them already converted to per-second rates by Dynamical.org, so an
archive built from ECMWF's own files has to apply the same conversion to give the same numbers.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

PRECIPITATION_INVALID_BELOW_MM_PER_S: Final[float] = -7e-5
"""Dynamical.org's threshold for precipitation, in millimetres per second (= kg m-2 s-1)."""

RADIATION_INVALID_BELOW_W_PER_M2: Final[float] = -50.0
"""Dynamical.org's threshold for downward radiation, in watts per square metre."""


@dataclass(frozen=True)
class DeaccumulatedRates:
    """The rates from `deaccumulate_to_rates`, with a count of what was corrected."""

    rates: np.ndarray
    """The average rate over the step ending at each index. Index 0 is NaN."""
    clamped: np.ndarray
    """A boolean array: the rate was slightly negative and was set to zero."""
    invalid: np.ndarray
    """A boolean array: the rate was more negative than the threshold and was set to NaN."""


def deaccumulate_to_rates(
    *,
    accumulations: np.ndarray,
    elapsed_seconds: np.ndarray,
    invalid_below_rate: float,
) -> DeaccumulatedRates:
    """Convert accumulated totals to the average rate over the step ending at each forecast step.

    This is the rule Dynamical.org applies (`deaccumulate_to_rates_inplace` in its `reformatters`
    repository, with the accumulation never reset). The rate at step `t` is the difference between
    the accumulation at `t` and at the step before, divided by the seconds between the two steps,
    so the change from 3-hourly to 6-hourly steps needs no special case. The accumulation before
    the first step is taken as zero, so the rate at step 1 is `accumulations[1] / seconds` and
    step 0 has no rate and is NaN. A negative rate can only come from the quantisation of the
    packed values: one above `invalid_below_rate` is set to zero, and one at or below it is set to
    NaN.

    Args:
        accumulations: The totals, with the forecast step as the first axis and any other axes
            after it. A tp total must be in millimetres so the rate comes out in mm per second.
        elapsed_seconds: The seconds from the forecast start to each step, one per step.
        invalid_below_rate: The threshold in the units of the rate. Negative.

    Returns:
        The rates and two masks of the corrected values.

    Raises:
        ValueError: If `elapsed_seconds` does not have one value per step, or does not increase.
    """
    if elapsed_seconds.shape != accumulations.shape[:1]:
        raise ValueError("elapsed_seconds needs one value per step of the first axis")
    step_seconds = np.diff(elapsed_seconds)
    if np.any(step_seconds <= 0):
        raise ValueError("elapsed_seconds must increase")
    totals = accumulations.astype(np.float64)
    previous = np.zeros_like(totals[1:])
    previous[1:] = totals[1:-1]
    seconds_shape = (-1,) + (1,) * (totals.ndim - 1)
    rates = np.full_like(totals, np.nan)
    rates[1:] = (totals[1:] - previous) / step_seconds.reshape(seconds_shape)
    negative = rates < 0
    clamped = negative & (rates > invalid_below_rate)
    invalid = negative & ~clamped
    rates[clamped] = 0.0
    rates[invalid] = np.nan
    return DeaccumulatedRates(rates=rates, clamped=clamped, invalid=invalid)
