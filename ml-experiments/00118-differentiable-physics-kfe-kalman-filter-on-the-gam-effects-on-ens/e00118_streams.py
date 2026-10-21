"""A substation's residual stream and effect regressors on the analysis proxy, for experiment 118.

The stream is observed power minus the chained monthly fits' prediction on the analysis proxy's
weather (the control member of the latest run, leads 9 to 33 hours), as experiment 111's filter
reads it. The regressors are the fits' effects on the same weather
(`studies.dp_effects_filter.stitched_effect_regressors`). Each fit supplies only the rows after its
own cut, so no residual is in-sample.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00118_common as common
import numpy as np
from studies.dp_calendar import DAY_TYPES
from studies.dp_effects_filter import (
    N_BANDS,
    N_WEIGHTS,
    regressor_scale,
    stitched_effect_regressors,
)
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import proxy_cache
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit

FIT_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The noise matrix is fitted on a stream cut here, so no scored-year residual reaches the fit."""
BANK_HOLIDAY: Final[int] = DAY_TYPES.index("bank_holiday")
"""The index of the bank-holiday day type."""
STREAM_TOLERANCE: Final[float] = 1e-9
"""How closely the decomposition's residual must match `proxy_residual_stream`."""


@dataclass(frozen=True)
class SiteStreams:
    """One substation's stream and regressors on the whole grid.

    Attributes:
        site: The substation's label.
        residual: Observed minus the chained fits' prediction on the proxy, in points; `NaN` where
            no residual can be made.
        contributions: The 14 regressors at a weight of 1, shape (grid rows, 14), `NaN` where the
            residual is `NaN`.
        band: The hour band of each grid row.
        tuning_mask: True on grid rows that count in the likelihood: finite residuals in the
            tuning months.
        scale: The root mean square of each regressor over the tuning mask, shape (14,).
        holiday_day: True on grid rows of a holiday day: a named holiday period (each bank holiday,
            Easter, bridge days, the Christmas to New Year run) or the bank-holiday day type.
    """

    site: str
    residual: np.ndarray
    contributions: np.ndarray
    band: np.ndarray
    tuning_mask: np.ndarray
    scale: np.ndarray
    holiday_day: np.ndarray

    @property
    def standardised(self) -> np.ndarray:
        """The regressors divided by their scale, shape (grid rows, 14)."""
        return self.contributions / self.scale

    def arm_residual(self, *, skip_holidays: bool) -> np.ndarray:
        """Return the residual stream an arm's filter reads.

        Holiday-day readings are missing for a day-type-aware arm.
        """
        return np.where(self.holiday_day, np.nan, self.residual) if skip_holidays else self.residual

    def fit_residual(self, *, cache: FeatureCache, skip_holidays: bool = False) -> np.ndarray:
        """Return the arm's residual stream cut at `FIT_END`, which is all a noise fit may see."""
        return np.where(
            cache.end_times <= naive_utc(moment=FIT_END),
            self.arm_residual(skip_holidays=skip_holidays),
            np.nan,
        )


def build_site_streams(
    *, cache: FeatureCache, setup: common.SiteSetup, fits: list[GamFit]
) -> SiteStreams:
    """Build a site's stream, regressors, tuning mask, and regressor scale.

    Args:
        cache: The feature cache.
        setup: The substation's setup.
        fits: The monthly fits of `gamk`, ascending by cut.

    Returns:
        The streams.

    Raises:
        ValueError: If the decomposition's residual disagrees with `proxy_residual_stream`.
    """
    proxy = proxy_cache(cache=cache, context=setup.context)
    pieces = stitched_effect_regressors(cache=proxy, fits=fits, site=setup.site)
    observed = cache.sites[setup.site].y
    prediction = pieces.remainder + pieces.contributions[:, N_BANDS:].sum(axis=1)
    residual = observed - prediction
    reference = proxy_residual_stream(cache=proxy, fits=fits, site=setup.site)
    both = np.isfinite(residual) & np.isfinite(reference)
    if not np.array_equal(np.isfinite(residual), np.isfinite(reference)) or (
        np.abs(residual[both] - reference[both]).max() > STREAM_TOLERANCE
    ):
        msg = f"{setup.site}: the decomposition's residual differs from proxy_residual_stream"
        raise ValueError(msg)
    tuning_mask = np.isfinite(residual) & common.in_months(cache=cache, months=common.TUNING_MONTHS)
    scale = regressor_scale(contributions=pieces.contributions, mask=tuning_mask)
    return SiteStreams(
        site=setup.site,
        residual=residual,
        contributions=pieces.contributions,
        band=cache.calendar.band.astype(np.int64),
        tuning_mask=tuning_mask,
        scale=scale,
        holiday_day=(cache.calendar.holiday_period >= 0)
        | (cache.calendar.day_type == BANK_HOLIDAY),
    )


def n_weights() -> int:
    """Return the number of weights, for scripts that size arrays."""
    return N_WEIGHTS
