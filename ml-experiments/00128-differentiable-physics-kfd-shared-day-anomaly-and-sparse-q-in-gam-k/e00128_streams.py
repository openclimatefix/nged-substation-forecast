"""A substation's residual stream on the analysis proxy, for experiment 128.

The stream is observed power minus the chained monthly fits' prediction on the analysis proxy's
weather (the control member of the latest run, leads 9 to 33 hours), as experiment 111's filter
reads it (`studies.dp_ens.proxy_residual_stream`). Each fit supplies only the rows after its own
cut, so no residual is in-sample. The stream also carries the PV regressor of the same fits (their
PV effect at a weight of 1), which the known-answer gate uses to shape an injected change.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00128_common as common
import numpy as np
from studies.dp_effects_filter import PV_INDEX, stitched_effect_regressors
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import proxy_cache
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit

FIT_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The hyperparameters are fitted on a stream cut here, so no scored-year residual reaches a fit."""


@dataclass(frozen=True)
class SiteStreams:
    """One substation's residual stream on the whole grid.

    Attributes:
        site: The substation's label.
        residual: Observed minus the chained fits' prediction on the proxy, in points; `NaN` where
            no residual can be made.
        pv_regressor: The PV effect of the same fits at a weight of 1, in points, `NaN` where the
            residual is `NaN`.
        band: The hour band of each grid row.
        day: The local calendar day of each grid row (`common.local_day_index`).
        tuning_mask: True on grid rows that count in the likelihood: finite residuals in the
            tuning months.
    """

    site: str
    residual: np.ndarray
    pv_regressor: np.ndarray
    band: np.ndarray
    day: np.ndarray
    tuning_mask: np.ndarray

    def fit_residual(self, *, cache: FeatureCache) -> np.ndarray:
        """Return the residual stream cut at `FIT_END`, which is all a fit may see."""
        return np.where(cache.end_times <= naive_utc(moment=FIT_END), self.residual, np.nan)


def build_site_streams(
    *, cache: FeatureCache, setup: common.SiteSetup, fits: list[GamFit]
) -> SiteStreams:
    """Build a site's stream, PV regressor, and tuning mask.

    Args:
        cache: The feature cache.
        setup: The substation's setup.
        fits: The monthly fits of `gamk`, ascending by cut.

    Returns:
        The streams.
    """
    proxy = proxy_cache(cache=cache, context=setup.context)
    residual = proxy_residual_stream(cache=proxy, fits=fits, site=setup.site)
    pieces = stitched_effect_regressors(cache=proxy, fits=fits, site=setup.site)
    return SiteStreams(
        site=setup.site,
        residual=residual,
        pv_regressor=np.where(np.isfinite(residual), pieces.contributions[:, PV_INDEX], np.nan),
        band=cache.calendar.band.astype(np.int64),
        day=common.local_day_index(cache=cache),
        tuning_mask=np.isfinite(residual)
        & common.in_months(cache=cache, months=common.TUNING_MONTHS),
    )


def load_site_streams(*, site: str) -> SiteStreams:
    """Build a site's streams from experiment 111's stored inputs."""
    cache = common.load_cache()
    setup = common.load_setup(cache=cache, site=site)
    return build_site_streams(cache=cache, setup=setup, fits=common.load_fits(site=site))
