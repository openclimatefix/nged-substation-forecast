"""A substation's ensemble runs and weather layer, as experiment 111 built them.

`SiteSetup` holds a substation's 821 ensemble runs, its analysis proxy, its CAMS climatology, and
its calibration. The member forecasts themselves are cached by `e00208_g_members.py`.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final

import e00208_g_common as common
import numpy as np
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import (
    AnalysisProxy,
    Calibration,
    EnsSiteRun,
    WeatherContext,
    ghi_climatology,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import ResidualSpread

CLIMATOLOGY_BEFORE: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
"""CAMS irradiance after this date never enters the climatology the solar shrink pulls towards."""
MIN_WINDOW_HALF_HOURS: Final[int] = 48
"""A run needs a day of grid after its origin; the last run of the scored year has less, because
the grid ends before the held-out period, and is left out."""
ZERO_SPREAD: Final[LeadHourSpread] = LeadHourSpread(
    by_lead_band=tuple(ResidualSpread(table=np.zeros((4, 4, 11))) for _ in range(4)),
    factors=np.ones(4),
)
"""A spread of zero, for a forecast that only needs the corrected mean."""


@dataclass(frozen=True)
class SiteSetup:
    """A substation's ensemble runs and the weather layer built from them.

    Attributes:
        site: The substation's label.
        run_dates: The 00:00 UTC initialisation time of each run.
        origins: Each run's `power_fcst_init_time`.
        runs: Each run's ensemble weather on the cache's grid.
        context: The calibration, climatology, and analysis proxy.
    """

    site: str
    run_dates: list[datetime]
    origins: list[datetime]
    runs: list[EnsSiteRun]
    context: WeatherContext

    def with_calibration(self, *, calibration: Calibration) -> SiteSetup:
        """Return the setup with another calibration."""
        return replace(self, context=replace(self.context, calibration=calibration))

    def indices(self, *, periods: set[str]) -> list[int]:
        """Return the runs that belong to the named periods."""
        return [
            index
            for index, run in enumerate(self.run_dates)
            if common.period_of_run(run=run) in periods
        ]


def load_setup(*, cache: FeatureCache, site: str) -> SiteSetup:
    """Load a substation's runs and build its proxy and climatology, with an identity calibration.

    Args:
        cache: The feature cache.
        site: The substation's label.

    Returns:
        The setup.
    """
    with np.load(common.STORE_DIR / f"{site}.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    run_dates = common.run_dates()
    origins = [common.origin_of(run=run) for run in run_dates]
    positions = np.searchsorted(cache.end_times, arrays["origins"])
    runs = [
        EnsSiteRun(
            origin_position=int(position),
            temperature_c=arrays["temperature"][index],
            shortwave_w_m2=arrays["shortwave"][index],
        )
        for index, position in enumerate(positions)
    ]
    context = WeatherContext(
        site=site,
        calibration=Calibration.identity(),
        climatology=ghi_climatology(cache=cache, site=site, before=CLIMATOLOGY_BEFORE),
        proxy=AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )
    return SiteSetup(site=site, run_dates=run_dates, origins=origins, runs=runs, context=context)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    levels = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)
