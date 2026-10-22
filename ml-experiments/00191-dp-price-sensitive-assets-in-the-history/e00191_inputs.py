"""The ensemble inputs of experiment 191: arrays per site, setup, and calibration.

Copied from experiment 152's `e00152_ens_inputs.py` (itself copied from experiments 111 and 136)
so this folder imports only itself. The decision Primaries read experiment 111's stored ensemble
arrays and calibrations, and D8 reads experiment 152's. Nothing is written.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00191_common as common
import numpy as np
from studies.dp_ens_weather import (
    AnalysisProxy,
    Calibration,
    EnsSiteRun,
    WeatherContext,
    ghi_climatology,
)
from studies.dp_features import FeatureCache

CLIMATOLOGY_BEFORE: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
"""CAMS irradiance after this date never enters the climatology the solar shrink pulls towards."""


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


def store_file(*, site: str) -> Path:
    """Return the array file of a site: experiment 111's if it has one, else experiment 152's."""
    own = common.E111_WORK / "ens_store" / f"{site}.npz"
    return own if own.exists() else common.E152_WORK / "ens_store" / f"{site}.npz"


def load_setup(*, cache: FeatureCache, site: str) -> SiteSetup:
    """Load a site's runs and build its proxy and climatology, with an identity calibration."""
    with np.load(store_file(site=site)) as stored:
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


def calibrated_setup(*, cache: FeatureCache, site: str) -> SiteSetup:
    """Return a site's setup with the calibration stored by experiment 111 or 152."""
    setup = load_setup(cache=cache, site=site)
    stored = common.E111_WORK / "calibration" / f"{site}.npz"
    if not stored.exists():
        stored = common.E152_WORK / "calibration" / f"{site}.npz"
    with np.load(stored) as arrays:
        calibration = Calibration(
            temperature_shift_c=arrays["shift"],
            irradiance_ratio=arrays["ratio"],
            solar_shrink=np.ones(4),
        )
    return setup.with_calibration(calibration=calibration)
