"""The unfiltered forecast of every run, built once per substation, and the arms' corrections.

The mean forecast of experiment 136's arm A2 is the T-all fit's prediction on the ensemble-mean
weather, plus the Kalman filter's level and transient at the run's cutoff. Only the filter differs
between this experiment's arms, so `build_base` computes the prediction without the filter once per
substation and `arm_frame` adds an arm's correction to it. `build_base` and `arm_frame` follow
`e00197_ens.forecast_mean_run` and `e00197_ens.forecast_frame` line for line; the control in
`e00197_run.py` checks that arm A0 reproduces the stored A2.
"""

import json
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00197_common as common
import e00197_ens as ens
import e00197_filters as filters
import e00197_tall as tall
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    CorrectionTrace,
    correction_at,
    usable_cutoff_positions,
)
from studies.dp_ens import choose_fit
from studies.dp_ens_weather import Calibration, RunWindow, proxy_cache
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import QUANTILE_COLUMNS

BASE_DIR: Final[Path] = common.WORK_DIR / "base"
"""One parquet per substation: every run's unfiltered forecast."""
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")


def read_tuned() -> dict:
    """Return experiment 111's tuned settings (solar shrink)."""
    return json.loads((common.SOURCE_WORK_DIR / "tuned.json").read_text())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, without its solar shrink."""
    with np.load(common.SOURCE_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        return Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )


def pool_settings() -> filters.FilterSettings:
    """Return experiment 133's single shared filter cell (POOL), as experiment 136's A2 uses."""
    hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


@dataclass(frozen=True)
class SiteContext:
    """A substation's stored inputs: the ensemble setup, the cache, and the monthly fits."""

    site: str
    cache: FeatureCache
    setup: ens.SiteSetup
    fits: list


def load_site(*, site: str, cache: FeatureCache | None = None) -> SiteContext:
    """Load a substation's ensemble setup with experiment 111's calibration and the T-all fits."""
    cache = cache if cache is not None else common.load_cache()
    calibration = load_calibration(site=site).with_solar_shrink(
        solar_shrink=read_tuned()["solar_shrink"]
    )
    setup = ens.load_setup(cache=cache, site=site).with_calibration(calibration=calibration)
    return SiteContext(
        site=site, cache=cache, setup=setup, fits=tall.load_fits(cadence="monthly", site=site)
    )


def tall_residual_stream(*, cache: FeatureCache, fits: list, site: str) -> np.ndarray:
    """Return observed minus the chained T-all fits' forecast on a cache, `NaN` where none.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
    """
    if not fits or any(later.cut <= earlier.cut for earlier, later in pairwise(fits)):
        msg = "the residual stream needs fits with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cuts = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=fit.cut), side="right")) - 1
        for fit in fits
    ]
    ends = [*cuts[1:], len(stream) - 1]
    for fit, cut, end in zip(fits, cuts, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            stream[rows] = features.y[rows] - tall.predict_tall(cache=cache, fit=fit, index=rows)
    return stream


def residual_stream(*, context: SiteContext) -> np.ndarray:
    """Return the site's residual stream against the analysis proxy (what the filter reads)."""
    proxy = proxy_cache(cache=context.cache, context=context.setup.context)
    return tall_residual_stream(cache=proxy, fits=context.fits, site=context.site)


def build_base(*, context: SiteContext) -> pl.DataFrame:
    """Return every run's unfiltered forecast rows at a substation, for all three periods.

    Columns: those of `ens.forecast_frame` except the quantiles and the filter's state, plus
    `period`, `base_mean` (the prediction without the filter), `target` (the grid position of the
    valid time), and `cutoff` (the newest grid position the run may read).
    """
    cache, setup = context.cache, context.setup
    features = cache.sites[setup.site]
    first = context.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    frames = []
    for period in PERIODS:
        for index in setup.indices(periods={period}):
            origin = setup.origins[index]
            run = setup.runs[index]
            if origin < first:
                continue
            if len(cache.end_times) - run.origin_position - 1 < ens.MIN_WINDOW_HALF_HOURS:
                continue
            fit = choose_fit(fits=context.fits, origin=origin)
            window = RunWindow.build(cache=cache, context=setup.context, run=run)
            driven = window.driven_caches(
                context=setup.context, run=ens.mean_run(run=run), members=[0]
            )[0]
            n_targets = window.window.stop - window.window.start
            means = ens.predict_mean(driven=driven, fit=fit, index=np.arange(n_targets))
            target = np.arange(window.window.start, window.window.stop)
            keep = features.present[target] & np.isfinite(features.y[target])
            if not keep.any():
                continue
            rows = target[keep]
            n = len(rows)
            cutoff = usable_cutoff_positions(
                end_times=cache.end_times,
                initialisation=np.array([naive_utc(moment=origin)]),
                delay_hours=DEFAULT_DELAY_HOURS,
            )[0]
            valid = cache.end_times[rows]
            origin64 = np.datetime64(origin.replace(tzinfo=None), "us")
            frames.append(
                pl.DataFrame(
                    {
                        "time_series_id": np.full(n, features.table.time_series_id, dtype=np.int32),
                        "run": np.full(n, index, dtype=np.int32),
                        "power_fcst_init_time": pl.Series(
                            np.full(n, origin64)
                        ).dt.replace_time_zone("UTC"),
                        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
                        "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(
                            np.float64
                        ),
                        "seed": np.zeros(n, dtype=np.int8),
                        "actual": features.y[rows].astype(np.float32),
                        "base_mean": np.asarray(means, dtype=np.float64)[keep],
                        "hour_band": cache.calendar.band[rows].astype(np.int8),
                        "quarter": cache.calendar.quarter[rows].astype(np.int8),
                        "switching_mask": features.switching_mask[rows],
                        "target": rows.astype(np.int64),
                        "cutoff": np.full(n, cutoff, dtype=np.int64),
                        "period": np.full(n, period),
                    }
                )
            )
    return pl.concat(frames)


def base_path(*, site: str) -> Path:
    """Return the file that holds a substation's unfiltered forecasts."""
    return BASE_DIR / f"{site}.parquet"


def correction_of(*, base: pl.DataFrame, trace: CorrectionTrace) -> tuple[np.ndarray, np.ndarray]:
    """Return the filter's level and transient at every row of `base`."""
    correction = correction_at(
        trace=trace,
        cutoff=base["cutoff"].to_numpy(),
        target=base["target"].to_numpy(),
        band=base["hour_band"].to_numpy().astype(np.int64),
    )
    return correction.level, correction.anomaly


def arm_mean_frame(*, base: pl.DataFrame, trace: CorrectionTrace) -> pl.DataFrame:
    """Return `base` with `mean` (prediction plus correction), `level`, and `anomaly`."""
    level, anomaly = correction_of(base=base, trace=trace)
    mean = base["base_mean"].to_numpy() + level + anomaly
    return base.with_columns(
        pl.Series("mean", mean.astype(np.float32)),
        pl.Series("level", level.astype(np.float32)),
        pl.Series("anomaly", anomaly.astype(np.float32)),
    )


def arm_frame(*, mean_frame: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's forecast frame: the corrected mean and quantiles from its own spread.

    The spread is fitted on the arm's tuning-month rows outside the switching events, every factor
    1, as experiment 136 builds it.
    """
    tuning = mean_frame.filter(pl.col("period") == "tuning")
    spread = ens.fit_spread_from_frame(frame=tuning)
    mean = mean_frame["mean"].to_numpy().astype(np.float64)
    quantiles = spread.quantiles(
        mean=mean,
        band=mean_frame["hour_band"].to_numpy().astype(np.int64),
        quarter=mean_frame["quarter"].to_numpy().astype(np.int64),
        lead_hours=mean_frame["lead_hours"].to_numpy(),
    )
    return mean_frame.with_columns(
        pl.Series(column, quantiles[:, position].astype(np.float32))
        for position, column in enumerate(QUANTILE_COLUMNS)
    )
