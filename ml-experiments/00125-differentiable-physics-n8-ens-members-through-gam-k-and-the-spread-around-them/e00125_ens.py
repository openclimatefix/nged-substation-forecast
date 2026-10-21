"""The ensemble-driven forecast of one substation, shared by experiment 125's scripts.

Copied from experiment 111.

`SiteSetup` holds a substation's 821 ensemble runs, its analysis proxy, its CAMS climatology, and
its calibration. `forecast_frame` forecasts a chosen set of runs with `studies.dp_ens.forecast_run`
and returns rows with the columns the scorers read.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Final

import e00125_common as common
import numpy as np
import polars as pl
from studies.dp_assimilation import CorrectionTrace
from studies.dp_ens import LeadHourSpread, forecast_run
from studies.dp_ens_weather import (
    HORIZON_HALF_HOURS,
    AnalysisProxy,
    Calibration,
    EnsSiteRun,
    WeatherContext,
    ghi_climatology,
    lead_band_of_hours,
    season_of_months,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS, GamFit, ResidualSpread

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
    with np.load(common.store_path(site=site)) as stored:
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


def fit_calibration(*, cache: FeatureCache, setup: SiteSetup) -> Calibration:
    """Fit the calibration of a substation on the tuning months' runs only.

    The ensemble mean of each tuning run is compared with ERA5 temperature and CAMS irradiance at
    the same half-hours, for the first 168 hours of the run.

    Args:
        cache: The feature cache.
        setup: The substation's setup.

    Returns:
        The calibration with a solar shrink of 1.
    """
    features = cache.sites[setup.site]
    pieces: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "origin",
            "season",
            "lead",
            "band",
            "ens_t",
            "era5_t",
            "ens_g",
            "cams_g",
        )
    }
    lead_hours = np.arange(1, HORIZON_HALF_HOURS + 1) / 2.0
    lead = lead_band_of_hours(lead_hours=lead_hours)
    for index in setup.indices(periods={"tuning"}):
        run = setup.runs[index]
        rows = np.arange(run.origin_position + 1, run.origin_position + 1 + HORIZON_HALF_HOURS)
        months = cache.calendar.local_date[rows].astype("datetime64[M]").astype(np.int64) % 12 + 1
        pieces["origin"].append(
            np.full(len(rows), np.datetime64(setup.origins[index].replace(tzinfo=None), "us"))
        )
        pieces["season"].append(season_of_months(months=months))
        pieces["lead"].append(lead)
        pieces["band"].append(cache.calendar.band[rows].astype(np.int64))
        pieces["ens_t"].append(np.nanmean(run.temperature_c[:, :HORIZON_HALF_HOURS], axis=0))
        pieces["era5_t"].append(features.temperature_c[rows])
        pieces["ens_g"].append(np.nanmean(run.shortwave_w_m2[:, :HORIZON_HALF_HOURS], axis=0))
        pieces["cams_g"].append(features.ghi_w_m2[rows])
    joined = {name: np.concatenate(values) for name, values in pieces.items()}
    return Calibration.fit(
        origin_time=joined["origin"],
        allowed_months=common.TUNING_MONTHS,
        season=joined["season"],
        lead_band=joined["lead"],
        hour_band=joined["band"],
        ens_mean_temperature_c=joined["ens_t"],
        target_temperature_c=joined["era5_t"],
        ens_mean_ghi=joined["ens_g"],
        target_ghi=joined["cams_g"],
    )


def forecast_frame(
    *,
    cache: FeatureCache,
    setup: SiteSetup,
    fits: list[GamFit],
    spread: LeadHourSpread,
    trace: CorrectionTrace | None,
    run_indices: list[int],
    arm: str,
) -> pl.DataFrame:
    """Forecast the chosen runs and return one row per scored half-hour.

    Args:
        cache: The feature cache.
        setup: The substation's setup.
        fits: The monthly fits, ascending by cut.
        spread: The model-error spread.
        trace: The filter's trace, or `None` for no correction.
        run_indices: The runs to forecast, as indices into `setup.runs`.
        arm: The arm's name.

    Returns:
        Columns `time_series_id`, `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`,
        `seed`, `arm`, `q01` to `q99`, `actual`, `mean`, `level`, `anomaly`, `hour_band`,
        `quarter`, and `switching_mask`, for every half-hour with a reading, in points.
    """
    features = cache.sites[setup.site]
    frames = []
    for index in run_indices:
        run = setup.runs[index]
        if len(cache.end_times) - run.origin_position - 1 < MIN_WINDOW_HALF_HOURS:
            continue
        forecast = forecast_run(
            cache=cache,
            context=setup.context,
            run=run,
            origin=setup.origins[index],
            fits=fits,
            spread=spread,
            trace=trace,
        )
        rows = np.arange(forecast.window.start, forecast.window.stop)
        keep = features.present[rows] & np.isfinite(features.y[rows])
        if not keep.any():
            continue
        rows = rows[keep]
        n = len(rows)
        valid = cache.end_times[rows]
        origin64 = np.datetime64(setup.origins[index].replace(tzinfo=None), "us")
        columns: dict[str, object] = {
            "time_series_id": np.full(n, features.table.time_series_id, dtype=np.int32),
            "run": np.full(n, index, dtype=np.int32),
            "power_fcst_init_time": pl.Series(np.full(n, origin64)).dt.replace_time_zone("UTC"),
            "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
            "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(np.float64),
            "seed": np.zeros(n, dtype=np.int8),
            "arm": np.full(n, arm),
            **{
                column: forecast.quantiles[keep][:, position].astype(np.float32)
                for position, column in enumerate(QUANTILE_COLUMNS)
            },
            "actual": features.y[rows].astype(np.float32),
            "mean": forecast.mean[keep].astype(np.float32),
            "level": forecast.level[keep].astype(np.float32),
            "anomaly": forecast.anomaly[keep].astype(np.float32),
            "hour_band": cache.calendar.band[rows].astype(np.int8),
            "quarter": cache.calendar.quarter[rows].astype(np.int8),
            "switching_mask": features.switching_mask[rows],
        }
        frames.append(pl.DataFrame(columns))
    return pl.concat(frames)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    levels = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def fit_spread_from_frame(*, frame: pl.DataFrame) -> LeadHourSpread:
    """Fit the centred spread from a forecast frame's residuals outside the switching masks.

    Args:
        frame: A frame from `forecast_frame` run with `ZERO_SPREAD`, so `mean` is the corrected
            ensemble mean.

    Returns:
        The spread, with every factor 1.
    """
    kept = frame.filter(~pl.col("switching_mask"))
    return LeadHourSpread.fit(
        residuals=(kept["actual"] - kept["mean"]).to_numpy().astype(np.float64),
        band=kept["hour_band"].to_numpy().astype(np.int64),
        quarter=kept["quarter"].to_numpy().astype(np.int64),
        lead_hours=kept["lead_hours"].to_numpy(),
    )
