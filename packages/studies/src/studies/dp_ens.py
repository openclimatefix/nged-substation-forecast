"""The convex demand model driven by ECMWF ensemble (ENS) weather, for Part 6 of the DP series.

`studies.dp_ens_weather` builds the weather arrays (the analysis proxy, the calibration, the driven
window). This module calls the fitted convex model on them: the per-member forecasts, the filter's
residual stream on proxy-driven weather, the forward-chained base margin, and the whole run's
quantile forecast.

**Forward chaining.** A fit made at a monthly cut predicts only the rows after the cut and before
the next cut. A forecast issued at an origin uses the fit of the latest cut at or before the origin
(`choose_fit`), and refuses a fit whose cut is less than the telemetry delay before the origin, so
the fit never trained on readings that a live service had not yet received.

**The model-error spread** of a corrected member is a centred residual spread per calibration lead
band (`LeadHourSpread`), with a factor per band that the experiment tunes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from itertools import pairwise

import numpy as np
import polars as pl

from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    CorrectionTrace,
    correction_at,
    usable_cutoff_positions,
)
from studies.dp_ens_weather import (
    HALF_HOURS_PER_HOUR,
    N_LEAD_BANDS,
    EnsSiteRun,
    RunWindow,
    WeatherContext,
    lead_band_of_hours,
)
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import QUANTILE_LEVELS, GamFit, ResidualSpread, predict_gam
from studies.ens_members import CONTROL_MEMBER, MIXTURE_LEVELS, mixture_quantiles

if tuple(QUANTILE_LEVELS) != MIXTURE_LEVELS:
    msg = "the DP's quantile levels and the mixture's must be the same 11 levels"
    raise ImportError(msg)


def latest_cut_index(*, cuts: Sequence[datetime], moment: datetime) -> int:
    """Return the index of the latest cut at or before `moment`, or -1.

    Args:
        cuts: The cuts, ascending.
        moment: The forecast's origin.

    Returns:
        The index; -1 if every cut is later.
    """
    index = -1
    for position, cut in enumerate(cuts):
        if cut <= moment:
            index = position
    return index


def assert_cut_not_after(
    *, fit: GamFit, origin: datetime, delay_hours: float = DEFAULT_DELAY_HOURS
) -> None:
    """Raise if a fit's cut is too close to, or after, the origin of the forecast that uses it.

    Args:
        fit: The fit.
        origin: The forecast's `power_fcst_init_time`.
        delay_hours: The telemetry delay: readings that ended later than this before the origin
            have not reached a live service.

    Raises:
        ValueError: If `fit.cut` is after `origin` minus the delay.
    """
    if fit.cut > origin - timedelta(hours=delay_hours):
        msg = (
            f"the fit's cut {fit.cut} is within {delay_hours} hours of, or after, the forecast's "
            f"origin {origin}: lookahead"
        )
        raise ValueError(msg)


def choose_fit(*, fits: Sequence[GamFit], origin: datetime) -> GamFit:
    """Return the fit of the latest monthly cut at or before the origin.

    This is the only place a forecast or a margin chooses its fit.

    Args:
        fits: The monthly fits, ascending by cut.
        origin: The run's `power_fcst_init_time`, timezone-aware.

    Returns:
        The fit.

    Raises:
        ValueError: If no fit has a cut at or before the origin, or the cut is within the
            telemetry delay of the origin.
    """
    chosen = latest_cut_index(cuts=[fit.cut for fit in fits], moment=origin)
    if chosen < 0:
        msg = f"no fit has a cut at or before {origin}"
        raise ValueError(msg)
    assert_cut_not_after(fit=fits[chosen], origin=origin)
    return fits[chosen]


@dataclass(frozen=True)
class MemberMeans:
    """The mean forecast of each member of one run.

    Attributes:
        means: Net power in points of effective capacity, shape (members, targets).
        members: The ensemble member of each row of `means`.
        window: The slice of the full grid the forecast covers.
        fit: The fit that made the forecast.
    """

    means: np.ndarray
    members: list[int]
    window: slice
    fit: GamFit


def member_means(
    *,
    cache: FeatureCache,
    context: WeatherContext,
    run: EnsSiteRun,
    origin: datetime,
    fits: Sequence[GamFit],
    members: Sequence[int] | None = None,
) -> MemberMeans:
    """Return each member's mean forecast, from the fit of the latest cut before the run.

    Args:
        cache: The full feature cache.
        context: The substation's weather context.
        run: The run.
        origin: The run's `power_fcst_init_time`, timezone-aware.
        fits: The monthly fits, ascending by cut.
        members: The members to run; defaults to those with complete weather (`usable_members`).

    Returns:
        The member means, the members they belong to, the window, and the fit used.
    """
    fit = choose_fit(fits=fits, origin=origin)
    wanted = run.usable_members() if members is None else list(members)
    window = RunWindow.build(cache=cache, context=context, run=run)
    driven = window.driven_caches(context=context, run=run, members=wanted)
    index = np.arange(window.window.stop - window.window.start)
    means = np.stack([predict_gam(cache=member, fit=fit, index=index) for member in driven])
    return MemberMeans(means=means, members=wanted, window=window.window, fit=fit)


def proxy_residual_stream(*, cache: FeatureCache, fits: Sequence[GamFit], site: str) -> np.ndarray:
    """Return observed minus the chained fits' forecast on a cache, `NaN` where it cannot be made.

    Pass the proxy cache for the ENS-driven residuals of the live filter, or the ERA5 and CAMS
    cache for `GAM-K-era5`. Fit `i` supplies the rows after its cut and up to the next cut, as
    `studies.dp_assimilation.stitch_out_of_sample_residuals` does, so no residual is in-sample.

    Args:
        cache: The cache whose weather arrays drive the fits.
        fits: The monthly fits, strictly ascending by cut.
        site: The substation's label.

    Returns:
        A stream on the grid in points.

    Raises:
        ValueError: If there is no fit or the cuts are not strictly ascending.
    """
    if not fits or any(later.cut <= earlier.cut for earlier, later in pairwise(fits)):
        msg = "the residual stream needs at least one fit, with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cut_positions = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=fit.cut), side="right")) - 1
        for fit in fits
    ]
    ends = [*cut_positions[1:], len(stream) - 1]
    for fit, cut, end in zip(fits, cut_positions, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            stream[rows] = features.y[rows] - predict_gam(cache=cache, fit=fit, index=rows)
    return stream


@dataclass(frozen=True)
class LeadHourSpread:
    """Centred residual quantiles per lead band, hour band, and quarter, with a spread factor.

    `studies.dp_assimilation.LeadSpread` keeps three lead bands counted in days ahead and has no
    factor. This spread keeps the four lead bands of the calibration, counted in hours from the
    origin, so that a run's 336 half-hours fall into bands the experiment can tune a factor for.

    Attributes:
        by_lead_band: One centred `ResidualSpread` per calibration lead band.
        factors: The spread factor of each lead band; 1 leaves the fitted spread unchanged.
    """

    by_lead_band: tuple[ResidualSpread, ...]
    factors: np.ndarray

    @staticmethod
    def fit(
        *,
        residuals: np.ndarray,
        band: np.ndarray,
        quarter: np.ndarray,
        lead_hours: np.ndarray,
    ) -> LeadHourSpread:
        """Fit one centred spread per lead band from out-of-sample residuals.

        Args:
            residuals: Observed minus corrected forecast, in points.
            band: The hour band of each residual.
            quarter: The calendar quarter of each residual.
            lead_hours: The lead of each residual, from the origin.

        Returns:
            The spread with every factor 1.

        Raises:
            ValueError: If a lead band holds no residual.
        """
        lead = lead_band_of_hours(lead_hours=lead_hours)
        spreads = []
        for index in range(N_LEAD_BANDS):
            chosen = lead == index
            if not chosen.any():
                msg = f"lead band {index} holds no residual to fit a spread on"
                raise ValueError(msg)
            spreads.append(
                ResidualSpread.fit(
                    residuals=residuals[chosen],
                    band=band[chosen],
                    quarter=quarter[chosen],
                    centre=True,
                )
            )
        return LeadHourSpread(by_lead_band=tuple(spreads), factors=np.ones(N_LEAD_BANDS))

    def with_factors(self, *, factors: Sequence[float]) -> LeadHourSpread:
        """Return the spread with new factors, one per lead band.

        Args:
            factors: The factors.

        Returns:
            The spread.

        Raises:
            ValueError: If there is not one factor per lead band.
        """
        if len(factors) != N_LEAD_BANDS:
            msg = f"the spread needs one factor per lead band, {N_LEAD_BANDS} in all"
            raise ValueError(msg)
        return replace(self, factors=np.asarray(factors, dtype=np.float64))

    def quantiles(
        self, *, mean: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
    ) -> np.ndarray:
        """Return the 11 quantiles around each mean, the spread scaled by its lead band's factor.

        Args:
            mean: The corrected mean forecast in points.
            band: The hour band of each row.
            quarter: The calendar quarter of each row.
            lead_hours: The lead of each row, from the origin.

        Returns:
            An array of shape (len(mean), 11), non-decreasing along the last axis for factors of 0
            or more.
        """
        lead = lead_band_of_hours(lead_hours=lead_hours)
        out = np.empty((len(mean), len(QUANTILE_LEVELS)))
        for index in range(N_LEAD_BANDS):
            chosen = lead == index
            if chosen.any():
                table = self.by_lead_band[index].table[band[chosen], quarter[chosen]]
                out[chosen] = mean[chosen, None] + self.factors[index] * table
        return out


def margin_frame(
    *,
    cache: FeatureCache,
    context: WeatherContext,
    run: EnsSiteRun,
    origin: datetime,
    fits: Sequence[GamFit],
) -> pl.DataFrame:
    """Return the DP's control-member forecast of one run as a base margin.

    The margin of a run comes from the fit of the latest monthly cut at or before the run's
    origin, driven by the control member the run's XGBoost rows see, so a training row and a
    forecast-time row get the margin a live service would have had.

    Args:
        cache: The full feature cache.
        context: The substation's weather context.
        run: The run.
        origin: The run's `power_fcst_init_time`, timezone-aware.
        fits: The monthly fits, ascending by cut.

    Returns:
        Columns `time_series_id`, `power_fcst_init_time`, `valid_time`, `margin_fraction`,
        `margin_mw`, and `fit_cut`, one row per half-hour of the run's window. `margin_fraction` is
        power over effective capacity, the unit of BG2's training label, so pass it as the base
        margin of a model trained on that label.
    """
    result = member_means(
        cache=cache, context=context, run=run, origin=origin, fits=fits, members=[CONTROL_MEMBER]
    )
    table = cache.sites[context.site].table
    n = result.means.shape[1]
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([table.time_series_id] * n, dtype=pl.Int32),
            "power_fcst_init_time": pl.Series([origin] * n).dt.replace_time_zone("UTC"),
            "valid_time": pl.Series(cache.end_times[result.window]).dt.replace_time_zone("UTC"),
            "margin_fraction": result.means[0] / 100.0,
            "margin_mw": result.means[0] * table.effective_capacity_mw / 100.0,
            "fit_cut": pl.Series([result.fit.cut] * n).dt.replace_time_zone("UTC"),
        }
    )


@dataclass(frozen=True)
class RunForecast:
    """The forecast of one run at one substation.

    Attributes:
        quantiles: The 11 quantiles of the member mixture, shape (targets, 11), in points.
        mean: The mean over members of the corrected forecast, in points.
        level: The filter's persistent correction at each target (zeros without a filter).
        anomaly: The filter's decaying correction at each target (zeros without a filter).
        window: The slice of the full grid the forecast covers.
        members: The members that entered the mixture.
    """

    quantiles: np.ndarray
    mean: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    window: slice
    members: list[int]


def forecast_run(
    *,
    cache: FeatureCache,
    context: WeatherContext,
    run: EnsSiteRun,
    origin: datetime,
    fits: Sequence[GamFit],
    spread: LeadHourSpread,
    trace: CorrectionTrace | None = None,
    delay_hours: float = DEFAULT_DELAY_HOURS,
) -> RunForecast:
    """Forecast one run: the DP per member, the filter's correction, and the member mixture.

    Each member's mean forecast comes from the fit of the latest monthly cut at or before the
    origin. The correction is the filter's state after the newest reading that ended at least
    `delay_hours` before the origin, and is the same for every member. Each corrected member gets
    the spread's quantiles around it, and the mixture pools the members with complete weather.

    Args:
        cache: The full feature cache.
        context: The substation's weather context.
        run: The run.
        origin: The run's `power_fcst_init_time`, timezone-aware.
        fits: The monthly fits, ascending by cut.
        spread: The model-error spread around each corrected member.
        trace: The filter's trace on the residual stream, or `None` for no correction.
        delay_hours: The telemetry delay.

    Returns:
        The forecast.
    """
    result = member_means(cache=cache, context=context, run=run, origin=origin, fits=fits)
    target = np.arange(result.window.start, result.window.stop)
    n_targets = len(target)
    level = np.zeros(n_targets)
    anomaly = np.zeros(n_targets)
    band = cache.calendar.band[target].astype(np.int64)
    if trace is not None:
        cutoff = usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=origin)]),
            delay_hours=delay_hours,
        )
        correction = correction_at(
            trace=trace, cutoff=np.full(n_targets, cutoff[0]), target=target, band=band
        )
        level, anomaly = correction.level, correction.anomaly
    corrected = result.means + (level + anomaly)[None, :]
    quarter = cache.calendar.quarter[target].astype(np.int64)
    lead_hours = np.arange(1, n_targets + 1) / HALF_HOURS_PER_HOUR
    per_member = np.stack(
        [
            spread.quantiles(mean=row, band=band, quarter=quarter, lead_hours=lead_hours)
            for row in corrected
        ],
        axis=1,
    )
    return RunForecast(
        quantiles=mixture_quantiles(member_quantiles=per_member),
        mean=corrected.mean(axis=0),
        level=level,
        anomaly=anomaly,
        window=result.window,
        members=result.members,
    )
