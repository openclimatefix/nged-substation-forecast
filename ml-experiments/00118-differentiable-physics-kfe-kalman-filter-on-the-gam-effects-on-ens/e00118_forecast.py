"""The ensemble forecast of every arm of experiment 118 at one substation.

For each run the script drives the convex fit with every ensemble member once, splits each member's
prediction into the filter's effects (`studies.dp_effects_filter.effect_regressors`), and then
applies every arm's filter state (the state after the newest reading at least 3.5 hours before the
origin) and spread. The 51 members are pooled by `studies.ens_members.mixture_quantiles`, as
experiment 111 does.

Arms (`ARM_SPECS`):

| Arm | Filter | Spread |
|---|---|---|
| GK | experiment 111's level-and-anomaly filter | experiment 111's stored spread and factors |
| GK-sp | the same filter | fitted by this experiment's procedure (the arms' own grid) |
| KEZ-shift | KEZ's state | experiment 111's GK spread (the shift-only check) |
| every other arm | its own fitted noise | fitted by this experiment's procedure |
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00118_arms as arms
import e00118_common as common
import e00118_noise as noise_io
import e00118_streams as streams
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, usable_cutoff_positions
from studies.dp_effects_filter import (
    N_BANDS,
    EffectsTrace,
    FilterNoise,
    correction_from_state,
    effect_regressors,
    run_effects_filter,
    state_at,
)
from studies.dp_ens import LeadHourSpread, choose_fit
from studies.dp_ens_weather import RunWindow, lead_band_of_hours
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import QUANTILE_COLUMNS, GamFit, ResidualSpread
from studies.ens_members import mixture_quantiles

MIN_WINDOW_HALF_HOURS: Final[int] = 48
"""A run needs a day of grid after its origin."""
ZERO_SPREAD: Final[LeadHourSpread] = LeadHourSpread(
    by_lead_band=tuple(ResidualSpread(table=np.zeros((4, 4, 11))) for _ in range(4)),
    factors=np.ones(4),
)
"""A spread of zero, for a forecast that only needs the corrected mean."""
SHIFT_ARM: Final[str] = "KEZ-shift"
SPREAD_ARM_OF: Final[dict[str, str]] = {"GK": "GK-111", SHIFT_ARM: "GK-111"}
"""Arms that use another arm's spread: experiment 111's stored GK spread."""
FORECAST_ARMS: Final[tuple[str, ...]] = (
    "GK",
    "GK-sp",
    SHIFT_ARM,
    *arms.FITTED_ARMS,
)
"""Every arm the experiment forecasts."""
STATE_ARM_OF: Final[dict[str, str]] = {"GK-sp": "GK", SHIFT_ARM: "KEZ"}
"""Arms whose filter state is another arm's."""
OWN_SPREAD_ARMS: Final[tuple[str, ...]] = tuple(
    name for name in FORECAST_ARMS if name not in SPREAD_ARM_OF
)
"""Arms whose spread this experiment fits on the tuning months."""


@dataclass(frozen=True)
class ArmFilter:
    """An arm's filter: its trace over the whole stream and its noise.

    Attributes:
        name: The arm whose state it is.
        trace: The filter's state after every reading.
        noise: The noise it ran with.
        day_type_aware: Whether readings on holiday days were left out and the level and anomaly
            corrections are zero on holiday target days.
    """

    name: str
    trace: EffectsTrace
    noise: FilterNoise
    day_type_aware: bool = False


def build_filters(*, cache: FeatureCache, data: streams.SiteStreams) -> dict[str, ArmFilter]:
    """Run every filtering arm over a site's whole residual stream.

    Args:
        cache: The feature cache (for the grid).
        data: The site's streams.

    Returns:
        A map from arm name (`GK` and every fitted arm) to its filter.
    """
    hyper = common.tuned_111()["filter_hyper"]["gamk"][data.site]
    out: dict[str, ArmFilter] = {}
    noises = {"GK": arms.gk_noise(hyper=hyper, scale=data.scale)}
    for name in arms.FITTED_ARMS:
        stored = noise_io.read_noise(arm=name)["sites"][data.site]
        noises[name] = FilterNoise(
            q=np.array(stored["q"]),
            phi=stored["phi"],
            sigma_z=stored["sigma_z"],
            sigma_eps=stored["sigma_eps"],
            initial_sd=np.array(stored["initial_sd"]),
            use_anomaly=arms.FITTED_ARMS[name].use_anomaly,
        )
    for name, noise in noises.items():
        skip = name in arms.FITTED_ARMS and arms.FITTED_ARMS[name].skip_holidays
        trace = run_effects_filter(
            residual=data.arm_residual(skip_holidays=skip),
            regressors=data.standardised,
            band=data.band,
            noise=noise,
        )
        out[name] = ArmFilter(name=name, trace=trace, noise=noise, day_type_aware=skip)
    return out


@dataclass(frozen=True)
class RunMembers:
    """One run's members before any correction.

    Attributes:
        index: The run's index in the setup.
        window: The grid slice the forecast covers.
        members: The ensemble members that entered.
        contributions: Each member's regressors at a weight of 1, shape (members, targets, 14).
        base_mean: Each member's prediction from the fit, shape (members, targets), in points.
        fit_cut: The cut of the fit used.
    """

    index: int
    window: slice
    members: list[int]
    contributions: np.ndarray
    base_mean: np.ndarray
    fit_cut: datetime


def run_members(
    *, cache: FeatureCache, setup: common.SiteSetup, fits: list[GamFit], index: int
) -> RunMembers | None:
    """Drive the fit with every usable member of a run and split each prediction into effects.

    Args:
        cache: The feature cache.
        setup: The substation's setup.
        fits: The monthly fits, ascending by cut.
        index: The run's index.

    Returns:
        The run's members, or `None` for a run with less than a day of grid after its origin.
    """
    run = setup.runs[index]
    if len(cache.end_times) - run.origin_position - 1 < MIN_WINDOW_HALF_HOURS:
        return None
    fit = choose_fit(fits=fits, origin=setup.origins[index])
    window = RunWindow.build(cache=cache, context=setup.context, run=run)
    members = run.usable_members()
    driven = window.driven_caches(context=setup.context, run=run, members=members)
    rows = np.arange(window.window.stop - window.window.start)
    pieces = [effect_regressors(cache=member, fit=fit, index=rows) for member in driven]
    contributions = np.stack([piece.contributions for piece in pieces])
    remainder = np.stack([piece.remainder for piece in pieces])
    return RunMembers(
        index=index,
        window=window.window,
        members=members,
        contributions=contributions,
        base_mean=remainder + contributions[:, :, N_BANDS:].sum(axis=2),
        fit_cut=fit.cut,
    )


@dataclass(frozen=True)
class Corrected:
    """A run's members after one arm's correction.

    Attributes:
        means: The corrected mean of each member, shape (members, targets), in points.
        level: The level weight of each target's hour band, in points.
        anomaly: The decayed anomaly of each target's hour band, in points.
        state_position: The grid position of the newest reading the state used, or -1.
    """

    means: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    state_position: int


def corrected_members(
    *,
    cache: FeatureCache,
    setup: common.SiteSetup,
    data: streams.SiteStreams,
    filtered: ArmFilter,
    members: RunMembers,
) -> Corrected:
    """Apply an arm's filter state at the run's origin to every member.

    Args:
        cache: The feature cache.
        setup: The substation's setup.
        data: The site's streams (for the regressor scale).
        filtered: The arm's filter.
        members: The run's members.

    Returns:
        The corrected members.
    """
    target = np.arange(members.window.start, members.window.stop)
    band = cache.calendar.band[target].astype(np.int64)
    origin = setup.origins[members.index]
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=np.array([naive_utc(moment=origin)]),
        delay_hours=DEFAULT_DELAY_HOURS,
    )
    state = state_at(trace=filtered.trace, cutoff=int(cutoff[0]))
    correction = correction_from_state(
        state=state,
        standardised_contributions=members.contributions / data.scale,
        scale=data.scale,
        band=band,
        target=target,
        phi=filtered.noise.phi,
    )
    shift, level, anomaly = correction.shift, correction.level, correction.anomaly
    if filtered.day_type_aware:
        holiday = data.holiday_day[target]
        shift = shift - np.where(holiday, level + anomaly, 0.0)[None, :]
        level = np.where(holiday, 0.0, level)
        anomaly = np.where(holiday, 0.0, anomaly)
    return Corrected(
        means=members.base_mean + shift, level=level, anomaly=anomaly, state_position=state.position
    )


def quantiles_of(
    *, cache: FeatureCache, members: RunMembers, corrected: np.ndarray, spread: LeadHourSpread
) -> np.ndarray:
    """Return the mixture's 11 quantiles of one run, shape (targets, 11), in points."""
    target = np.arange(members.window.start, members.window.stop)
    band = cache.calendar.band[target].astype(np.int64)
    quarter = cache.calendar.quarter[target].astype(np.int64)
    lead_hours = np.arange(1, len(target) + 1) / common.HALF_HOURS_PER_HOUR
    per_member = np.stack(
        [
            spread.quantiles(mean=row, band=band, quarter=quarter, lead_hours=lead_hours)
            for row in corrected
        ],
        axis=1,
    )
    return mixture_quantiles(member_quantiles=per_member)


def frame_of(
    *,
    cache: FeatureCache,
    setup: common.SiteSetup,
    members: RunMembers,
    corrected: Corrected,
    quantiles: np.ndarray,
    arm: str,
) -> pl.DataFrame | None:
    """Return one row per scored half-hour of a run for one arm, or `None` if no half-hour is read.

    Columns: `time_series_id`, `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `seed`,
    `arm`, `q01` to `q99`, `actual`, `mean`, `level`, `anomaly`, `hour_band`, `quarter`, and
    `switching_mask`, in points.
    """
    features = cache.sites[setup.site]
    rows = np.arange(members.window.start, members.window.stop)
    keep = features.present[rows] & np.isfinite(features.y[rows])
    if not keep.any():
        return None
    rows = rows[keep]
    n = len(rows)
    valid = cache.end_times[rows]
    origin64 = np.datetime64(setup.origins[members.index].replace(tzinfo=None), "us")
    columns: dict[str, object] = {
        "time_series_id": np.full(n, features.table.time_series_id, dtype=np.int32),
        "run": np.full(n, members.index, dtype=np.int32),
        "power_fcst_init_time": pl.Series(np.full(n, origin64)).dt.replace_time_zone("UTC"),
        "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
        "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(np.float64),
        "seed": np.zeros(n, dtype=np.int8),
        "arm": np.full(n, arm),
        **{
            column: quantiles[keep][:, position].astype(np.float32)
            for position, column in enumerate(QUANTILE_COLUMNS)
        },
        "actual": features.y[rows].astype(np.float32),
        "mean": corrected.means.mean(axis=0)[keep].astype(np.float32),
        "level": corrected.level[keep].astype(np.float32),
        "anomaly": corrected.anomaly[keep].astype(np.float32),
        "hour_band": cache.calendar.band[rows].astype(np.int8),
        "quarter": cache.calendar.quarter[rows].astype(np.int8),
        "switching_mask": features.switching_mask[rows],
    }
    return pl.DataFrame(columns)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    levels = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def band_scores(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the sum and count of quantile scores per calibration lead band, shape (4, 2).

    Rows inside the hand-filtered switching events do not count.
    """
    kept = frame.filter(~pl.col("switching_mask"))
    scores = quantile_score(
        actual=kept["actual"].to_numpy().astype(np.float64),
        quantiles=kept.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64),
    )
    band = lead_band_of_hours(lead_hours=kept["lead_hours"].to_numpy())
    out = np.zeros((common.N_LEAD_BANDS, 2))
    for index in range(common.N_LEAD_BANDS):
        out[index] = (scores[band == index].sum(), (band == index).sum())
    return out


def fit_spread(*, frame: pl.DataFrame) -> LeadHourSpread:
    """Fit the centred spread from a zero-spread frame's residuals outside the switching masks."""
    kept = frame.filter(~pl.col("switching_mask"))
    return LeadHourSpread.fit(
        residuals=(kept["actual"] - kept["mean"]).to_numpy().astype(np.float64),
        band=kept["hour_band"].to_numpy().astype(np.int64),
        quarter=kept["quarter"].to_numpy().astype(np.int64),
        lead_hours=kept["lead_hours"].to_numpy(),
    )
