"""The ensemble forecast of every arm of experiment 128 at one substation.

For each run the script drives the convex fit with every ensemble member once (`predict_gam`), then
adds each arm's correction: the level and anomaly of the filter state after the newest reading at
least 3.5 hours before the origin, held over the horizon, plus the day-level state d decayed by
`phi_d` per local day (`e00128_filter.correction_from_state`). The correction does not depend on the
member, because the filter has no regressors. The 51 members are pooled by
`studies.ens_members.mixture_quantiles`, as experiment 111 does.

The arms are in `e00128_arms`. **KDS-shuf** replaces d in the forecast by the d of another day of
the same calendar month, taken from that day's state at the same clock time (`shuffle_partners`).
**KDS-nod** leaves d out of the forecast and keeps KDS's levels and anomalies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00128_arms as arm_module
import e00128_common as common
import e00128_filter as joint
import e00128_fit as fit
import e00128_streams as streams
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, usable_cutoff_positions
from studies.dp_ens import LeadHourSpread, choose_fit
from studies.dp_ens_weather import RunWindow, lead_band_of_hours
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import QUANTILE_COLUMNS, GamFit, ResidualSpread, predict_gam
from studies.ens_members import mixture_quantiles

MIN_WINDOW_HALF_HOURS: Final[int] = 48
"""A run needs a day of grid after its origin."""
SHUFFLE_SEED: Final[int] = 128
ZERO_SPREAD: Final[LeadHourSpread] = LeadHourSpread(
    by_lead_band=tuple(ResidualSpread(table=np.zeros((4, 4, 11))) for _ in range(4)),
    factors=np.ones(4),
)
"""A spread of zero, for a forecast that only needs the corrected mean."""


@dataclass(frozen=True)
class ArmFilter:
    """An arm's filter: its trace over the whole stream.

    Attributes:
        name: The fitted arm (or `GK`) whose constants the trace used.
        trace: The filter's state after every reading.
    """

    name: str
    trace: joint.JointTrace


def shuffle_partners(*, origins: list[datetime]) -> dict[datetime, datetime]:
    """Return, for each origin, the origin of another day of the same calendar month.

    The partner comes from a fixed permutation (seed 128) of the month's origins, so the check
    repeats exactly.

    Args:
        origins: Every run's `power_fcst_init_time`.

    Returns:
        A map from origin to the origin whose state supplies d.
    """
    by_month: dict[tuple[int, int], list[datetime]] = {}
    for origin in origins:
        by_month.setdefault((origin.year, origin.month), []).append(origin)
    partners: dict[datetime, datetime] = {}
    for (year, month), members in by_month.items():
        order = np.random.default_rng([SHUFFLE_SEED, year, month]).permutation(len(members))
        for position, member in enumerate(members):
            partners[member] = members[int(order[position])]
    return partners


def build_filters(
    *, cache: FeatureCache, data: streams.SiteStreams, wanted: tuple[str, ...]
) -> dict[str, ArmFilter]:
    """Run the filters the wanted arms need over a site's whole residual stream.

    Args:
        cache: The feature cache (unused beyond the grid; kept for a uniform signature).
        data: The site's streams.
        wanted: The arms to forecast.

    Returns:
        A map from the fitted arm name (`GK` for GAM-K) to its filter. An arm whose constants have
        not been fitted is left out.
    """
    del cache
    keys = {arm_module.ARM_SPECS[name].params_arm or "GK" for name in wanted}
    out: dict[str, ArmFilter] = {}
    for key in sorted(keys):
        if key == "GK":
            params = joint.gamk_params(hyper=common.gamk_hyper(site=data.site))
        elif fit.hyper_path(arm=key).exists():
            params = fit.read_params(arm=key, site=data.site)
        else:
            continue
        trace = joint.run_joint_filter(
            residual=data.residual, band=data.band, day=data.day, params=params
        )
        out[key] = ArmFilter(name=key, trace=trace)
    return out


def available_arms(*, filters: dict[str, ArmFilter], arms_wanted: tuple[str, ...]) -> list[str]:
    """Return the wanted arms whose filter exists."""
    return [arm for arm in arms_wanted if (arm_module.ARM_SPECS[arm].params_arm or "GK") in filters]


@dataclass(frozen=True)
class RunMembers:
    """One run's members before any correction.

    Attributes:
        index: The run's index in the setup.
        window: The grid slice the forecast covers.
        members: The ensemble members that entered.
        base_mean: Each member's prediction from the fit, shape (members, targets), in points.
        fit_cut: The cut of the fit used.
    """

    index: int
    window: slice
    members: list[int]
    base_mean: np.ndarray
    fit_cut: datetime


def run_members(
    *, cache: FeatureCache, setup: common.SiteSetup, fits: list[GamFit], index: int
) -> RunMembers | None:
    """Drive the fit with every usable member of a run.

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
    fit_used = choose_fit(fits=fits, origin=setup.origins[index])
    window = RunWindow.build(cache=cache, context=setup.context, run=run)
    members = run.usable_members()
    driven = window.driven_caches(context=setup.context, run=run, members=members)
    rows = np.arange(window.window.stop - window.window.start)
    return RunMembers(
        index=index,
        window=window.window,
        members=members,
        base_mean=np.stack(
            [predict_gam(cache=member, fit=fit_used, index=rows) for member in driven]
        ),
        fit_cut=fit_used.cut,
    )


@dataclass(frozen=True)
class Corrected:
    """A run's members after one arm's correction.

    Attributes:
        means: The corrected mean of each member, shape (members, targets), in points.
        level: The held level of each target's hour band, in points.
        anomaly: The decayed anomaly of each target's hour band, in points.
        day_term: The decayed day-level state of each target, in points (0 when d is not read).
        state_position: The grid position of the newest reading the state used, or -1.
    """

    means: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    day_term: np.ndarray
    state_position: int


def cutoff_of(*, cache: FeatureCache, origin: datetime) -> int:
    """Return the newest grid row a forecast issued at `origin` may see (3.5 hours earlier)."""
    return int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=origin)]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )[0]
    )


def corrected_members(
    *,
    cache: FeatureCache,
    setup: common.SiteSetup,
    data: streams.SiteStreams,
    filtered: ArmFilter,
    members: RunMembers,
    day_mode: arm_module.DayModeType,
    partners: dict[datetime, datetime] | None = None,
) -> Corrected:
    """Apply an arm's filter state at the run's origin to every member.

    Args:
        cache: The feature cache.
        setup: The substation's setup.
        data: The site's streams (for the bands and local days).
        filtered: The arm's filter.
        members: The run's members.
        day_mode: `own` reads d from the state, `none` leaves it out, `shuffled` reads the d of the
            partner origin's state.
        partners: The shuffle partners of every origin, needed for `shuffled`.

    Returns:
        The corrected members.

    Raises:
        ValueError: If `shuffled` is asked for without partners.
    """
    target = np.arange(members.window.start, members.window.stop)
    origin = setup.origins[members.index]
    state = joint.state_at(trace=filtered.trace, cutoff=cutoff_of(cache=cache, origin=origin))
    day_value = None
    if day_mode == "shuffled":
        if partners is None:
            msg = "the shuffled mode needs the partner map"
            raise ValueError(msg)
        other = joint.state_at(
            trace=filtered.trace, cutoff=cutoff_of(cache=cache, origin=partners[origin])
        )
        day_value = float(other.x[joint.D_INDEX]) if other.position >= 0 else 0.0
    correction = joint.correction_from_state(
        state=state,
        params=filtered.trace.params,
        band=data.band[target],
        target=target,
        target_day=data.day[target],
        day_value=day_value,
    )
    day_term = np.zeros(len(target)) if day_mode == "none" else correction.day_term
    return Corrected(
        means=members.base_mean + (correction.level + correction.anomaly + day_term)[None, :],
        level=correction.level,
        anomaly=correction.anomaly,
        day_term=day_term,
        state_position=state.position,
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
    `arm`, `q01` to `q99`, `actual`, `mean`, `level`, `anomaly`, `day_term`, `hour_band`,
    `quarter`, and `switching_mask`, in points.
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
        "day_term": corrected.day_term[keep].astype(np.float32),
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
