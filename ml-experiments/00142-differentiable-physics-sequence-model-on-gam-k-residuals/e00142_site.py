"""One substation's cached runs on flat arrays, with the weather and the residual stream.

`SiteData` holds everything an arm of experiment 115 needs at a substation: the cached member
forecasts of GAM-K, the proxy-driven residual stream the filters read, the grid's calendar arrays,
and one row per (run, target half-hour), flat across runs, so a correction is one array of
`n_rows` values. Rows are those of `studies.dp_ens.forecast_run`'s window: up to 336 half-hours
(168 hours) after each run's origin.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from zoneinfo import ZoneInfo

import e00142_common as common
import e00142_ens as ens
import e00142_members as members
import e00142_shape as shape
import numpy as np
from studies.dp_assimilation import usable_cutoff_positions
from studies.dp_ens import LeadHourSpread, proxy_residual_stream
from studies.dp_ens_weather import lead_band_of_hours, proxy_cache
from studies.dp_features import FeatureCache, SiteFeatures

LONDON: Final[ZoneInfo] = ZoneInfo("Europe/London")
HORIZON_HOURS: Final[float] = 168.0
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0
"""Day D+1 is the UTC day after the origin's: leads 15.5 to 39 hours from the 09:00 UTC origin."""


@dataclass(frozen=True)
class SiteData:
    """A substation's runs, rows, and residual stream.

    Attributes:
        site: The label.
        cache: The feature cache.
        features: The site's arrays on the grid.
        stream: Observed minus the chained fits' forecast, driven by the analysis proxy, on the
            grid, in points; `NaN` where there is none.
        half_hour: The local half-hour of day of each grid row.
        band_of_half_hour: The hour band of each local half-hour of day.
        month_code: `100 * year + month` of each grid row's half-hour start.
        runs: The cached run indices, ascending.
        run_origins: Each cached run's `power_fcst_init_time` as naive UTC `datetime64[us]`.
        run_period: Each cached run's period (`tuning`, `replicate`, `scored`, `stride`, or
            `warm_up`).
        run_of_row: The index into `runs` of each row.
        offsets: Each run's first row, with the total at the end, shape (runs + 1,).
        position: The grid position of each row's valid time.
        lead_hours: The lead of each row from its run's origin.
        actual: The reading at each row, in points, `NaN` where there is none.
        scorable: True where the row has a reading and lies outside the switching masks.
        member_mean: The mean over members of GAM-K's forecast at each row, in points.
        member_means: Each run's member forecasts, shape (members, targets).
        kalman_hyper: GAM-K's hyperparameters as experiment 111 fitted them.
    """

    site: str
    cache: FeatureCache
    features: SiteFeatures
    stream: np.ndarray
    half_hour: np.ndarray
    band_of_half_hour: np.ndarray
    month_code: np.ndarray
    runs: np.ndarray
    run_origins: np.ndarray
    run_period: np.ndarray
    run_of_row: np.ndarray
    offsets: np.ndarray
    position: np.ndarray
    lead_hours: np.ndarray
    actual: np.ndarray
    scorable: np.ndarray
    member_mean: np.ndarray
    member_means: list[np.ndarray]
    kalman_hyper: shape.ShapeHyper

    @property
    def n_rows(self) -> int:
        """The number of rows."""
        return len(self.position)

    @property
    def lead_band(self) -> np.ndarray:
        """The calibration lead band of each row."""
        return lead_band_of_hours(lead_hours=self.lead_hours)

    @property
    def hour_band(self) -> np.ndarray:
        """The hour band of each row's target."""
        return self.cache.calendar.band[self.position].astype(np.int64)

    @property
    def quarter(self) -> np.ndarray:
        """The calendar quarter of each row's target."""
        return self.cache.calendar.quarter[self.position].astype(np.int64)

    @property
    def row_half_hour(self) -> np.ndarray:
        """The local half-hour of day of each row's target."""
        return self.half_hour[self.position]

    def period_rows(self, *, period: str) -> np.ndarray:
        """Return the rows a period scores, as a boolean mask.

        A tuning or replicate row needs its origin and its valid time both in the period's months,
        so a long lead from a late-month origin never scores the other period's targets. A scored
        row is any row of a scored run.

        Args:
            period: `tuning`, `replicate`, or `scored`.

        Returns:
            The mask over rows, before the reading and switching tests.
        """
        run_in = (self.run_period == period)[self.run_of_row]
        if period == "scored":
            return run_in
        months = common.TUNING_MONTHS if period == "tuning" else common.REPLICATE_MONTHS
        codes = np.array([100 * year + month for year, month in months])
        return run_in & np.isin(self.month_code[self.position], codes)

    def d1_rows(self) -> np.ndarray:
        """Return the rows of day D+1: leads of 15.5 to 39 hours."""
        return (self.lead_hours >= D1_FIRST_LEAD) & (self.lead_hours <= D1_LAST_LEAD)

    def cutoffs(self, *, delay_hours: float, day_only: bool = False) -> np.ndarray:
        """Return each run's newest usable grid position.

        Args:
            delay_hours: The telemetry delay; a half-hour is usable when it ended at least this long
                before the origin.
            day_only: True to also stop at local midnight before the origin, as an arm that sees
                no readings from the day of the forecast does.

        Returns:
            One grid position per run, -1 where none is usable.
        """
        cutoff = usable_cutoff_positions(
            end_times=self.cache.end_times, initialisation=self.run_origins, delay_hours=delay_hours
        )
        if not day_only:
            return cutoff
        midnights = []
        for origin in self.run_origins:
            local = origin.astype("datetime64[us]").astype(datetime).replace(tzinfo=UTC)
            midnight = datetime.combine(
                local.astimezone(LONDON).date(), datetime.min.time(), LONDON
            )
            midnights.append(np.datetime64(midnight.astimezone(UTC).replace(tzinfo=None), "us"))
        midnight_position = (
            np.searchsorted(self.cache.end_times, np.array(midnights), side="right") - 1
        )
        return np.minimum(cutoff, midnight_position)


def band_of_half_hour(*, cache: FeatureCache) -> np.ndarray:
    """Return the hour band of each local half-hour of day, shape (48,)."""
    half_hour = cache.calendar.half_hour.astype(np.int64)
    return np.array(
        [int(cache.calendar.band[np.flatnonzero(half_hour == h)[0]]) for h in range(48)]
    )


def month_codes(*, cache: FeatureCache) -> np.ndarray:
    """Return `100 * year + month` of each grid row's half-hour start."""
    starts = cache.end_times - np.timedelta64(30, "m")
    year = starts.astype("datetime64[Y]").astype(np.int64) + 1970
    month = starts.astype("datetime64[M]").astype(np.int64) % 12 + 1
    return year * 100 + month


def load_site(*, site: str) -> SiteData:
    """Load a substation's cached runs and build its flat rows and residual stream.

    Args:
        site: The substation's label.

    Returns:
        The site data.
    """
    cache = common.load_cache()
    features = cache.sites[site]
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=members.load_calibration(site=site)
    )
    fits = members.load_fits(site=site)
    stream = proxy_residual_stream(
        cache=proxy_cache(cache=cache, context=setup.context), fits=fits, site=site
    )
    cached = members.load_members(site=site)
    runs = np.array(sorted(cached))
    run_dates = common.run_dates()
    origins = np.array(
        [np.datetime64(common.origin_of(run=run_dates[i]).replace(tzinfo=None), "us") for i in runs]
    )
    period = np.array([common.period_of_run(run=run_dates[i]) for i in runs])
    sizes = [cached[int(i)].means.shape[1] for i in runs]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    run_of_row = np.repeat(np.arange(len(runs)), sizes)
    position = np.concatenate(
        [
            np.arange(cached[int(i)].start, cached[int(i)].start + n)
            for i, n in zip(runs, sizes, strict=True)
        ]
    )
    lead_hours = np.concatenate([np.arange(1, n + 1) / 2.0 for n in sizes])
    actual = np.where(features.present[position], features.y[position], np.nan)
    scorable = np.isfinite(actual) & ~features.switching_mask[position]
    member_means = [cached[int(i)].means for i in runs]
    member_mean = np.concatenate([means.mean(axis=0) for means in member_means])
    tuned = json.loads((common.SRC_WORK_DIR / "tuned.json").read_text())
    kalman = shape.ShapeHyper(*tuned["filter_hyper"]["gamk"][site], log_q_s=None)
    return SiteData(
        site=site,
        cache=cache,
        features=features,
        stream=stream,
        half_hour=cache.calendar.half_hour.astype(np.int64),
        band_of_half_hour=band_of_half_hour(cache=cache),
        month_code=month_codes(cache=cache),
        runs=runs,
        run_origins=origins,
        run_period=period,
        run_of_row=run_of_row,
        offsets=offsets,
        position=position,
        lead_hours=lead_hours,
        actual=actual,
        scorable=scorable,
        member_mean=member_mean,
        member_means=member_means,
        kalman_hyper=kalman,
    )


def stored_spread(*, site: str) -> LeadHourSpread:
    """Return experiment 111's stored GAM-K spread with its tuned factors."""
    with np.load(common.SRC_WORK_DIR / "spread" / f"gamk_{site}.npz") as stored:
        tables = stored["tables"]
    factors = json.loads((common.SRC_WORK_DIR / "tuned.json").read_text())["spread_factors"]["gamk"]
    from studies.dp_gam import ResidualSpread

    return LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables),
        factors=np.array(factors),
    )
