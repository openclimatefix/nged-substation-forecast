"""The ECMWF-ensemble (ENS) weather layer of the differentiable-physics (DP) series, for Part 6.

Parts 1 to 5 drove the convex demand model with ERA5 temperature and CAMS irradiance. This module
drives the same fitted model with the weather forecasts a live service would have had: the 51 ENS
members of one daily run for the future, and a stitched "analysis proxy" for the past.

**One run is usable from its `power_fcst_init_time`.** A run initialised at 00:00 UTC is published
by 09:00 UTC (`origin`), and its frame holds the half-hours that end after the origin. The forecast
issued at the origin may use nothing published later.

**The analysis proxy replaces ERA5 for the past.** For each past half-hour the proxy takes the
control member of the latest run whose origin is before the half-hour, so the 48 half-hours that end
in `(origin_d, origin_d + 24 h]` come from run `d` at NWP lead 9.5 to 33 hours. Half-hours up to and
including the origin of run `d` therefore never come from run `d` or later, and the module's tests
assert that. ERA5 is never read here.

**Calibration maps ENS onto the weather the model was trained on.** The DP is fitted on ERA5 and
CAMS. ENS temperature takes an additive shift and ENS irradiance a multiplicative ratio clipped to
`[0, clear sky]`, per season, lead band, and hour band, and then the forecast solar anomaly is
shrunk towards the CAMS month-by-half-hour climatology by a coefficient per lead band.

**Forward chaining.** A fit made at a monthly cut predicts only the rows after the cut and before
the next cut. A forecast issued at an origin uses the fit of the latest cut at or before the origin.
`latest_cut_index` is the only place that choice is made, and `assert_cut_not_after` checks it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Final

import numpy as np
import polars as pl

from studies.dp_assimilation import (
    CorrectionTrace,
    correction_at,
    usable_cutoff_positions,
)
from studies.dp_calendar import SMOOTHING_HOURS, CalendarColumns
from studies.dp_features import FLEET, FeatureCache, SiteFeatures
from studies.dp_gam import QUANTILE_LEVELS, GamFit, ResidualSpread, naive_utc, predict_gam
from studies.ens_members import CONTROL_MEMBER, ENSEMBLE_SIZE
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw, solar_position

DEFAULT_DELAY_HOURS: Final[float] = 3.5
"""BG2's telemetry delay: the newest reading a forecast sees ended this long before the origin."""

HALF_HOURS_PER_HOUR: Final[int] = 2
HALF_HOURS_PER_DAY: Final[int] = 48
HORIZON_HALF_HOURS: Final[int] = 336
"""The forecast horizon: 168 hours of half-hours, from the origin."""

HISTORY_HALF_HOURS: Final[int] = HALF_HOURS_PER_HOUR * max(SMOOTHING_HOURS)
"""Half-hours of past temperature the longest trailing mean needs."""

LEAD_BAND_UPPER_HOURS: Final[tuple[float, ...]] = (24.0, 48.0, 144.0, 168.0)
"""Upper edges of the calibration's lead bands: 0-24, 24-48, 48-144, and 144-168 hours."""

N_LEAD_BANDS: Final[int] = len(LEAD_BAND_UPPER_HOURS)
N_SEASONS: Final[int] = 4
N_HOUR_BANDS: Final[int] = 4
"""The number of hour bands of `studies.dp_calendar`."""

MIN_CELL_ROWS: Final[int] = 200
"""A calibration cell with fewer daylight rows keeps the identity map."""

MIN_DAYLIGHT_W_M2: Final[float] = 5.0
"""An ENS or CAMS half-hour needs more than this to enter an irradiance ratio."""

HAURWITZ_SCALE_W_M2: Final[float] = 1098.0
HAURWITZ_EXTINCTION: Final[float] = 0.059
"""The two constants of Haurwitz's clear-sky formula `1098 cos(z) exp(-0.059 / cos(z))`."""


def season_of_months(*, months: np.ndarray) -> np.ndarray:
    """Return the season index of calendar months: 0 for December to February, then March to May.

    Args:
        months: Calendar months, 1 to 12.

    Returns:
        Season indices 0 (December to February), 1 (March to May), 2 (June to August), and 3
        (September to November).
    """
    return ((months % 12) // 3).astype(np.int64)


def lead_band_of_hours(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the calibration lead band of leads in hours: 0 to 3, over (0, 168] hours.

    Args:
        lead_hours: Leads from the origin, in hours.

    Returns:
        Band indices; a lead beyond 168 hours takes the last band.
    """
    edges = np.array(LEAD_BAND_UPPER_HOURS)
    return np.minimum(np.searchsorted(edges, lead_hours, side="left"), N_LEAD_BANDS - 1).astype(
        np.int64
    )


@dataclass(frozen=True)
class EnsSiteRun:
    """One daily ENS run at one substation's grid cell, on the cache's half-hourly grid.

    Attributes:
        origin: The run's `power_fcst_init_time` as a naive UTC `datetime64[us]`.
        origin_position: The cache grid position of the half-hour that ends at the origin.
        temperature_c: Temperature in degrees Celsius, shape (members, targets), float64, for the
            half-hours that end after the origin.
        shortwave_w_m2: Downward shortwave flux in W/m², the same shape.
    """

    origin: np.datetime64
    origin_position: int
    temperature_c: np.ndarray
    shortwave_w_m2: np.ndarray

    @property
    def n_targets(self) -> int:
        """Return the number of half-hours the run covers after its origin."""
        return self.temperature_c.shape[1]


def read_site_run(*, frame: pl.DataFrame, origin: datetime, end_times: np.ndarray) -> EnsSiteRun:
    """Build an `EnsSiteRun` from one series' rows of a run's frame.

    Args:
        frame: Columns `ensemble_member`, `valid_time`, `temperature_2m`, and `shortwave` for one
            series and one run, all members.
        origin: The run's `power_fcst_init_time`, timezone-aware.
        end_times: The cache's half-hour ends.

    Returns:
        The run on the grid.

    Raises:
        ValueError: If the frame does not hold every member at the same consecutive half-hours that
            start 30 minutes after the origin, or the origin is not on the cache's grid.
    """
    ordered = frame.sort("ensemble_member", "valid_time")
    members = ordered["ensemble_member"].unique().sort().to_list()
    if members != list(range(ENSEMBLE_SIZE)):
        msg = f"the frame holds members {members}, not 0 to {ENSEMBLE_SIZE - 1}"
        raise ValueError(msg)
    n_targets = ordered.height // ENSEMBLE_SIZE
    times = ordered["valid_time"].dt.replace_time_zone(None).to_numpy().reshape(ENSEMBLE_SIZE, -1)
    if not (times == times[0]).all():
        msg = "the members do not share the same valid times"
        raise ValueError(msg)
    origin64 = naive_utc(moment=origin)
    expected = origin64 + np.arange(1, n_targets + 1) * np.timedelta64(30, "m")
    if not (times[0] == expected).all():
        msg = "the valid times are not consecutive half-hours starting 30 minutes after the origin"
        raise ValueError(msg)
    position = int(np.searchsorted(end_times, origin64))
    if position >= len(end_times) or end_times[position] != origin64:
        msg = f"the origin {origin} is not a half-hour end of the cache's grid"
        raise ValueError(msg)
    return EnsSiteRun(
        origin=origin64,
        origin_position=position,
        temperature_c=ordered["temperature_2m"]
        .to_numpy()
        .astype(np.float64)
        .reshape(ENSEMBLE_SIZE, -1),
        shortwave_w_m2=ordered["shortwave"]
        .to_numpy()
        .astype(np.float64)
        .reshape(ENSEMBLE_SIZE, -1),
    )


@dataclass(frozen=True)
class AnalysisProxy:
    """The control members of consecutive runs, stitched into one past-weather series.

    Attributes:
        temperature_c: Temperature on the whole cache grid, `NaN` where no run supplies a value.
        shortwave_w_m2: Shortwave flux on the whole cache grid, `NaN` likewise.
        run_origin_position: For each grid position, the origin position of the run that supplied
            it, or -1. The leakage tests read this.
    """

    temperature_c: np.ndarray
    shortwave_w_m2: np.ndarray
    run_origin_position: np.ndarray

    @staticmethod
    def build(*, runs: Sequence[EnsSiteRun], n_rows: int) -> AnalysisProxy:
        """Stitch the runs' control members, each over the 24 hours after its origin.

        Run `d` supplies the 48 half-hours that end in `(origin_d, origin_d + 24 h]`, at NWP lead
        9.5 to 33 hours, so a half-hour is never supplied by a run whose origin is at or after it.

        Args:
            runs: The runs, any order.
            n_rows: The length of the cache grid.

        Returns:
            The proxy.

        Raises:
            ValueError: If two runs share an origin.
        """
        temperature = np.full(n_rows, np.nan)
        shortwave = np.full(n_rows, np.nan)
        supplier = np.full(n_rows, -1, dtype=np.int64)
        ordered = sorted(runs, key=lambda run: run.origin_position)
        positions = [run.origin_position for run in ordered]
        if len(set(positions)) != len(positions):
            msg = "two runs share an origin"
            raise ValueError(msg)
        for run in ordered:
            count = min(HALF_HOURS_PER_DAY, run.n_targets, n_rows - run.origin_position - 1)
            span = slice(run.origin_position + 1, run.origin_position + 1 + count)
            temperature[span] = run.temperature_c[CONTROL_MEMBER, :count]
            shortwave[span] = run.shortwave_w_m2[CONTROL_MEMBER, :count]
            supplier[span] = run.origin_position
        return AnalysisProxy(
            temperature_c=temperature, shortwave_w_m2=shortwave, run_origin_position=supplier
        )

    def assert_unseen_by(self, *, origin_position: int, through: int) -> None:
        """Raise if a proxy value up to `through` came from a run that was not yet published.

        Args:
            origin_position: The origin of the forecast that reads the proxy.
            through: The newest grid position the forecast reads.

        Raises:
            ValueError: If a value at or before `through` came from a run with a later origin than
                `origin_position`, or `through` is after the origin.
        """
        if through > origin_position:
            msg = "a forecast may read the proxy only up to its own origin"
            raise ValueError(msg)
        supplied = self.run_origin_position[: through + 1]
        if (supplied > origin_position).any():
            msg = "the proxy holds a value from a run published after the forecast's origin"
            raise ValueError(msg)


@dataclass(frozen=True)
class Calibration:
    """The map from ENS onto the training weather, for one substation.

    Attributes:
        temperature_shift_c: Added to ENS temperature, shape (seasons, lead bands, hour bands).
        irradiance_ratio: Multiplies ENS irradiance, the same shape.
        solar_shrink: The share of the forecast solar anomaly kept, one per lead band: 1 keeps
            ENS as calibrated and 0 replaces it with the climatology.
    """

    temperature_shift_c: np.ndarray
    irradiance_ratio: np.ndarray
    solar_shrink: np.ndarray

    @staticmethod
    def identity() -> Calibration:
        """Return the calibration that changes nothing."""
        shape = (N_SEASONS, N_LEAD_BANDS, N_HOUR_BANDS)
        return Calibration(
            temperature_shift_c=np.zeros(shape),
            irradiance_ratio=np.ones(shape),
            solar_shrink=np.ones(N_LEAD_BANDS),
        )

    @staticmethod
    def fit(
        *,
        season: np.ndarray,
        lead_band: np.ndarray,
        hour_band: np.ndarray,
        ens_mean_temperature_c: np.ndarray,
        target_temperature_c: np.ndarray,
        ens_mean_ghi: np.ndarray,
        target_ghi: np.ndarray,
        solar_shrink: np.ndarray | None = None,
    ) -> Calibration:
        """Fit the shift and the ratio from the ensemble mean against the training weather.

        Args:
            season: Season index of each row (`season_of_months`).
            lead_band: Lead band of each row.
            hour_band: Hour band of each row.
            ens_mean_temperature_c: The ensemble-mean ENS temperature of each row.
            target_temperature_c: ERA5 temperature at the same half-hour.
            ens_mean_ghi: The ensemble-mean ENS irradiance of each row.
            target_ghi: CAMS irradiance at the same half-hour.
            solar_shrink: The kept anomaly share per lead band; defaults to 1.

        Returns:
            The calibration. A cell with fewer than `MIN_CELL_ROWS` usable rows keeps the identity.
        """
        calibration = Calibration.identity()
        shift = calibration.temperature_shift_c
        ratio = calibration.irradiance_ratio
        finite_temperature = np.isfinite(ens_mean_temperature_c) & np.isfinite(target_temperature_c)
        daylight = (
            np.isfinite(ens_mean_ghi)
            & np.isfinite(target_ghi)
            & (ens_mean_ghi > MIN_DAYLIGHT_W_M2)
            & (target_ghi > MIN_DAYLIGHT_W_M2)
        )
        for s in range(N_SEASONS):
            for lead in range(N_LEAD_BANDS):
                for band in range(N_HOUR_BANDS):
                    cell = (season == s) & (lead_band == lead) & (hour_band == band)
                    warm = cell & finite_temperature
                    if warm.sum() >= MIN_CELL_ROWS:
                        shift[s, lead, band] = float(
                            np.mean(target_temperature_c[warm] - ens_mean_temperature_c[warm])
                        )
                    bright = cell & daylight
                    if bright.sum() >= MIN_CELL_ROWS:
                        ratio[s, lead, band] = float(
                            target_ghi[bright].sum() / ens_mean_ghi[bright].sum()
                        )
        kept = np.ones(N_LEAD_BANDS) if solar_shrink is None else np.asarray(solar_shrink, float)
        return replace(calibration, solar_shrink=kept)

    def apply_temperature(
        self,
        *,
        temperature_c: np.ndarray,
        season: np.ndarray,
        lead_band: np.ndarray,
        hour_band: np.ndarray,
    ) -> np.ndarray:
        """Return temperatures shifted by their cell's additive shift.

        Args:
            temperature_c: ENS temperatures.
            season: Season index of each value.
            lead_band: Lead band of each value.
            hour_band: Hour band of each value.

        Returns:
            The calibrated temperatures.
        """
        return temperature_c + self.temperature_shift_c[season, lead_band, hour_band]

    def apply_irradiance(
        self,
        *,
        ghi: np.ndarray,
        clear_sky_ghi: np.ndarray,
        climatology_ghi: np.ndarray,
        season: np.ndarray,
        lead_band: np.ndarray,
        hour_band: np.ndarray,
        shrink: bool = True,
    ) -> np.ndarray:
        """Return irradiance scaled by its cell's ratio, clipped, and shrunk to the climatology.

        Args:
            ghi: ENS irradiance in W/m².
            clear_sky_ghi: Clear-sky irradiance at the same half-hours. A value the ratio would lift
                above the larger of this and the input itself is held there, so a ratio of 1 changes
                nothing even where the input is above the clear-sky formula.
            climatology_ghi: The CAMS month-by-half-hour climatology at the same half-hours.
            season: Season index of each value.
            lead_band: Lead band of each value.
            hour_band: Hour band of each value.
            shrink: False to skip the anomaly shrink (the analysis proxy is not shrunk).

        Returns:
            The calibrated irradiance, never negative and never above the larger of clear sky and
            the input.
        """
        ceiling = np.maximum(np.maximum(clear_sky_ghi, ghi), 0.0)
        scaled = np.clip(ghi * self.irradiance_ratio[season, lead_band, hour_band], 0.0, ceiling)
        if not shrink:
            return scaled
        kept = self.solar_shrink[lead_band]
        mixed = climatology_ghi + kept * (scaled - climatology_ghi)
        return np.clip(mixed, 0.0, np.maximum(ceiling, climatology_ghi))


def clear_sky_ghi(*, end_times: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """Return the Haurwitz clear-sky irradiance at the midpoint of each half-hour.

    Args:
        end_times: Half-hour ends, naive UTC `datetime64[us]`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        W/m², zero when the sun is below the horizon.
    """
    stamps = pl.Series(end_times).dt.replace_time_zone("UTC").dt.offset_by("-15m")
    zenith, _ = solar_position(stamps=stamps, latitude=latitude, longitude=longitude)
    cosine = np.cos(np.radians(np.asarray(zenith, dtype=np.float64)))
    up = cosine > 0.0
    out = np.zeros(len(cosine))
    out[up] = HAURWITZ_SCALE_W_M2 * cosine[up] * np.exp(-HAURWITZ_EXTINCTION / cosine[up])
    return out


def ghi_climatology(*, cache: FeatureCache, site: str, before: datetime) -> np.ndarray:
    """Return the mean CAMS irradiance per calendar month and UTC half-hour of day.

    Args:
        cache: The feature cache.
        site: The substation's label.
        before: Only half-hours ending at or before this moment contribute.

    Returns:
        Shape (12, 48), W/m², `NaN` for an empty cell.
    """
    ghi = cache.sites[site].ghi_w_m2
    ends = cache.end_times
    use = (ends <= naive_utc(moment=before)) & np.isfinite(ghi)
    stamps = ends[use]
    months = stamps.astype("datetime64[M]").astype(np.int64) % 12
    minutes = (stamps - stamps.astype("datetime64[D]")).astype("timedelta64[m]").astype(np.int64)
    slot = (minutes // 30 - 1) % HALF_HOURS_PER_DAY
    table = np.full((12, HALF_HOURS_PER_DAY), np.nan)
    total = np.zeros((12, HALF_HOURS_PER_DAY))
    count = np.zeros((12, HALF_HOURS_PER_DAY))
    np.add.at(total, (months, slot), ghi[use])
    np.add.at(count, (months, slot), 1.0)
    np.divide(total, count, out=table, where=count > 0)
    return table


def climatology_at(*, table: np.ndarray, end_times: np.ndarray) -> np.ndarray:
    """Look up the month-by-half-hour climatology at half-hour ends.

    Args:
        table: `ghi_climatology`'s output.
        end_times: Half-hour ends, naive UTC.

    Returns:
        W/m² at each end.
    """
    months = end_times.astype("datetime64[M]").astype(np.int64) % 12
    minutes = (end_times - end_times.astype("datetime64[D]")).astype("timedelta64[m]")
    slot = (minutes.astype(np.int64) // 30 - 1) % HALF_HOURS_PER_DAY
    return table[months, slot]


def trailing_means(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return the trailing mean over `window` samples ending at each sample, `NaN` if incomplete.

    Args:
        values: A half-hourly series.
        window: Samples per window.

    Returns:
        An array of the same length; the first `window - 1` entries and any window that holds a
        non-finite value are `NaN`.
    """
    finite = np.isfinite(values)
    summed = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    counted = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    totals = summed[window:] - summed[:-window]
    counts = counted[window:] - counted[:-window]
    out[window - 1 :] = np.where(counts == window, totals / window, np.nan)
    return out


def smoothed_temperatures(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return the trailing means over `SMOOTHING_HOURS` of a half-hourly series, shape (n, 4)."""
    return np.stack(
        [
            trailing_means(values=temperature_c, window=HALF_HOURS_PER_HOUR * hours)
            for hours in SMOOTHING_HOURS
        ],
        axis=1,
    )


def unit_pv_shape_and_ghi(
    *, end_times: np.ndarray, ghi_w_m2: np.ndarray, latitude: float, longitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fleet's unit PV shape and the half-hourly irradiance of a block of hours.

    The block must start 30 minutes after an hour and hold whole hours. Each pair of half-hours
    ending at :30 and :00 becomes the hour-ending mean, as CAMS is stamped, and
    `half_hour_sun_and_sky` spreads the hour back over its two half-hours by the sun's height.

    Args:
        end_times: Half-hour ends, naive UTC, an even number starting at :30.
        ghi_w_m2: Irradiance at those half-hours.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The unit shape and the irradiance, per half-hour; `NaN` for an hour with a missing value.

    Raises:
        ValueError: If the block does not start at :30 or holds an odd number of half-hours.
    """
    if len(end_times) % 2 or _minute(end_times[0]) != 30:
        msg = "the block must hold whole hours and start 30 minutes past an hour"
        raise ValueError(msg)
    pairs = ghi_w_m2.reshape(-1, 2)
    hourly_mean = pairs.mean(axis=1)
    hour_ends = pl.Series(end_times[1::2]).dt.replace_time_zone("UTC")
    known = np.isfinite(hourly_mean)
    shape = np.full(len(end_times), np.nan)
    ghi = np.full(len(end_times), np.nan)
    if known.any():
        sky = half_hour_sun_and_sky(
            hourly=pl.DataFrame(
                {"time": hour_ends.filter(pl.Series(known)), "ghi_w_m2": hourly_mean[known]}
            ),
            latitude=latitude,
            longitude=longitude,
        )
        keep = np.repeat(known, 2)
        shape[keep] = plant_power_mw(sky=sky, parameters=FLEET)
        ghi[keep] = sky.ghi_w_m2
    return shape, ghi


def slice_cache(
    *, cache: FeatureCache, start: int, stop: int, sites: Sequence[str]
) -> FeatureCache:
    """Return the cache restricted to the grid rows `[start, stop)` and to `sites`.

    Args:
        cache: The feature cache.
        start: The first row.
        stop: One past the last row.
        sites: The substations to keep.

    Returns:
        A cache whose arrays are views of the original.
    """
    window = slice(start, stop)
    calendar = CalendarColumns(
        **{f.name: getattr(cache.calendar, f.name)[window] for f in fields(CalendarColumns)}
    )
    kept = {}
    for label in sites:
        original = cache.sites[label]
        arrays = {
            f.name: getattr(original, f.name)[window]
            for f in fields(SiteFeatures)
            if f.name != "table"
        }
        kept[label] = SiteFeatures(table=original.table, **arrays)
    return FeatureCache(
        key=cache.key,
        end_times=cache.end_times[window],
        calendar=calendar,
        sites=kept,
        node_ids={label: cache.node_ids[label] for label in sites if label in cache.node_ids},
    )


@dataclass(frozen=True)
class WeatherContext:
    """What the weather layer needs about one substation.

    Attributes:
        site: The substation's label.
        calibration: The ENS-to-training-weather map.
        climatology: `ghi_climatology`'s table, for the solar shrink.
        proxy: The analysis proxy of the substation's grid cell.
    """

    site: str
    calibration: Calibration
    climatology: np.ndarray
    proxy: AnalysisProxy


def _bands(*, cache: FeatureCache, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the season and hour band of grid positions."""
    months = cache.calendar.local_date[positions].astype("datetime64[M]").astype(np.int64) % 12 + 1
    return season_of_months(months=months), cache.calendar.band[positions].astype(np.int64)


def proxy_cache(*, cache: FeatureCache, context: WeatherContext) -> FeatureCache:
    """Return the cache with the site's weather replaced by the calibrated analysis proxy.

    Positions the proxy does not cover, and positions whose trailing temperature windows are
    incomplete, hold `NaN`. The proxy is mapped with the first lead band's calibration and is not
    shrunk, because its NWP lead is 9.5 to 33 hours from the run and so 0.5 to 24 hours from the
    origin.

    Args:
        cache: The full feature cache.
        context: The substation's weather context.

    Returns:
        A cache whose site arrays `temperature_c`, `smoothed_temperature_c`, `ghi_w_m2`, and
        `pv_unit_shape` come from the proxy.
    """
    features = cache.sites[context.site]
    n = len(cache.end_times)
    positions = np.arange(n)
    season, hour_band = _bands(cache=cache, positions=positions)
    lead = np.zeros(n, dtype=np.int64)
    temperature = context.calibration.apply_temperature(
        temperature_c=context.proxy.temperature_c,
        season=season,
        lead_band=lead,
        hour_band=hour_band,
    )
    sky = clear_sky_ghi(
        end_times=cache.end_times,
        latitude=features.table.latitude,
        longitude=features.table.longitude,
    )
    ghi = context.calibration.apply_irradiance(
        ghi=context.proxy.shortwave_w_m2,
        clear_sky_ghi=sky,
        climatology_ghi=climatology_at(table=context.climatology, end_times=cache.end_times),
        season=season,
        lead_band=lead,
        hour_band=hour_band,
        shrink=False,
    )
    shape = np.full(n, np.nan)
    flux = np.full(n, np.nan)
    covered = np.flatnonzero(np.isfinite(ghi))
    if len(covered):
        first = covered[0] + (0 if _minute(cache.end_times[covered[0]]) == 30 else 1)
        last = covered[-1] - (0 if _minute(cache.end_times[covered[-1]]) == 0 else 1)
        count = (last - first + 1) // 2 * 2
        span = slice(first, first + count)
        shape[span], flux[span] = unit_pv_shape_and_ghi(
            end_times=cache.end_times[span],
            ghi_w_m2=ghi[span],
            latitude=features.table.latitude,
            longitude=features.table.longitude,
        )
    return replace_site(
        cache=cache,
        site=context.site,
        temperature_c=temperature,
        smoothed_temperature_c=smoothed_temperatures(temperature_c=temperature),
        ghi_w_m2=flux,
        pv_unit_shape=shape,
    )


def _minute(stamp: np.datetime64) -> int:
    """Return the minute past the hour of a naive UTC timestamp."""
    return int((stamp - stamp.astype("datetime64[h]")).astype("timedelta64[m]").astype(np.int64))


def replace_site(*, cache: FeatureCache, site: str, **arrays: np.ndarray) -> FeatureCache:
    """Return the cache with some of one substation's arrays replaced.

    Args:
        cache: The feature cache.
        site: The substation's label.
        **arrays: Replacement arrays, keyed by `SiteFeatures` field name.

    Returns:
        A new cache sharing every other array.
    """
    return replace(cache, sites={**cache.sites, site: replace(cache.sites[site], **arrays)})


def driven_window(
    *, cache: FeatureCache, context: WeatherContext, run: EnsSiteRun, member: int
) -> tuple[FeatureCache, slice]:
    """Return the cache window of one member's forecast for one run, driven by that member.

    Temperature before the origin comes from the analysis proxy and after it from the member, both
    calibrated, and the trailing means run over the stitched series. Irradiance is the member's,
    calibrated and shrunk towards the climatology. The window holds the rows that end after the
    origin, up to the run's horizon.

    Args:
        cache: The full feature cache.
        context: The substation's weather context.
        run: The run.
        member: The ensemble member, 0 to 50.

    Returns:
        The windowed cache, and the slice of the full grid that the window covers.

    Raises:
        ValueError: If the window is shorter than a day, or the proxy lacks the temperature history.
    """
    features = cache.sites[context.site]
    first = run.origin_position + 1
    count = min(HORIZON_HALF_HOURS, run.n_targets, len(cache.end_times) - first)
    count -= count % 2
    if count < HALF_HOURS_PER_DAY:
        msg = "the run's window is shorter than a day"
        raise ValueError(msg)
    window = slice(first, first + count)
    past = slice(first - HISTORY_HALF_HOURS, first)
    context.proxy.assert_unseen_by(origin_position=run.origin_position, through=run.origin_position)
    ends = cache.end_times[window]
    season, hour_band = _bands(cache=cache, positions=np.arange(first, first + count))
    lead_hours = (np.arange(1, count + 1)) / HALF_HOURS_PER_HOUR
    lead = lead_band_of_hours(lead_hours=lead_hours)
    past_season, past_band = _bands(cache=cache, positions=np.arange(past.start, past.stop))
    past_temperature = context.calibration.apply_temperature(
        temperature_c=context.proxy.temperature_c[past],
        season=past_season,
        lead_band=np.zeros(HISTORY_HALF_HOURS, dtype=np.int64),
        hour_band=past_band,
    )
    future_temperature = context.calibration.apply_temperature(
        temperature_c=run.temperature_c[member, :count],
        season=season,
        lead_band=lead,
        hour_band=hour_band,
    )
    stitched = np.concatenate([past_temperature, future_temperature])
    if not np.isfinite(past_temperature).all():
        msg = "the analysis proxy lacks the past week's temperature before the origin"
        raise ValueError(msg)
    sky = clear_sky_ghi(
        end_times=ends, latitude=features.table.latitude, longitude=features.table.longitude
    )
    ghi = context.calibration.apply_irradiance(
        ghi=run.shortwave_w_m2[member, :count],
        clear_sky_ghi=sky,
        climatology_ghi=climatology_at(table=context.climatology, end_times=ends),
        season=season,
        lead_band=lead,
        hour_band=hour_band,
    )
    shape, flux = unit_pv_shape_and_ghi(
        end_times=ends,
        ghi_w_m2=ghi,
        latitude=features.table.latitude,
        longitude=features.table.longitude,
    )
    sliced = slice_cache(cache=cache, start=window.start, stop=window.stop, sites=[context.site])
    driven = replace_site(
        cache=sliced,
        site=context.site,
        temperature_c=stitched[HISTORY_HALF_HOURS:],
        smoothed_temperature_c=smoothed_temperatures(temperature_c=stitched)[HISTORY_HALF_HOURS:],
        ghi_w_m2=flux,
        pv_unit_shape=shape,
    )
    return driven, window


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


def assert_cut_not_after(*, fit: GamFit, origin: datetime) -> None:
    """Raise if a fit's cut is after the origin of the forecast that uses it.

    Args:
        fit: The fit.
        origin: The forecast's `power_fcst_init_time`.

    Raises:
        ValueError: If `fit.cut` is after `origin`.
    """
    if fit.cut > origin:
        msg = f"the fit's cut {fit.cut} is after the forecast's origin {origin}: lookahead"
        raise ValueError(msg)


def member_means(
    *,
    cache: FeatureCache,
    context: WeatherContext,
    run: EnsSiteRun,
    origin: datetime,
    fits: Sequence[GamFit],
    members: Sequence[int] | None = None,
) -> tuple[np.ndarray, slice]:
    """Return each member's mean forecast in points, from the fit of the latest cut before the run.

    Args:
        cache: The full feature cache.
        context: The substation's weather context.
        run: The run.
        origin: The run's `power_fcst_init_time`, timezone-aware.
        fits: The monthly fits, ascending by cut.
        members: The members to run; defaults to all 51.

    Returns:
        An array of shape (members, targets) in points of effective capacity, and the slice of the
        full grid it covers.

    Raises:
        ValueError: If no fit is at or before the origin.
    """
    chosen = latest_cut_index(cuts=[fit.cut for fit in fits], moment=origin)
    if chosen < 0:
        msg = f"no fit has a cut at or before {origin}"
        raise ValueError(msg)
    fit = fits[chosen]
    assert_cut_not_after(fit=fit, origin=origin)
    wanted = list(range(ENSEMBLE_SIZE)) if members is None else list(members)
    rows: list[np.ndarray] = []
    window = slice(0, 0)
    for member in wanted:
        driven, window = driven_window(cache=cache, context=context, run=run, member=member)
        rows.append(predict_gam(cache=driven, fit=fit, index=np.arange(len(driven.end_times))))
    return np.stack(rows), window


def proxy_residual_stream(*, cache: FeatureCache, fits: Sequence[GamFit], site: str) -> np.ndarray:
    """Return observed minus the chained fits' forecast on a cache, `NaN` where it cannot be made.

    Pass the proxy cache for the ENS-driven residuals of the live filter, or the ERA5 and CAMS
    cache for `GAM-K-era5`. Fit `i` supplies the rows after its cut and up to the next cut, as
    `studies.dp_assimilation.stitch_out_of_sample_residuals` does.

    Args:
        cache: The cache whose weather arrays drive the fits.
        fits: The monthly fits, ascending by cut.
        site: The substation's label.

    Returns:
        A stream on the grid in points.
    """
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


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool the members' quantiles into the mixture's 11 quantiles.

    Each member contributes its 11 quantiles as atoms of mass `masses / members`, where a level's
    mass is the interval between the mid-points to its neighbours. The mixture's level-`p` quantile
    is the smallest atom value whose cumulative mass reaches `p`.

    Args:
        member_quantiles: Shape (cases, members, 11).

    Returns:
        Shape (cases, 11), sorted along the levels.
    """
    levels = np.array(QUANTILE_LEVELS)
    masses = np.diff(np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [1.0]]))
    cases, members, _ = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * len(levels))
    weights = np.tile(masses, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (levels - 1e-9)).argmax(axis=1)
    return np.sort(np.take_along_axis(sorted_values, picks, axis=1), axis=1)


@dataclass(frozen=True)
class LeadHourSpread:
    """Centred residual quantiles per lead band, hour band, and quarter, with a spread factor.

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
        """Return the spread with new factors, one per lead band."""
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
    """Return the DP's control-member forecast of one run as a base margin in MW.

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
        Columns `time_series_id`, `power_fcst_init_time`, `valid_time`, `margin_mw`, and
        `fit_cut`, one row per half-hour of the run's window.
    """
    means, window = member_means(
        cache=cache, context=context, run=run, origin=origin, fits=fits, members=[CONTROL_MEMBER]
    )
    table = cache.sites[context.site].table
    chosen = fits[latest_cut_index(cuts=[fit.cut for fit in fits], moment=origin)]
    return pl.DataFrame(
        {
            "time_series_id": pl.Series([table.time_series_id] * means.shape[1], dtype=pl.Int32),
            "power_fcst_init_time": pl.Series([origin] * means.shape[1]).dt.replace_time_zone(
                "UTC"
            ),
            "valid_time": pl.Series(cache.end_times[window]).dt.replace_time_zone("UTC"),
            "margin_mw": means[0] * table.effective_capacity_mw / 100.0,
            "fit_cut": pl.Series([chosen.cut] * means.shape[1]).dt.replace_time_zone("UTC"),
        }
    )


ACTUAL_TOLERANCE_POINTS: Final[float] = 1e-6
"""How closely experiment 87's `actual` column and the cache's `y` must agree."""


def assert_actual_matches_cache(
    *, valid_time: np.ndarray, actual_points: np.ndarray, cache: FeatureCache, site: str
) -> int:
    """Raise if a frame's observed power disagrees with the cache's on the rows they share.

    Experiment 87 and the cache both divide power by `effective_capacity_mw`. A different capacity
    or a different timestamp convention would make every DP score and every XGBoost score differ
    by a hidden factor.

    Args:
        valid_time: The frame's valid times as naive UTC `datetime64[us]`.
        actual_points: The frame's observed power in points of effective capacity.
        cache: The feature cache.
        site: The substation's label.

    Returns:
        The number of rows compared.

    Raises:
        ValueError: If no row is shared or any shared row differs by more than
            `ACTUAL_TOLERANCE_POINTS`.
    """
    position = np.searchsorted(cache.end_times, valid_time)
    inside = position < len(cache.end_times)
    matched = np.zeros(len(valid_time), dtype=bool)
    matched[inside] = cache.end_times[position[inside]] == valid_time[inside]
    cached = np.full(len(valid_time), np.nan)
    cached[matched] = cache.sites[site].y[position[matched]]
    shared = matched & np.isfinite(cached) & np.isfinite(actual_points)
    if not shared.any():
        msg = f"no row of {site} is shared between the frame and the cache"
        raise ValueError(msg)
    worst = float(np.abs(cached[shared] - actual_points[shared]).max())
    if worst > ACTUAL_TOLERANCE_POINTS:
        msg = f"the frame's actual differs from the cache's by up to {worst:.3g} points at {site}"
        raise ValueError(msg)
    return int(shared.sum())


@dataclass(frozen=True)
class RunForecast:
    """The forecast of one run at one substation.

    Attributes:
        quantiles: The 11 quantiles of the member mixture, shape (targets, 11), in points.
        mean: The mean over members of the corrected forecast, in points.
        level: The filter's persistent correction at each target (zeros without a filter).
        anomaly: The filter's decaying correction at each target (zeros without a filter).
        window: The slice of the full grid the forecast covers.
    """

    quantiles: np.ndarray
    mean: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    window: slice


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
    the spread's quantiles around it, and the mixture pools the members.

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
    means, window = member_means(cache=cache, context=context, run=run, origin=origin, fits=fits)
    target = np.arange(window.start, window.stop)
    n_targets = len(target)
    level = np.zeros(n_targets)
    anomaly = np.zeros(n_targets)
    if trace is not None:
        cutoff = usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=origin)]),
            delay_hours=delay_hours,
        )
        correction = correction_at(
            trace=trace,
            cutoff=np.full(n_targets, cutoff[0]),
            target=target,
            band=cache.calendar.band[target].astype(np.int64),
        )
        level, anomaly = correction.level, correction.anomaly
    corrected = means + (level + anomaly)[None, :]
    band = cache.calendar.band[target].astype(np.int64)
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
        window=window,
    )
