"""The ECMWF-ensemble (ENS) weather layer of the differentiable-physics (DP) series, for Part 6.

Parts 1 to 5 drove the convex demand model with ERA5 temperature and CAMS irradiance. This module
turns the weather forecasts a live service would have had into the same arrays: the 51 ENS members
of one daily run for the future, and a stitched "analysis proxy" for the past. It needs no torch;
`studies.dp_ens` fits nothing either but calls the fitted model, and holds everything that does.

**The DP is fitted on ERA5 and CAMS and driven here by calibrated ENS.** That departs from the rule
to train each weather product on its own history. The arm that follows the rule, the GAM fitted on
ENS control rows (GAM-ens), is the experiment's test of the departure.

**One run is usable from its `power_fcst_init_time`.** A run initialised at 00:00 UTC is published
by 09:00 UTC (the origin), and its frame holds the half-hours that end after the origin. The
forecast issued at the origin may use nothing published later.

**The analysis proxy replaces ERA5 for the past.** For each past half-hour the proxy takes the
control member of the latest run whose origin is before the half-hour, so the 48 half-hours that end
in `(origin_d, origin_d + 24 h]` come from run `d` at NWP lead 9.5 to 33 hours. Half-hours up to and
including the origin of run `d` therefore never come from run `d` or later, and the module's tests
assert that. ERA5 is never read for the weather of a forecast.

**Calibration maps ENS onto the weather the DP was trained on.** ENS temperature takes an additive
shift and ENS irradiance a multiplicative ratio clipped to `[0, clear sky]`, per season, lead band,
and hour band, and then the forecast solar anomaly is shrunk towards the CAMS month-by-half-hour
climatology by a coefficient per lead band.

**What this module leaves to the experiment's scripts.** It fills no `WeatherInputLayer` hook, so
the pooled arms of branch B1 would need their own ENS path. It does not derate the PV shape by the
member's temperature, because the cache's unit PV shape (CAMS) is not derated either and the nesting
test needs the two to match; the convex model derates the shape itself, from the temperature it is
given. The scripts carry the X-bar+K invariance check, the check that no valid time on or after
2026-07-01 is scored, and the reproduction of Part 4's stored predictions on the real cache.

**Cost.** Everything that depends on the run but not on the member (the sun's position, clear-sky
irradiance, bands, climatology) is computed once per run in `RunWindow`, and the members' irradiance
is turned into PV shapes in one batch.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Final

import numpy as np
import polars as pl
import pvlib

from studies.baselines import haurwitz_w_m2
from studies.dp_calendar import (
    BAND_NAMES,
    HALF_HOURS_PER_DAY,
    SMOOTHING_HOURS,
    CalendarColumns,
)
from studies.dp_features import FLEET, FeatureCache, SiteFeatures, naive_utc
from studies.ens_members import CONTROL_MEMBER, ENSEMBLE_SIZE
from studies.power import HALF_HOURS_PER_HOUR
from studies.pv_physics import PlantParameters, SunAndSky, plant_power_mw, solar_position

HORIZON_HALF_HOURS: Final[int] = 336
"""The forecast horizon: 168 hours of half-hours, from the origin."""

HISTORY_HALF_HOURS: Final[int] = HALF_HOURS_PER_HOUR * max(SMOOTHING_HOURS)
"""Half-hours of past temperature the longest trailing mean needs."""

LEAD_BAND_UPPER_HOURS: Final[tuple[float, ...]] = (24.0, 48.0, 144.0, 168.0)
"""Upper edges of the calibration's lead bands: 0-24, 24-48, 48-144, and 144-168 hours."""

N_LEAD_BANDS: Final[int] = len(LEAD_BAND_UPPER_HOURS)
"""The number of calibration lead bands."""

N_SEASONS: Final[int] = 4
"""The number of calibration seasons: December to February, March to May, and so on."""

N_HOUR_BANDS: Final[int] = len(BAND_NAMES)
"""The number of hour bands of `studies.dp_calendar`."""

MIN_CELL_ROWS: Final[int] = 200
"""A calibration cell with fewer usable rows keeps the identity map."""

MIN_DAYLIGHT_W_M2: Final[float] = 5.0
"""An ENS or CAMS half-hour needs more than this to enter an irradiance ratio."""


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


def _minute(stamp: np.datetime64) -> int:
    """Return the minute past the hour of a naive UTC timestamp."""
    return int((stamp - stamp.astype("datetime64[h]")).astype("timedelta64[m]").astype(np.int64))


def _month_and_slot(*, end_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the calendar month (0 to 11) and the UTC half-hour-of-day slot of half-hour ends.

    The slot of the half-hour that ends at 00:30 is 0 and of the one that ends at 00:00 is 47.
    """
    months = end_times.astype("datetime64[M]").astype(np.int64) % 12
    minutes = (end_times - end_times.astype("datetime64[D]")).astype("timedelta64[m]")
    return months, (minutes.astype(np.int64) // 30 - 1) % HALF_HOURS_PER_DAY


@dataclass(frozen=True)
class EnsSiteRun:
    """One daily ENS run at one substation's grid cell, on the cache's half-hourly grid.

    Attributes:
        origin_position: The cache grid position of the half-hour that ends at the origin.
        temperature_c: Temperature in degrees Celsius, shape (members, targets), float64, for the
            half-hours that end after the origin.
        shortwave_w_m2: Downward shortwave flux in W/m², the same shape.
    """

    origin_position: int
    temperature_c: np.ndarray
    shortwave_w_m2: np.ndarray

    @property
    def n_targets(self) -> int:
        """Return the number of half-hours the run covers after its origin."""
        return self.temperature_c.shape[1]

    def usable_members(self) -> list[int]:
        """Return the members whose temperature and shortwave are finite at every half-hour.

        A member with any missing value is dropped from the run at this substation, and the
        mixture pools the members that remain. The control member feeds the analysis proxy and the
        base margin, so a run whose control member is incomplete is refused.

        Returns:
            The usable members, ascending.

        Raises:
            ValueError: If the control member has a missing value.
        """
        finite = np.isfinite(self.temperature_c).all(axis=1) & np.isfinite(self.shortwave_w_m2).all(
            axis=1
        )
        if not finite[CONTROL_MEMBER]:
            msg = "the control member has a missing value, so the run cannot be used"
            raise ValueError(msg)
        return np.flatnonzero(finite).tolist()


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

    def assert_covers(self, *, first: int, last: int) -> None:
        """Raise if any grid position from `first` to `last` has no finite proxy value.

        One missing run leaves a 24-hour hole, and every forecast whose trailing 168-hour mean
        spans the hole fails, so the scripts check coverage before they start.

        Args:
            first: The first grid position that must be covered.
            last: The last grid position that must be covered.

        Raises:
            ValueError: If a position has a missing temperature or shortwave value.
        """
        span = slice(first, last + 1)
        missing = ~(np.isfinite(self.temperature_c[span]) & np.isfinite(self.shortwave_w_m2[span]))
        if missing.any():
            msg = (
                f"the analysis proxy has {int(missing.sum())} uncovered half-hours between "
                f"positions {first} and {last}, the first at {first + int(missing.argmax())}"
            )
            raise ValueError(msg)

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


def assert_months_allowed(*, valid_time: np.ndarray, allowed: Collection[tuple[int, int]]) -> None:
    """Raise if any row falls in a (year, month) outside `allowed`.

    The calibration is fitted on the tuning months only, never on a month that is scored.

    Args:
        valid_time: Naive UTC `datetime64` times of the rows.
        allowed: The permitted (year, month) pairs.

    Raises:
        ValueError: If a row is in another month.
    """
    months = valid_time.astype("datetime64[M]")
    years = months.astype("datetime64[Y]").astype(np.int64) + 1970
    month_numbers = months.astype(np.int64) % 12 + 1
    pairs = set(zip(years.tolist(), month_numbers.tolist(), strict=True))
    extra = pairs - set(allowed)
    if extra:
        msg = f"rows from months {sorted(extra)} are not allowed in this fit"
        raise ValueError(msg)


@dataclass(frozen=True)
class Calibration:
    """The map from ENS onto the training weather, for one substation.

    Attributes:
        temperature_shift_c: Added to ENS temperature, shape (seasons, lead bands, hour bands).
        irradiance_ratio: Multiplies ENS irradiance, the same shape.
        solar_shrink: The share of the forecast solar anomaly kept, one per lead band: 1 keeps
            ENS as calibrated and 0 replaces it with the climatology. It is tuned, not fitted.
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
        origin_time: np.ndarray,
        allowed_months: Collection[tuple[int, int]],
        season: np.ndarray,
        lead_band: np.ndarray,
        hour_band: np.ndarray,
        ens_mean_temperature_c: np.ndarray,
        target_temperature_c: np.ndarray,
        ens_mean_ghi: np.ndarray,
        target_ghi: np.ndarray,
    ) -> Calibration:
        """Fit the shift and the ratio from the ensemble mean against the training weather.

        Args:
            origin_time: The naive UTC `power_fcst_init_time` of the run each row belongs to.
            allowed_months: The (year, month) pairs of run origins the fit may use: the tuning
                months.
            season: Season index of each row (`season_of_months`).
            lead_band: Lead band of each row.
            hour_band: Hour band of each row.
            ens_mean_temperature_c: The ensemble-mean ENS temperature of each row.
            target_temperature_c: ERA5 temperature at the same half-hour.
            ens_mean_ghi: The ensemble-mean ENS irradiance of each row.
            target_ghi: CAMS irradiance at the same half-hour.

        Returns:
            The calibration with `solar_shrink` 1. A cell with fewer than `MIN_CELL_ROWS` usable
            rows keeps the identity.

        Raises:
            ValueError: If a row is in a month outside `allowed_months`.
        """
        assert_months_allowed(valid_time=origin_time, allowed=allowed_months)
        calibration = Calibration.identity()
        shift = calibration.temperature_shift_c
        ratio = calibration.irradiance_ratio
        has_temperature_pair = np.isfinite(ens_mean_temperature_c) & np.isfinite(
            target_temperature_c
        )
        has_daylight_pair = (
            np.isfinite(ens_mean_ghi)
            & np.isfinite(target_ghi)
            & (ens_mean_ghi > MIN_DAYLIGHT_W_M2)
            & (target_ghi > MIN_DAYLIGHT_W_M2)
        )
        for season_index in range(N_SEASONS):
            for lead in range(N_LEAD_BANDS):
                for band in range(N_HOUR_BANDS):
                    cell = (season == season_index) & (lead_band == lead) & (hour_band == band)
                    temperature_rows = cell & has_temperature_pair
                    if temperature_rows.sum() >= MIN_CELL_ROWS:
                        shift[season_index, lead, band] = float(
                            np.mean(
                                target_temperature_c[temperature_rows]
                                - ens_mean_temperature_c[temperature_rows]
                            )
                        )
                    daylight_rows = cell & has_daylight_pair
                    if daylight_rows.sum() >= MIN_CELL_ROWS:
                        ratio[season_index, lead, band] = float(
                            target_ghi[daylight_rows].sum() / ens_mean_ghi[daylight_rows].sum()
                        )
        return calibration

    def with_solar_shrink(self, *, solar_shrink: Sequence[float]) -> Calibration:
        """Return the calibration with a new kept-anomaly share per lead band."""
        if len(solar_shrink) != N_LEAD_BANDS:
            msg = f"the solar shrink needs one value per lead band, {N_LEAD_BANDS} in all"
            raise ValueError(msg)
        return replace(self, solar_shrink=np.asarray(solar_shrink, dtype=np.float64))

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
            clear_sky_ghi: Clear-sky irradiance at the same half-hours. A value the ratio would
                lift above the larger of this and the input itself is held there, so a ratio of 1
                changes nothing even where the input is above the clear-sky formula.
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


def ghi_climatology(*, cache: FeatureCache, site: str, before: datetime) -> np.ndarray:
    """Return the mean CAMS irradiance per calendar month and UTC half-hour of day.

    Args:
        cache: The feature cache.
        site: The substation's label.
        before: Only half-hours ending at or before this moment contribute. Pass a moment no later
            than the first origin the climatology serves.

    Returns:
        Shape (12, 48), W/m², `NaN` for an empty cell.
    """
    ghi = cache.sites[site].ghi_w_m2
    use = (cache.end_times <= naive_utc(moment=before)) & np.isfinite(ghi)
    months, slot = _month_and_slot(end_times=cache.end_times[use])
    total = np.zeros((12, HALF_HOURS_PER_DAY))
    count = np.zeros((12, HALF_HOURS_PER_DAY))
    np.add.at(total, (months, slot), ghi[use])
    np.add.at(count, (months, slot), 1.0)
    table = np.full((12, HALF_HOURS_PER_DAY), np.nan)
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
    months, slot = _month_and_slot(end_times=end_times)
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


@dataclass(frozen=True)
class SunGeometry:
    """The sun's position over a block of whole hours at one place, computed once.

    The block starts 30 minutes past an hour and holds an even number of half-hours, so each pair
    of half-hours is one hour that ends on the hour.

    Attributes:
        zenith_deg: The apparent zenith at the midpoint of each half-hour.
        azimuth_deg: The sun's azimuth there.
        share: Each half-hour's share of its hour's irradiance, from the cosines of the zenith, so
            the two half-hours average back to the hour's value.
        day_of_year: The day of the year of each midpoint.
        clear_sky_w_m2: The Haurwitz clear-sky irradiance at each midpoint.
        end_times: The half-hour ends.
    """

    zenith_deg: np.ndarray
    azimuth_deg: np.ndarray
    share: np.ndarray
    day_of_year: np.ndarray
    clear_sky_w_m2: np.ndarray
    end_times: np.ndarray

    @staticmethod
    def build(*, end_times: np.ndarray, latitude: float, longitude: float) -> SunGeometry:
        """Compute the geometry of a block of whole hours.

        Args:
            end_times: Half-hour ends, naive UTC, an even number starting at :30.
            latitude: Degrees north.
            longitude: Degrees east.

        Returns:
            The geometry.

        Raises:
            ValueError: If the block does not start at :30 or holds an odd number of half-hours.
        """
        if len(end_times) % 2 or _minute(end_times[0]) != 30:
            msg = "the block must hold whole hours and start 30 minutes past an hour"
            raise ValueError(msg)
        stamps = pl.Series(end_times).dt.replace_time_zone("UTC").dt.offset_by("-15m")
        zenith, azimuth = solar_position(stamps=stamps, latitude=latitude, longitude=longitude)
        cosine = np.clip(np.cos(np.radians(zenith)), 0.0, None)
        hour_mean = np.repeat(cosine.reshape(-1, 2).mean(axis=1), 2)
        share = np.divide(cosine, hour_mean, out=np.zeros_like(cosine), where=hour_mean > 0)
        return SunGeometry(
            zenith_deg=zenith,
            azimuth_deg=azimuth,
            share=share,
            day_of_year=stamps.dt.ordinal_day().to_numpy(),
            clear_sky_w_m2=haurwitz_w_m2(apparent_zenith_deg=zenith),
            end_times=end_times,
        )

    def pv_shape_and_ghi(
        self, *, ghi_w_m2: np.ndarray, plant: PlantParameters = FLEET
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the fleet's unit PV shape and the half-hourly irradiance of several series.

        Each pair of half-hours ending at :30 and :00 becomes the hour-ending mean, as CAMS is
        stamped, and the sun's height spreads the hour back over its two half-hours.

        Args:
            ghi_w_m2: Irradiance at the block's half-hours, shape (series, half-hours).
            plant: The plant whose output is the shape; the generic south-facing fleet by default.

        Returns:
            The unit shape and the irradiance, both of shape (series, half-hours); `NaN` for an
            hour with a missing value.
        """
        series, n = ghi_w_m2.shape
        hourly = ghi_w_m2.reshape(series, -1, 2).mean(axis=2)
        known = np.repeat(np.isfinite(hourly), 2, axis=1)
        hour_ghi = np.repeat(np.nan_to_num(hourly), 2, axis=1)
        flux = np.clip(hour_ghi * self.share[None, :], 0.0, None)
        zenith = np.tile(self.zenith_deg, series)
        split = pvlib.irradiance.erbs(
            ghi=flux.ravel(), zenith=zenith, datetime_or_doy=np.tile(self.day_of_year, series)
        )
        sky = SunAndSky(
            half_hour_end_time=np.tile(self.end_times, series),
            ghi_w_m2=flux.ravel(),
            dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
            dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
            zenith_deg=zenith,
            sun_azimuth_deg=np.tile(self.azimuth_deg, series),
        )
        shape = plant_power_mw(sky=sky, parameters=plant).reshape(series, n)
        return np.where(known, shape, np.nan), np.where(known, flux, np.nan)


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


@dataclass(frozen=True)
class WeatherContext:
    """What the weather layer needs about one substation.

    Attributes:
        site: The substation's label.
        calibration: The ENS-to-training-weather map.
        climatology: `ghi_climatology`'s table, for the solar shrink.
        proxy: The analysis proxy of the substation's grid cell.
        plant: The plant whose output is the unit PV shape; the generic south-facing fleet by
            default.
    """

    site: str
    calibration: Calibration
    climatology: np.ndarray
    proxy: AnalysisProxy
    plant: PlantParameters = FLEET


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
    season, hour_band = _bands(cache=cache, positions=np.arange(n))
    lead = np.zeros(n, dtype=np.int64)
    temperature = context.calibration.apply_temperature(
        temperature_c=context.proxy.temperature_c,
        season=season,
        lead_band=lead,
        hour_band=hour_band,
    )
    shape = np.full(n, np.nan)
    flux = np.full(n, np.nan)
    covered = np.flatnonzero(np.isfinite(context.proxy.shortwave_w_m2))
    if len(covered):
        first = covered[0] + (0 if _minute(cache.end_times[covered[0]]) == 30 else 1)
        last = covered[-1] - (0 if _minute(cache.end_times[covered[-1]]) == 0 else 1)
        span = slice(first, first + (last - first + 1) // 2 * 2)
        sun = SunGeometry.build(
            end_times=cache.end_times[span],
            latitude=features.table.latitude,
            longitude=features.table.longitude,
        )
        ghi = context.calibration.apply_irradiance(
            ghi=context.proxy.shortwave_w_m2[span],
            clear_sky_ghi=sun.clear_sky_w_m2,
            climatology_ghi=climatology_at(table=context.climatology, end_times=sun.end_times),
            season=season[span],
            lead_band=lead[span],
            hour_band=hour_band[span],
            shrink=False,
        )
        block_shape, block_flux = sun.pv_shape_and_ghi(ghi_w_m2=ghi[None, :], plant=context.plant)
        shape[span], flux[span] = block_shape[0], block_flux[0]
    return replace_site(
        cache=cache,
        site=context.site,
        temperature_c=temperature,
        smoothed_temperature_c=smoothed_temperatures(temperature_c=temperature),
        ghi_w_m2=flux,
        pv_unit_shape=shape,
    )


@dataclass(frozen=True)
class RunWindow:
    """Everything about one run's forecast window that does not depend on the member.

    Attributes:
        window: The slice of the full grid the window covers: the rows that end after the origin.
        cache: The cache cut to the window and to the site.
        sun: The sun's geometry over the window.
        season: Season index of each row.
        hour_band: Hour band of each row.
        lead_band: Calibration lead band of each row.
        past_temperature_c: The calibrated proxy temperature of the `HISTORY_HALF_HOURS` before
            the window.
        climatology_w_m2: The CAMS climatology at each row.
    """

    window: slice
    cache: FeatureCache
    sun: SunGeometry
    season: np.ndarray
    hour_band: np.ndarray
    lead_band: np.ndarray
    past_temperature_c: np.ndarray
    climatology_w_m2: np.ndarray

    @staticmethod
    def build(*, cache: FeatureCache, context: WeatherContext, run: EnsSiteRun) -> RunWindow:
        """Compute the member-independent parts of a run's window.

        Args:
            cache: The full feature cache.
            context: The substation's weather context.
            run: The run.

        Returns:
            The window.

        Raises:
            ValueError: If the window is shorter than a day, the proxy lacks the past week's
                temperature, or the proxy holds a value from a run published after the origin.
        """
        features = cache.sites[context.site]
        first = run.origin_position + 1
        count = min(HORIZON_HALF_HOURS, run.n_targets, len(cache.end_times) - first)
        count -= count % 2
        if count < HALF_HOURS_PER_DAY:
            msg = "the run's window is shorter than a day"
            raise ValueError(msg)
        window = slice(first, first + count)
        context.proxy.assert_unseen_by(
            origin_position=run.origin_position, through=run.origin_position
        )
        past = np.arange(first - HISTORY_HALF_HOURS, first)
        past_season, past_band = _bands(cache=cache, positions=past)
        past_temperature = context.calibration.apply_temperature(
            temperature_c=context.proxy.temperature_c[past],
            season=past_season,
            lead_band=np.zeros(HISTORY_HALF_HOURS, dtype=np.int64),
            hour_band=past_band,
        )
        if not np.isfinite(past_temperature).all():
            msg = "the analysis proxy lacks the past week's temperature before the origin"
            raise ValueError(msg)
        season, hour_band = _bands(cache=cache, positions=np.arange(first, first + count))
        lead_hours = np.arange(1, count + 1) / HALF_HOURS_PER_HOUR
        sun = SunGeometry.build(
            end_times=cache.end_times[window],
            latitude=features.table.latitude,
            longitude=features.table.longitude,
        )
        return RunWindow(
            window=window,
            cache=slice_cache(
                cache=cache, start=window.start, stop=window.stop, sites=[context.site]
            ),
            sun=sun,
            season=season,
            hour_band=hour_band,
            lead_band=lead_band_of_hours(lead_hours=lead_hours),
            past_temperature_c=past_temperature,
            climatology_w_m2=climatology_at(table=context.climatology, end_times=sun.end_times),
        )

    def driven_caches(
        self, *, context: WeatherContext, run: EnsSiteRun, members: Sequence[int]
    ) -> list[FeatureCache]:
        """Return the window cache driven by each member, calibrated and stitched.

        Temperature before the origin comes from the analysis proxy and after it from the member,
        both calibrated, and the trailing means run over the stitched series. Irradiance is the
        member's, calibrated and shrunk towards the climatology.

        Args:
            context: The substation's weather context.
            run: The run.
            members: The ensemble members, each 0 to 50.

        Returns:
            One cache per member, in the order of `members`.
        """
        count = self.window.stop - self.window.start
        chosen = list(members)
        temperature = context.calibration.apply_temperature(
            temperature_c=run.temperature_c[chosen, :count],
            season=self.season[None, :],
            lead_band=self.lead_band[None, :],
            hour_band=self.hour_band[None, :],
        )
        ghi = context.calibration.apply_irradiance(
            ghi=run.shortwave_w_m2[chosen, :count],
            clear_sky_ghi=self.sun.clear_sky_w_m2[None, :],
            climatology_ghi=self.climatology_w_m2[None, :],
            season=self.season[None, :],
            lead_band=self.lead_band[None, :],
            hour_band=self.hour_band[None, :],
        )
        shape, flux = self.sun.pv_shape_and_ghi(ghi_w_m2=ghi, plant=context.plant)
        caches = []
        for index in range(len(chosen)):
            stitched = np.concatenate([self.past_temperature_c, temperature[index]])
            caches.append(
                replace_site(
                    cache=self.cache,
                    site=context.site,
                    temperature_c=temperature[index],
                    smoothed_temperature_c=smoothed_temperatures(temperature_c=stitched)[
                        HISTORY_HALF_HOURS:
                    ],
                    ghi_w_m2=flux[index],
                    pv_unit_shape=shape[index],
                )
            )
        return caches


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
