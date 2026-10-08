"""One-megawatt battery schedules driven by a public signal, for the unmetered-battery study.

Every template is a power in megawatts per megawatt of rating at each half-hour of a grid, positive
for export to the network and negative for charging, on the grid of half-hour end times that the
study passes in. A battery of `a` megawatts adds `a * template` to the network's export.

A template depends on a signal (a tariff window, a price) and on two physical parameters: the
usable duration `d`, in hours at full power (usable energy in MWh per MW), and the round-trip
efficiency `eta`. The one-way efficiency is `sqrt(eta)`, so the energy a template discharges equals
the energy it charges times `eta`: every template here is energy-neutral over its cycle.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from typing import Final, Literal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from studies.battery_dispatch import lp_schedule

LOCAL_TIME_ZONE: Final[ZoneInfo] = ZoneInfo("Europe/London")
"""The clock that tariff windows and delivery days follow."""
HOURS_PER_HALF_HOUR: Final[float] = 0.5
HALF_HOURS_PER_DAY: Final[int] = 48
DOMESTIC_SPREAD_SD_LOG: Final[float] = 0.25
"""The standard deviation, in natural log, of the spread of durations that softens the end of a
domestic charge block: homes on one tariff do not all hold the same energy."""
SPREAD_NODES: Final[int] = 5
"""Gauss-Hermite nodes used to average a template over the spread of durations."""

TariffNameType = Literal["intelligent_octopus_go", "octopus_go", "octopus_flux", "red_band"]


@dataclass(frozen=True)
class TariffWindow:
    """A daily charge window and discharge window, as hours of the UK local clock.

    A negative hour is on the evening before: `-0.5` is 23:30 the previous day. The charge window
    and the discharge window of one cycle belong to one local date `D`.

    Attributes:
        charge_start_hour: When charging starts, in hours after local midnight of `D`.
        charge_end_hour: When the charge window closes.
        discharge_start_hour: When discharging starts.
        discharge_end_hour: When the discharge window closes.
        weekdays_only: True if the cycle runs only on Monday to Friday of `D`.
    """

    charge_start_hour: float
    charge_end_hour: float
    discharge_start_hour: float
    discharge_end_hour: float
    weekdays_only: bool = False


TARIFF_WINDOWS: Final[dict[TariffNameType, TariffWindow]] = {
    "intelligent_octopus_go": TariffWindow(-0.5, 5.5, 16.0, 23.5),
    "octopus_go": TariffWindow(0.5, 5.5, 16.0, 23.5),
    "octopus_flux": TariffWindow(2.0, 5.0, 16.0, 19.0),
    "red_band": TariffWindow(0.0, 7.0, 16.0, 19.0, weekdays_only=True),
}
"""Intelligent Octopus Go cheap from 23:30 to 05:30, Octopus Go cheap from 00:30 to 05:30, Octopus
Flux cheap from 02:00 to 05:00 with its high export rate from 16:00 to 19:00 (the window times as
published on the tariff pages), and a commercial and industrial battery that discharges across the
weekday distribution-charge red band (16:00 to 19:00 for NGED's East Midlands licence area) and
recharges overnight. The fleets discharge from 16:00 until empty, up to the window's close."""


@dataclass(frozen=True)
class LpSettings:
    """The fixed physical limits that the price-taker schedules use.

    Attributes:
        soc_min: The lowest state of charge, as a fraction of the energy capacity.
        soc_max: The highest state of charge, as a fraction.
        cycles_per_day_cap: The most energy discharged in a day, in multiples of the usable energy.
    """

    soc_min: float = 0.05
    soc_max: float = 0.95
    cycles_per_day_cap: float = 1.0


STANDARD_LP_SETTINGS: Final[LpSettings] = LpSettings()
SENSITIVITY_LP_SETTINGS: Final[LpSettings] = LpSettings(
    soc_min=0.0, soc_max=1.0, cycles_per_day_cap=2.0
)


def _grid_start(*, half_hour_end_time: pl.Series) -> datetime:
    """Return the start instant of the grid's first half-hour, in UTC."""
    first = half_hour_end_time.dt.replace_time_zone(None)[0]
    return first.replace(tzinfo=UTC) - timedelta(minutes=30)


def _local_instant_hours(*, local_date: date, hour: float, grid_start: datetime) -> float:
    """Return the hours from `grid_start` to a UK wall-clock time on a local date.

    Args:
        local_date: The local date.
        hour: Hours after local midnight (negative for the evening before).
        grid_start: The grid's first instant, UTC.

    Returns:
        The signed offset in hours.
    """
    # Adding a timedelta to an aware datetime moves the wall clock, which is what a tariff means.
    wall = datetime.combine(local_date, time.min, tzinfo=LOCAL_TIME_ZONE) + timedelta(hours=hour)
    instant = wall.astimezone(UTC)
    return (instant - grid_start).total_seconds() / 3600.0


def _local_dates(*, half_hour_end_time: pl.Series, grid_start: datetime) -> list[date]:
    """Return every local date whose cycle can touch the grid, one day either side."""
    grid_end = grid_start + timedelta(minutes=30 * len(half_hour_end_time))
    first = grid_start.astimezone(LOCAL_TIME_ZONE).date() - timedelta(days=1)
    last = grid_end.astimezone(LOCAL_TIME_ZONE).date() + timedelta(days=1)
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def _add_intervals(
    *, power: np.ndarray, starts_h: np.ndarray, lengths_h: np.ndarray, sign: float
) -> None:
    """Add `sign` times the fraction of each half-hour covered by each interval, in place.

    Args:
        power: The template being built, one entry per half-hour of the grid.
        starts_h: Each interval's start, in hours from the grid start.
        lengths_h: Each interval's length in hours.
        sign: +1 for discharge, -1 for charge.
    """
    ends_h = starts_h + lengths_h
    first_bin = np.floor(starts_h / HOURS_PER_HALF_HOUR).astype(int)
    last_bin = np.ceil(ends_h / HOURS_PER_HALF_HOUR).astype(int)
    for offset in range(int((last_bin - first_bin).max(initial=0)) + 1):
        bins = first_bin + offset
        overlap = np.clip(
            np.minimum(ends_h, (bins + 1) * HOURS_PER_HALF_HOUR)
            - np.maximum(starts_h, bins * HOURS_PER_HALF_HOUR),
            0.0,
            None,
        )
        inside = (bins >= 0) & (bins < len(power))
        np.add.at(power, bins[inside], sign * overlap[inside] / HOURS_PER_HALF_HOUR)


def effective_duration_hours(
    *, window: TariffWindow, duration_hours: float, round_trip_efficiency: float
) -> float:
    """Return the usable duration after the windows' lengths cap it.

    The battery cannot store more than the charge window lets it charge, nor more than the discharge
    window lets it discharge, so a long duration is cut to what both windows allow.

    Args:
        window: The tariff window.
        duration_hours: The usable duration at full power.
        round_trip_efficiency: The round-trip efficiency.

    Returns:
        The duration actually cycled each day, in hours at full power.
    """
    one_way = np.sqrt(round_trip_efficiency)
    charge_hours = window.charge_end_hour - window.charge_start_hour
    discharge_hours = window.discharge_end_hour - window.discharge_start_hour
    return float(min(duration_hours, one_way * charge_hours, discharge_hours / one_way))


def window_template(
    *,
    half_hour_end_time: pl.Series,
    window: TariffWindow,
    duration_hours: float,
    round_trip_efficiency: float,
) -> np.ndarray:
    """Return the schedule of a battery that charges from a window's start, then discharges.

    Each local day the battery charges at full power from the window's start until it holds its
    usable energy, then discharges at full power from the discharge window's start until empty.
    Both blocks start exactly at their window's edge, and a partly covered half-hour carries the
    covered share of full power.

    Args:
        half_hour_end_time: The UTC end time of each half-hour of the grid.
        window: The tariff window.
        duration_hours: The usable duration at full power.
        round_trip_efficiency: The round-trip efficiency.

    Returns:
        The per-unit power at each half-hour.
    """
    grid_start = _grid_start(half_hour_end_time=half_hour_end_time)
    one_way = float(np.sqrt(round_trip_efficiency))
    stored = effective_duration_hours(
        window=window, duration_hours=duration_hours, round_trip_efficiency=round_trip_efficiency
    )
    dates = [
        d
        for d in _local_dates(half_hour_end_time=half_hour_end_time, grid_start=grid_start)
        if not window.weekdays_only or d.isoweekday() <= 5
    ]
    charge_starts = np.array(
        [
            _local_instant_hours(local_date=d, hour=window.charge_start_hour, grid_start=grid_start)
            for d in dates
        ]
    )
    discharge_starts = np.array(
        [
            _local_instant_hours(
                local_date=d, hour=window.discharge_start_hour, grid_start=grid_start
            )
            for d in dates
        ]
    )
    power = np.zeros(len(half_hour_end_time))
    _add_intervals(
        power=power,
        starts_h=charge_starts,
        lengths_h=np.full(len(dates), stored / one_way),
        sign=-1.0,
    )
    _add_intervals(
        power=power,
        starts_h=discharge_starts,
        lengths_h=np.full(len(dates), stored * one_way),
        sign=1.0,
    )
    return power


def charge_only_template(
    *, half_hour_end_time: pl.Series, window: TariffWindow, charge_hours: float
) -> np.ndarray:
    """Return a charge block at a window's start with no discharge, the electric-vehicle nuisance.

    An electric vehicle on the same tariff steps at the same window edge but never discharges, so a
    charge-only column lets the estimator credit that step to the vehicle rather than a battery.

    Args:
        half_hour_end_time: The UTC end time of each half-hour of the grid.
        window: The tariff window whose start the block begins at.
        charge_hours: How long the block lasts, in hours at full power.

    Returns:
        The per-unit power: zero or negative at every half-hour.
    """
    grid_start = _grid_start(half_hour_end_time=half_hour_end_time)
    dates = _local_dates(half_hour_end_time=half_hour_end_time, grid_start=grid_start)
    starts = np.array(
        [
            _local_instant_hours(local_date=d, hour=window.charge_start_hour, grid_start=grid_start)
            for d in dates
        ]
    )
    power = np.zeros(len(half_hour_end_time))
    _add_intervals(
        power=power, starts_h=starts, lengths_h=np.full(len(dates), charge_hours), sign=-1.0
    )
    return power


@cache
def _log_normal_spread(*, sd_log: float, nodes: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return multipliers `exp(sd_log * z)` and weights from Gauss-Hermite quadrature."""
    z, weights = np.polynomial.hermite_e.hermegauss(nodes)
    return tuple(np.exp(sd_log * z)), tuple(weights / weights.sum())


def spread_mean(
    *,
    template_for_duration: Callable[[float], np.ndarray],
    duration_hours: float,
    sd_log: float = DOMESTIC_SPREAD_SD_LOG,
    nodes: int = SPREAD_NODES,
) -> np.ndarray:
    """Average a template over a log-normal spread of durations around `duration_hours`.

    Args:
        template_for_duration: Builds the template for one duration.
        duration_hours: The median duration.
        sd_log: The standard deviation of the log of the duration. Zero returns the template at
            `duration_hours` itself.
        nodes: The Gauss-Hermite nodes averaged over.

    Returns:
        The weighted mean of the templates.
    """
    if sd_log == 0.0:
        return template_for_duration(duration_hours)
    multipliers, weights = _log_normal_spread(sd_log=sd_log, nodes=nodes)
    mean = 0.0 * template_for_duration(duration_hours)
    for multiplier, weight in zip(multipliers, weights, strict=True):
        mean = mean + weight * template_for_duration(duration_hours * multiplier)
    return mean


def merchant_template(
    *,
    day_ahead_prices: np.ndarray,
    duration_hours: float,
    round_trip_efficiency: float,
    settings: LpSettings = STANDARD_LP_SETTINGS,
) -> np.ndarray:
    """Return the schedule of a price taker on the day-ahead price.

    The battery's nameplate energy is scaled up so that the energy between the state-of-charge
    limits equals `duration_hours` at full power, which makes the duration mean the same thing for
    every template.

    Args:
        day_ahead_prices: The price at each half-hour of the grid, in whole UTC days. A day with a
            missing price gets a zero schedule.
        duration_hours: The usable duration at full power.
        round_trip_efficiency: The round-trip efficiency.
        settings: The state-of-charge limits and cycle cap.

    Returns:
        The per-unit power at each half-hour.
    """
    return lp_schedule(
        prices=day_ahead_prices,
        energy_hours=duration_hours / (settings.soc_max - settings.soc_min),
        eta_one_way=float(np.sqrt(round_trip_efficiency)),
        soc_min=settings.soc_min,
        soc_max=settings.soc_max,
        cycles_per_day_cap=settings.cycles_per_day_cap,
    )


def agile_template(
    *,
    half_hour_end_time: pl.Series,
    agile_prices: pl.DataFrame,
    duration_hours: float,
    round_trip_efficiency: float,
    settings: LpSettings = STANDARD_LP_SETTINGS,
) -> np.ndarray:
    """Return the schedule of a price taker on each 23:00-to-23:00 Agile delivery day.

    Each delivery day starts at 23:00 UK local time on the evening before. A day with fewer than
    48 half-hours (the spring clock change) or with a missing price gets a zero schedule, and the
    two surplus half-hours of the 50-half-hour autumn day get zero.

    Args:
        half_hour_end_time: The UTC end time of each half-hour of the grid.
        agile_prices: Columns `time` (the half-hour start, UTC) and `price_inc_vat_p_per_kwh`.
        duration_hours: The usable duration at full power.
        round_trip_efficiency: The round-trip efficiency.
        settings: The state-of-charge limits and cycle cap.

    Returns:
        The per-unit power at each half-hour of the grid.
    """
    grid_start = _grid_start(half_hour_end_time=half_hour_end_time)
    n_grid = len(half_hour_end_time)
    price_by_start = dict(
        zip(
            agile_prices["time"].dt.epoch("s").to_list(),
            agile_prices["price_inc_vat_p_per_kwh"].to_list(),
            strict=True,
        )
    )
    dates = _local_dates(half_hour_end_time=half_hour_end_time, grid_start=grid_start)
    # Each delivery day starts at 23:00 UK local time on the evening before its date.
    day_starts = [
        (datetime.combine(d, time.min, tzinfo=LOCAL_TIME_ZONE) - timedelta(hours=1)).astimezone(UTC)
        for d in dates
    ]
    day_prices = np.full((len(dates), HALF_HOURS_PER_DAY), np.nan)
    slot_grid_index = np.full((len(dates), HALF_HOURS_PER_DAY), -1)
    for row, day_start in enumerate(day_starts):
        next_start = day_starts[row + 1] if row + 1 < len(day_starts) else None
        for slot in range(HALF_HOURS_PER_DAY):
            slot_start = day_start + timedelta(minutes=30 * slot)
            if next_start is not None and slot_start >= next_start:
                continue
            day_prices[row, slot] = price_by_start.get(int(slot_start.timestamp()), np.nan)
            slot_grid_index[row, slot] = round((slot_start - grid_start).total_seconds() / 1800.0)
    schedule = merchant_template(
        day_ahead_prices=day_prices.ravel(),
        duration_hours=duration_hours,
        round_trip_efficiency=round_trip_efficiency,
        settings=settings,
    ).reshape(day_prices.shape)
    power = np.zeros(n_grid)
    inside = (slot_grid_index >= 0) & (slot_grid_index < n_grid)
    power[slot_grid_index[inside]] = schedule[inside]
    return power


DomesticTariffNameType = Literal["intelligent_octopus_go", "octopus_go", "octopus_flux", "agile"]
DOMESTIC_TARIFFS: Final[tuple[DomesticTariffNameType, ...]] = (
    "intelligent_octopus_go",
    "octopus_go",
    "octopus_flux",
    "agile",
)
"""The four tariffs a domestic battery is assumed to follow, in template-column order."""


def domestic_template(
    *,
    tariff: DomesticTariffNameType,
    half_hour_end_time: pl.Series,
    agile_prices: pl.DataFrame,
    duration_hours: float,
    round_trip_efficiency: float,
    spread_sd_log: float = DOMESTIC_SPREAD_SD_LOG,
    settings: LpSettings = STANDARD_LP_SETTINGS,
) -> np.ndarray:
    """Return a domestic battery's schedule on one tariff, averaged over a spread of durations.

    Args:
        tariff: Which tariff the battery follows.
        half_hour_end_time: The UTC end time of each half-hour of the grid.
        agile_prices: The Agile prices, read only when `tariff` is `agile`.
        duration_hours: The median usable duration at full power.
        round_trip_efficiency: The round-trip efficiency.
        spread_sd_log: The standard deviation of the log of the duration. Zero uses one duration.
        settings: The state-of-charge limits and cycle cap of the Agile price taker.

    Returns:
        The per-unit power at each half-hour.
    """

    def for_duration(duration: float) -> np.ndarray:
        if tariff == "agile":
            return agile_template(
                half_hour_end_time=half_hour_end_time,
                agile_prices=agile_prices,
                duration_hours=duration,
                round_trip_efficiency=round_trip_efficiency,
                settings=settings,
            )
        return window_template(
            half_hour_end_time=half_hour_end_time,
            window=TARIFF_WINDOWS[tariff],
            duration_hours=duration,
            round_trip_efficiency=round_trip_efficiency,
        )

    return spread_mean(
        template_for_duration=for_duration, duration_hours=duration_hours, sd_log=spread_sd_log
    )


@dataclass(frozen=True)
class FleetSpec:
    """How a simulated fleet of domestic batteries is drawn, home by home.

    Attributes:
        tariff_shares: The share of homes on each tariff; the values sum to 1.
        home_power_kw: The lowest and highest battery power of one home, in kilowatts, drawn
            uniformly.
        duration_median_hours: The median usable duration at full power.
        duration_sd_log: The standard deviation of the log of the duration across homes.
        round_trip_efficiency_mean: The mean round-trip efficiency across homes.
        round_trip_efficiency_sd: The standard deviation of the round-trip efficiency across homes.
        duration_resolution_hours: Durations are rounded to a multiple of this, so homes share
            templates and the Agile price taker is solved a few dozen times rather than per home.
        efficiency_resolution: Round-trip efficiencies are rounded to a multiple of this.
    """

    tariff_shares: dict[DomesticTariffNameType, float]
    home_power_kw: tuple[float, float] = (3.0, 5.0)
    duration_median_hours: float = 2.6
    duration_sd_log: float = 0.3
    round_trip_efficiency_mean: float = 0.82
    round_trip_efficiency_sd: float = 0.03
    duration_resolution_hours: float = 0.25
    efficiency_resolution: float = 0.02


@dataclass(frozen=True)
class Fleet:
    """A simulated fleet.

    Attributes:
        homes: One row per home: `tariff`, `power_mw`, `duration_hours`, `round_trip_efficiency`.
        output_mw: The fleet's power at each half-hour, positive for export: the sum of the homes.
    """

    homes: pl.DataFrame
    output_mw: np.ndarray


MIN_HOME_DURATION_HOURS: Final[float] = 0.5
MAX_HOME_DURATION_HOURS: Final[float] = 6.0
MIN_HOME_EFFICIENCY: Final[float] = 0.7
MAX_HOME_EFFICIENCY: Final[float] = 0.95


def draw_homes(*, n_homes: int, spec: FleetSpec, rng: np.random.Generator) -> pl.DataFrame:
    """Draw the homes of a fleet.

    Args:
        n_homes: How many homes.
        spec: How the homes are drawn.
        rng: The random generator.

    Returns:
        One row per home: `tariff`, `power_mw`, `duration_hours`, `round_trip_efficiency`.
    """
    tariffs = rng.choice(
        list(spec.tariff_shares), size=n_homes, p=list(spec.tariff_shares.values())
    )
    low, high = spec.home_power_kw
    durations = spec.duration_median_hours * np.exp(
        spec.duration_sd_log * rng.standard_normal(n_homes)
    )
    efficiencies = spec.round_trip_efficiency_mean + spec.round_trip_efficiency_sd * (
        rng.standard_normal(n_homes)
    )
    durations = np.clip(
        np.round(durations / spec.duration_resolution_hours) * spec.duration_resolution_hours,
        MIN_HOME_DURATION_HOURS,
        MAX_HOME_DURATION_HOURS,
    )
    efficiencies = np.clip(
        np.round(efficiencies / spec.efficiency_resolution) * spec.efficiency_resolution,
        MIN_HOME_EFFICIENCY,
        MAX_HOME_EFFICIENCY,
    )
    return pl.DataFrame(
        {
            "tariff": tariffs,
            "power_mw": rng.uniform(low, high, n_homes) / 1000.0,
            "duration_hours": durations,
            "round_trip_efficiency": efficiencies,
        }
    )


def simulate_fleet(
    *,
    half_hour_end_time: pl.Series,
    agile_prices: pl.DataFrame,
    n_homes: int,
    spec: FleetSpec,
    rng: np.random.Generator,
    settings: LpSettings = STANDARD_LP_SETTINGS,
) -> Fleet:
    """Simulate a fleet home by home: each home runs its own tariff, duration, and efficiency.

    A home's schedule has no spread of durations, because the fleet's spread comes from the homes.
    Homes with the same tariff, duration, and efficiency share one schedule.

    Args:
        half_hour_end_time: The UTC end time of each half-hour of the grid.
        agile_prices: The Agile prices.
        n_homes: How many homes.
        spec: How the homes are drawn.
        rng: The random generator.
        settings: The state-of-charge limits and cycle cap of the Agile price taker.

    Returns:
        The homes and their summed output.
    """
    homes = draw_homes(n_homes=n_homes, spec=spec, rng=rng)
    output = np.zeros(len(half_hour_end_time))
    groups = homes.group_by("tariff", "duration_hours", "round_trip_efficiency").agg(
        total_power_mw=pl.col("power_mw").sum()
    )
    for row in groups.iter_rows(named=True):
        output += row["total_power_mw"] * domestic_template(
            tariff=row["tariff"],
            half_hour_end_time=half_hour_end_time,
            agile_prices=agile_prices,
            duration_hours=row["duration_hours"],
            round_trip_efficiency=row["round_trip_efficiency"],
            spread_sd_log=0.0,
            settings=settings,
        )
    return Fleet(homes=homes, output_mw=output)
