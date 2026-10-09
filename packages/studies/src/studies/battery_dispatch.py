"""Daily battery schedules built from day-ahead prices alone, in per-unit power.

Both schedules return megawatts per megawatt of power rating at each half-hour: positive is
export to the grid, negative is charging. A day is 48 half-hours, so a price array must hold a
whole number of days and start at the first half-hour of a day. A day with any missing price
(NaN) gets a zero schedule.
"""

from typing import Final

import numpy as np
from scipy.optimize import linprog

HALF_HOURS_PER_DAY: Final[int] = 48
HOURS_PER_HALF_HOUR: Final[float] = 0.5
DEFAULT_ROUND_TRIP_EFFICIENCY: Final[float] = 0.85
"""The rank rule idles a day whose discharge price times this does not exceed its charge price."""


def _days(*, prices: np.ndarray) -> np.ndarray:
    """Return the prices as one row per day.

    Args:
        prices: The price at every half-hour, in a whole number of days.

    Returns:
        An array of shape (days, 48).

    Raises:
        ValueError: If the prices do not fill whole days.
    """
    if len(prices) % HALF_HOURS_PER_DAY != 0:
        raise ValueError(f"{len(prices)} prices do not fill whole days of {HALF_HOURS_PER_DAY}")
    return np.asarray(prices, dtype=float).reshape(-1, HALF_HOURS_PER_DAY)


def rank_rule_schedule(
    *,
    prices: np.ndarray,
    duration_half_hours: int,
    round_trip_efficiency: float = DEFAULT_ROUND_TRIP_EFFICIENCY,
) -> np.ndarray:
    """Charge in each day's cheapest half-hours and discharge in its dearest.

    Each day charges at full power (-1) in the `duration_half_hours` cheapest half-hours and
    discharges at full power (+1) in the `duration_half_hours` dearest. Among equal prices, the
    charge block takes the earlier half-hours and the discharge block takes the later ones. The day
    is idle when the charge block does not come before the discharge block (the mean time of the
    charge half-hours is not earlier than the mean time of the discharge half-hours), or when the
    mean discharge price times the round-trip efficiency does not exceed
    the mean charge price.

    Args:
        prices: The price at every half-hour, in a whole number of days.
        duration_half_hours: The battery's energy in half-hours at full power; also the number of
            half-hours charged and discharged per day.
        round_trip_efficiency: The efficiency used for the idle test.

    Returns:
        The per-unit power at every half-hour.

    Raises:
        ValueError: If the duration is not between 1 and 24 half-hours.
    """
    if not 1 <= duration_half_hours <= HALF_HOURS_PER_DAY // 2:
        raise ValueError(f"duration_half_hours must be 1 to 24, got {duration_half_hours}")
    days = _days(prices=prices)
    schedule = np.zeros_like(days)
    for day, row in enumerate(days):
        if np.isnan(row).any():
            continue
        order = np.argsort(row, kind="stable")
        charge = np.sort(order[:duration_half_hours])
        discharge = np.sort(order[::-1][:duration_half_hours])
        if np.intersect1d(charge, discharge).size > 0 or charge.mean() >= discharge.mean():
            continue
        if row[discharge].mean() * round_trip_efficiency <= row[charge].mean():
            continue
        schedule[day, charge] = -1.0
        schedule[day, discharge] = 1.0
    return schedule.ravel()


def lp_schedule(
    *,
    prices: np.ndarray,
    energy_hours: float,
    eta_one_way: float = 0.92,
    soc_min: float = 0.05,
    soc_max: float = 0.95,
    cycles_per_day_cap: float = 1.0,
    initial_soc: float | None = None,
) -> np.ndarray:
    """Maximise each day's revenue as a price taker, carrying the state of charge across days.

    Each day solves a linear programme over its 48 half-hours: charge and discharge powers between
    0 and 1 per unit, a state of charge between `soc_min` and `soc_max`, a one-way efficiency on
    both charging and discharging, and a cap on the energy discharged. The day must end at the
    state of charge it started at, so the state of charge handed to the next day is the state of
    charge this day started with. The first day starts at `initial_soc`.

    Args:
        prices: The price at every half-hour, in a whole number of days.
        energy_hours: The battery's energy capacity in hours at full power (MWh per MW).
        eta_one_way: The efficiency of charging and, separately, of discharging.
        soc_min: The lowest state of charge, as a fraction of `energy_hours`.
        soc_max: The highest state of charge, as a fraction of `energy_hours`.
        cycles_per_day_cap: The most energy discharged in a day, in multiples of the usable
            energy `(soc_max - soc_min) * energy_hours`.
        initial_soc: The first day's starting state of charge, as a fraction; `soc_min` when None.

    Returns:
        The per-unit power (discharge minus charge) at every half-hour.

    Raises:
        RuntimeError: If a day's linear programme has no solution.
    """
    days = _days(prices=prices)
    size = HALF_HOURS_PER_DAY
    soc = soc_min if initial_soc is None else initial_soc
    # The state of charge after each half-hour, minus the day's start, is `cumulative @ x`.
    lower = np.tril(np.ones((size, size))) * HOURS_PER_HALF_HOUR / energy_hours
    cumulative = np.hstack([lower * eta_one_way, -lower / eta_one_way])
    discharge_energy = np.concatenate([np.zeros(size), np.full(size, HOURS_PER_HALF_HOUR)])
    usable = (soc_max - soc_min) * energy_hours
    schedule = np.zeros_like(days)
    for day, row in enumerate(days):
        if np.isnan(row).any():
            continue
        cost = np.concatenate([row, -row]) * HOURS_PER_HALF_HOUR
        result = linprog(
            c=cost,
            A_ub=np.vstack([cumulative, -cumulative, discharge_energy]),
            b_ub=np.concatenate(
                [
                    np.full(size, soc_max - soc),
                    np.full(size, soc - soc_min),
                    [cycles_per_day_cap * usable],
                ]
            ),
            A_eq=cumulative[-1:],
            b_eq=[0.0],
            bounds=(0.0, 1.0),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Day {day}: {result.message}")
        schedule[day] = result.x[size:] - result.x[:size]
    return schedule.ravel()


def state_of_charge(
    *, schedule: np.ndarray, energy_hours: float, eta_one_way: float, initial_soc: float
) -> np.ndarray:
    """Return the state of charge after each half-hour of a per-unit schedule.

    Args:
        schedule: The per-unit power, positive for discharge.
        energy_hours: The battery's energy capacity in hours at full power.
        eta_one_way: The efficiency of charging and, separately, of discharging.
        initial_soc: The state of charge before the first half-hour, as a fraction.

    Returns:
        The state of charge as a fraction of `energy_hours`.
    """
    stored = np.where(schedule < 0, -schedule * eta_one_way, -schedule / eta_one_way)
    return initial_soc + np.cumsum(stored) * HOURS_PER_HALF_HOUR / energy_hours
