"""Read NOAA GFS from Dynamical.org's native store: its radiation windows and its served runs.

Two conventions of the native store need pure, tested helpers.

**Radiation is a mean since the last reset, and the lead labels the end of the window.** GFS resets
its radiation average every 6 hours of lead (at leads that are multiples of 6, which fall at
00, 06, 12, and 18 UTC because every run starts at one of those hours). The value at lead `L` is the
mean over the window `[L - w, L]`, where `w = ((L - 1) % 6) + 1` is 1 to 6 hours. The store steps
every hour to lead 120 and every 3 hours from lead 123 to lead 384. `step_means` inverts those
windows to the mean over each step (the hour, or the 3 hours, before its label), which is the
quantity the hourly-mean convention of the past-weather studies wants.

**Which run serves a target hour is set by the lead day.** `served_init_time` and
`served_lead_hours` apply the rule the other arms of the matched-lead comparison follow: day `N`
reads the 00 UTC run issued `N` days before the target hour's own day, and day 0 reads the
freshest run.
"""

from typing import Final, Literal

import numpy as np
import polars as pl

DomainType = Literal["solar", "wind"]

LAST_HOURLY_LEAD_HOURS: Final[int] = 120
"""The last lead GFS publishes every hour; it publishes every 3 hours beyond it."""

LAST_LEAD_HOURS: Final[int] = 384
"""The last lead of a GFS run."""

COARSE_STEP_HOURS: Final[int] = 3
"""GFS's step beyond `LAST_HOURLY_LEAD_HOURS`."""

RESET_HOURS: Final[int] = 6
"""How often GFS resets its radiation average, in hours of lead."""

FRESHEST_RUN_CYCLE_HOURS: Final[int] = 6
"""GFS starts a run every 6 hours (00, 06, 12, and 18 UTC)."""

HOURLY_SERVED_LAST_DAY: Final[int] = 4
"""The last lead day whose target hours all fall on hourly leads: day 4's solar band ends at lead
120."""


def gfs_leads(*, last_lead: int = LAST_LEAD_HOURS) -> np.ndarray:
    """Return the leads of a GFS run that carry radiation, in hours.

    Args:
        last_lead: The last lead wanted, at most `LAST_LEAD_HOURS`.

    Returns:
        Every hour from 1 to `LAST_HOURLY_LEAD_HOURS`, then every 3 hours from
        `LAST_HOURLY_LEAD_HOURS + 3`, up to `last_lead`. Lead 0 is absent because a mean since the
        reset has no period to cover there.

    Raises:
        ValueError: If `last_lead` is not a lead of the run.
    """
    if not 1 <= last_lead <= LAST_LEAD_HOURS:
        msg = f"{last_lead} is not a GFS lead that carries radiation"
        raise ValueError(msg)
    hourly = np.arange(1, min(last_lead, LAST_HOURLY_LEAD_HOURS) + 1)
    coarse = np.arange(LAST_HOURLY_LEAD_HOURS + COARSE_STEP_HOURS, last_lead + 1, COARSE_STEP_HOURS)
    leads = np.concatenate([hourly, coarse])
    if leads[-1] != last_lead:
        msg = f"{last_lead} is not a GFS lead that carries radiation"
        raise ValueError(msg)
    return leads


def window_hours(*, leads: np.ndarray) -> np.ndarray:
    """Return how many hours each lead's radiation window spans.

    Args:
        leads: Leads in whole hours.

    Returns:
        `((lead - 1) % 6) + 1`: 1 at the first lead after a reset and 6 at the reset itself.
    """
    return (np.asarray(leads) - 1) % RESET_HOURS + 1


def step_means(*, values: np.ndarray, leads: np.ndarray, negative_floor: float = 0.0) -> np.ndarray:
    """Convert since-reset window means to the mean over each step.

    The value at lead `L` is the mean over `[L - w, L]`. The step ending at `L` starts at the
    previous lead `P`, so its mean is the window's total less the total already accumulated at `P`,
    over the step's width: `(w * C(L) - (P - (L - w)) * C(P)) / (L - P)`, where the second term is
    zero when `P` lies at or before the window's start (a step that begins at a reset). A missing
    value affects only its own step and the next step of the same window, because each window is
    inverted on its own.

    Args:
        values: Shape (n_series, n_leads), the window mean at each of `leads`, NaN where missing.
        leads: Shape (n_leads,), which must equal `gfs_leads(last_lead=leads[-1])`, so the step
            before the first lead starts at lead 0.
        negative_floor: Radiation cannot be negative, so a step mean that noise in two window means
            would push below this floor is clipped to it.

    Returns:
        Shape (n_series, n_leads), the mean over the step ending at each lead: 1 hour wide to lead
        120 and 3 hours wide beyond.

    Raises:
        ValueError: If `leads` is not a GFS run's radiation leads from lead 1.
    """
    leads = np.asarray(leads)
    if leads.size == 0 or not np.array_equal(leads, gfs_leads(last_lead=int(leads[-1]))):
        msg = f"leads must be a GFS run's radiation leads from lead 1, got {leads!r}"
        raise ValueError(msg)
    previous = np.concatenate([[0], leads[:-1]])
    window = window_hours(leads=leads)
    already_accumulated = np.maximum(previous - (leads - window), 0)
    previous_values = np.concatenate([np.zeros((values.shape[0], 1)), values[:, :-1]], axis=1)
    carried = np.where(already_accumulated > 0, already_accumulated * previous_values, 0.0)
    means = (window * values - carried) / (leads - previous)
    return np.clip(means, negative_floor, None)


def three_hour_means(*, hourly: np.ndarray, leads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put a run's step means on the 3-hourly grid its later leads already use.

    Args:
        hourly: `step_means`'s result, shape (n_series, n_leads).
        leads: The leads `hourly` is on, as `gfs_leads` returns them.

    Returns:
        The multiples of 3 hours among `leads`, and shape (n_series, n_multiples), the mean over the
        3 hours before each: the mean of three hourly steps up to lead 120, and the value already
        given beyond it. A missing hourly step makes its 3-hour mean NaN.
    """
    leads = np.asarray(leads)
    index_of = {int(lead): position for position, lead in enumerate(leads)}
    three_leads = leads[leads % COARSE_STEP_HOURS == 0]
    columns = []
    for lead in three_leads:
        if lead <= LAST_HOURLY_LEAD_HOURS:
            positions = [index_of[int(lead) - back] for back in range(COARSE_STEP_HOURS)]
            columns.append(hourly[:, positions].mean(axis=1))
        else:
            columns.append(hourly[:, index_of[int(lead)]])
    return three_leads, np.stack(columns, axis=1)


def served_init_time(*, time: pl.Expr, day: int, domain: DomainType) -> pl.Expr:
    """Return the start of the GFS run that serves a target hour at a lead day.

    A solar hour is labelled by its end, so the run is chosen from the instant an hour earlier; a
    wind hour is labelled by the instant it describes.

    Args:
        time: The target hour's label, timezone-aware.
        day: The lead day. For 1 or more, the 00 UTC run issued `day` days before the label's own
            day. For 0, the freshest run at or before the instant the value describes (a run every
            6 hours), which leaves a lead of 1 to 6 hours for solar and 0 to 5 hours for wind.
        domain: `solar` or `wind`.

    Returns:
        The run's start, timezone-aware.
    """
    instant = time - pl.duration(hours=1) if domain == "solar" else time
    if day == 0:
        return instant.dt.truncate(f"{FRESHEST_RUN_CYCLE_HOURS}h")
    return instant.dt.truncate("1d") - pl.duration(days=day)


def served_lead_hours(*, time: pl.Expr, day: int, domain: DomainType) -> pl.Expr:
    """Return the lead in whole hours at which a target hour reads its serving run.

    Args:
        time: The target hour's label, timezone-aware.
        day: The lead day, as in `served_init_time`.
        domain: `solar` or `wind`.

    Returns:
        `24 * day + 1` to `24 * day + 24` for solar and `24 * day` to `24 * day + 23` for wind at
        days 1 or more; 1 to 6 for solar and 0 to 5 for wind at day 0.
    """
    init = served_init_time(time=time, day=day, domain=domain)
    return (time - init).dt.total_hours().cast(pl.Int32)
