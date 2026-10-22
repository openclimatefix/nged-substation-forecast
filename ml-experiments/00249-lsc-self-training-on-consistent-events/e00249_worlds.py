"""Place drawn events on the residual, build injected worlds, and measure event features.

The residual of a frozen convex model is fixed, so an event's effect is `residual + moved MW`
(the fast path of experiment 130), which is exact for a model not refitted on the injected period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from e00249_common import (
    ADJACENCY,
    DAY,
    GRID_LENGTH,
    SERIES_IDS,
    explained_masks,
    stamp_hour_weekday,
    stamp_index,
    stamp_months,
)
from e00249_injector import DrawnEvent, InjectorFit, draw_event, moved_profile

QUIET: int = 7 * DAY
MIN_PRESENT: float = 0.7


@dataclass(frozen=True)
class Placed:
    """A drawn event placed at a stamp, with the MW it adds to each series."""

    event: DrawnEvent
    onset: int
    end: int
    """First stamp after the event."""
    delta: dict[int, np.ndarray]

    @property
    def series(self) -> tuple[int, ...]:
        """The donor and the metered receivers."""
        return (self.event.donor, *self.event.receivers)


class Calendar:
    """Local hour, weekday, and day of year of every stamp, and a start-time lookup."""

    def __init__(self) -> None:
        """Precompute the calendar arrays and the start-time lookup."""
        self.hour, self.weekday = stamp_hour_weekday()
        ordinal = stamp_index().dt.convert_time_zone("Europe/London").dt.ordinal_day()
        self.day_of_year: np.ndarray = ordinal.to_numpy()
        self.months = stamp_months()
        self._lookup: dict[tuple[int, int], np.ndarray] = {}
        slot = np.round(self.hour * 2).astype(int) % DAY
        for weekday in range(7):
            for s in range(DAY):
                self._lookup[(weekday, s)] = np.flatnonzero((self.weekday == weekday) & (slot == s))

    def candidates(self, *, hour: float, weekday: int) -> np.ndarray:
        """Stamps whose local weekday and half-hour slot match the drawn onset."""
        return self._lookup[(weekday, round(hour * 2) % DAY)]


def place(
    *,
    event: DrawnEvent,
    onset: int,
    power: dict[int, np.ndarray],
    sigma: dict[int, float],
    calendar: Calendar,
    rng: np.random.Generator,
) -> Placed | None:
    """Return the event placed at `onset`, or None if it does not fit on the grid."""
    length = max(4, round(event.hours * 2))
    end = onset + length
    if end >= GRID_LENGTH:
        return None
    span = slice(onset, end)
    moved = moved_profile(
        event=event,
        length=length,
        donor_power=power[event.donor][span],
        local_hours=calendar.hour[span],
        day_of_year=calendar.day_of_year[span],
        sigma_donor=sigma[event.donor],
        rng=rng,
    )
    delta = {event.donor: -moved}
    for receiver, weight in zip(event.receivers, event.weights, strict=True):
        delta[receiver] = weight * moved
    return Placed(event=event, onset=onset, end=end, delta=delta)


def quiet_ok(
    *, placed: Placed, residual: dict[int, np.ndarray], explained: dict[int, np.ndarray]
) -> bool:
    """True when every series of the event has readings and no logged or labelled event nearby."""
    lo, hi = max(placed.onset - QUIET, 0), min(placed.end + QUIET, GRID_LENGTH)
    for s in placed.series:
        if explained[s][lo:hi].any():
            return False
        if np.isfinite(residual[s][placed.onset : placed.end]).mean() < MIN_PRESENT:
            return False
    return True


def sample_placed(
    *,
    fit: InjectorFit,
    residual: dict[int, np.ndarray],
    power: dict[int, np.ndarray],
    sigma: dict[int, float],
    calendar: Calendar,
    explained: dict[int, np.ndarray],
    rng: np.random.Generator,
    months: np.ndarray | None = None,
    allow_months: frozenset[int] | None = None,
) -> Placed:
    """Draw and place one event that fits: quiet on every series, in an allowed month."""
    while True:
        event = draw_event(fit=fit, sigma=sigma, rng=rng)
        if event is None:
            continue
        stamps = calendar.candidates(hour=event.onset_hour, weekday=event.onset_weekday)
        if stamps.size == 0:
            continue
        onset = int(rng.choice(stamps))
        if allow_months is not None and calendar.months[onset] not in allow_months:
            continue
        placed = place(
            event=event, onset=onset, power=power, sigma=sigma, calendar=calendar, rng=rng
        )
        if placed is not None and quiet_ok(placed=placed, residual=residual, explained=explained):
            return placed


def inject(*, residual: dict[int, np.ndarray], events: list[Placed]) -> dict[int, np.ndarray]:
    """The residual with every event's MW added."""
    out = {s: v.copy() for s, v in residual.items()}
    for placed in events:
        for s, d in placed.delta.items():
            out[s][placed.onset : placed.end] += d
    return out


def disjoint_world(
    *,
    fit: InjectorFit,
    n_events: int,
    residual: dict[int, np.ndarray],
    power: dict[int, np.ndarray],
    sigma: dict[int, float],
    calendar: Calendar,
    rng: np.random.Generator,
    allow_months: frozenset[int] | None,
) -> list[Placed]:
    """Draw events that share no series within 7 days of each other, so each is scored alone."""
    explained = explained_masks()
    events: list[Placed] = []
    busy = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    tries = 0
    while len(events) < n_events and tries < 50 * n_events:
        tries += 1
        placed = sample_placed(
            fit=fit,
            residual=residual,
            power=power,
            sigma=sigma,
            calendar=calendar,
            explained=explained,
            rng=rng,
            allow_months=allow_months,
        )
        lo, hi = max(placed.onset - QUIET, 0), min(placed.end + QUIET, GRID_LENGTH)
        # An event's neighbours' windows share its signal, so keep them clear too.
        used = set(placed.series) | {n for s in placed.series for n in ADJACENCY[s]}
        if any(busy[s][lo:hi].any() for s in used):
            continue
        for s in used:
            busy[s][lo:hi] = True
        events.append(placed)
    return events


def step_features(
    *,
    residual: dict[int, np.ndarray],
    sigma: dict[int, float],
    series: tuple[int, ...],
    onset: int,
    end: int,
    calendar: Calendar,
) -> dict[str, float] | None:
    """Features of one event measured from the residual alone (the same for real and injected).

    The main series is the event's series with the largest step. Returns None when no series of
    the event has readings before and during the event.
    """
    length = end - onset
    first_day = slice(onset, onset + min(length, DAY))
    pre = slice(max(onset - 2 * DAY, 0), max(onset - 2, 1))

    def step(s: int) -> float:
        during, before = residual[s][first_day], residual[s][pre]
        if np.isfinite(during).sum() < 4 or np.isfinite(before).sum() < 20:
            return float("nan")
        return float(np.nanmean(during) - np.nanmean(before))

    steps = {s: step(s) for s in series if s in residual and np.isfinite(sigma.get(s, np.nan))}
    steps = {s: v for s, v in steps.items() if np.isfinite(v)}
    if not steps:
        return None
    main = max(steps, key=lambda s: abs(steps[s]) / sigma[s])
    sign = np.sign(steps[main])
    opposite_mw, n_opposite = 0.0, 0
    for n in ADJACENCY.get(main, ()):
        if n in residual and np.isfinite(sigma.get(n, np.nan)):
            v = step(n)
            if np.isfinite(v) and -sign * v / sigma[n] > 1.5:
                opposite_mw += v
                n_opposite += 1
    span = slice(onset, end)
    hours, values = calendar.hour[span], residual[main][span]
    day = np.isfinite(values) & (hours >= 10) & (hours <= 16)
    night = np.isfinite(values) & ((hours >= 22) | (hours <= 5))
    day_night = (
        float((values[day].mean() - values[night].mean()) * sign / sigma[main])
        if length >= DAY and day.sum() > 4 and night.sum() > 4
        else 0.0
    )
    onset_hour = calendar.hour[onset]
    return {
        "log_depth_z": float(np.log(abs(steps[main]) / sigma[main] + 1e-3)),
        "log_depth_mw": float(np.log(abs(steps[main]) + 1e-3)),
        "log_hours": float(np.log(length / 2)),
        "onset_sin": float(np.sin(2 * np.pi * onset_hour / 24)),
        "onset_cos": float(np.cos(2 * np.pi * onset_hour / 24)),
        "weekend": float(calendar.weekday[onset] >= 5),
        "day_night": day_night,
        "n_opposite": float(n_opposite),
        "balance": float(np.clip(-opposite_mw / steps[main], -3, 3)),
        "n_series": float(len(steps)),
    }
