"""The injection harness: add known load transfers to power, and score detectors against them.

A copy of experiment 43's injection harness. `inject_power` returns the injected power
and the event table, so a residual model that is refitted on the injected period sees the
transfers. `Injection.fast_residual` is the shortcut `residual + moved MW`, valid only for a fixed
lagless weather model that was not trained on the injected period (the fast path of this
experiment).

A transfer takes a share of the donor's own power (so it follows the donor's daily shape), splits it
across 1 to 3 receivers, and the share drifts slowly between 0.5 and 2 times its start. Durations
and sheet sides come from the real logged operations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00132_common import ADJACENCY, BLOCK, GRID_LENGTH, GRID_START, LABELLED_PATH

QUIET_STAMPS: Final[int] = 7 * BLOCK
SIZE_BANDS: Final[dict[str, tuple[float, float]]] = {
    "under 10%": (0.05, 0.10),
    "10% to 25%": (0.10, 0.25),
    "25% or more": (0.25, 0.40),
}
DURATION_BANDS: Final[dict[str, tuple[float, float]]] = {
    "2 to 6 h": (2.0, 6.0),
    "6 to 24 h": (6.0, 24.0),
    "over 1 d": (24.0, 30 * 24.0),
}
HEADLINE_BAND: Final[tuple[str, str]] = ("25% or more", "over 1 d")
HEADLINE_EVENTS: Final[int] = 160
OTHER_BAND_EVENTS: Final[int] = 55
"""600 events: 160 in the headline band and 55 in each of the other eight bands."""
P_EXTRA_UNMETERED: Final[float] = 0.1


@dataclass(frozen=True)
class Event:
    """One injected transfer. `end` is the first stamp after the event."""

    event_id: int
    donor: int
    receivers: tuple[int, ...]
    weights: tuple[float, ...]
    """Share of the moved load each receiver takes; the rest goes to an unmetered receiver."""
    onset: int
    end: int
    size_band: str
    duration_band: str
    fraction: float
    """Moved load as a share of the receiving load (or half the donor's, if none is metered)."""

    @property
    def series(self) -> tuple[int, ...]:
        """The donor and the metered receivers."""
        return (self.donor, *self.receivers)


@dataclass
class Injection:
    """Injected power, the event table, and the power added to each series."""

    power: dict[int, np.ndarray]
    delta: dict[int, np.ndarray]
    events: list[Event]

    def fast_residual(self, residual: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """`residual + moved MW`: valid for a fixed lagless model not trained on these periods."""
        return {s: residual[s] + self.delta[s] for s in residual}


def operations() -> pl.DataFrame:
    """The real logged operations of 2 hours to 30 days: sheet sides and duration in hours."""
    return (
        pl.read_parquet(LABELLED_PATH)
        .filter(pl.col("kind") == "real_operation")
        .with_columns(
            hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60
        )
        .filter(pl.col("hours").is_between(2, 30 * 24))
        .select("side_a", "side_b", "hours")
    )


def _walk(*, length: int, rng: np.random.Generator) -> np.ndarray:
    """A slow random walk between 0.5 and 2 times one, reflected at both ends."""
    log = np.zeros(length)
    low, high = np.log(0.5), np.log(2.0)
    for i in range(1, length):
        log[i] = log[i - 1] + rng.normal(0, 0.03)
        log[i] = 2 * high - log[i] if log[i] > high else log[i]
        log[i] = 2 * low - log[i] if log[i] < low else log[i]
    return np.exp(np.clip(log, low, high))


def _share(
    *, event: Event, power: dict[int, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """The moved share at each stamp of the event, and the donor's power there (gaps filled)."""
    stamps = np.arange(event.onset, event.end)
    donor = np.nan_to_num(power[event.donor][stamps], nan=np.nanmedian(power[event.donor]))
    receivers = [np.nanmean(power[r][stamps]) for r in event.receivers]
    load = float(np.nansum(receivers)) if receivers else 0.5 * float(np.nanmean(donor))
    start = min(0.8, event.fraction * load / max(float(np.mean(donor)), 1e-9))
    return np.minimum(start * _walk(length=len(stamps), rng=rng), 0.8), donor


def _moved(*, event: Event, power: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """The MW moved at each stamp of the event: a drifting share of the donor's power."""
    share, donor = _share(event=event, power=power, rng=rng)
    return share * donor


def draw_events(
    *,
    quiet: dict[int, np.ndarray],
    power: dict[int, np.ndarray],
    seed: int,
    sheets: pl.DataFrame | None = None,
    headline_only: int | None = None,
    first_id: int = 0,
    onset_ok: Callable[[int], bool] | None = None,
) -> list[Event]:
    """Draw 600 events, stratified by size and duration band, each quiet on every series it uses.

    With `headline_only`, draw that many events in the headline band and none in the other bands.
    `first_id` numbers the events from that id, and `onset_ok` rejects an event whose onset stamp
    it returns False for. The default arguments reproduce experiment 43's draw exactly.
    """
    rng = np.random.default_rng(seed)
    sheets = sheets if sheets is not None else operations()
    wanted = {
        (size, duration): HEADLINE_EVENTS
        if (size, duration) == HEADLINE_BAND
        else OTHER_BAND_EVENTS
        for size in SIZE_BANDS
        for duration in DURATION_BANDS
    }
    if headline_only is not None:
        wanted = {HEADLINE_BAND: headline_only}
    events: list[Event] = []
    for (size, duration), count in wanted.items():
        hours_low, hours_high = DURATION_BANDS[duration]
        pool = sheets.filter(pl.col("hours").is_between(hours_low, hours_high))
        made = 0
        while made < count:
            event = _try_event(
                pool=pool,
                size=size,
                duration=duration,
                quiet=quiet,
                rng=rng,
                event_id=first_id + len(events),
                onset_ok=onset_ok,
            )
            if event is not None:
                events.append(event)
                made += 1
    return events


def _try_event(
    *,
    pool: pl.DataFrame,
    size: str,
    duration: str,
    quiet: dict[int, np.ndarray],
    rng: np.random.Generator,
    event_id: int,
    onset_ok: Callable[[int], bool] | None = None,
) -> Event | None:
    row = pool.row(int(rng.integers(0, pool.height)), named=True)
    sides = [int(s) for s in (row["side_a"], row["side_b"]) if s is not None and s in ADJACENCY]
    if not sides:
        return None
    donor = int(rng.choice(sides))
    receivers = [s for s in sides if s != donor]
    extras = [n for n in ADJACENCY[donor] if n not in receivers]
    rng.shuffle(extras)
    if receivers:
        receivers += extras[: int(rng.integers(0, 3))]
    n_weights = len(receivers) + (1 if receivers and rng.random() < P_EXTRA_UNMETERED else 0)
    weights = rng.dirichlet(np.ones(n_weights)) if n_weights else np.array([])
    length = max(4, int(row["hours"] * 2))
    onset = int(rng.integers(14 * BLOCK, GRID_LENGTH - length - BLOCK))
    if onset_ok is not None and not onset_ok(onset):
        return None
    series = (donor, *receivers)
    span = slice(max(onset - QUIET_STAMPS, 0), onset + length + QUIET_STAMPS)
    if any(quiet[s][span].any() for s in series):
        return None
    low, high = SIZE_BANDS[size]
    return Event(
        event_id=event_id,
        donor=donor,
        receivers=tuple(receivers),
        weights=tuple(float(w) for w in weights[: len(receivers)]),
        onset=onset,
        end=onset + length,
        size_band=size,
        duration_band=duration,
        fraction=float(rng.uniform(low, high)),
    )


def batches(*, events: list[Event]) -> list[list[Event]]:
    """Greedily split events into batches that share no series within 7 days of each other."""
    out: list[list[Event]] = []
    for event in events:
        for batch in out:
            if all(
                set(event.series).isdisjoint(other.series)
                or event.onset - QUIET_STAMPS > other.end
                or other.onset - QUIET_STAMPS > event.end
                for other in batch
            ):
                batch.append(event)
                break
        else:
            out.append([event])
    return out


def inject_power(*, power: dict[int, np.ndarray], events: list[Event], seed: int) -> Injection:
    """Return the power with the events added, and the MW added to each series."""
    rng = np.random.default_rng(seed)
    out = {s: v.copy() for s, v in power.items()}
    delta = {s: np.zeros(GRID_LENGTH) for s in power}
    for event in events:
        moved = _moved(event=event, power=power, rng=rng)
        stamps = np.arange(event.onset, event.end)
        out[event.donor][stamps] -= moved
        delta[event.donor][stamps] -= moved
        for receiver, weight in zip(event.receivers, event.weights, strict=True):
            out[receiver][stamps] += weight * moved
            delta[receiver][stamps] += weight * moved
    return Injection(power=out, delta=delta, events=events)


def events_frame(*, events: list[Event]) -> pl.DataFrame:
    """The event table."""
    return pl.DataFrame(
        [
            {
                "event_id": e.event_id,
                "donor": e.donor,
                "receivers": ",".join(map(str, e.receivers)),
                "onset": e.onset,
                "end": e.end,
                "hours": (e.end - e.onset) / 2,
                "size_band": e.size_band,
                "duration_band": e.duration_band,
                "fraction": e.fraction,
                "metered_receivers": len(e.receivers),
            }
            for e in events
        ]
    )


def stamp_of_month(stamp: int) -> int:
    """Year and month of a grid stamp as an integer such as 202203."""
    moment = GRID_START + timedelta(minutes=30 * stamp)
    return moment.year * 100 + moment.month
