"""The long set: injected transfers that last 15.6 to 60 days, drawn like experiment 58's.

Experiment 58's injected transfers last 30 days at most, and only 17 of its 1,160 headline-band
transfers last over 14 days, so they cannot tell a cap of 14 days from a cap of 30 or 60. The long
set draws its durations from the labelled real operations that last 15.6 to 60 days, in the headline
size band, with the quietness rules of `draw_events`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00148_common import ADJACENCY, BLOCK, GRID_LENGTH, LABELLED_PATH
from e00148_inject import HEADLINE_BAND, P_EXTRA_UNMETERED, SIZE_BANDS, Event, stamp_of_month

LONG_EVENTS: Final[int] = 400
FIRST_ID: Final[int] = 5000
SEED_LONG: Final[int] = 7700
"""Seed of the draw; batch `i` is injected with `SEED_LONG + 1000 + i`."""
LONG_BAND: Final[str] = "15.6 to 60 d"
MIN_DAYS: Final[float] = 15.6
MAX_DAYS: Final[int] = 60
MAX_ATTEMPTS: Final[int] = 400_000
EDGE_STAMPS: Final[int] = 7 * BLOCK
"""Quietness and data are checked 7 days either side of the onset and of the end."""
MAX_PER_MONTH: Final[int] = 16
"""At most this many transfers start in one calendar month, so few months cannot decide one."""
MIN_PRESENT: Final[float] = 0.9
"""Share of the power that must be present in each edge window, on every series the event uses."""


def long_operations() -> pl.DataFrame:
    """The labelled real operations lasting 15.6 to 60 days: sheet sides and duration in hours."""
    return (
        pl.read_parquet(LABELLED_PATH)
        .filter(pl.col("kind") == "real_operation")
        .with_columns(
            hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60
        )
        .filter(pl.col("hours").is_between(MIN_DAYS * 24, MAX_DAYS * 24))
        .select("side_a", "side_b", "hours")
    )


def _edge_ok(*, series: tuple[int, ...], onset: int, end: int, quiet: dict, power: dict) -> bool:
    """Whether both edge windows are quiet and have data on every series."""
    for centre in (onset, end):
        span = slice(max(centre - EDGE_STAMPS, 0), min(centre + EDGE_STAMPS, GRID_LENGTH))
        for s in series:
            if quiet[s][span].any() or np.isfinite(power[s][span]).mean() < MIN_PRESENT:
                return False
    return True


def _try_long(
    *, pool: pl.DataFrame, quiet: dict, power: dict, rng: np.random.Generator, event_id: int
) -> Event | None:
    """Draw one transfer like `inject._try_event`, with the quietness rule on the two edges only."""
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
    if not _edge_ok(
        series=(donor, *receivers), onset=onset, end=onset + length, quiet=quiet, power=power
    ):
        return None
    low, high = SIZE_BANDS[HEADLINE_BAND[0]]
    return Event(
        event_id=event_id,
        donor=donor,
        receivers=tuple(receivers),
        weights=tuple(float(w) for w in weights[: len(receivers)]),
        onset=onset,
        end=onset + length,
        size_band=HEADLINE_BAND[0],
        duration_band=LONG_BAND,
        fraction=float(rng.uniform(low, high)),
    )


def draw_long_events(
    *,
    quiet: dict[int, np.ndarray],
    power: dict[int, np.ndarray],
    seed: int = SEED_LONG,
    first_id: int = FIRST_ID,
) -> list[Event]:
    """Draw `LONG_EVENTS` headline-size transfers lasting 15.6 to 60 days, numbered from `first_id`.

    Unlike `draw_events`, the quiet and data rules apply to the two windows around the onset and
    the end, not to the whole transfer, because 7 quiet days either side of a transfer of 15.6 days
    or more leave only a few clustered months.
    """
    pool = long_operations()
    if pool.height != 25 or pool.filter(pl.col("hours") > 30 * 24).height != 5:
        raise RuntimeError("the pool of long labelled operations differs from the plan's 25 and 5")
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    attempts = 0
    while len(events) < LONG_EVENTS:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise RuntimeError(f"only {len(events)} long events after {attempts} draws")
        event = _try_long(
            pool=pool, quiet=quiet, power=power, rng=rng, event_id=first_id + len(events)
        )
        if event is None:
            continue
        month = stamp_of_month(event.onset)
        if sum(stamp_of_month(e.onset) == month for e in events) < MAX_PER_MONTH:
            events.append(event)
    return events


def days(event: Event) -> float:
    """The duration of an event in days."""
    return (event.end - event.onset) / BLOCK
