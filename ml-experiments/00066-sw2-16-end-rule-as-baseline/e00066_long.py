"""The long set: injected transfers that last 14 to 60 days, drawn like experiment 58's.

Experiment 58's injected transfers last 30 days at most, and only 17 of its 1,160 headline-band
transfers last over 14 days, so they cannot tell a cap of 14 days from a cap of 30 or 60. The long
set draws its durations from the labelled real operations that last 14 to 60 days, in the headline
size band, with the quietness rules of `draw_events`.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00066_common import BLOCK, LABELLED_PATH
from e00066_inject import HEADLINE_BAND, Event, _try_event

LONG_EVENTS: Final[int] = 400
FIRST_ID: Final[int] = 5000
SEED_LONG: Final[int] = 7700
"""Seed of the draw; batch `i` is injected with `SEED_LONG + 1000 + i`."""
LONG_BAND: Final[str] = "14 to 60 d"
MIN_DAYS: Final[int] = 14
MAX_DAYS: Final[int] = 60
MAX_ATTEMPTS: Final[int] = 400_000


def long_operations() -> pl.DataFrame:
    """The labelled real operations lasting 14 to 60 days: sheet sides and duration in hours."""
    return (
        pl.read_parquet(LABELLED_PATH)
        .filter(pl.col("kind") == "real_operation")
        .with_columns(
            hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60
        )
        .filter(pl.col("hours").is_between(MIN_DAYS * 24, MAX_DAYS * 24))
        .select("side_a", "side_b", "hours")
    )


def draw_long_events(*, quiet: dict[int, np.ndarray]) -> list[Event]:
    """Draw `LONG_EVENTS` headline-size transfers lasting 14 to 60 days, quiet on every series."""
    pool = long_operations()
    rng = np.random.default_rng(SEED_LONG)
    events: list[Event] = []
    attempts = 0
    while len(events) < LONG_EVENTS:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise RuntimeError(f"only {len(events)} long events after {attempts} draws")
        event = _try_event(
            pool=pool,
            size=HEADLINE_BAND[0],
            duration=LONG_BAND,
            quiet=quiet,
            rng=rng,
            event_id=FIRST_ID + len(events),
        )
        if event is not None:
            events.append(event)
    return events


def days(event: Event) -> float:
    """The duration of an event in days."""
    return (event.end - event.onset) / BLOCK
