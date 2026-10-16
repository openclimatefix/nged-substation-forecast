"""World B: injected transfers that are part of the data the lagless XGBoost model trains on.

The transfers are a fresh draw of headline-band transfers from experiment 58's generator that
start in even months, reduced greedily to a set with no two transfers on one series within 1 day of
each other. (Experiment 58's own 620 even-month transfers overlap too much on one series: only 103
fit with a gap of 1 day.) They take the
shapes S2, S3, and S4 in turn. The MW added to each series is written for `e00068_fit.py`, and the
event table (with the MW moved and the shape) for the scorer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00068_common import (
    GRID_LENGTH,
    ML_EXPERIMENTS_DATA_DIR,
    WORK_DIR,
    load_panel,
    stamp_months,
)
from e00068_detect import (
    SEED_43,
    SEED_NEW,
    new_events,
    original_events,
    quiet_masks,
    seeded_batches,
)
from e00068_inject import BLOCK, HEADLINE_BAND, Event, draw_events
from e00068_profile import NON_MATCHING, fit_profile_shares, inject_shaped

EXPERIMENT_58_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
MIN_GAP: Final[int] = BLOCK
"""Transfers that share a series are at least this many half-hours apart in world B."""
SEED_B: Final[int] = 6800
N_DRAWN: Final[int] = 4000
FIRST_ID: Final[int] = 6000


def headline_even_ids() -> set[int]:
    """Experiment 58's headline-band, even-month, non-holiday transfers (experiment 61's 620)."""
    events = pl.read_parquet(EXPERIMENT_58_DIR / "events.parquet")
    month = stamp_months()[events["onset"].to_numpy()]
    return set(
        events.with_columns(month=pl.Series(month))
        .filter(
            pl.col("size_band") == HEADLINE_BAND[0],
            pl.col("duration_band") == HEADLINE_BAND[1],
            pl.col("month") % 2 == 0,
            pl.col("set") != "holiday",
        )["event_id"]
        .to_list()
    )


def world_a_batches(*, power: dict[int, np.ndarray]) -> list[tuple[int, list[Event]]]:
    """Experiment 58's 620 transfers in its injection batches, each with its seed."""
    original = original_events(power=power)
    extra, holiday = new_events(power=power)
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    wanted = headline_even_ids()
    kept = [(seed, [e for e in batch if e.event_id in wanted]) for seed, batch in groups]
    return [(seed, batch) for seed, batch in kept if batch]


def draw(*, power: dict[int, np.ndarray]) -> list[Event]:
    """Draw `N_DRAWN` headline-band transfers, keeping those that lie within one even month."""
    months = stamp_months()
    drawn = draw_events(
        quiet=quiet_masks(),
        power=power,
        seed=SEED_B,
        headline_only=N_DRAWN,
        first_id=FIRST_ID,
        onset_ok=lambda onset: bool(months[onset] % 2 == 0),
    )
    return [e for e in drawn if months[e.end - 1] % 2 == 0 and months[e.end - 1] == months[e.onset]]


def spread_out(*, events: list[Event]) -> list[Event]:
    """Greedily keep events with no other kept event on a shared series within `MIN_GAP`."""
    kept: list[Event] = []
    for event in sorted(events, key=lambda e: e.event_id):
        if all(
            set(event.series).isdisjoint(other.series)
            or event.onset - MIN_GAP > other.end
            or other.onset - MIN_GAP > event.end
            for other in kept
        ):
            kept.append(event)
    return kept


def build(*, out: Path = WORK_DIR) -> None:
    """Write world B's delta table and event table."""
    panel = load_panel()
    power = panel["power"]
    chosen = spread_out(events=draw(power=power))
    shares = fit_profile_shares(panel=panel)
    delta = {s: np.zeros(GRID_LENGTH) for s in power}
    rows = []
    for i, shape in enumerate(NON_MATCHING):
        group = chosen[i :: len(NON_MATCHING)]
        injection = inject_shaped(
            power=power, events=group, seed=SEED_B + i, shape=shape, shares=shares
        )
        for s in power:
            delta[s] += injection.delta[s]
        for event in group:
            span = slice(event.onset, event.end)
            rows.append(
                {
                    "event_id": event.event_id,
                    "shape": shape,
                    "donor": event.donor,
                    "series": ",".join(map(str, event.series)),
                    "onset": event.onset,
                    "end": event.end,
                    "metered_receivers": len(event.receivers),
                    "moved_mw": float(-np.mean(injection.delta[event.donor][span])),
                }
            )
    pieces = []
    for s, values in delta.items():
        stamps = np.flatnonzero(values)
        pieces.append(
            pl.DataFrame({"series": s, "stamp": stamps, "delta_mw": values[stamps]}).cast(
                {"series": pl.Int64, "stamp": pl.Int64}
            )
        )
    out.mkdir(parents=True, exist_ok=True)
    pl.concat(pieces).write_parquet(out / "world_b_delta.parquet")
    events_table = pl.DataFrame(rows)
    events_table.write_parquet(out / "world_b_events.parquet")
    print(
        f"world B: {events_table.height} of {N_DRAWN} drawn headline even-month transfers kept; "
        + ", ".join(
            f"{shape} {events_table.filter(pl.col('shape') == shape).height}"
            for shape in NON_MATCHING
        )
    )


if __name__ == "__main__":
    build()
