"""The second pass: injected transfers lasting 1 to 60 days, so the detector has time to see them.

The main draw takes durations from NGED's logged operations, and most last under 24 hours. The
causal detector needs 24 hours of data after an onset, so this pass draws transfers of four
duration bins (1 to 3 days, 3 to 14 days, 14 to 30 days, and 30 to 60 days) with the same
quiet-period rules, shapes, and months as the main draw, and writes their rows with
`e00091_rows.run_job`.

    uv run python ml-experiments/00091-*/e00091_longrows.py [--workers 4]
"""

import argparse
import pickle
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00091_common as common
import e00091_grid as grid
import e00091_harness as harness
import e00091_rows as rows
import numpy as np
from e00091_inject import (
    ADJACENCY,
    P_EXTRA_UNMETERED,
    QUIET_STAMPS,
    SIZE_BANDS,
    Event,
    batches,
    operations,
)

LONG_BINS: Final[dict[str, tuple[float, float]]] = {
    "1 to 3 d": (24.0, 72.0),
    "3 to 14 d": (72.0, 14 * 24.0),
    "14 to 30 d": (14 * 24.0, 30 * 24.0),
    "30 to 60 d": (30 * 24.0, 60 * 24.0),
}
SIZES: Final[tuple[str, ...]] = ("10% to 25%", "25% or more")
PER_BIN_PER_MONTH: Final[int] = 8
SEED_LONG: Final[int] = 9100
"""Month `m` is drawn with `SEED_LONG + m`; batch `i` is injected with `SEED_LONG + 1000 + i`."""
FIRST_ID: Final[int] = 91_000
SHAPES: Final[tuple[str, ...]] = ("S2", "S3", "S4", "S6")
MAX_ATTEMPTS: Final[int] = 200_000
EVENTS_PATH: Final = common.WORK_DIR / "long_events.pkl"


def try_event(
    *,
    sheets_row: dict,
    hours: float,
    size: str,
    quiet: dict,
    rng: np.random.Generator,
    event_id: int,
    onset_ok: Callable[[int], bool],
) -> Event | None:
    """Draw one transfer like `e00091_inject._try_event`, with the duration set by the caller."""
    sides = [
        int(s)
        for s in (sheets_row["side_a"], sheets_row["side_b"])
        if s is not None and s in ADJACENCY
    ]
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
    length = max(4, int(hours * 2))
    onset = int(rng.integers(14 * grid.BLOCK, grid.GRID_LENGTH - length - grid.BLOCK))
    if not onset_ok(onset):
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
        duration_band=next(n for n, (lo, hi) in LONG_BINS.items() if lo <= hours < hi),
        fraction=float(rng.uniform(low, high)),
    )


def draw_long_events(*, power: dict) -> list[Event]:
    """Draw `PER_BIN_PER_MONTH` transfers per duration bin and month of the validation year."""
    first, last = harness.validation_stamps()
    months = grid.stamp_months()
    quiet = harness.quiet_masks()
    sheets = operations()
    events: list[Event] = []
    for month in harness.DRAW_MONTHS:
        rng = np.random.default_rng(SEED_LONG + month)
        for name, (low, high) in LONG_BINS.items():
            made = attempts = 0
            while made < PER_BIN_PER_MONTH and attempts < MAX_ATTEMPTS:
                attempts += 1
                hours = float(np.exp(rng.uniform(np.log(low), np.log(high))))
                event = try_event(
                    sheets_row=sheets.row(int(rng.integers(0, sheets.height)), named=True),
                    hours=hours,
                    size=SIZES[made % 2],
                    quiet=quiet,
                    rng=rng,
                    event_id=FIRST_ID + len(events),
                    onset_ok=lambda onset, m=month: (
                        first <= onset < last and int(months[onset]) == m
                    ),
                )
                if event is not None and harness.has_data(event=event, power=power):
                    events.append(event)
                    made += 1
            print(f"month {month} {name}: {made} of {PER_BIN_PER_MONTH}", flush=True)
    return events


def main() -> None:
    """Draw the long transfers, then write the rows of every batch and shape."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    state = harness.make_state()
    if EVENTS_PATH.exists():
        events = pickle.loads(EVENTS_PATH.read_bytes())
    else:
        events = draw_long_events(power=state["panel"]["power"])
        EVENTS_PATH.write_bytes(pickle.dumps(events))
    print(f"{len(events)} long transfers")
    seeded = [(SEED_LONG + 1000 + i, b) for i, b in enumerate(batches(events=events))]
    jobs = [(seed, batch, shape) for shape in SHAPES for seed, batch in seeded]
    print(f"{len(seeded)} batches, {len(jobs)} jobs", flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=rows.init_worker) as pool:
        for done, line in enumerate(pool.map(rows.run_job, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
