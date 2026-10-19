"""Draw the synthetic training transfers of experiment 00096 and turn them into dense-grid arrays.

The transfers start in odd calendar months from 2024-07 to 2026-05, quiet on every series they
touch (the quiet masks of experiment 00088), and are injected with the shapes S2, S3, S4, and S6 in
turn. The number of injected series-intervals is 3 times the number of real labelled transfer
intervals that start between 2024-07-01 and 2026-06-01. Test transfers start in even months only.

    uv run python ml-experiments/00096-*/e00096_synth.py
"""

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import e00096_common as common
import e00096_grid as grid
import e00096_harness as harness
import e00096_labels as labelling
import numpy as np
import polars as pl
from e00096_inject import BLOCK, QUIET_STAMPS, Event, draw_events
from scipy.ndimage import maximum_filter1d
from e00096_profile import Shape, fit_profile_shares, inject_shaped

SEED: Final[int] = 9600
FIRST_ID: Final[int] = 96_000
SHAPES: Final[tuple[Shape, ...]] = ("S2", "S3", "S4", "S6")
RANGE_START: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
RANGE_END: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
FIRST_YEAR_MONTH: Final[int] = 202407
LAST_YEAR_MONTH: Final[int] = 202605
MULTIPLE: Final[int] = 3
"""Injected series-intervals as a multiple of the real series-intervals in the same range."""
PER_MONTH_COUNTS: Final[tuple[int, int]] = (24, 12)
"""Events drawn per odd month: in the headline band, and in each of the other eight bands."""
PAD_STAMPS: Final[int] = BLOCK
"""A synthetic transfer keeps one day clear of every logged closure (a real transfer is corrected by
its labels, so a nearby real transfer does no harm in training) and two days clear of every alarm."""
PATH: Final = common.WORK_DIR / "synthetic.pkl"


class _NarrowedMask:
    """A quiet mask whose look-up window is cut by the 7 days that `draw_events` adds each side."""

    def __init__(self, *, mask: np.ndarray) -> None:
        self._mask = mask

    def __getitem__(self, span: slice) -> np.ndarray:
        trim = QUIET_STAMPS
        return self._mask[span.start + trim : span.stop - trim]


def training_quiet_masks() -> dict[int, _NarrowedMask]:
    """Return masks that a training transfer must not touch, padded by one day or two days."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    closed = labelling.load_labels(series=series)
    ids = {int(r["time_series_id"]): r["site"] for r in series.iter_rows(named=True)}
    logged = grid.near_log_masks(pad_stamps=PAD_STAMPS)
    alarmed = harness.alarm_masks()
    out = {}
    for sid in grid.SERIES_IDS:
        flags = closed[ids[sid]].closed[harness.OFFSET : harness.OFFSET + grid.GRID_LENGTH]
        labelled = maximum_filter1d(flags, size=2 * PAD_STAMPS + 1)
        out[sid] = _NarrowedMask(mask=logged[sid] | alarmed[sid] | labelled)
    return out


@dataclass(frozen=True)
class SyntheticArrays:
    """One series' synthetic transfers on the dense half-hourly grid.

    Attributes:
        delta: The injected power over effective capacity (zero where nothing is injected).
        signed_mw: Each transfer's median injected MW over capacity, constant over the transfer.
        in_event: True where a synthetic transfer covers the half-hour.
    """

    delta: np.ndarray
    signed_mw: np.ndarray
    in_event: np.ndarray


def real_interval_count(*, labels: dict[str, labelling.SeriesLabels]) -> int:
    """Count the real labelled transfer intervals that start in the training range, per series."""
    first = int(common.grid_index(times=RANGE_START))
    last = int(common.grid_index(times=RANGE_END))
    total = 0
    for series_labels in labels.values():
        flags = series_labels.in_event.astype(np.int8)
        onsets = np.flatnonzero(np.diff(flags, prepend=0) == 1)
        total += int(((onsets >= first) & (onsets < last)).sum())
    return total


def _overlaps(*, event: Event, other: Event) -> bool:
    """Whether two events share a series and lie within 7 days of each other."""
    return not set(event.series).isdisjoint(other.series) and not (
        event.onset - QUIET_STAMPS > other.end or other.onset - QUIET_STAMPS > event.end
    )


def draw_training_events(*, power: dict[int, np.ndarray], quota: int) -> list[Event]:
    """Draw odd-month events, drop overlaps, and take them month by month until the quota is met.

    Args:
        power: The stored panel's observed power by series.
        quota: The number of series-intervals to inject.

    Returns:
        The events, each quiet on its series, no two within 7 days on a shared series.
    """
    months = grid.stamp_months()
    year_months = grid.stamp_year_months()
    quiet = training_quiet_masks()
    by_month: list[list[Event]] = []
    number = 0
    for ym in sorted(set(year_months.tolist())):
        if not FIRST_YEAR_MONTH <= ym <= LAST_YEAR_MONTH or ym % 2 == 0:
            continue
        drawn = draw_events(
            quiet=quiet,  # ty: ignore[invalid-argument-type]
            power=power,
            seed=SEED + number,
            first_id=FIRST_ID + 1000 * number,
            onset_ok=lambda onset, y=ym: int(year_months[onset]) == y and int(months[onset]) % 2 == 1,
            counts=PER_MONTH_COUNTS,
        )
        by_month.append([e for e in drawn if harness.has_data(event=e, power=power)])
        number += 1
    chosen: list[Event] = []
    total = 0
    for rank in range(max(len(month) for month in by_month)):
        for month in by_month:
            if rank >= len(month) or total >= quota:
                continue
            event = month[rank]
            if any(_overlaps(event=event, other=other) for other in chosen):
                continue
            chosen.append(event)
            total += len(event.series)
    return chosen


def synthetic_arrays(
    *, events: list[Event], panel_power: dict[int, np.ndarray], shares: list[float]
) -> tuple[dict[str, SyntheticArrays], list[Shape]]:
    """Inject the events (shapes in turn) and return each Primary series' dense-grid arrays.

    Args:
        events: The training events.
        panel_power: The stored panel's observed power by series.
        shares: The profile shares fitted to the labelled transfers (shape S4).

    Returns:
        The arrays by series label, and the shape given to each event in order.
    """
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    arrays = common.load_series_arrays(series=series)
    rows = {int(r["time_series_id"]): r for r in series.iter_rows(named=True)}
    out = {
        row["site"]: SyntheticArrays(
            delta=np.zeros_like(arrays[row["site"]].power),
            signed_mw=np.zeros_like(arrays[row["site"]].power),
            in_event=np.zeros(arrays[row["site"]].power.size, dtype=bool),
        )
        for row in rows.values()
    }
    shapes = [SHAPES[i % len(SHAPES)] for i in range(len(events))]
    for k, shape in enumerate(SHAPES):
        subset = [e for e, s in zip(events, shapes, strict=True) if s == shape]
        if not subset:
            continue
        injection = inject_shaped(
            power=panel_power, events=subset, seed=SEED + 100 + k, shape=shape, shares=shares
        )
        for event in subset:
            for sid in event.series:
                row = rows[sid]
                capacity = row["effective_capacity_mw"]
                target = out[row["site"]]
                moved = injection.delta[sid][event.onset : event.end]
                span = slice(harness.OFFSET + event.onset, harness.OFFSET + event.end)
                target.delta[span] += (moved / capacity).astype(np.float32)
                target.signed_mw[span] += np.float32(np.median(moved) / capacity)
                target.in_event[span] = True
    return out, shapes


def load_or_build() -> tuple[list[Event], dict[str, SyntheticArrays]]:
    """Return the training events and arrays, building and saving them on the first call."""
    if PATH.exists():
        return pickle.loads(PATH.read_bytes())
    panel = grid.load_panel()
    series = common.series_table().filter(pl.col("substation_type").is_in(common.GROUPS))
    labels = labelling.load_labels(series=series)
    real = real_interval_count(labels=labels)
    events = draw_training_events(power=panel["power"], quota=MULTIPLE * real)
    arrays, shapes = synthetic_arrays(
        events=events, panel_power=panel["power"], shares=fit_profile_shares(panel=panel)
    )
    injected = sum(len(e.series) for e in events)
    print(
        f"real series-intervals in range: {real}; injected events: {len(events)} "
        f"({injected} series-intervals), shapes {dict(zip(*np.unique(shapes, return_counts=True), strict=True))}"
    )
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PATH.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps((events, arrays)))
    temporary.rename(PATH)
    return events, arrays


if __name__ == "__main__":
    # Import by name so the pickle holds `e00096_synth.SyntheticArrays`, not `__main__`.
    import e00096_synth

    e00096_synth.load_or_build()
