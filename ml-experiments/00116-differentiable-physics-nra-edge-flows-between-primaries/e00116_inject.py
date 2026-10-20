"""Draw injected transfers for experiment 116 and build their true flows.

A transfer moves power from a donor to a receiver across an interconnection edge for a while. It
is added to the residuals of the two ends (the receiver gains the moved megawatts and the donor
loses them), which is exact for a frozen fit that did not train on the injected period.

**Four disjoint sets of 50 seeds**, `T` (tuning), `G` (gate), `C1`, and `C3`, each hold up to 4
transfers per gate edge and seed, at least 7 days apart, never within 48 hours of a hand-filtered
event at either end or inside one of D10's two unlabelled windows. The draws depend on the seed
and on which rows hold readings, never on the residuals. Sets `T` and `G` are balanced over a grid
of 4 sizes and 3 durations. Sets `C1` and `C3` draw size and duration jointly from the hand-filtered
events, so they match the real distribution.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal

import e00116_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SetNameType = Literal["T", "G", "C1", "C3"]
"""The four disjoint sets of seeds."""
ShapeType = Literal["donor_shaped", "flat"]
"""The donor-shaped share of experiment 43, or a flat block."""

SET_IDS: Final[dict[str, int]] = {"T": 1, "G": 2, "C1": 3, "C3": 4}
SEEDS_PER_SET: Final[int] = 50
TRANSFERS_PER_EDGE_AND_SEED: Final[int] = 4
MIN_SPACING_HALF_HOURS: Final[int] = 7 * 48
"""Transfers on one edge start at least 7 days apart."""
EVENT_BUFFER_HALF_HOURS: Final[int] = 2 * 48
"""A transfer stays this far from a hand-filtered event at either end."""
MARGIN_HALF_HOURS: Final[int] = 14 * 48
"""A transfer starts at least this far after the year's first row, so the filter has history."""
POST_HALF_HOURS: Final[int] = 2 * 48
"""Rows after a transfer's end that must hold readings, for the post-event rows of C3."""
MIN_PRESENT_SHARE: Final[float] = 0.9
"""Share of the transfer's rows and its surrounding rows that must hold readings at both ends."""
SIZE_LEVELS: Final[tuple[str, ...]] = ("0.5MW", "10%", "25%", "40%")
SIZE_FRACTION: Final[dict[str, float]] = {"10%": 0.10, "25%": 0.25, "40%": 0.40}
DURATION_BANDS_HOURS: Final[dict[str, tuple[float, float]]] = {
    "3-6h": (3.0, 6.0),
    "6-24h": (6.0, 24.0),
    "over 1 day": (24.0, 120.0),
}
CELLS: Final[tuple[tuple[str, str], ...]] = tuple(
    (size, duration) for size in SIZE_LEVELS for duration in DURATION_BANDS_HOURS
)
"""The 12 size-by-duration cells of the balanced sets."""
HALF_MW: Final[float] = 0.5
WALK_STEP: Final[float] = 0.03
"""Standard deviation of the log random walk per half-hour (experiment 43)."""
WALK_LOW: Final[float] = 0.5
WALK_HIGH: Final[float] = 2.0
MAX_DONOR_SHARE: Final[float] = 0.8
"""The donor-shaped share never exceeds this part of the donor's power."""


@dataclass(frozen=True)
class RealEvents:
    """The 66 hand-filtered events' sizes and durations, for the realistic sets.

    Attributes:
        moved_mw: Each event's median moved megawatts.
        hours: Each event's duration in hours.
    """

    moved_mw: np.ndarray
    hours: np.ndarray


def real_events() -> RealEvents:
    """Return the sizes and durations of experiment 101's 66 events.

    The filter is `real_events` of experiment 101: not a sentinel, a real operation, and visible as
    conserved or one-sided. Events with no moved-megawatt estimate are dropped.
    """
    frame = (
        pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
        )
        .filter(~pl.col("is_sentinel"), pl.col("kind") == "real_operation")
        .filter(pl.col("visibility").is_in(["conserved", "one_sided"]))
        .with_columns(hours=(pl.col("end_utc") - pl.col("onset_utc")).dt.total_minutes() / 60.0)
        .filter(pl.col("moved_mw_p50").is_finite(), pl.col("hours") > 0)
    )
    return RealEvents(
        moved_mw=frame["moved_mw_p50"].abs().to_numpy(), hours=frame["hours"].to_numpy()
    )


def _walk(*, length: int, rng: np.random.Generator) -> np.ndarray:
    """Return a slow multiplicative random walk between 0.5 and 2, reflected at both ends."""
    log = np.zeros(length)
    low, high = np.log(WALK_LOW), np.log(WALK_HIGH)
    steps = rng.normal(0.0, WALK_STEP, length)
    for i in range(1, length):
        value = log[i - 1] + steps[i]
        value = 2 * high - value if value > high else value
        log[i] = 2 * low - value if value < low else value
    return np.exp(np.clip(log, low, high))


def moved_mw(
    *, transfer: dict, donor_power_mw: np.ndarray, receiver_power_mw: np.ndarray
) -> np.ndarray:
    """Return the megawatts a transfer moves at each half-hour of its duration.

    Args:
        transfer: One row of the transfers table.
        donor_power_mw: The donor's measured power over the transfer's rows.
        receiver_power_mw: The receiver's measured power over the same rows.

    Returns:
        The moved megawatts, non-negative.
    """
    length = int(transfer["end"] - transfer["onset"])
    walk = _walk(length=length, rng=np.random.default_rng(int(transfer["walk_seed"])))
    if transfer["size"] == "0.5MW":
        start = HALF_MW
    else:
        start = SIZE_FRACTION[transfer["size"]] * float(np.nanmean(receiver_power_mw))
    start = float(transfer["size_mw"]) if transfer["size_mw"] > 0 else start
    if transfer["shape"] == "flat":
        return start * walk
    donor = np.maximum(np.nan_to_num(donor_power_mw, nan=np.nanmean(donor_power_mw)), 1e-6)
    share = min(MAX_DONOR_SHARE, start / float(np.mean(donor)))
    return np.minimum(share * walk, MAX_DONOR_SHARE) * donor


def edge_flow(*, transfer: dict, moved: np.ndarray, length: int) -> np.ndarray:
    """Return the true flow on the transfer's edge, positive onto the edge's first node.

    Args:
        transfer: One row of the transfers table.
        moved: The moved megawatts over the transfer's rows.
        length: The length of the grid slice the flow is placed on.

    Returns:
        A flow of `length` rows, zero outside the transfer, with `transfer["onset"]` and
        `transfer["end"]` taken as positions on that slice.
    """
    flow = np.zeros(length)
    sign = 1.0 if transfer["receiver"] == transfer["node_a"] else -1.0
    flow[int(transfer["onset"]) : int(transfer["end"])] = sign * moved
    return flow


def _usable_rows(
    *, cache: FeatureCache, node_a: str, node_b: str, first: int, stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return rows where both ends have readings, and rows near a hand-filtered event or jump."""
    a, b = cache.sites[node_a], cache.sites[node_b]
    present = a.present & b.present & np.isfinite(a.y) & np.isfinite(b.y)
    blocked = a.switching_mask | b.switching_mask
    for node in (node_a, node_b):
        if node == "D10":
            for start, end in common.D10_JUMP_WINDOWS:
                blocked = blocked | (
                    (cache.end_times > common.to_64(moment=start))
                    & (cache.end_times <= common.to_64(moment=end + timedelta(days=0)))
                )
    near = np.convolve(
        blocked.astype(np.int32), np.ones(2 * EVENT_BUFFER_HALF_HOURS + 1, dtype=np.int32), "same"
    )
    window = np.zeros(len(present), dtype=bool)
    window[first:stop] = True
    return present & window, (near > 0) | ~window


def _place(
    *,
    present: np.ndarray,
    blocked: np.ndarray,
    first: int,
    stop: int,
    length: int,
    taken: list[int],
    rng: np.random.Generator,
) -> int | None:
    """Return a start position for a transfer, or `None` after 200 failed tries."""
    for _ in range(200):
        onset = int(rng.integers(first + MARGIN_HALF_HOURS, stop - length - POST_HALF_HOURS))
        span = slice(onset - 12, onset + length + POST_HALF_HOURS)
        if blocked[
            onset - EVENT_BUFFER_HALF_HOURS : onset + length + EVENT_BUFFER_HALF_HOURS
        ].any():
            continue
        if present[span].mean() < MIN_PRESENT_SHARE:
            continue
        if any(abs(onset - other) < MIN_SPACING_HALF_HOURS for other in taken):
            continue
        return onset
    return None


def draw_set(
    *, set_name: SetNameType, year_name: str, cache: FeatureCache, events: RealEvents | None = None
) -> pl.DataFrame:
    """Draw one set's transfers in one scored year.

    Args:
        set_name: `T`, `G`, `C1`, or `C3`.
        year_name: `tuning`, `replicate`, or `scored`.
        cache: The feature cache, for the grid and the rows with readings.
        events: The real events, for the realistic sets; read from disk if not given.

    Returns:
        One row per transfer with `transfer_id`, `set`, `seed`, `node_a`, `node_b` (the edge, in
        name order), `donor`, `receiver`, `onset` and `end` (grid positions), `size`, `size_mw`
        (zero for a size set by the receiving load), `duration`, `shape`, and `walk_seed`.
    """
    realistic = set_name in ("C1", "C3")
    events = events if events is not None else (real_events() if realistic else None)
    cut = common.CUTS[year_name]
    year = common.year_slice(cache=cache, cut=cut)
    cell_order = _balanced_cells(set_name=set_name)
    rows: list[dict] = []
    counter = 0
    for seed in range(SEEDS_PER_SET):
        rng = np.random.default_rng([SET_IDS[set_name], seed, 116])
        for edge in common.GATE_EDGES:
            node_a, node_b = sorted(edge)
            present, blocked = _usable_rows(
                cache=cache, node_a=node_a, node_b=node_b, first=year.start, stop=year.stop
            )
            taken: list[int] = []
            for _ in range(TRANSFERS_PER_EDGE_AND_SEED):
                if realistic:
                    assert events is not None
                    pick = int(rng.integers(0, len(events.hours)))
                    hours, size, size_mw = (
                        float(events.hours[pick]),
                        "event",
                        float(events.moved_mw[pick]),
                    )
                    duration = "event"
                else:
                    size, duration = cell_order[counter % len(cell_order)]
                    low, high = DURATION_BANDS_HOURS[duration]
                    hours = float(np.exp(rng.uniform(np.log(low), np.log(high))))
                    size_mw = HALF_MW if size == "0.5MW" else 0.0
                counter += 1
                length = max(4, round(hours * 2))
                onset = _place(
                    present=present,
                    blocked=blocked,
                    first=year.start,
                    stop=year.stop,
                    length=length,
                    taken=taken,
                    rng=rng,
                )
                if onset is None:
                    continue
                taken.append(onset)
                donor_is_a = bool(rng.random() < 0.5)
                rows.append(
                    {
                        "transfer_id": f"{set_name}-{seed:02d}-{len(rows):04d}",
                        "set": set_name,
                        "seed": seed,
                        "year": year_name,
                        "node_a": node_a,
                        "node_b": node_b,
                        "donor": node_a if donor_is_a else node_b,
                        "receiver": node_b if donor_is_a else node_a,
                        "onset": onset,
                        "end": onset + length,
                        "size": "0.5MW" if (size == "event" and size_mw == HALF_MW) else size,
                        "size_mw": size_mw,
                        "duration": duration,
                        "shape": "donor_shaped" if rng.random() < 0.5 else "flat",
                        "walk_seed": int(rng.integers(0, 2**31 - 1)),
                    }
                )
    return pl.DataFrame(rows)


def _balanced_cells(*, set_name: SetNameType) -> list[tuple[str, str]]:
    """Return a shuffled cycle of the 12 cells, so each cell gets 33 or 34 of 400 transfers."""
    rng = np.random.default_rng([SET_IDS[set_name], 999, 116])
    order = rng.permutation(len(CELLS))
    return [CELLS[i] for i in order]
