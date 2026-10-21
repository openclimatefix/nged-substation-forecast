"""Draw injected transfers for experiment 131 and build their true flows.

A transfer moves power from a donor to a receiver across an interconnection edge for a while. It
is added to the residuals of the two ends (the receiver gains the moved megawatts and the donor
loses them), which is exact for a frozen fit that did not train on the injected period.

**Three disjoint sets of seeds.** `T` (tuning) and `G` (gate) hold 400 transfers each: 50 seeds,
2 edges, 4 transfers per edge and seed, balanced over 4 sizes and 2 durations (3 to 24 hours and
24 to 72 hours, log-uniform within each band). `L` holds 200 transfers of 25% and 40% of the
receiving load that last 7 to 30 days, log-uniform (25 seeds, 2 edges, 4 transfers per edge and
seed). A transfer stays at least 7 days from another transfer on its edge, and never within 48
hours of a hand-filtered event at either end or inside one of D10's two unlabelled windows. The
draws depend on the seed and on which rows hold readings, never on the residuals.
"""

from datetime import timedelta
from typing import Final, Literal

import e00131_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

SetNameType = Literal["T", "G", "L"]
"""The three disjoint sets of seeds."""
ShapeType = Literal["donor_shaped", "flat"]
"""The donor-shaped share of experiment 43, or a flat block."""

SET_IDS: Final[dict[str, int]] = {"T": 1, "G": 2, "L": 5}
SEEDS_PER_SET: Final[dict[str, int]] = {"T": 50, "G": 50, "L": 25}
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
    "3-24h": (3.0, 24.0),
    "1-3 days": (24.0, 72.0),
}
DURATION_SHORT_BAND: Final[str] = "3-24h"
DURATION_LONG_BAND: Final[str] = "1-3 days"
LONG_BAND: Final[tuple[str, tuple[float, float]]] = ("7-30 days", (7 * 24.0, 30 * 24.0))
CELLS: Final[tuple[tuple[str, str], ...]] = tuple(
    (size, duration) for size in SIZE_LEVELS for duration in DURATION_BANDS_HOURS
)
"""The 8 size-by-duration cells of the balanced sets."""
LONG_CELLS: Final[tuple[tuple[str, str], ...]] = tuple(
    (size, LONG_BAND[0]) for size in ("25%", "40%")
)
"""The 2 cells of the 7 to 30 day set."""
HALF_MW: Final[float] = 0.5
WALK_STEP: Final[float] = 0.03
"""Standard deviation of the log random walk per half-hour (experiment 43)."""
WALK_LOW: Final[float] = 0.5
WALK_HIGH: Final[float] = 2.0
MAX_DONOR_SHARE: Final[float] = 0.8
"""The donor-shaped share never exceeds this part of the donor's power."""


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
    if transfer["size_mw"] > 0:
        start = float(transfer["size_mw"])
    else:
        start = SIZE_FRACTION[transfer["size"]] * float(np.nanmean(receiver_power_mw))
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
    taken: list[tuple[int, int]],
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
        if any(
            onset < other_end + MIN_SPACING_HALF_HOURS
            and other_onset < onset + length + MIN_SPACING_HALF_HOURS
            for other_onset, other_end in taken
        ):
            continue
        return onset
    return None


def draw_set(*, set_name: SetNameType, year_name: str, cache: FeatureCache) -> pl.DataFrame:
    """Draw one set's transfers in one scored year.

    Args:
        set_name: `T`, `G`, or `L`.
        year_name: `tuning`, `replicate`, or `scored`.
        cache: The feature cache, for the grid and the rows with readings.

    Returns:
        One row per transfer with `transfer_id`, `set`, `seed`, `node_a`, `node_b` (the edge, in
        name order), `donor`, `receiver`, `onset` and `end` (grid positions), `size`, `size_mw`
        (zero for a size set by the receiving load), `duration`, `shape`, and `walk_seed`.
    """
    cut = common.CUTS[year_name]
    year = common.year_slice(cache=cache, cut=cut)
    cells = LONG_CELLS if set_name == "L" else CELLS
    cell_order = _balanced_cells(set_name=set_name, cells=cells)
    bands = {LONG_BAND[0]: LONG_BAND[1], **DURATION_BANDS_HOURS}
    rows: list[dict] = []
    counter = 0
    for seed in range(SEEDS_PER_SET[set_name]):
        rng = np.random.default_rng([SET_IDS[set_name], seed, 131])
        for edge in common.GATE_EDGES:
            node_a, node_b = sorted(edge)
            present, blocked = _usable_rows(
                cache=cache, node_a=node_a, node_b=node_b, first=year.start, stop=year.stop
            )
            taken: list[tuple[int, int]] = []
            for _ in range(TRANSFERS_PER_EDGE_AND_SEED):
                size, duration = cell_order[counter % len(cell_order)]
                low, high = bands[duration]
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
                taken.append((onset, onset + length))
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
                        "size": size,
                        "size_mw": size_mw,
                        "duration": duration,
                        "shape": "donor_shaped" if rng.random() < 0.5 else "flat",
                        "walk_seed": int(rng.integers(0, 2**31 - 1)),
                    }
                )
    return pl.DataFrame(rows)


def _balanced_cells(
    *, set_name: SetNameType, cells: tuple[tuple[str, str], ...]
) -> list[tuple[str, str]]:
    """Return a shuffled cycle of the cells, so each cell gets an equal share of the transfers."""
    rng = np.random.default_rng([SET_IDS[set_name], 999, 131])
    order = rng.permutation(len(cells))
    return [cells[i] for i in order]
