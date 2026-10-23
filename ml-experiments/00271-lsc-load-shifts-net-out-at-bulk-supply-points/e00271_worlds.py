"""Inject load shifts between Primary substations and push them up to the bulk supply points.

**A world is a set of rectangular load shifts conserved in megawatts between two Primaries.** Each
shift lowers the donor's readings and raises the receiver's by the same megawatts, for 1, 2, 4, or
8 days. The harness reruns each arm's Kalman filter on the injected readings of the affected
Primaries, and this module adds up, at each bulk supply point, how much each arm's forecast moved
against the same arm on the real record. Nothing here reads a score.

A bulk supply point (BSP) forecast of an arm is experiment 48's node forecast plus that sum, so only
the injected shifts move a forecast. The BSP's observed power moves only through a shift that
crosses the boundary between the two BSPs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from typing import Final

import e00271_data as dat
import e00271_harness as h
import numpy as np
import polars as pl
from studies.sources import REPO_DATA_DIR

HIERARCHY_DIR: Final = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15"
SHARES: Final[tuple[float, ...]] = (0.15, 0.30, 0.45)
"""Moved power as a share of the donor's mean load."""
DURATION_DAYS: Final[tuple[int, ...]] = (1, 2, 4, 8)
SHIFTS_PER_WORLD: Final[int] = 8
ONSET_LAG: Final[int] = 96
"""Half-hours from a shift's onset to the stamp the live service knows of it (experiment 208)."""
ARMS: Final[tuple[str, ...]] = ("K0", "GAM_ONLY", "PERFECT", "PERFECT_LAG", "LB20_LIKE")
KINDS: Final[tuple[str, ...]] = ("within", "cross")
PLACEMENTS: Final[tuple[str, ...]] = ("random", "targeted")
RANDOM_WINDOW: Final[tuple[dt.datetime, dt.datetime]] = (
    dt.datetime(2025, 8, 10, tzinfo=dt.UTC),
    dt.datetime(2026, 6, 25, tzinfo=dt.UTC),
)
TARGET_LEAD_HOURS: Final[tuple[float, float]] = (6.0, 72.0)
"""A targeted shift ends this many hours before the start of an event day."""


def position(*, moment: dt.datetime) -> int:
    """The grid position of a time (the harness labels a half-hour by its start)."""
    return int((moment - dat.GRID_START).total_seconds() // 1800)


@dataclass(frozen=True)
class Shift:
    """One injected load shift, in grid positions.

    Attributes:
        donor: Series id of the Primary that loses load.
        receiver: Series id of the Primary that gains it.
        onset: First position inside the shift.
        end: First position after the shift.
        moved_mw: Megawatts moved.
        detected: Whether the stand-in for the carried mask finds this shift.
    """

    donor: int
    receiver: int
    onset: int
    end: int
    moved_mw: float
    detected: bool


@lru_cache(maxsize=1)
def children() -> pl.DataFrame:
    """Series id, D-label, BSP name, and effective capacity of the 15 Primaries of the harness."""
    series = pl.read_parquet(h.SERIES_PATH).select(
        "time_series_id", "site", "effective_capacity_mw"
    )
    monitors = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        pl.col("TimeSeriesID").alias("time_series_id"),
        pl.col("SubstationNumber").alias("substation"),
    )
    substations = pl.read_csv(HIERARCHY_DIR / "Substations.csv").select(
        pl.col("substation_number").alias("substation"), "bsp_name"
    )
    return (
        series.join(monitors, on="time_series_id")
        .join(substations, on="substation", how="left")
        .filter(pl.col("time_series_id").is_in(h.SERIES))
        .sort("time_series_id")
    )


def bsp_of_series() -> dict[int, str]:
    """BSP name of each harness series."""
    t = children()
    return dict(zip(t["time_series_id"].to_list(), t["bsp_name"].to_list(), strict=True))


@lru_cache(maxsize=1)
def candidate_pairs() -> dict[str, list[tuple[int, int]]]:
    """Pairs of harness series that are interconnected or share a hand-filtered real event.

    Returns:
        `within` (same BSP) and `cross` (different BSPs), each sorted, smaller id first.
    """
    t = children()
    by_substation = dict(zip(t["substation"].to_list(), t["time_series_id"].to_list(), strict=True))
    pairs = {
        tuple(sorted((by_substation[a], by_substation[b])))
        for a, b in pl.read_csv(HIERARCHY_DIR / "Primary Substation Interconnections.csv").rows()
        if a in by_substation and b in by_substation
    }
    events = pl.read_parquet(
        dat.DATA
        / "00101-ro-nc-detector-on-real-hand-filtered-switching-events"
        / "event_table.parquet"
    )
    for text in events["series"].to_list():
        ids = [int(x) for x in text.split(",")]
        if len(ids) == 2 and all(i in h.SERIES for i in ids):
            pairs.add(tuple(sorted(ids)))
    bsp = bsp_of_series()
    out: dict[str, list[tuple[int, int]]] = {"within": [], "cross": []}
    for a, b in sorted(pairs):
        out["within" if bsp[a] == bsp[b] else "cross"].append((a, b))
    return out


@lru_cache(maxsize=1)
def donor_mean_mw() -> dict[int, float]:
    """Mean load of each series in MW over the readings before the scored window."""
    stop = position(moment=dt.datetime(2025, 9, 1, tzinfo=dt.UTC))
    out = {}
    for s in h.SERIES:
        d = h.load_series(series=s)
        out[s] = float(np.nanmean(d.actual[:stop])) * d.capacity_mw / 100.0
    return out


def draw_world(
    *,
    kind: str,
    placement: str,
    index: int,
    event_days: dict[str, np.ndarray],
) -> list[Shift]:
    """Draw one world's shifts; the seed names the kind, the placement, and the index.

    Args:
        kind: `within` or `cross`.
        placement: `random` (end anywhere in the window) or `targeted` (end 6 to 72 hours before
            the start of a day with a scored event at the BSP of the donor).
        index: World number.
        event_days: Grid positions of the first half-hour of each event day, by BSP name.

    Returns:
        The shifts, sorted by onset.
    """
    rng = np.random.default_rng([271, KINDS.index(kind), PLACEMENTS.index(placement), index])
    pairs = candidate_pairs()[kind]
    bsp = bsp_of_series()
    loads = donor_mean_mw()
    lo, hi = (position(moment=m) for m in RANDOM_WINDOW)
    shifts = []
    for _ in range(SHIFTS_PER_WORLD):
        a, b = pairs[int(rng.integers(len(pairs)))]
        donor, receiver = (a, b) if rng.random() < 0.5 else (b, a)
        duration = int(rng.choice(DURATION_DAYS)) * 48
        share = float(rng.choice(SHARES))
        if placement == "random":
            end = int(rng.integers(lo + duration, hi))
        else:
            days = event_days[bsp[donor] if kind == "within" else bsp[(a, b)[int(rng.integers(2))]]]
            day = int(days[int(rng.integers(len(days)))])
            end = day - round(rng.uniform(*TARGET_LEAD_HOURS) * 2)
        shifts.append(
            Shift(
                donor, receiver, end - duration, end, share * loads[donor], bool(rng.random() < 0.5)
            )
        )
    return sorted(shifts, key=lambda s: s.onset)


def inject(*, data: h.SeriesData, shifts: list[Shift]) -> h.SeriesData:
    """The series with every shift added to or taken from its readings (NaN stays NaN)."""
    actual = data.actual.copy()
    points = 100.0 / data.capacity_mw
    for s in shifts:
        if s.donor == data.series:
            actual[s.onset : s.end] -= s.moved_mw * points
        if s.receiver == data.series:
            actual[s.onset : s.end] += s.moved_mw * points
    return replace(data, actual=actual)


def spans_for(*, shifts: list[Shift], series: int, arm: str) -> list[h.Span]:
    """The mask spans an arm holds for a series: the shifts it knows, as it would know them."""
    out = []
    for s in shifts:
        if series not in (s.donor, s.receiver):
            continue
        if arm == "PERFECT":
            out.append(h.Span(s.onset, s.end, s.onset, s.end))
        elif arm == "PERFECT_LAG" or (arm == "LB20_LIKE" and s.detected):
            start = s.onset + ONSET_LAG
            out.append(h.Span(s.onset, s.end, start, max(s.end + h.END_LAG, start + 1)))
    return out


@cache
def real_k0(*, series: int) -> np.ndarray:
    """The no-mask correction of one series on the real record, shape (origins, leads)."""
    band = h.band_grid()
    data = h.load_series(series=series)
    return h.k0_correction(trace=h.k0_trace(data=data, band=band), band=band)


@lru_cache(maxsize=1)
def gam_only_deltas() -> dict[str, np.ndarray]:
    """GAM_ONLY's forecast change against the K0 base at each BSP: minus the real K0 correction."""
    bsp = bsp_of_series()
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in children().iter_rows(named=True)
    }
    _, _, targets = h.origin_grid()
    out = {n: np.zeros(targets.shape, dtype=np.float64) for n in sorted(set(bsp.values()))}
    for series in h.SERIES:
        out[bsp[series]] -= real_k0(series=series).astype(np.float64) * capacity[series] / 100.0
    return out


def node_deltas(
    *, shifts: list[Shift]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Each arm's forecast change at each BSP, and the change in each BSP's observed power.

    The base is experiment 48's node forecast, which stands for the filter with no knowledge on the
    real record (K0). An arm's change is the sum, over the Primaries it re-forecasts, of its
    correction on the injected readings minus K0's correction on the real record.

    Args:
        shifts: The world's shifts.

    Returns:
        The forecast change in MW by arm and BSP name, shape (origins, leads), and the change in
        the BSP's observed power in MW by BSP name, shape (grid positions,).
    """
    band = h.band_grid()
    bsp = bsp_of_series()
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in children().iter_rows(named=True)
    }
    _, _, targets = h.origin_grid()
    names = sorted(set(bsp.values()))
    forecast = {arm: {n: np.zeros(targets.shape, dtype=np.float64) for n in names} for arm in ARMS}
    forecast["GAM_ONLY"] = {n: v.copy() for n, v in gam_only_deltas().items()}
    observed = {n: np.zeros(h.N_GRID, dtype=np.float64) for n in names}
    affected = sorted({s.donor for s in shifts} | {s.receiver for s in shifts})
    for series in affected:
        base = h.load_series(series=series)
        injected = inject(data=base, shifts=shifts)
        weight = capacity[series] / 100.0
        for shift in shifts:
            sign = (shift.receiver == series) - (shift.donor == series)
            if sign:
                observed[bsp[series]][shift.onset : shift.end] += sign * shift.moved_mw
        trace = h.k0_trace(data=injected, band=band)
        reference = real_k0(series=series)
        for arm in ARMS:
            if arm == "GAM_ONLY":
                continue
            if arm == "K0":
                correction = h.k0_correction(trace=trace, band=band)
            else:
                spans = spans_for(shifts=shifts, series=series, arm=arm)
                correction = (
                    h.mask_correction(data=injected, spans=spans, band=band, k0=trace)
                    if spans
                    else h.k0_correction(trace=trace, band=band)
                )
            forecast[arm][bsp[series]] += (correction - reference).astype(np.float64) * weight
    return forecast, observed
