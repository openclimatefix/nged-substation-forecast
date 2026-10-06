"""Read the three UKV-on-CEDA stores of 6-hourly runs at the hours a study's rows need.

**Each hour is read from the freshest run that has started at or before the hour.** The runs start
at 00, 06, 12, and 18 UTC, so an hour's lead is 0 to 5 hours. The hour at a run's own start reads
that run at lead 0, never the previous run at lead 6.

**The record is split across three stores, each holding a run only in its own range of slots.** The
second store starts at the first store's slot count and the third at the second's, so
`make_stores` takes each store's range from the lengths of the status arrays. A slot's status is
read from the store that owns the slot only, because every store carries a status array as long as
its own axis and holds `0` (never archived) at the slots before its range.

**A run that is not complete is never replaced by an older run.** Serving an hour from the run
before would change the hour's lead part-way through the record, so the hour is reported as
unusable and the caller drops it. A run is unusable when its status is partial or missing, when CEDA
holds no directory for it (status 0 inside a store's range), and when it lies beyond the last
store's last slot.

**An hour built from the two instants at its ends needs both instants' runs to be complete.** The
05 UTC instant reads the 00 UTC run at lead 5 and the 06 UTC instant reads the 06 UTC run at lead
0, so a missing 00 UTC run removes the hour labelled 06 UTC although the 06 UTC run is intact.

`studies.ukv_ceda_profiles` owns the product's name, slot epoch, cycle, and status codes, and this
module takes them from there.

**The UKV physics versions are era boundaries.** `with_ukv_eras` drops the two months that straddle
a change, labels the three eras, and cuts the folds inside each era.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple, Protocol

import numpy as np
import polars as pl

from studies.cross_validation import (
    UKV_UPGRADE_MONTH,
    calendar_month_coverage,
    cut_eras,
    raise_on_uncovered_months,
    search_fold_offsets,
)
from studies.ukv_ceda_profiles import DEFAULT_PROFILE, STATUS_COMPLETE, Profile

SECONDS_PER_HOUR: Final[int] = 3600

ERA_FIRST_MONTHS: Final[tuple[str, str]] = ("2020-01", UKV_UPGRADE_MONTH)
"""The first whole month of the second and third eras, in `%Y-%m` form.

The Met Office's Parallel Suite 43 took effect on 2019-12-04, so the first era is the months before
2019-12. The second era runs from 2020-01 to 2025-12, and the third starts at the first whole month
after the 2026-01-21 upgrade.
"""

STRADDLING_MONTHS: Final[tuple[str, str]] = ("2019-12", "2026-01")
"""The two months in which a UKV physics version changed part-way, dropped from every row."""

EXPECTED_ERA_CODES: Final[frozenset[int]] = frozenset({0, 1, 2})
"""The `era_code` values a study's rows hold, one per UKV era."""


class ArrayGroup(Protocol):
    """A group of arrays indexed by name, as a Zarr group or a dictionary of arrays is."""

    def __getitem__(self, name: str, /) -> Any:
        """Return the array of this name, indexed by slot, then lead, then cell."""
        ...


class Store(NamedTuple):
    """One UKV-on-CEDA store, and the slots it owns.

    Attributes:
        group: The store's arrays, each of shape (slots, leads, cells).
        statuses: Each slot's status in this store: `0` never archived, else a `STATUS_*` code.
        first_slot: The first slot this store owns.
        end_slot: One past the last slot this store owns.
    """

    group: ArrayGroup
    statuses: np.ndarray
    first_slot: int
    end_slot: int


def make_stores(
    *, groups: Sequence[ArrayGroup], statuses: Sequence[np.ndarray]
) -> tuple[Store, ...]:
    """Pair each store's arrays with the range of slots it owns.

    Args:
        groups: The stores' arrays, in date order.
        statuses: Each store's status array, in the same order. The lengths must strictly increase,
            because each store's axis runs from slot 0 to the end of its own range.

    Returns:
        One `Store` per input, the first owning slots from 0 and each later one owning the slots
        after the store before it.

    Raises:
        ValueError: If the two sequences differ in length, or a status array is no longer than the
            one before it.
    """
    if len(groups) != len(statuses):
        msg = f"{len(groups)} stores but {len(statuses)} status arrays"
        raise ValueError(msg)
    stores: list[Store] = []
    first_slot = 0
    for group, status in zip(groups, statuses, strict=True):
        if len(status) <= first_slot:
            msg = f"a status array of {len(status)} slots cannot extend the stores before it"
            raise ValueError(msg)
        stores.append(
            Store(group=group, statuses=status, first_slot=first_slot, end_slot=len(status))
        )
        first_slot = len(status)
    return tuple(stores)


def store_index(*, stores: Sequence[Store], slot: int) -> int | None:
    """Return the position of the store that owns a slot.

    Args:
        stores: The stores, from `make_stores`.
        slot: A slot of the `init_time` axis.

    Returns:
        The store's position, or `None` for a slot before the first store or beyond the last.
    """
    for position, store in enumerate(stores):
        if store.first_slot <= slot < store.end_slot:
            return position
    return None


def instant_runs(*, instants: pl.Series, profile: Profile = DEFAULT_PROFILE) -> pl.DataFrame:
    """Map each instant to the freshest run that started at or before it.

    Args:
        instants: Whole-hour instants, as a UTC datetime series.
        profile: The product's slot epoch and cycle.

    Returns:
        One row per instant with `instant`, `slot`, `init_time` and `lead_hours`. An instant at a
        run's own start has lead 0.

    Raises:
        ValueError: If an instant is not on a whole hour.
    """
    cycle_seconds = profile.cycle_hours * SECONDS_PER_HOUR
    epoch_seconds = int(profile.slot_epoch.timestamp())
    seconds = instants.dt.epoch("s").to_numpy().astype(np.int64)
    slots = (seconds - epoch_seconds) // cycle_seconds
    since_run = seconds - (epoch_seconds + slots * cycle_seconds)
    if (since_run % SECONDS_PER_HOUR).any():
        msg = "every instant must lie on a whole hour"
        raise ValueError(msg)
    init_seconds = pl.Series(seconds - since_run, dtype=pl.Int64)
    return pl.DataFrame(
        {
            "instant": instants,
            "slot": pl.Series(slots, dtype=pl.Int64),
            "init_time": pl.from_epoch(init_seconds, time_unit="s").dt.replace_time_zone("UTC"),
            "lead_hours": pl.Series(since_run // SECONDS_PER_HOUR, dtype=pl.Int64),
        }
    )


def run_status(*, stores: Sequence[Store], slots: np.ndarray) -> np.ndarray:
    """Return each slot's status, read from the store that owns the slot.

    Args:
        stores: The stores, from `make_stores`.
        slots: The slots to look up.

    Returns:
        An `int8` array, `0` for a slot no store owns.
    """
    status = np.zeros(len(slots), dtype=np.int8)
    for store in stores:
        owned = (slots >= store.first_slot) & (slots < store.end_slot)
        status[owned] = store.statuses[slots[owned]]
    return status


def usable_instants(
    *, stores: Sequence[Store], instants: pl.Series, profile: Profile = DEFAULT_PROFILE
) -> np.ndarray:
    """Say whether each instant's freshest run is complete.

    Args:
        stores: The stores, from `make_stores`.
        instants: Whole-hour instants, as a UTC datetime series.
        profile: The product's slot epoch and cycle.

    Returns:
        A boolean array, true where the freshest run has status `STATUS_COMPLETE`.
    """
    runs = instant_runs(instants=instants, profile=profile)
    return run_status(stores=stores, slots=runs["slot"].to_numpy()) == STATUS_COMPLETE


def usable_hours(
    *,
    stores: Sequence[Store],
    hours: pl.Series,
    offsets_hours: Sequence[int],
    profile: Profile = DEFAULT_PROFILE,
) -> np.ndarray:
    """Say whether every instant an hour is built from has a complete run.

    Args:
        stores: The stores, from `make_stores`.
        hours: The hours' labels, as a UTC datetime series.
        offsets_hours: Each instant the hour reads, as an offset from its label: `(0,)` for an
            instantaneous value at the label, and `(-1, 0)` for the mean of the instants at both
            ends of the hour ending at the label.
        profile: The product's slot epoch and cycle.

    Returns:
        A boolean array, true where every instant's freshest run is complete.
    """
    usable = np.ones(len(hours), dtype=bool)
    for offset in offsets_hours:
        usable &= usable_instants(
            stores=stores, instants=hours.dt.offset_by(f"{offset}h"), profile=profile
        )
    return usable


class InstantReads(NamedTuple):
    """The values read at a set of instants.

    Attributes:
        usable: Whether each instant's freshest run is complete.
        lead_hours: Each instant's lead in the freshest run.
        values: Each variable's values, shape (instants, cells), NaN where the run is unusable.
    """

    usable: np.ndarray
    lead_hours: np.ndarray
    values: dict[str, np.ndarray]


def read_at_instants(
    *,
    stores: Sequence[Store],
    instants: pl.Series,
    variables: Sequence[str],
    cells: np.ndarray,
    profile: Profile = DEFAULT_PROFILE,
) -> InstantReads:
    """Read variables at the given cells and instants, each from its freshest complete run.

    Args:
        stores: The stores, from `make_stores`.
        instants: Whole-hour instants, as a UTC datetime series.
        variables: The array names to read.
        cells: Flat cell indices to keep.
        profile: The product's slot epoch and cycle.

    Returns:
        The usable flags, the leads, and the values. An unusable instant holds NaN in every
        variable, so a run that is partial or missing is never read.
    """
    runs = instant_runs(instants=instants, profile=profile)
    slots = runs["slot"].to_numpy()
    leads = runs["lead_hours"].to_numpy()
    usable = run_status(stores=stores, slots=slots) == STATUS_COMPLETE
    values = {name: np.full((len(instants), len(cells)), np.nan) for name in variables}
    for slot in np.unique(slots[usable]):
        position = store_index(stores=stores, slot=int(slot))
        if position is None:
            continue
        rows = np.flatnonzero(slots == slot)
        for name in variables:
            block = np.asarray(stores[position].group[name][int(slot)])
            values[name][rows] = block[np.ix_(leads[rows], cells)]
    return InstantReads(usable=usable, lead_hours=leads, values=values)


def check_era_codes(*, frame: pl.DataFrame) -> None:
    """Raise unless a frame holds exactly the three eras and no month that straddles a change.

    Args:
        frame: Rows carrying `era_code` and `month`.

    Raises:
        ValueError: If `era_code` is not exactly `EXPECTED_ERA_CODES`, or a straddling month
            survives.
    """
    codes = set(frame["era_code"].unique().to_list())
    if codes != EXPECTED_ERA_CODES:
        msg = f"era_code holds {sorted(codes)}, not {sorted(EXPECTED_ERA_CODES)}"
        raise ValueError(msg)
    kept = sorted(set(frame["month"].unique().to_list()) & set(STRADDLING_MONTHS))
    if kept:
        msg = f"rows from the straddling months {kept} survive"
        raise ValueError(msg)


def ukv_fold_designs(*, frame: pl.DataFrame) -> list[Mapping[int, int]]:
    """List every fold rotation that leaves every calendar month a training row.

    Args:
        frame: Rows carrying `site`, `month` (a `%Y-%m` string) and `time`.

    Returns:
        The rotations `search_fold_offsets` finds, fewest rotations first. The list is empty if no
        rotation covers every calendar month.
    """
    kept = frame.filter(~pl.col("month").is_in(STRADDLING_MONTHS))
    return search_fold_offsets(frame=kept, first_months=ERA_FIRST_MONTHS)


def with_ukv_eras(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, Mapping[int, int]]:
    """Drop the straddling months, label the three eras, and cut folds that cover every month.

    The straddling months are dropped before the eras are cut, because otherwise the second era
    would hold the part of 2026-01 that carries the upgraded UKV.

    Args:
        frame: Rows carrying `site`, `month` (a `%Y-%m` string) and `time`.

    Returns:
        The frame with `era_code`, `era` and `fold`, and the rotation of each era's folds.

    Raises:
        ValueError: If the rows do not hold all three eras, or no fold rotation leaves every
            calendar month with a training row.
    """
    designs = ukv_fold_designs(frame=frame)
    if not designs:
        msg = "no fold rotation leaves every calendar month with a training row"
        raise ValueError(msg)
    offsets = dict(designs[0])
    kept = frame.filter(~pl.col("month").is_in(STRADDLING_MONTHS))
    cut = cut_eras(frame=kept, first_months=ERA_FIRST_MONTHS, fold_offsets=offsets)
    check_era_codes(frame=cut)
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=cut))
    return cut, offsets
