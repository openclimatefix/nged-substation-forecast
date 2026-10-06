"""Tests for `studies.ukv_ceda_stores`.

Every store is a dictionary of arrays whose values encode the store, the slot, and the lead they
sit at, so a test that reads the wrong store, the wrong run, or the wrong lead reads a different
number. The three stores split the slot axis at synthetic boundaries (8 and 16 slots), unlike the
real stores' boundaries, so no test passes by reading the store that the real record happens to
make the last one.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.ukv_ceda_profiles import (
    DEFAULT_PROFILE,
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_PARTIAL,
)
from studies.ukv_ceda_stores import (
    ERA_FIRST_MONTHS,
    STRADDLING_MONTHS,
    ArrayGroup,
    Store,
    check_era_codes,
    instant_runs,
    make_stores,
    read_at_instants,
    run_status,
    store_index,
    usable_hours,
    usable_instants,
    with_ukv_eras,
)

EPOCH: Final[datetime] = DEFAULT_PROFILE.slot_epoch
SLOT_COUNTS: Final[tuple[int, int, int]] = (8, 16, 24)
N_LEADS: Final[int] = 6
N_CELLS: Final[int] = 3


def _value(*, store: int, slot: int, lead: int, cell: int) -> float:
    return 10_000.0 * store + 100.0 * slot + 10.0 * lead + cell


def _stores(*, statuses: dict[int, int] | None = None) -> tuple[Store, ...]:
    """Three stores whose slots 0 to 23 are complete, apart from the given statuses."""
    overrides = statuses or {}
    groups: list[ArrayGroup] = []
    status_arrays: list[np.ndarray] = []
    first = 0
    for index, count in enumerate(SLOT_COUNTS):
        array = np.array(
            [
                [[_value(store=index, slot=s, lead=lead, cell=c) for c in range(N_CELLS)]]
                for s in range(count)
                for lead in range(N_LEADS)
            ]
        ).reshape(count, N_LEADS, N_CELLS)
        groups.append({"wind": array})
        status = np.zeros(count, dtype=np.int8)
        status[first:] = STATUS_COMPLETE
        for slot, code in overrides.items():
            if first <= slot < count:
                status[slot] = code
        status_arrays.append(status)
        first = count
    return make_stores(groups=groups, statuses=status_arrays)


def _instants(*hours_after_epoch: int) -> pl.Series:
    return pl.Series(
        [EPOCH + timedelta(hours=h) for h in hours_after_epoch], dtype=pl.Datetime("us", "UTC")
    )


# --- the freshest run -----------------------------------------------------------------------------


def test_an_hour_at_a_runs_own_start_reads_that_run_at_lead_zero_never_the_run_before_at_lead_six():
    runs = instant_runs(instants=_instants(5, 6, 7, 11, 12))

    assert runs["slot"].to_list() == [0, 1, 1, 1, 2]
    assert runs["lead_hours"].to_list() == [5, 0, 1, 5, 0]
    assert runs["init_time"].to_list() == [EPOCH + timedelta(hours=h) for h in (0, 6, 6, 6, 12)]


def test_an_instant_off_the_hour_raises():
    instants = pl.Series([EPOCH + timedelta(minutes=30)], dtype=pl.Datetime("us", "UTC"))

    with pytest.raises(ValueError, match="whole hour"):
        instant_runs(instants=instants)


# --- the stores -----------------------------------------------------------------------------------


def test_each_store_owns_the_slots_after_the_store_before_it():
    stores = _stores()

    assert [(s.first_slot, s.end_slot) for s in stores] == [(0, 8), (8, 16), (16, 24)]
    assert [store_index(stores=stores, slot=slot) for slot in (0, 7, 8, 15, 16, 23, 24, -1)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        None,
        None,
    ]


def test_a_status_array_no_longer_than_the_one_before_it_raises():
    groups: list[ArrayGroup] = [{}, {}]

    with pytest.raises(ValueError, match="cannot extend"):
        make_stores(
            groups=groups, statuses=[np.zeros(8, dtype=np.int8), np.zeros(8, dtype=np.int8)]
        )


def test_each_slot_is_read_from_the_store_that_owns_it():
    # Slots 1, 9 and 20 lie in the first, second and third store. A reader that always used the
    # last store, or the first, would read another store's 10,000s digit.
    hours = _instants(6 * 1 + 2, 6 * 9 + 3, 6 * 20 + 5)

    reads = read_at_instants(
        stores=_stores(), instants=hours, variables=["wind"], cells=np.array([2, 0])
    )

    assert reads.values["wind"].tolist() == [
        [_value(store=0, slot=1, lead=2, cell=c) for c in (2, 0)],
        [_value(store=1, slot=9, lead=3, cell=c) for c in (2, 0)],
        [_value(store=2, slot=20, lead=5, cell=c) for c in (2, 0)],
    ]
    assert reads.lead_hours.tolist() == [2, 3, 5]
    assert reads.usable.all()


def test_a_status_is_read_from_the_owning_store_only():
    stores = _stores(statuses={3: STATUS_PARTIAL})
    # The third store's array holds a complete status at slot 3, which the first store owns.
    stores[2].statuses[3] = STATUS_COMPLETE

    assert run_status(stores=stores, slots=np.array([3])).tolist() == [STATUS_PARTIAL]


# --- runs that cannot serve an hour ---------------------------------------------------------------


@pytest.mark.parametrize("code", [STATUS_PARTIAL, STATUS_MISSING, 0])
def test_a_run_that_is_not_complete_is_dropped_and_never_served_by_an_older_run(code: int):
    stores = _stores(statuses={4: code})
    hours = _instants(6 * 4 + 1, 6 * 3 + 1)

    reads = read_at_instants(stores=stores, instants=hours, variables=["wind"], cells=np.array([0]))

    assert reads.usable.tolist() == [False, True]
    assert np.isnan(reads.values["wind"][0, 0])
    assert reads.values["wind"][1, 0] == _value(store=0, slot=3, lead=1, cell=0)


def test_an_hour_beyond_the_last_store_is_unusable():
    last_slot = SLOT_COUNTS[-1]

    usable = usable_instants(stores=_stores(), instants=_instants(6 * last_slot - 1, 6 * last_slot))

    assert usable.tolist() == [True, False]


def test_an_hour_built_from_two_instants_needs_both_runs_complete():
    # The hour labelled 06 UTC averages the 05 UTC instant (run 0, lead 5) and the 06 UTC instant
    # (run 1, lead 0). Run 0 is partial, so the hour is unusable although run 1 is intact.
    stores = _stores(statuses={0: STATUS_PARTIAL})
    hours = _instants(6, 7)

    both_ends = usable_hours(stores=stores, hours=hours, offsets_hours=(-1, 0))
    label_only = usable_hours(stores=stores, hours=hours, offsets_hours=(0,))

    assert both_ends.tolist() == [False, True]
    assert label_only.tolist() == [True, True]


# --- eras -----------------------------------------------------------------------------------------


def _daily_rows() -> pl.DataFrame:
    days = pl.datetime_range(
        datetime(2019, 9, 17, 12, tzinfo=UTC),
        datetime(2026, 4, 30, 12, tzinfo=UTC),
        interval="1d",
        time_zone="UTC",
        eager=True,
    )
    return pl.concat(
        pl.DataFrame({"site": site, "time": days}) for site in ("A", "B")
    ).with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def test_the_straddling_months_are_dropped_and_the_eras_are_exactly_three():
    cut, _ = with_ukv_eras(frame=_daily_rows())

    assert set(cut["era_code"].unique().to_list()) == {0, 1, 2}
    assert not set(cut["month"].unique().to_list()) & set(STRADDLING_MONTHS)
    era_by_month = dict(cut.group_by("month").agg(pl.col("era_code").first()).iter_rows())
    assert era_by_month["2019-11"] == 0
    assert era_by_month[ERA_FIRST_MONTHS[0]] == 1
    assert era_by_month["2025-12"] == 1
    assert era_by_month[ERA_FIRST_MONTHS[1]] == 2


def test_a_row_set_with_no_post_upgrade_months_does_not_pass_as_three_eras():
    rows = _daily_rows().filter(pl.col("month") < "2026-01")

    with pytest.raises(ValueError, match=r"era_code holds \[0, 1\]"):
        with_ukv_eras(frame=rows)


def test_a_surviving_straddling_month_raises():
    cut, _ = with_ukv_eras(frame=_daily_rows())
    straddling = cut.head(1).with_columns(month=pl.lit(STRADDLING_MONTHS[1]))

    with pytest.raises(ValueError, match="straddling"):
        check_era_codes(frame=pl.concat([cut, straddling]))
