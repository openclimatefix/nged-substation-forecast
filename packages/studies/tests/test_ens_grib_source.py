from datetime import date

import pytest
from studies.ens_grib_source import (
    GRIB_FILE_NAMES,
    IdxEntry,
    choose_pilot_dates,
    complete_dates,
    find_gaps,
    parse_idx,
    parse_listing,
    prefix_range_header,
)

CONTROL_LINE = (
    '{"type": "cf", "param": "z", "step": 3, "levtype": "pl", "_offset": 2076642.0, '
    '"_length": 2076642, "levelist": 500}'
)
PERTURBED_LINE = (
    '{"type": "pf", "param": "2t", "step": 0, "levtype": "sfc", "_offset": 0.0, '
    '"_length": 100, "number": 26}'
)
PREFIX = "dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens"


def _all_files(*, day: str, leave_out: str | None = None) -> str:
    names = [name for grib in GRIB_FILE_NAMES for name in (grib, f"{grib}.idx")]
    return "\n".join(
        f"2026-06-07 21:05:44 1234 {PREFIX}/{day}/{name}" for name in names if name != leave_out
    )


def test_a_control_line_has_member_zero_and_an_integer_offset():
    (entry,) = parse_idx(text=CONTROL_LINE + "\n")

    assert entry == IdxEntry(
        parameter="z",
        level_type="pl",
        level=500,
        step_hours=3,
        member=0,
        offset=2_076_642,
        length=2_076_642,
    )


def test_a_perturbed_surface_line_has_its_member_and_level_zero():
    (entry,) = parse_idx(text=PERTURBED_LINE)

    assert (entry.member, entry.level) == (26, 0)


def test_messages_that_tile_the_file_have_no_gaps():
    entries = parse_idx(text=CONTROL_LINE)
    first = IdxEntry(
        parameter="t",
        level_type="pl",
        level=500,
        step_hours=0,
        member=0,
        offset=0,
        length=2_076_642,
    )

    assert find_gaps(entries=[entries[0], first]) == []


def test_a_gap_and_a_first_message_not_at_zero_are_reported():
    late = IdxEntry(
        parameter="t", level_type="pl", level=500, step_hours=0, member=0, offset=10, length=5
    )
    after_gap = IdxEntry(
        parameter="t", level_type="pl", level=500, step_hours=3, member=0, offset=20, length=5
    )

    assert len(find_gaps(entries=[late, after_gap])) == 2


def test_the_range_header_starts_at_the_message_and_is_inclusive():
    (entry,) = parse_idx(text=CONTROL_LINE)

    assert prefix_range_header(entry=entry, prefix_bytes=100) == "bytes=2076642-2076741"


def test_a_prefix_longer_than_the_message_is_refused():
    (entry,) = parse_idx(text=PERTURBED_LINE)

    with pytest.raises(ValueError, match="outside"):
        prefix_range_header(entry=entry, prefix_bytes=101)


def test_a_date_is_complete_only_with_every_file_and_sidecar():
    listing = (
        _all_files(day="2023-06-26")
        + "\n"
        + _all_files(day="2023-06-27", leave_out="pf_pl.grib.idx")
    )

    assert complete_dates(files_by_date=parse_listing(text=listing)) == [date(2023, 6, 26)]


def test_a_listing_line_that_is_not_in_a_date_folder_is_skipped():
    listing = f"2026-01-01 00:00:00 5 {PREFIX}/README.md\n" + _all_files(day="2023-06-26")

    assert list(parse_listing(text=listing)) == [date(2023, 6, 26)]


def _year_of_dates() -> list[date]:
    return [date.fromordinal(date(2022, 1, 1).toordinal() + offset) for offset in range(365)]


def test_the_same_seed_draws_the_same_dates_and_the_always_dates_are_added_once():
    complete = _year_of_dates()
    always = (date(2022, 3, 3),)

    first = choose_pilot_dates(
        complete=complete,
        first=date(2022, 1, 1),
        last=date(2022, 12, 31),
        sample_size=20,
        seed=1,
        always=always,
    )
    second = choose_pilot_dates(
        complete=complete,
        first=date(2022, 1, 1),
        last=date(2022, 12, 31),
        sample_size=20,
        seed=1,
        always=always,
    )

    assert first == second
    assert date(2022, 3, 3) in first
    assert first == sorted(set(first))
    assert len(first) in (20, 21)


def test_an_always_date_that_is_not_complete_is_refused():
    with pytest.raises(ValueError, match="not complete"):
        choose_pilot_dates(
            complete=_year_of_dates(),
            first=date(2022, 1, 1),
            last=date(2022, 12, 31),
            sample_size=2,
            seed=1,
            always=(date(2030, 1, 1),),
        )


def test_too_few_dates_in_range_is_refused():
    with pytest.raises(ValueError, match="complete dates"):
        choose_pilot_dates(
            complete=_year_of_dates(),
            first=date(2022, 1, 1),
            last=date(2022, 1, 5),
            sample_size=6,
            seed=1,
            always=(),
        )
