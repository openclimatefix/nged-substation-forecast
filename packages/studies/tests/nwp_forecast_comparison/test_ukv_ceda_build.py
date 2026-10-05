from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
import zarr
import zarr.storage

_STUDIES_DIR = Path(__file__).resolve().parents[3] / "studies"

import build_ukv_ceda_inputs as build  # noqa: E402
from nwp_forecast_comparison import DomainType  # noqa: E402
from studies.baselines import haurwitz_w_m2  # noqa: E402
from studies.solar import zenith  # noqa: E402

NAN = float("nan")


def _series(*, values: dict[int, float], n_runs: int = 1) -> np.ndarray:
    """One run's values at the given leads and NaN at every other lead."""
    row = np.full(build.N_LEADS, NAN)
    for lead, value in values.items():
        row[lead] = value
    return np.tile(row, (n_runs, 1))


def _native(*, value_of_lead: Callable[[int], float]) -> np.ndarray:
    return _series(values={int(lead): value_of_lead(int(lead)) for lead in build.NATIVE_LEADS})


# --- the leads the store holds --------------------------------------------------------------------


def test_the_store_lacks_exactly_the_hours_between_its_3_hourly_steps_after_lead_48():
    assert build.FILLED_LEADS.tolist()[:6] == [49, 50, 52, 53, 55, 56]
    assert build.FILLED_LEADS.tolist()[-2:] == [118, 119]
    assert all(lead > 48 for lead in build.FILLED_LEADS)
    assert build.N_LEADS == 121
    assert set(build.NATIVE_LEADS.tolist()) | set(build.FILLED_LEADS.tolist()) == set(range(121))
    assert not set(build.NATIVE_LEADS.tolist()) & set(build.FILLED_LEADS.tolist())


def test_the_anchors_are_every_third_hour_from_lead_48_to_120():
    assert build.ANCHOR_LEADS.tolist() == list(range(48, 121, 3))


# --- fill_linear ----------------------------------------------------------------------------------


def test_a_straight_line_is_rebuilt_exactly_and_native_leads_are_untouched():
    values = _native(value_of_lead=lambda lead: 2.0 * lead + 1.0)

    filled = build.fill_linear(values=values)

    assert np.allclose(filled[0], 2.0 * np.arange(build.N_LEADS) + 1.0)
    native = build.NATIVE_LEADS
    assert np.array_equal(filled[0, native], values[0, native])


def test_a_missing_anchor_blanks_only_the_rebuilt_leads_either_side_of_it():
    values = _native(value_of_lead=float)
    values[0, 51] = NAN

    filled = build.fill_linear(values=values)

    blanked = np.flatnonzero(np.isnan(filled[0])).tolist()
    assert blanked == [49, 50, 51, 52, 53]


def test_the_bracketing_test_needs_both_anchors():
    values = _native(value_of_lead=float)
    values[0, 54] = NAN

    finite = build.bracketing_anchors_finite(values=values)[0]

    leads = build.FILLED_LEADS
    assert not finite[np.isin(leads, [52, 53, 55, 56])].any()
    assert finite[np.isin(leads, [49, 50, 58, 59])].all()


# --- fill_radiation -------------------------------------------------------------------------------


def _clear_sky(*, daylight_utc_hours: range, level: float = 600.0) -> np.ndarray:
    """A clear sky of `level` W/m2 in the given UTC hours of every day, for a 03 UTC run."""
    utc_hour = (build.RUN_HOUR + np.arange(build.N_LEADS)) % 24
    return np.tile(np.where(np.isin(utc_hour, list(daylight_utc_hours)), level, 0.0), (1, 1))


def test_a_fixed_share_of_the_clear_sky_is_rebuilt_as_that_share_of_each_leads_clear_sky():
    clear = np.tile(
        [600.0 * max(0.0, np.sin(np.pi * ((3 + lead) % 24 - 6) / 12)) for lead in range(121)],
        (1, 1),
    )
    snapshots = np.where(np.isin(np.arange(121), build.NATIVE_LEADS), 0.5 * clear, NAN)

    filled = build.fill_radiation(snapshots=snapshots, clear_sky=clear)

    assert np.allclose(filled, 0.5 * clear)


def test_a_rising_clear_sky_index_is_rebuilt_at_each_leads_own_instant():
    # A constant index would hide an anchor position shifted by 1.5 hours (ENS's mistake, where a
    # value is the mean of the step ending at its label); a rising index would not.
    clear = _clear_sky(daylight_utc_hours=range(24))
    index = 0.2 + 0.005 * (np.arange(build.N_LEADS) - 48)
    snapshots = np.where(np.isin(np.arange(121), build.NATIVE_LEADS), index * 600.0, NAN)[None, :]

    filled = build.fill_radiation(snapshots=snapshots, clear_sky=clear)

    assert filled[0, build.FILLED_LEADS] == pytest.approx(index[build.FILLED_LEADS] * 600.0)


def test_the_clear_sky_is_evaluated_at_each_leads_own_instant():
    init = datetime(2026, 6, 21, 3, tzinfo=UTC)
    latitude, longitude = 52.0, -1.0
    leads = [6, 9, 30, 57, 105]

    clear = build.clear_sky_by_lead(init_times=[init], latitude=latitude, longitude=longitude)

    expected = [
        float(
            haurwitz_w_m2(
                apparent_zenith_deg=zenith(
                    stamps=pl.Series(
                        [init + timedelta(hours=lead)], dtype=pl.Datetime("us", "UTC")
                    ),
                    latitude=latitude,
                    longitude=longitude,
                )
            )[0]
        )
        for lead in leads
    ]
    assert expected[0] > 100.0
    assert clear[0, leads] == pytest.approx(expected)


def test_native_radiation_leads_pass_through_unchanged_even_below_the_daylight_floor():
    clear = np.full((1, 121), 30.0)
    snapshots = np.where(np.isin(np.arange(121), build.NATIVE_LEADS), 7.0, NAN)[None, :]

    filled = build.fill_radiation(snapshots=snapshots, clear_sky=clear)

    assert np.array_equal(filled[0, build.NATIVE_LEADS], snapshots[0, build.NATIVE_LEADS])


def test_a_missing_daylight_anchor_blanks_its_neighbours_and_is_not_filled_from_another_step():
    clear = _clear_sky(daylight_utc_hours=range(24))
    snapshots = np.where(np.isin(np.arange(121), build.NATIVE_LEADS), 300.0, NAN)[None, :]
    snapshots[0, 60] = NAN

    filled = build.fill_radiation(snapshots=snapshots, clear_sky=clear)

    assert np.isnan(filled[0, [58, 59, 61, 62]]).all()
    assert np.isfinite(filled[0, [55, 56, 64, 65]]).all()


def test_whether_an_anchor_is_before_noon_comes_from_the_utc_hour_not_from_the_lead():
    # Daylight is UTC 0 to 11, with a 30 W/m2 shoulder anchor (below the floor) at UTC 12. The
    # shoulder is in the afternoon, so it takes the last daylight index (0.5), not the next one
    # (0.9 from the following day's first daylight anchor). Read as `lead % 24 < 12`, UTC 12 would
    # count as a morning step.
    utc_hour = (build.RUN_HOUR + np.arange(build.N_LEADS)) % 24
    clear = np.where(utc_hour < 12, 600.0, np.where(utc_hour == 12, 30.0, 0.0))[None, :]
    index = np.where(np.arange(build.N_LEADS) < 66, 0.5, 0.9)[None, :]
    snapshots = np.where(np.isin(np.arange(121), build.NATIVE_LEADS), index * clear, NAN)

    filled = build.fill_radiation(snapshots=snapshots, clear_sky=clear)

    lead_at_utc_10 = 55  # (3 + 55) % 24 == 10, between the anchors at UTC 9 and UTC 12
    assert utc_hour[lead_at_utc_10] == 10
    assert filled[0, lead_at_utc_10] == pytest.approx(0.5 * 600.0)


def test_a_night_is_rebuilt_as_zero():
    clear = _clear_sky(daylight_utc_hours=range(6, 18))
    native = np.isin(np.arange(121), build.NATIVE_LEADS)
    snapshots = np.where(native, np.where(clear[0] > 0, 300.0, 0.0), NAN)[None, :]

    filled = build.fill_radiation(snapshots=snapshots, clear_sky=clear)

    assert (filled[0][clear[0] == 0.0] == 0.0).all()


# --- fill_wind ------------------------------------------------------------------------------------


def test_a_steady_wind_is_rebuilt_unchanged():
    speed = _native(value_of_lead=lambda lead: 8.0)
    direction = _native(value_of_lead=lambda lead: 270.0)

    out_speed, out_direction = build.fill_wind(speed=speed, direction_deg=direction)

    assert np.allclose(out_speed, 8.0)
    assert np.allclose(out_direction, 270.0)


def test_a_wind_turning_through_north_is_rebuilt_through_north_and_not_through_south():
    speed = _native(value_of_lead=lambda lead: 10.0)
    direction = _native(value_of_lead=lambda lead: 350.0 if lead <= 51 else 10.0)

    out_speed, out_direction = build.fill_wind(speed=speed, direction_deg=direction)

    midway = out_direction[0, 52]  # a third of the way from 51 (350 degrees) to 54 (10 degrees)
    assert min(midway, 360.0 - midway) < 10.0
    assert out_speed[0, 52] < 10.0
    assert out_speed[0, 52] > 9.7


def test_a_wind_with_a_missing_anchor_is_missing_at_the_leads_it_brackets():
    speed = _native(value_of_lead=lambda lead: 5.0)
    direction = _native(value_of_lead=lambda lead: 90.0)
    direction[0, 57] = NAN

    out_speed, out_direction = build.fill_wind(speed=speed, direction_deg=direction)

    assert np.isnan(out_speed[0, [55, 56, 58, 59]]).all()
    assert np.isnan(out_direction[0, [55, 56, 58, 59]]).all()
    assert np.isfinite(out_speed[0, [49, 50, 61, 62]]).all()


# --- rows, runs and causes ------------------------------------------------------------------------


def _keys(*times: datetime, site: str = "A") -> pl.DataFrame:
    return pl.DataFrame({"site": [site] * len(times), "time": list(times)}).with_columns(
        pl.col("time").dt.replace_time_zone("UTC")
    )


def test_a_row_reads_the_03_utc_run_of_its_run_day_at_a_lead_three_hours_shorter():
    keys = _keys(datetime(2026, 3, 10, 14))

    run = build.with_run(frame=keys, day=2, domain="wind").row(0, named=True)

    assert run["init_time"] == datetime(2026, 3, 8, 3, tzinfo=UTC)
    assert run["lead_hours"] == 59
    expected_slot = int((run["init_time"] - build.T120_PROFILE.slot_epoch) / timedelta(hours=12))
    assert run["slot"] == expected_slot
    assert run["key"] == f"A|{expected_slot}"


def test_the_slot_is_the_runs_place_on_the_twelve_hourly_grid_from_the_first_03_utc_run():
    epoch = build.T120_PROFILE.slot_epoch

    assert build.slot_init_time(slot=0) == epoch
    assert build.slot_init_time(slot=3) == epoch + timedelta(hours=36)
    frame = pl.DataFrame({"t": [epoch + timedelta(hours=24)]})
    assert frame.select(build.slot_of(init_time=pl.col("t")))["t"].to_list() == [2]


def test_a_slot_beyond_the_stored_slots_has_status_zero():
    statuses = np.array([1, 2, 3], dtype=np.int8)

    assert build.status_at(statuses=statuses, slot=2) == 3
    assert build.status_at(statuses=statuses, slot=3) == 0
    assert build.status_at(statuses=statuses, slot=-1) == 0


def _day_frame(*, slots: list[int], leads: list[int], values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"slot": slots, "lead_hours": leads, "ghi": values},
        schema={"slot": pl.Int64, "lead_hours": pl.Int32, "ghi": pl.Float64},
    )


def test_each_missing_value_is_labelled_by_its_runs_status():
    statuses = np.array([0, 1, 2, 3], dtype=np.int8)
    frame = _day_frame(
        slots=[0, 1, 2, 3, 3, 1, 9],
        leads=[30, 30, 30, 30, 30, 30, 30],
        values=[None, None, None, None, 1.0, 1.0, None],
    )

    labelled = build.attribute_causes(frame=frame, statuses=statuses, columns=["ghi"])

    assert labelled["cause"].to_list() == [
        "run not listed by CEDA",
        "complete run lacks a value",
        "run partial",
        "run missing",
        None,
        None,
        "run not listed by CEDA",
    ]


def test_a_nan_wind_value_is_labelled_by_its_runs_status_and_stored_as_null():
    slots, _ = _slots_and_init(n=1)
    shape = (1, build.N_LEADS, 1)
    series = {
        name: np.full(shape, np.nan)
        for name in (
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_speed_925hpa",
            "wind_direction_925hpa",
        )
    }
    hourly = build.wind_instants(slots=slots, series=series, sites=["W1"])
    init = build.slot_init_time(slot=slots[0])
    keys = _keys((init + timedelta(hours=30)).replace(tzinfo=None), site="W1")

    built = build.build_day(
        keys=keys,
        day=1,
        domain="wind",
        hourly=hourly,
        statuses=np.array([build.STATUS_MISSING] * 4, dtype=np.int8),
    )

    assert built["ukv_ceda_day1_cause"].to_list() == ["run missing"]
    assert built["ukv_ceda_day1_speed_10m"].is_null().all()


def test_a_lead_beyond_the_store_is_its_own_cause_before_the_status_is_read():
    frame = _day_frame(slots=[1], leads=[121], values=[None])

    labelled = build.attribute_causes(
        frame=frame, statuses=np.array([0, 1], dtype=np.int8), columns=["ghi"]
    )

    assert labelled["cause"].to_list() == ["lead beyond the store"]


def test_a_complete_run_that_lacks_a_value_stops_the_build():
    columns = pl.DataFrame(
        {
            f"ukv_ceda_day{day}_cause": ["complete run lacks a value", None]
            for day in build.LEAD_DAYS
        }
    )

    with pytest.raises(ValueError, match="complete run lacks a value"):
        build.check_complete_runs_hold_values(columns=columns, domain="wind")


def test_runs_with_every_value_or_a_known_gap_do_not_stop_the_build():
    columns = pl.DataFrame(
        {f"ukv_ceda_day{day}_cause": ["run missing", None] for day in build.LEAD_DAYS}
    )

    build.check_complete_runs_hold_values(columns=columns, domain="wind")


# --- the coverage guard ---------------------------------------------------------------------------


def _slot_of_date(*, day: date) -> int:
    init = datetime(day.year, day.month, day.day, 3, tzinfo=UTC)
    return int((init - build.T120_PROFILE.slot_epoch) / timedelta(hours=12))


def test_the_guard_refuses_a_store_the_fetcher_has_not_taken_back_to_the_window():
    statuses = np.zeros(_slot_of_date(day=date(2026, 3, 1)) + 1, dtype=np.int8)
    statuses[_slot_of_date(day=date(2026, 1, 6)) :] = 1
    needed = [_slot_of_date(day=date(2024, 11, 27)), _slot_of_date(day=date(2026, 2, 1))]

    with pytest.raises(ValueError, match="has not reached the window"):
        build.check_window_covered(statuses=statuses, needed_slots=needed, unlisted_days=[])


def test_the_guard_refuses_a_never_archived_run_until_its_day_is_named_as_unlisted():
    statuses = np.ones(_slot_of_date(day=date(2026, 3, 1)) + 1, dtype=np.int8)
    gap = _slot_of_date(day=date(2026, 2, 3))
    statuses[gap] = 0
    needed = [gap - 2, gap, gap + 2]

    with pytest.raises(ValueError, match="2026-02-03"):
        build.check_window_covered(statuses=statuses, needed_slots=needed, unlisted_days=[])
    accepted = build.check_window_covered(
        statuses=statuses, needed_slots=needed, unlisted_days=[date(2026, 2, 3)]
    )
    assert accepted == [date(2026, 2, 3)]


def test_the_guard_passes_a_fully_archived_window_with_missing_and_partial_runs_in_it():
    statuses = np.ones(_slot_of_date(day=date(2026, 3, 1)) + 1, dtype=np.int8)
    statuses[100] = 3
    statuses[102] = 2

    assert (
        build.check_window_covered(
            statuses=statuses, needed_slots=[100, 102, 104], unlisted_days=[]
        )
        == []
    )


# --- the loss table and the day-5 share -----------------------------------------------------------


def test_every_row_is_counted_under_the_first_cause_that_applies():
    candidates = pl.DataFrame(
        {
            "site": ["A"] * 5,
            "time": [datetime(2026, 3, 1, hour, tzinfo=UTC) for hour in range(5)],
            "power_mw": [None, 1.0, 1.0, 1.0, 1.0],
            "ens_mean_day1_ghi": [None, None, 1.0, 1.0, 1.0],
            "ens_mean_day1_temp": [1.0] * 5,
        }
    )
    built = candidates.select("site", "time").with_columns(
        **{
            f"ukv_ceda_day{day}_cause": pl.Series(
                [None, "run missing", "run partial", None, "run not listed by CEDA"],
                dtype=pl.String,
            )
            for day in build.LEAD_DAYS
        }
    )
    candidates = candidates.with_columns(
        **{f"ens_mean_day{day}_ghi": pl.col("ens_mean_day1_ghi") for day in (2, 3, 4)},
        **{f"ens_mean_day{day}_temp": pl.col("ens_mean_day1_temp") for day in (2, 3, 4)},
    )

    table = build.loss_table(candidates=candidates, built=built, domain="solar")

    day1 = table.filter(pl.col("day") == 1).row(0, named=True)
    assert day1["candidates"] == 5
    assert day1["target absent"] == 1
    assert day1["ENS absent"] == 1
    assert day1["run partial"] == 1
    assert day1["run not listed by CEDA"] == 1
    assert day1["run missing"] == 0
    assert day1["kept"] == 1
    counted = sum(day1[name] for name in (*build.CAUSES, "complete run lacks a value", "kept"))
    assert counted == day1["candidates"]


def test_the_03_utc_run_reaches_day_5_only_for_wind_hours_0_to_3():
    hours = [datetime(2026, 3, 10, hour) for hour in range(24)]
    keys = _keys(*hours)

    assert build.beyond_day_share(candidates=keys, domain="wind") == pytest.approx(4 / 24)
    assert (
        build.beyond_day_share(candidates=_keys(datetime(2026, 3, 10, 12)), domain="solar") == 0.0
    )


# --- hourly series --------------------------------------------------------------------------------


def _slots_and_init(*, n: int) -> tuple[list[int], list[datetime]]:
    slots = [2 * index for index in range(n)]
    return slots, [build.slot_init_time(slot=slot) for slot in slots]


def test_a_solar_label_is_the_mean_of_its_two_snapshots_and_needs_both():
    slots, _ = _slots_and_init(n=1)
    shortwave = np.full((1, build.N_LEADS, 1), NAN)
    temperature = np.full((1, build.N_LEADS, 1), NAN)
    for lead in range(49):
        shortwave[0, lead, 0] = 10.0 * lead
        temperature[0, lead, 0] = 273.15 + lead
    shortwave[0, 7, 0] = NAN

    hourly = build.solar_hourly(
        slots=slots,
        series={"shortwave_down": shortwave, "temperature_1p5m": temperature},
        sites=["A"],
        coordinates={"A": (54.0, -1.5)},
    )

    init = build.slot_init_time(slot=slots[0])
    by_lead = {
        int((time - init) / timedelta(hours=1)): ghi
        for time, ghi in hourly.select("time", "ghi").iter_rows()
    }
    assert by_lead[10] == pytest.approx(95.0)
    assert 7 not in by_lead
    assert 8 not in by_lead
    assert 6 in by_lead
    assert hourly["key"].unique().to_list() == [f"A|{slots[0]}"]
    temp = {
        int((time - init) / timedelta(hours=1)): value
        for time, value in hourly.select("time", "temp").iter_rows()
    }
    assert temp[10] == pytest.approx(9.5)


def test_a_wind_hour_is_the_instant_at_its_label_with_the_direction_as_sine_and_cosine():
    slots, _ = _slots_and_init(n=1)
    shape = (1, build.N_LEADS, 1)
    series = {
        "wind_speed_10m": np.full(shape, 6.0),
        "wind_direction_10m": np.full(shape, 90.0),
        "wind_speed_925hpa": np.full(shape, 12.0),
        "wind_direction_925hpa": np.full(shape, 180.0),
    }

    hourly = build.wind_instants(slots=slots, series=series, sites=["W1"])

    row = hourly.filter(
        pl.col("time") == build.slot_init_time(slot=slots[0]) + timedelta(hours=30)
    ).row(0, named=True)
    assert row["speed_10m"] == pytest.approx(6.0)
    assert row["sin_10m"] == pytest.approx(1.0)
    assert row["cos_10m"] == pytest.approx(0.0, abs=1e-12)
    assert row["speed_925hpa"] == pytest.approx(12.0)
    assert hourly.height == build.N_LEADS


# --- writing --------------------------------------------------------------------------------------


def _write_published(*, folder: Path) -> None:
    folder.mkdir()
    for domain in build.DOMAINS:
        pl.DataFrame({"a": [1]}).write_parquet(folder / f"{domain}_forecast_inputs.parquet")
        pl.DataFrame({"a": [1]}).write_parquet(folder / f"{domain}_extra_lead_inputs.parquet")


def test_the_outputs_are_written_once_with_a_stamp_naming_the_snapshot_and_every_hash(
    tmp_path: Path,
):
    published, day4 = tmp_path / "published", tmp_path / "day4"
    _write_published(folder=published)
    _write_published(folder=day4)
    outputs = {
        domain: pl.DataFrame({"site": ["A"], "time": [datetime(2026, 3, 1, tzinfo=UTC)]})
        for domain in build.DOMAINS
    }
    store = build.StoreRead(
        group=cast("zarr.Group", None), snapshot_id="SNAP", statuses=np.zeros(0, dtype=np.int8)
    )
    output_dir = tmp_path / "ukv_ceda_blends"

    def write() -> None:
        build.write_outputs(
            output_dir=output_dir,
            outputs=outputs,
            tables=["| table |"],
            store=store,
            published_dir=published,
            day4_dir=day4,
            unlisted_days=[date(2026, 6, 27)],
        )

    write()

    import json

    stamp = json.loads((output_dir / "build.json").read_text())
    assert stamp["snapshot_id"] == "SNAP"
    assert stamp["coverage_guard_passed"] is True
    assert stamp["unlisted_days"] == ["2026-06-27"]
    assert set(stamp["inputs_sha256"]) == {"solar", "wind"}
    assert "SNAP" in (output_dir / "README.md").read_text()
    with pytest.raises(FileExistsError):
        write()


def test_the_build_writes_only_to_a_folder_named_for_the_study(tmp_path: Path):
    reads = [tmp_path / "published"]

    build.check_output_dir(output_dir=tmp_path / "ukv_ceda_blends", read_only=reads)
    with pytest.raises(ValueError, match="writes only to a folder named"):
        build.check_output_dir(output_dir=tmp_path / "elsewhere", read_only=reads)
    with pytest.raises(ValueError, match="writes only to a folder named"):
        build.check_output_dir(output_dir=reads[0], read_only=reads)


def _store_with_init_times(*, seconds: list[int]) -> build.StoreRead:
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    array = group.create_array("init_time", shape=(len(seconds),), dtype="int64")
    array[:] = np.array(seconds, dtype=np.int64)
    return build.StoreRead(group=group, snapshot_id="X", statuses=np.zeros(len(seconds), np.int8))


def test_the_slot_arithmetic_must_match_the_stores_own_init_time_coordinate():
    epoch = int(build.T120_PROFILE.slot_epoch.timestamp())
    good = _store_with_init_times(seconds=[epoch + 12 * 3600 * slot for slot in range(4)])

    build.check_slot_times(store=good, slots=[0, 1, 3, 9])

    shifted = _store_with_init_times(
        seconds=[epoch + 12 * 3600 * slot + (3600 if slot == 2 else 0) for slot in range(4)]
    )
    with pytest.raises(ValueError, match=r"slots \[2\]"):
        build.check_slot_times(store=shifted, slots=[0, 1, 2])


# --- the post hoc older run -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "label", "day", "init", "lead"),
    [
        # Wind label 12:00 on 10 March, lead day 1: the 15 UTC run of 8 March, 45 hours before.
        ("wind", datetime(2026, 3, 10, 12), 1, datetime(2026, 3, 8, 15), 45),
        # Wind label at midnight, lead day 3: the 15 UTC run of 6 March, 3 days and 9 hours before.
        ("wind", datetime(2026, 3, 10, 0), 3, datetime(2026, 3, 6, 15), 81),
        # Solar label 13:00 names the hour ending then, which starts at 12:00 on 10 March. Lead day
        # 2 reads the 15 UTC run of 7 March, 70 hours before the label.
        ("solar", datetime(2026, 3, 10, 13), 2, datetime(2026, 3, 7, 15), 70),
        # Solar label 00:00 on 11 March names the hour that started at 23:00 on 10 March, whose
        # own day is 10 March, so lead day 1 reads the 15 UTC run of 8 March, 57 hours before.
        ("solar", datetime(2026, 3, 11, 0), 1, datetime(2026, 3, 8, 15), 57),
        # The longest lead the older run needs: lead day 3, the last solar hour of the day.
        ("solar", datetime(2026, 3, 11, 0), 3, datetime(2026, 3, 6, 15), 105),
    ],
)
def test_the_older_run_lead_mapping_is_the_15_utc_run_of_the_day_before_ens_runs_day(
    domain: DomainType, label: datetime, day: int, init: datetime, lead: int
):
    keys = _keys(label)

    run = build.with_run(frame=keys, day=day, domain=domain, spec=build.OLDER_RUN).row(
        0, named=True
    )

    assert run["init_time"] == init.replace(tzinfo=UTC)
    assert run["lead_hours"] == lead
    assert run["slot"] % 2 == 1
    assert build.slot_init_time(slot=run["slot"]) == run["init_time"]
    # The planned run of the same row starts 12 hours after the older run of the day before.
    planned = build.with_run(frame=keys, day=day, domain=domain).row(0, named=True)
    assert planned["init_time"] - run["init_time"] == timedelta(hours=12)
    assert run["lead_hours"] - planned["lead_hours"] == 12


def test_the_older_run_starts_nine_hours_before_ens_run_whose_lead_it_shares():
    label = datetime(2026, 3, 10, 12, tzinfo=UTC)
    ens_init = datetime(2026, 3, 9, tzinfo=UTC)  # ENS's 00 UTC run at lead day 1
    run = build.with_run(frame=_keys(label), day=1, domain="wind", spec=build.OLDER_RUN)

    assert ens_init - run["init_time"][0] == timedelta(hours=9)


def test_the_older_run_is_built_for_lead_days_one_to_three_in_its_own_folder_and_columns():
    assert build.OLDER_RUN.lead_days == (1, 2, 3)
    assert build.OLDER_RUN.run_hour == 15
    assert build.OLDER_RUN.extra_days == 1
    assert build.OLDER_RUN.column_prefix == "ukv_ceda_run15"
    assert build.OLDER_RUN.output_dir_name != build.PLANNED_RUN.output_dir_name
    assert build.PLANNED_RUN.lead_days == build.LEAD_DAYS == (1, 2, 3, 4)
    assert build.PLANNED_RUN.run_hour == build.RUN_HOUR == 3


def test_the_older_run_reaches_day_4_for_wind_hours_0_to_15_only():
    hours = [datetime(2026, 3, 10, hour) for hour in range(24)]

    share = build.beyond_day_share(candidates=_keys(*hours), domain="wind", spec=build.OLDER_RUN)

    # Lead at day 4 is 120 + h - 15 = 105 + h hours, inside the store's 120 for h of 0 to 15.
    assert share == pytest.approx(16 / 24)


def test_the_older_runs_columns_carry_its_own_prefix_and_the_planned_columns_do_not_change():
    joined = pl.DataFrame(
        {
            "site": ["A"],
            "time": [datetime(2026, 3, 10, 12, tzinfo=UTC)],
            "init_time": [datetime(2026, 3, 8, 15, tzinfo=UTC)],
            "cause": [None],
            "ghi": [1.0],
            "temp": [2.0],
        },
        schema_overrides={"cause": pl.String},
    )

    older = build.day_columns(joined=joined, day=1, domain="solar", spec=build.OLDER_RUN)
    planned = build.day_columns(joined=joined, day=1, domain="solar")

    assert older.columns == [
        "site",
        "time",
        "ukv_ceda_run15_day1_ghi",
        "ukv_ceda_run15_day1_temp",
        "ukv_ceda_run15_day1_init_time",
        "ukv_ceda_run15_day1_cause",
    ]
    assert planned.columns[2:4] == ["ukv_ceda_day1_ghi", "ukv_ceda_day1_temp"]


def test_the_morning_flag_of_a_rebuilt_radiation_follows_the_runs_start_hour(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[np.ndarray] = []

    def spy(**kwargs: np.ndarray) -> np.ndarray:
        seen.append(np.asarray(kwargs["morning"]))
        return np.zeros((1, len(build.FILLED_LEADS)))

    monkeypatch.setattr(build, "clear_sky_index_resample", spy)
    snapshots = np.ones((1, build.N_LEADS))
    clear_sky = np.ones((1, build.N_LEADS))

    build.fill_radiation(snapshots=snapshots, clear_sky=clear_sky)
    build.fill_radiation(snapshots=snapshots, clear_sky=clear_sky, run_hour=15)

    leads = build.ANCHOR_LEADS
    assert seen[0].tolist() == [(3 + lead) % 24 < 12 for lead in leads]
    assert seen[1].tolist() == [(15 + lead) % 24 < 12 for lead in leads]
    # Lead 48 of a 03 UTC run is 03:00 UTC (morning); of a 15 UTC run it is 15:00 UTC (afternoon).
    assert seen[0][0]
    assert not seen[1][0]


def test_the_loss_table_of_the_older_run_has_one_row_per_older_lead_day_and_its_own_causes():
    candidates = pl.DataFrame(
        {
            "site": ["A"] * 3,
            "time": [datetime(2026, 3, 1, hour, tzinfo=UTC) for hour in range(3)],
            "power_mw": [1.0] * 3,
            **{
                f"ens_mean_day{day}_{field}": [1.0] * 3
                for day in (1, 2, 3, 4)
                for field in ("ghi", "temp")
            },
        }
    )
    built = candidates.select("site", "time").with_columns(
        **{
            f"ukv_ceda_run15_day{day}_cause": pl.Series(
                [None, "run missing", "run not listed by CEDA"], dtype=pl.String
            )
            for day in build.OLDER_RUN.lead_days
        }
    )

    table = build.loss_table(
        candidates=candidates, built=built, domain="solar", spec=build.OLDER_RUN
    )

    assert table["day"].to_list() == [1, 2, 3]
    day1 = table.row(0, named=True)
    assert (day1["kept"], day1["run missing"], day1["run not listed by CEDA"]) == (1, 1, 1)


def test_each_run_writes_only_to_its_own_folder(tmp_path: Path):
    reads = [tmp_path / "published"]

    build.check_output_dir(
        output_dir=tmp_path / "ukv_ceda_blends_run15", read_only=reads, spec=build.OLDER_RUN
    )
    with pytest.raises(ValueError, match="ukv_ceda_blends_run15"):
        build.check_output_dir(
            output_dir=tmp_path / "ukv_ceda_blends", read_only=reads, spec=build.OLDER_RUN
        )
    with pytest.raises(ValueError, match="folder named ukv_ceda_blends,"):
        build.check_output_dir(output_dir=tmp_path / "ukv_ceda_blends_run15", read_only=reads)


def test_the_older_run_outputs_are_written_once_under_its_own_names_and_run_hour(tmp_path: Path):
    import json

    published, day4 = tmp_path / "published", tmp_path / "day4"
    _write_published(folder=published)
    _write_published(folder=day4)
    outputs = {
        domain: pl.DataFrame({"site": ["A"], "time": [datetime(2026, 3, 1, tzinfo=UTC)]})
        for domain in build.DOMAINS
    }
    store = build.StoreRead(
        group=cast("zarr.Group", None), snapshot_id="SNAP", statuses=np.zeros(0, dtype=np.int8)
    )
    output_dir = tmp_path / "ukv_ceda_blends_run15"

    def write() -> None:
        build.write_outputs(
            output_dir=output_dir,
            outputs=outputs,
            tables=["| table |"],
            store=store,
            published_dir=published,
            day4_dir=day4,
            unlisted_days=[],
            spec=build.OLDER_RUN,
        )

    write()

    stamp = json.loads((output_dir / "build.json").read_text())
    assert (stamp["run_hour"], stamp["extra_days"]) == (15, 1)
    assert (output_dir / "solar_ukv_ceda_run15_inputs.parquet").exists()
    assert not (output_dir / "solar_ukv_ceda_inputs.parquet").exists()
    assert stamp["inputs_sha256"]["wind"] == build.sha256_of(
        path=output_dir / "wind_ukv_ceda_run15_inputs.parquet"
    )
    readme = (output_dir / "README.md").read_text()
    assert "15 UTC run" in readme
    assert "`D - N - 1`" in readme
    with pytest.raises(FileExistsError):
        write()


def test_the_solar_hourly_series_rebuilds_its_radiation_for_the_runs_own_start_hour(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[int] = []
    real = build.fill_radiation

    def spy(*, snapshots: np.ndarray, clear_sky: np.ndarray, run_hour: int = 3) -> np.ndarray:
        seen.append(run_hour)
        return real(snapshots=snapshots, clear_sky=clear_sky, run_hour=run_hour)

    monkeypatch.setattr(build, "fill_radiation", spy)
    slots = [1]
    series = {
        "shortwave_down": np.full((1, build.N_LEADS, 1), 1.0),
        "temperature_1p5m": np.full((1, build.N_LEADS, 1), 280.0),
    }
    coordinates = {"A": (54.0, -1.5)}

    build.solar_hourly(slots=slots, series=series, sites=["A"], coordinates=coordinates)
    build.solar_hourly(
        slots=slots, series=series, sites=["A"], coordinates=coordinates, run_hour=15
    )

    assert seen == [3, 15]
