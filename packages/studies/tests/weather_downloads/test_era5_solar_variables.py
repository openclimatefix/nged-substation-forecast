"""Tests for the ERA5 solar-variables download and its validator.

No test touches the network or `data/`. Each test fails on the bug it exists for: a request over
the store's field limit, an accumulation requested alongside instantaneous fields, an `expver`
read from the wrong slice, a seam step the profile misses, a NaN share that ignores cloud cover, an
end month typed in instead of read, and a resume that skips an empty file for good.
"""

import zipfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import fetch_cams_eac4_aod as eac4
import numpy as np
import polars as pl
import pytest
import validate_era5_solar_variables as validator
import xarray as xr
from era5_solar_variables import (
    CDS_FIELD_LIMIT,
    CELL_COUNT,
    SEAM_HOURS,
    TIERS,
    VARIABLES,
    VARIABLES_BY_NAME,
    PlannedChunk,
    balanced_groups,
    chunk_is_valid,
    collapse_expver,
    dataset_to_frames,
    expected_hours,
    expected_rows,
    hour_profile,
    last_final_month,
    nan_share_by_tcc,
    pilot_chunks,
    plan_chunks,
    read_archive,
    request_body,
    retrieve_with_cleanup,
    run_chunks,
    step_flags,
)
from studies.era5_grid import GRID_LATITUDES, GRID_LONGITUDES

FIRST: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)
N_TIMES: Final[int] = 6


def _dataset(*, expver: object, with_dimension: bool = False) -> xr.Dataset:
    """Return a small ERA5-shaped file: 6 hours over the 20 public cells, one variable `tcc`."""
    times = np.array(
        [np.datetime64("2025-06-01T00:00") + np.timedelta64(h, "h") for h in range(N_TIMES)]
    )
    coords = {
        "valid_time": times,
        "latitude": np.array(GRID_LATITUDES),
        "longitude": np.array(GRID_LONGITUDES),
    }
    shape = (N_TIMES, len(GRID_LATITUDES), len(GRID_LONGITUDES))
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000
    if with_dimension:
        stacked = np.full((2, *shape), np.nan, dtype=np.float32)
        stacked[0, :4] = values[:4]  # the first four hours are final (1)
        stacked[1, 4:] = values[4:]  # the last two are preliminary (5)
        return xr.Dataset(
            {"tcc": (("expver", "valid_time", "latitude", "longitude"), stacked)},
            coords={**coords, "expver": np.array([1, 5])},
        )
    dataset = xr.Dataset({"tcc": (("valid_time", "latitude", "longitude"), values)}, coords=coords)
    return dataset.assign_coords(expver=expver) if isinstance(expver, str) else dataset


def test_no_request_exceeds_the_store_limit_and_the_longest_ones_use_four_variables() -> None:
    chunks = plan_chunks(tiers=TIERS)

    assert max(chunk.cost_fields for chunk in chunks) <= CDS_FIELD_LIMIT
    assert max(len(chunk.variables) for chunk in chunks) == 4
    assert max(chunk.cost_fields for chunk in chunks) > 100_000


def test_accumulations_never_share_a_request_with_instantaneous_fields() -> None:
    for chunk in plan_chunks(tiers=TIERS):
        kinds = {VARIABLES_BY_NAME[name].kind for name in chunk.variables}
        assert kinds == {chunk.kind}


def test_every_variable_is_requested_once_per_half_year() -> None:
    chunks = plan_chunks(tiers=TIERS)
    seen = [(name, chunk.period.name) for chunk in chunks for name in chunk.variables]

    assert len(seen) == len(set(seen))
    assert {name for name, _ in seen} == {v.short_name for v in VARIABLES}
    assert len({period for _, period in seen}) == 15


def test_a_chunk_never_crosses_a_year_and_the_span_starts_in_september_2019() -> None:
    chunks = plan_chunks(tiers=["tier1a"])

    assert chunks[0].period.year == 2019
    assert chunks[0].period.months == (9, 10, 11, 12)
    assert chunks[-1].period.months == (7, 8, 9)
    assert all(len({chunk.period.year}) == 1 for chunk in chunks)
    assert request_body(chunk=chunks[0])["month"] == ["09", "10", "11", "12"]


def test_already_held_variables_are_not_requested() -> None:
    requested = {name for chunk in plan_chunks(tiers=TIERS) for name in chunk.variables}

    assert requested.isdisjoint({"ssrd", "fdir", "t2m"})


def test_balanced_groups_split_evenly_under_the_limit() -> None:
    assert balanced_groups(names=list("abcdefghi"), max_size=4) == [
        ("a", "b", "c"),
        ("d", "e", "f"),
        ("g", "h", "i"),
    ]
    assert balanced_groups(names=[], max_size=4) == []


def test_the_span_holds_61608_hours_and_20_cells() -> None:
    first = datetime(2019, 9, 1, tzinfo=UTC)
    last = datetime(2026, 9, 10, 23, tzinfo=UTC)

    assert expected_hours(first_hour=first, last_hour=last) == 61_608
    assert expected_rows(first_hour=first, last_hour=last) == CELL_COUNT * 61_608


def test_expver_is_read_from_a_dimension_by_taking_the_slice_that_holds_data() -> None:
    collapsed, per_hour = collapse_expver(dataset=_dataset(expver=None, with_dimension=True))

    assert list(per_hour) == ["0001"] * 4 + ["0005"] * 2
    assert "expver" not in collapsed.dims
    assert not np.isnan(collapsed["tcc"].values).any()


def test_expver_is_read_from_a_per_hour_variable_and_from_a_scalar() -> None:
    per_time = _dataset(expver=None).assign_coords(
        expver=("valid_time", ["0001", "0001", "0001", "0005", "0005", "0005"])
    )
    scalar = _dataset(expver="0005")

    assert list(collapse_expver(dataset=per_time)[1]) == ["0001"] * 3 + ["0005"] * 3
    assert list(collapse_expver(dataset=scalar)[1]) == ["0005"] * N_TIMES


def test_expver_integers_and_padded_strings_give_the_same_label() -> None:
    dataset = _dataset(expver=None).assign_coords(expver=("valid_time", [1, 1, 1, 5, 5, 5]))

    assert list(collapse_expver(dataset=dataset)[1]) == ["0001"] * 3 + ["0005"] * 3


def test_an_hour_with_data_under_two_expver_labels_raises() -> None:
    dataset = _dataset(expver=None, with_dimension=True)
    dataset["tcc"].values[0, 4] = 0.5  # hour 4 is now filled under both labels

    with pytest.raises(ValueError, match="no `expver` label or under more than one"):
        collapse_expver(dataset=dataset)


def test_a_file_without_expver_raises_instead_of_guessing_final() -> None:
    with pytest.raises(ValueError, match="no `expver`"):
        collapse_expver(dataset=_dataset(expver=None))


def test_frames_hold_every_cell_hour_with_the_hour_ending_stamp_and_float32_values() -> None:
    frames = dataset_to_frames(dataset=_dataset(expver=None, with_dimension=True)).frames["tcc"]

    assert frames.height == N_TIMES * CELL_COUNT
    assert frames.schema["value"] == pl.Float32
    assert frames["time"].min() == FIRST
    assert frames.filter(pl.col("time") >= FIRST + timedelta(hours=4))[
        "expver"
    ].unique().to_list() == ["0005"]


def test_frames_keep_a_nan_value_as_nan() -> None:
    dataset = _dataset(expver="0001")
    dataset["tcc"].values[0, 0, 0] = np.nan
    frame = dataset_to_frames(dataset=dataset).frames["tcc"]

    assert frame["value"].is_nan().sum() == 1
    assert frame["value"].null_count() == 0


def test_last_final_month_is_the_month_before_the_first_preliminary_hour() -> None:
    hours = pl.datetime_range(
        datetime(2026, 5, 30, tzinfo=UTC), datetime(2026, 8, 2, tzinfo=UTC), "1d", eager=True
    )
    labels = ["0001" if hour < datetime(2026, 7, 10, tzinfo=UTC) else "0005" for hour in hours]
    frame = pl.DataFrame({"time": hours, "expver": labels})

    assert last_final_month(frame=frame) == "2026-06"
    assert last_final_month(frame=frame.with_columns(expver=pl.lit("0001"))) == "2026-08"
    assert last_final_month(frame=frame.with_columns(expver=pl.lit("0005"))) is None


def _profile_frame(*, step_hour: int | None, scale: float = 1.0) -> pl.DataFrame:
    """Return three days for one cell whose value rises by 1 each hour, plus a jump at a seam."""
    times = pl.datetime_range(
        datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 3, 23, tzinfo=UTC), "1h", eager=True
    )
    value, values = 0.0, []
    for time in times:
        value += scale * (11.0 if time.hour == step_hour else 1.0)
        values.append(value)
    return pl.DataFrame(
        {"time": times, "latitude": 53.5, "longitude": 0.0, "value": values}
    ).with_columns(pl.col("value").cast(pl.Float32))


def test_the_hour_profile_shows_a_step_at_a_seam_and_the_flag_catches_it() -> None:
    profile = hour_profile(frame=_profile_frame(step_hour=7))
    flagged = step_flags(profile=profile, seam_hours=SEAM_HOURS["cloud"], rule="median_of_others")

    assert profile[7] == pytest.approx(11.0)
    assert profile[8] == pytest.approx(1.0)
    assert flagged == [7]


def test_a_smooth_profile_raises_no_flag() -> None:
    profile = hour_profile(frame=_profile_frame(step_hour=None))

    assert (
        step_flags(profile=profile, seam_hours=SEAM_HOURS["cloud"], rule="median_of_others") == []
    )


def test_a_sunrise_ramp_is_not_a_step_under_the_neighbour_rule_but_a_spike_is() -> None:
    ramp = [0.0] * 6 + [2.0, 10.0, 30.0, 40.0, 40.0, 30.0, 10.0, 2.0] + [0.0] * 10
    spike = list(ramp)
    spike[7] = 400.0

    assert step_flags(profile=ramp, seam_hours=(7,), rule="larger_neighbour") == []
    assert step_flags(profile=spike, seam_hours=(7,), rule="larger_neighbour") == [7]
    assert step_flags(profile=ramp, seam_hours=(7,), rule="median_of_others") == [7]


def test_nan_share_is_split_by_total_cloud_cover_below_and_above_the_threshold() -> None:
    times = pl.datetime_range(
        datetime(2025, 6, 1, tzinfo=UTC), datetime(2025, 6, 1, 3, tzinfo=UTC), "1h", eager=True
    )
    cell = {"latitude": 53.5, "longitude": 0.0}
    tcc = pl.DataFrame({"time": times, **cell, "value": [0.0, 0.01, 0.5, 1.0]})
    cbh = pl.DataFrame({"time": times, **cell, "value": [np.nan, np.nan, 900.0, np.nan]})

    split = nan_share_by_tcc(frame=cbh, tcc=tcc)

    assert split["tcc_below"] == {"rows": 2, "nan_rows": 2, "share": 1.0}
    assert split["tcc_at_or_above"] == {"rows": 2, "nan_rows": 1, "share": 0.5}
    assert split["tcc_missing"]["share"] is None


def test_the_end_month_is_the_latest_year_and_month_the_constraints_list() -> None:
    older = [{"year": ["2003", "2024"], "month": ["01", "06"]}]
    newer = [
        {"year": ["2003", "2025"], "month": ["01", "08"]},
        {"year": ["2019"], "month": ["12"]},
    ]

    assert eac4.parse_latest_month(constraints=older) == (2024, 6)
    assert eac4.parse_latest_month(constraints=newer) == (2025, 8)


def test_the_end_month_is_read_from_date_ranges_too() -> None:
    constraints = {"date": ["2003-01-01/2026-02-28"], "variable": ["x"]}

    assert eac4.parse_latest_month(constraints=constraints) == (2026, 2)


def test_constraints_that_list_no_date_raise_instead_of_returning_a_default() -> None:
    with pytest.raises(ValueError, match="no year and month"):
        eac4.parse_latest_month(constraints=[{"variable": ["x"]}])


def test_the_eac4_box_encloses_the_era5_box_on_its_own_grid_lines() -> None:
    north, west, south, east = eac4.eac4_area()
    cells = eac4.eac4_cells()

    assert (north, west, south, east) == (54.0, -0.75, 52.5, 0.75)
    assert len(cells) == 9
    assert north >= max(GRID_LATITUDES)
    assert south <= min(GRID_LATITUDES)
    assert west <= min(GRID_LONGITUDES)
    assert east >= max(GRID_LONGITUDES)


def test_eac4_requests_one_calendar_year_each_and_stop_at_the_end_month() -> None:
    chunks = eac4.plan_chunks(first_month=(2019, 9), last_month=(2025, 8))
    body = eac4.request_body(chunk=chunks[-1])

    assert len(chunks) == 7
    assert body["date"] == ["2025-01-01/2025-08-31"]
    assert body["time"] == [f"{h:02d}:00" for h in range(0, 24, 3)]
    assert eac4.request_body(chunk=chunks[0])["date"] == ["2019-09-01/2019-12-31"]


def _write_valid_archive(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.nc", b"CDF\x01 not read in this test")


def _chunk() -> PlannedChunk:
    return plan_chunks(tiers=["tier1a"], first_month=(2025, 6), last_month=(2025, 6))[0]


def test_a_valid_chunk_is_skipped_and_an_empty_or_truncated_one_is_fetched_again(
    tmp_path: Path,
) -> None:
    chunk = _chunk()
    destination = tmp_path / f"{chunk.chunk_id}.zip"
    calls: list[str] = []

    def download(requested: PlannedChunk, path: Path) -> None:
        calls.append(requested.chunk_id)
        _write_valid_archive(path)

    run_chunks(chunks=[chunk], chunk_dir=tmp_path, download=download, log=lambda _: None)
    records = run_chunks(chunks=[chunk], chunk_dir=tmp_path, download=download, log=lambda _: None)
    assert calls == [chunk.chunk_id]
    assert records[0].skipped

    destination.write_bytes(b"")
    run_chunks(chunks=[chunk], chunk_dir=tmp_path, download=download, log=lambda _: None)
    assert len(calls) == 2
    assert chunk_is_valid(path=destination)

    destination.write_bytes(destination.read_bytes()[:20])
    run_chunks(chunks=[chunk], chunk_dir=tmp_path, download=download, log=lambda _: None)
    assert len(calls) == 3


def test_a_download_that_is_not_an_archive_raises_and_leaves_no_final_file(tmp_path: Path) -> None:
    chunk = _chunk()

    def download(_: PlannedChunk, path: Path) -> None:
        path.write_bytes(b"<html>error</html>")

    with pytest.raises(RuntimeError, match="not a valid archive"):
        run_chunks(chunks=[chunk], chunk_dir=tmp_path, download=download, log=lambda _: None)
    assert not (tmp_path / f"{chunk.chunk_id}.zip").exists()


def test_each_chunk_logs_its_id_fields_bytes_and_seconds(tmp_path: Path) -> None:
    lines: list[str] = []
    chunk = _chunk()

    run_chunks(
        chunks=[chunk],
        chunk_dir=tmp_path,
        download=lambda _, path: _write_valid_archive(path),
        log=lines.append,
    )

    assert chunk.chunk_id in lines[0]
    assert f"{chunk.variable_hours} variable-hours" in lines[0]
    assert f"({chunk.cost_fields} store fields)" in lines[0]
    assert "bytes" in lines[0]
    assert " s" in lines[0]


def _validation_frame(*, values: list[float], variable_hours: int) -> pl.DataFrame:
    times = pl.datetime_range(
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 1, tzinfo=UTC) + timedelta(hours=variable_hours - 1),
        "1h",
        eager=True,
    )
    return pl.DataFrame(
        {"time": times, "latitude": 53.5, "longitude": 0.0, "value": values}
    ).with_columns(pl.col("value").cast(pl.Float32))


def _daily_solar(*, running: bool) -> list[float]:
    """Return 3 days of an hourly solar total: a daytime bell, or the same as a running sum."""
    out = []
    for _ in range(3):
        day = [
            max(0.0, 3.0e5 * np.sin(np.pi * (h - 4) / 16)) if 4 <= h <= 20 else 0.0
            for h in range(24)
        ]
        out.extend(np.cumsum(day) if running else day)
    return out


def test_a_solar_accumulation_that_only_rises_is_failed_as_a_running_total() -> None:
    cdir = VARIABLES_BY_NAME["cdir"]
    hourly = _validation_frame(values=_daily_solar(running=False), variable_hours=72)
    running = _validation_frame(values=_daily_solar(running=True), variable_hours=72)

    assert validator.check_accumulation(frame=hourly, variable=cdir)[0] == "PASS"
    status, details = validator.check_accumulation(frame=running, variable=cdir)
    assert status == "FAIL"
    assert details["looks_like_running_total"]


def test_a_clear_sky_radiation_value_at_night_is_failed() -> None:
    values = _daily_solar(running=False)
    values[1] = 5000.0  # 01 UTC, dark
    frame = _validation_frame(values=values, variable_hours=72)

    status, details = validator.check_accumulation(frame=frame, variable=VARIABLES_BY_NAME["ssrdc"])

    assert status == "FAIL"
    assert details["maximum_at_night_hours"] == pytest.approx(5000.0)


def test_the_range_check_fails_a_cloud_cover_above_one() -> None:
    frame = _validation_frame(values=[0.2, 0.5, 1.4], variable_hours=3)

    assert (
        validator.check_physical_range(frame=frame, variable=VARIABLES_BY_NAME["tcc"])[0] == "FAIL"
    )
    assert (
        validator.check_physical_range(
            frame=frame.with_columns(value=pl.col("value").clip(upper_bound=1.0)),
            variable=VARIABLES_BY_NAME["tcc"],
        )[0]
        == "PASS"
    )


def test_a_nan_in_a_variable_that_should_have_none_fails_and_in_cbh_is_reported() -> None:
    frame = _validation_frame(values=[0.2, float("nan"), 0.4], variable_hours=3)

    assert (
        validator.check_missing_values(frame=frame, variable=VARIABLES_BY_NAME["tcc"], tcc=None)[0]
        == "FAIL"
    )
    status, details = validator.check_missing_values(
        frame=frame, variable=VARIABLES_BY_NAME["cbh"], tcc=None
    )
    assert status == "INFO"
    assert details["nan_rows"] == 1


def test_the_row_check_fails_when_one_hour_of_one_cell_is_missing() -> None:
    first, last = FIRST, FIRST + timedelta(hours=N_TIMES - 1)
    cells = pl.DataFrame(
        {
            "latitude": [lat for lat in GRID_LATITUDES for _ in GRID_LONGITUDES],
            "longitude": [lon for _ in GRID_LATITUDES for lon in GRID_LONGITUDES],
        }
    )
    times = pl.datetime_range(first, last, "1h", eager=True).to_frame("time")
    full = (
        times.join(cells, how="cross")
        .with_columns(value=pl.lit(0.1, dtype=pl.Float32), expver=pl.lit("0001"))
        .select("time", "latitude", "longitude", "value", "expver")
    )

    assert validator.check_rows_and_keys(frame=full, first_hour=first, last_hour=last)[0] == "PASS"
    assert (
        validator.check_rows_and_keys(frame=full.slice(1), first_hour=first, last_hour=last)[0]
        == "FAIL"
    )


def test_the_float32_tolerance_is_a_millionth_of_the_largest_finite_value() -> None:
    values = pl.Series([0.0, -2.0e6, float("nan")], dtype=pl.Float32)

    assert validator.float32_tolerance(values=values) == pytest.approx(2.0)


@pytest.mark.parametrize("engine", ["scipy", "netcdf4"])
def test_an_archive_written_as_netcdf_is_read_back_with_its_units_and_expver(
    tmp_path: Path, engine: Literal["scipy", "netcdf4"]
) -> None:
    """Scipy writes netCDF3; netCDF4 writes the HDF5 files CDS serves, and is skipped if absent."""
    if engine == "netcdf4":
        pytest.importorskip("netCDF4")
    dataset = _dataset(expver=None, with_dimension=True)
    dataset["tcc"].attrs["units"] = "(0 - 1)"
    netcdf = tmp_path / "data.nc"
    dataset.to_netcdf(netcdf, engine=engine)
    archive = tmp_path / "chunk.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.write(netcdf, "data_stream-oper_stepType-instant.nc")

    parsed = read_archive(path=archive, scratch_dir=tmp_path / "scratch")

    assert parsed.units == {"tcc": "(0 - 1)"}
    frame = parsed.frames["tcc"]
    assert frame.height == N_TIMES * CELL_COUNT
    assert frame.group_by("expver").len().sort("expver")["len"].to_list() == [
        4 * CELL_COUNT,
        2 * CELL_COUNT,
    ]


def test_the_pilot_is_one_month_of_the_three_groups_and_never_mixes_kinds() -> None:
    chunks = pilot_chunks()

    assert [(c.tier, c.kind) for c in chunks] == [
        ("tier1a", "instantaneous"),
        ("tier1b", "accumulation"),
        ("tier2", "instantaneous"),
    ]
    assert {c.period.name for c in chunks} == {"2025_06_06"}
    assert chunks[1].variables == ("ssrdc", "cdir", "strd")


def test_no_planned_archive_holds_both_an_accumulation_and_an_instantaneous_field() -> None:
    for chunk in [*plan_chunks(tiers=TIERS), *pilot_chunks()]:
        kinds = {VARIABLES_BY_NAME[name].kind for name in chunk.variables}
        assert len(kinds) == 1


def test_the_cloud_family_checks_the_window_boundary_hours_and_the_hour_after() -> None:
    assert SEAM_HOURS["cloud"] == (6, 7, 9, 10, 18, 19, 21, 22)
    assert SEAM_HOURS["analysed"] == (9, 10, 21, 22)


def test_the_uvb_limit_admits_a_sunny_summer_noon() -> None:
    assert (VARIABLES_BY_NAME["uvb"].maximum or 0) >= 2.5e5


def test_each_record_is_passed_on_as_soon_as_its_chunk_finishes(tmp_path: Path) -> None:
    chunks = plan_chunks(tiers=["tier1a"], first_month=(2025, 6), last_month=(2025, 7))
    seen: list[tuple[str, int]] = []

    def download(_: PlannedChunk, path: Path) -> None:
        seen.append(("download", len(seen)))
        _write_valid_archive(path)

    run_chunks(
        chunks=chunks,
        chunk_dir=tmp_path,
        download=download,
        log=lambda _: None,
        on_record=lambda record: seen.append((record.chunk_id, len(seen))),
    )

    assert [name for name, _ in seen] == [
        "download",
        chunks[0].chunk_id,
        "download",
        chunks[1].chunk_id,
    ]


class _FakeRemote:
    def __init__(self, *, ready_after: int, fail_download: bool = False) -> None:
        self.request_id = "job-1"
        self.deleted = False
        self._checks = 0
        self._ready_after = ready_after
        self._fail_download = fail_download

    @property
    def results_ready(self) -> bool:
        self._checks += 1
        return self._checks > self._ready_after

    def get_results(self) -> _FakeRemote:
        return self

    def download(self, target: str) -> None:
        if self._fail_download:
            raise OSError("broken")
        Path(target).write_bytes(b"x")

    def delete(self) -> None:
        self.deleted = True


class _FakeClient:
    def __init__(self, remote: _FakeRemote) -> None:
        self.remote = remote

    def submit(self, collection: str, request: dict[str, object]) -> _FakeRemote:
        return self.remote


def _retrieve(
    *,
    remote: _FakeRemote,
    tmp_path: Path,
    sleep: Callable[[float], None] = lambda _: None,
    **kwargs: float,
) -> None:
    retrieve_with_cleanup(
        client=_FakeClient(remote),
        collection="c",
        request={},
        destination=tmp_path / "out.zip",
        log=lambda _: None,
        sleep=sleep,
        **kwargs,
    )


def test_a_finished_job_is_downloaded_and_not_deleted(tmp_path: Path) -> None:
    remote = _FakeRemote(ready_after=2)

    _retrieve(remote=remote, tmp_path=tmp_path)

    assert (tmp_path / "out.zip").exists()
    assert not remote.deleted


def test_an_interrupt_while_waiting_deletes_the_remote_job(tmp_path: Path) -> None:
    remote = _FakeRemote(ready_after=10)

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _retrieve(remote=remote, tmp_path=tmp_path, sleep=interrupt)
    assert remote.deleted


def test_a_wait_over_the_limit_times_out_and_deletes_the_job(tmp_path: Path) -> None:
    remote = _FakeRemote(ready_after=10_000)

    with pytest.raises(TimeoutError):
        _retrieve(remote=remote, tmp_path=tmp_path, max_wait_seconds=90, poll_seconds=30)
    assert remote.deleted


def test_a_failed_download_deletes_the_remote_job(tmp_path: Path) -> None:
    remote = _FakeRemote(ready_after=0, fail_download=True)

    with pytest.raises(OSError, match="broken"):
        _retrieve(remote=remote, tmp_path=tmp_path)
    assert remote.deleted


def test_eac4_asks_for_zipped_netcdf_and_has_no_download_format() -> None:
    body = eac4.request_body(
        chunk=eac4.plan_chunks(first_month=(2019, 9), last_month=(2019, 12))[0]
    )

    assert body["data_format"] == "netcdf_zip"
    assert "download_format" not in body


def test_the_eac4_end_month_falls_back_to_the_form_widgets_max_end() -> None:
    form = [
        {"name": "variable", "details": {}},
        {"name": "date", "details": {"maxEnd": "2025-12-31"}},
    ]

    assert eac4.parse_form_end(form=form) == (2025, 12)
    with pytest.raises(ValueError, match="maxEnd"):
        eac4.parse_form_end(form=[{"name": "x", "details": {}}])
