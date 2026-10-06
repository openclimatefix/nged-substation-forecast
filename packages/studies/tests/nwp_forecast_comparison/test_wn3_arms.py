"""Tests for the WeatherNext 3 build, verify and fit code under `studies/nwp_forecast_comparison/`.

Each test is written to fail on the defect it names. The store is a tiny synthetic copy whose
value at each run and lead is a known number, so a wrong run, lead, unit or site shows up as a
wrong number.
"""

import importlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final

import h3
import numpy as np
import polars as pl
import pytest
import xarray as xr

from studies import ens_members

RUNS: Final[np.ndarray] = np.array(["2026-03-01T00", "2026-03-02T00"], dtype="datetime64[h]")
"""The two 00 UTC runs of the synthetic copy."""

RUN_STRIDE: Final[float] = 1000.0
"""How far apart, in the synthetic values, consecutive runs are."""


def _load(*, name: str) -> ModuleType:
    """Import a study script by name, from the study folder pytest puts on `sys.path`."""
    return importlib.import_module(name)


efh = _load(name="ens_forecast_horizons")
w = _load(name="build_wn3_inputs")
fa = _load(name="fit_aifs")
vw = _load(name="verify_wn3_steps")
charts = _load(name="nwp_forecast_charts")
bfi = _load(name="build_forecast_inputs")
driver = _load(name="fit_day5_aifs_wn3")

DAY5_FOLDER_NAME: Final[str] = driver.OUTPUT_DIR.name
"""The real day-5 folder's name, which the tests recreate under a temporary directory."""


@pytest.fixture(autouse=True)
def day5_folder_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both day-5 guards at a temporary folder, because the guards compare paths."""
    folder = tmp_path / DAY5_FOLDER_NAME
    monkeypatch.setattr(driver, "OUTPUT_DIR", folder)
    monkeypatch.setattr(w, "NFC_DAY5_AIFS_WN3_DIR", folder)


def _dataset(*, nan_cell: tuple[int, int] | None = None) -> xr.Dataset:
    """Return a 2-run, 360-lead, 2x2-cell copy whose value is `RUN_STRIDE * run + lead`.

    Radiation is that value times 3600 (J m-2), temperature is 273.15 plus it (K), and every wind
    component is 3 (u) or 4 (v), so the speed is 5 everywhere. `nan_cell` blanks one cell.
    """
    leads = np.arange(1, w.N_LEADS + 1)
    base = RUN_STRIDE * np.arange(len(RUNS))[:, None] + leads[None, :]
    shape = (len(RUNS), len(leads), 2, 2)
    grid = np.broadcast_to(base[:, :, None, None], shape).astype(np.float32)
    values = {
        w.RADIATION: grid * 3600.0,
        w.TEMPERATURE: grid + np.float32(w.KELVIN),
        **{
            name: np.full(shape, u, dtype=np.float32)
            for name, u in zip(w.WIND_COMPONENTS["100m"], (3.0, 4.0), strict=True)
        },
        **{
            name: np.full(shape, u, dtype=np.float32)
            for name, u in zip(w.WIND_COMPONENTS["10m"], (6.0, 8.0), strict=True)
        },
    }
    if nan_cell is not None:
        for array in values.values():
            array[:, :, *nan_cell] = np.nan
    dims = (w.fetch.INIT_TIME, w.fetch.LEAD_TIME, w.fetch.LATITUDE, w.fetch.LONGITUDE)
    return xr.Dataset(
        {name: (dims, array) for name, array in values.items()},
        coords={
            w.fetch.INIT_TIME: RUNS.astype("datetime64[ns]"),
            w.fetch.LEAD_TIME: leads,
            w.fetch.LATITUDE: [0.0, 0.1],
            w.fetch.LONGITUDE: [0.0, 0.1],
        },
    )


def _weights(*, cells: list[tuple[str, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        cells,
        schema=["site", "lat_index", "lon_index", "weight"],
        orient="row",
    ).with_columns(pl.col("lat_index", "lon_index").cast(pl.Int64))


def test_site_cube_nan_outside_a_sites_cells_does_not_reach_the_site() -> None:
    dataset = _dataset(nan_cell=(1, 1))
    weights = _weights(cells=[("A", 0, 0, 1.0), ("B", 1, 1, 1.0)])
    cube = w.site_cube(dataset=dataset, weights=weights, sites=["A", "B"], name=w.RADIATION)
    assert np.isfinite(cube[..., 0]).all()
    assert np.isnan(cube[..., 1]).all()


def test_site_cube_is_the_overlap_weighted_mean() -> None:
    dataset = _dataset()
    weights = _weights(cells=[("A", 0, 0, 0.25), ("A", 1, 1, 0.75)])
    cube = w.site_cube(dataset=dataset, weights=weights, sites=["A"], name=w.TEMPERATURE)
    assert cube[1, 9, 0] == pytest.approx(RUN_STRIDE + 10 + w.KELVIN, rel=1e-6)


def _keys(*, times: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"site": ["A"] * len(times), "time": times}).with_columns(
        pl.col("time").dt.replace_time_zone("UTC")
    )


def _cubes(*, dataset: xr.Dataset) -> dict[str, np.ndarray]:
    weights = _weights(cells=[("A", 0, 0, 1.0)])
    return {
        name: w.site_cube(dataset=dataset, weights=weights, sites=["A"], name=name)
        for name in w.COPIED_VARIABLES
    }


def test_solar_hour_reads_the_run_of_the_day_before_and_converts_the_unit() -> None:
    dataset = _dataset()
    # The hour ending 12:00 on 3 March is labelled by its end, so its own day is 3 March and the
    # day-1 run is the one of 2 March, 36 h before.
    keys = _keys(times=[datetime(2026, 3, 3, 12)])
    frame = w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=keys,
        domain="solar",
        day=1,
    )
    assert frame["wn3_mean_day1_ghi"][0] == pytest.approx(RUN_STRIDE + 36, rel=1e-5)
    # The temperature is the mean of the two hourly values that bracket the hour's midpoint.
    assert frame["wn3_mean_day1_temp"][0] == pytest.approx(RUN_STRIDE + 35.5, rel=1e-5)
    assert frame["wn3_mean_day1_init_time"][0] == datetime(2026, 3, 2, tzinfo=UTC)


def test_solar_midnight_hour_reads_the_run_of_its_own_start_day() -> None:
    dataset = _dataset()
    # The hour ending 00:00 on 3 March starts on 2 March, so the day-1 run is 1 March, 48 h back.
    keys = _keys(times=[datetime(2026, 3, 3, 0)])
    frame = w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=keys,
        domain="solar",
        day=1,
    )
    assert frame["wn3_mean_day1_ghi"][0] == pytest.approx(48.0, rel=1e-5)
    assert frame["wn3_mean_day1_init_time"][0] == datetime(2026, 3, 1, tzinfo=UTC)


def test_wind_speed_is_the_length_of_the_mean_vector_and_direction_follows_the_convention() -> None:
    dataset = _dataset()
    keys = _keys(times=[datetime(2026, 3, 3, 12)])
    frame = w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=keys,
        domain="wind",
        day=1,
    )
    assert frame["wn3_mean_day1_speed_100m"][0] == pytest.approx(5.0)
    assert frame["wn3_mean_day1_sin_100m"][0] == pytest.approx(-3.0 / 5.0)
    assert frame["wn3_mean_day1_cos_100m"][0] == pytest.approx(-4.0 / 5.0)
    assert frame["wn3_mean_day1_speed_10m"][0] == pytest.approx(10.0)


def test_hour_without_a_copied_run_is_null_not_a_neighbouring_runs_value() -> None:
    dataset = _dataset()
    # The day-1 run of 5 March is 4 March, which the copy does not hold.
    keys = _keys(times=[datetime(2026, 3, 5, 12), datetime(2026, 3, 3, 12)])
    frame = w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=keys,
        domain="solar",
        day=1,
    )
    assert frame["wn3_mean_day1_ghi"][0] is None
    assert frame["wn3_mean_day1_init_time"][0] is None
    assert frame["wn3_mean_day1_ghi"][1] is not None


def test_identity_check_passes_on_built_values_and_fails_on_a_shifted_lead() -> None:
    dataset = _dataset()
    weights = _weights(cells=[("A", 0, 0, 0.5), ("A", 1, 0, 0.5)])
    times = [datetime(2026, 3, 3, hour) for hour in range(1, 24)] + [
        datetime(2026, 3, 2 + day, hour) for day in (1, 2) for hour in range(1, 24)
    ]
    keys = _keys(times=times * 12)
    built = w.wn3_arm_frame(
        cubes={
            name: w.site_cube(dataset=dataset, weights=weights, sites=["A"], name=name)
            for name in w.COPIED_VARIABLES
        },
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=keys,
        domain="solar",
        day=1,
    )
    w.check_against_store(built=built, dataset=dataset, weights=weights, domain="solar", day=1)
    wrong = built.with_columns(pl.col("wn3_mean_day1_ghi") + 1.0)
    with pytest.raises(ValueError, match="differ from the local copy"):
        w.check_against_store(built=wrong, dataset=dataset, weights=weights, domain="solar", day=1)


def test_check_runs_requires_the_wn3_stamp() -> None:
    frame = pl.DataFrame(
        {"time": [datetime(2026, 3, 2, 12, tzinfo=UTC)], "era_code": [0]},
        schema_overrides={"era_code": pl.Int8},
    )
    with pytest.raises(ValueError, match="init_time is missing"):
        fa.check_runs(frame=frame, domain="wind", row_set="wn3", arms=("wn3_mean_day1",))


def test_wn3_arms_add_the_mean_vector_reference_for_wind_at_every_day_and_never_for_solar() -> None:
    for day in fa.WN3_DAYS:
        assert f"ens_meanvec_day{day}" in fa.wn3_arms(domain="wind", day=day)
        assert not any("meanvec" in arm for arm in fa.wn3_arms(domain="solar", day=day))


def test_the_wn3_fit_covers_days_one_two_seven_and_fourteen() -> None:
    assert fa.WN3_DAYS == (1, 2, 7, 14)
    assert w.WN3_DAYS == (1, 2, 7, 14)


def test_wn3_row_set_is_kept_out_of_the_old_row_set_loops() -> None:
    assert "wn3" not in fa.ROW_SETS
    assert fa.ROW_SET_SPECS["wn3"].deciding is None


def test_offset_check_picks_the_true_offset() -> None:
    times = [datetime(2026, 3, 3, hour, tzinfo=UTC) for hour in range(6, 19)]
    ghi = [float(x * x % 7) + 10.0 * x for x in range(len(times))]
    inputs = pl.DataFrame({"site": "A", "time": times, "wn3_mean_day1_ghi": ghi})
    era5 = pl.DataFrame({"site": "A", "time": times, "era5": ghi})
    scores = vw.offset_scores(inputs=inputs, era5=era5, domain="solar", day=1)
    assert vw.best_is_zero(scores=scores, domain="solar")
    shifted = era5.with_columns(pl.col("time") + pl.duration(hours=1))
    scores = vw.offset_scores(inputs=inputs, era5=shifted, domain="solar", day=1)
    assert not vw.best_is_zero(scores=scores, domain="solar")


def test_minor_grid_is_every_half_point_that_is_not_a_tick() -> None:
    values = charts.minor_grid_values(x_ticks=[8.0, 9.0, 10.0], x_domain=(6.0, 11.0))
    assert values == [8.5, 9.5, 10.5, 11.0]


def test_row_set_rows_name_the_months_and_keep_the_same_rows_ens_mean_as_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canned = pl.DataFrame(
        {
            "product": ["ENS mean", "ENS mean", "WeatherNext 3 mean", "WeatherNext 3 mean"],
            "day": [1, 2, 1, 2],
            "value": [9.0, 10.0, 8.0, 9.5],
            "lower_95": [8.0, 9.0, 7.0, 8.5],
            "upper_95": [10.0, 11.0, 9.0, 10.5],
            "n_months": [7, 7, 7, 7],
        }
    )
    monkeypatch.setattr(charts, "lead_board_rows", lambda *, losses: canned)
    losses = pl.DataFrame({"arm": ["wn3_mean_day1", "ens_mean_day1", "unrelated_day1"]})
    rows = charts.row_set_board_rows(marks=[charts.RowSetMarks(slug="wn3_mean", losses=losses)])
    assert set(rows["product"]) == {"WeatherNext 3 mean (7 months)"}
    marks = rows.filter(pl.col("kind") == "mark")
    ticks = rows.filter(pl.col("kind") == "ens_same_rows")
    assert marks.sort("day")["value"].to_list() == [8.0, 9.5]
    assert ticks.sort("day")["value"].to_list() == [9.0, 10.0]


def test_row_set_rows_refuse_ticks_from_other_days_than_the_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canned = pl.DataFrame(
        {
            "product": ["ENS mean", "WeatherNext 3 mean", "WeatherNext 3 mean"],
            "day": [1, 1, 2],
            "value": [9.0, 8.0, 9.5],
            "lower_95": [8.0, 7.0, 8.5],
            "upper_95": [10.0, 9.0, 10.5],
            "n_months": [7, 7, 7],
        }
    )
    monkeypatch.setattr(charts, "lead_board_rows", lambda *, losses: canned)
    losses = pl.DataFrame({"arm": ["wn3_mean_day1"]})
    with pytest.raises(ValueError, match="other \\(lead day, months\\)"):
        charts.row_set_board_rows(marks=[charts.RowSetMarks(slug="wn3_mean", losses=losses)])


def test_row_set_marks_refuse_losses_with_no_device_column(tmp_path: Path) -> None:
    for day in (1, 2, 7, 14):
        pl.DataFrame(
            {
                "arm": [f"wn3_mean_day{day}"],
                "site": ["A"],
                "time": [datetime(2026, 9, 1, 12, tzinfo=UTC)],
                "month": ["2026-09"],
            }
        ).write_parquet(tmp_path / f"solar_wn3_day{day}_losses.parquet")
    with pytest.raises(ValueError, match="no device column"):
        charts.load_row_set_marks(blends_dir=None, wn3_dir=tmp_path, domain="solar")


def _wind_extract() -> pl.DataFrame:
    """Return a 51-member extract at two sites and two runs, 3-hourly to 72 h.

    Site `A` has every member blowing at 90 degrees at the speed `lead / 10 + 1000 * run`, so the
    mean vector's length identifies the lead and the run. Site `B` has 26 members at 0 degrees and
    25 at 180 degrees, all at one speed, so the mean vector is one member's length over 51.
    """
    records = []
    for run in range(2):
        init = datetime(2026, 3, 1 + run)
        for lead in range(0, 73, 3):
            for member in range(efh.ENSEMBLE_SIZE):
                a_speed = lead / 10 + 1000.0 * run
                b_direction = 0.0 if member % 2 == 0 else 180.0
                for site, speed, direction in (
                    ("A", a_speed, 90.0),
                    ("B", 51.0, b_direction),
                ):
                    records.append(
                        {
                            "site": site,
                            "init_time": init,
                            "ensemble_member": member,
                            "lead_hours": lead,
                            "speed_100m": speed,
                            "direction_100m": direction,
                            "speed_10m": speed,
                            "direction_10m": direction,
                        }
                    )
    return pl.DataFrame(records)


def test_ens_mean_vector_frame_keeps_each_site_run_and_lead_together() -> None:
    frame = w.ens_vector_mean_frame(extract=_wind_extract(), day=1)
    speed = "ens_meanvec_day1_speed_100m"
    stamp = "ens_meanvec_day1_init_time"
    for run in range(2):
        init = datetime(2026, 3, 1 + run, tzinfo=UTC)
        for hour in (0, 7, 23):
            time = init + timedelta(hours=24 + hour)
            row = frame.filter((pl.col("site") == "A") & (pl.col("time") == time))
            assert row[speed][0] == pytest.approx((24 + hour) / 10 + 1000.0 * run, rel=1e-6)
            assert row[stamp][0] == init


def test_ens_mean_vector_frame_speed_is_the_length_of_the_mean_of_member_vectors() -> None:
    frame = w.ens_vector_mean_frame(extract=_wind_extract(), day=1)
    site_b = frame.filter(pl.col("site") == "B")
    assert site_b["ens_meanvec_day1_speed_100m"].to_list() == pytest.approx(
        [1.0] * site_b.height, rel=1e-5
    )


def _store_axes(*, written: list[bool], source_shift: int = 0) -> tuple[np.ndarray, ...]:
    first = int(np.datetime64("2026-01-01T00", "h").astype("int64"))
    hours = first + 24 * np.arange(len(written))
    source = hours.copy()
    source[1] += source_shift
    return hours, np.array(written), source


def test_run_provenance_accepts_written_runs_followed_by_unwritten_future_slots() -> None:
    hours, written, source = _store_axes(written=[True, True, True, False, False])
    w.log_run_provenance(init_hours=hours, written=written, source_hours=source)


def test_run_provenance_stops_on_a_gap_before_the_last_written_run() -> None:
    hours, written, source = _store_axes(written=[True, False, True, False])
    with pytest.raises(ValueError, match="1 00 UTC runs are unwritten"):
        w.log_run_provenance(init_hours=hours, written=written, source_hours=source)


def test_run_provenance_stops_when_a_source_init_time_differs_from_the_init_time() -> None:
    hours, written, source = _store_axes(written=[True, True, True], source_shift=-6)
    with pytest.raises(ValueError, match="1 have a source_init_time"):
        w.log_run_provenance(init_hours=hours, written=written, source_hours=source)


def test_physical_range_rejects_radiation_left_in_joules() -> None:
    frame = pl.DataFrame(
        {"wn3_mean_day1_ghi": [0.0, 900.0 * 3600], "wn3_mean_day1_temp": [10.0, 12.0]}
    )
    with pytest.raises(ValueError, match="outside"):
        w.check_physical_range(built=frame, domain="solar", day=1)


def test_physical_range_rejects_radiation_in_the_wrong_small_unit() -> None:
    frame = pl.DataFrame({"wn3_mean_day1_ghi": [0.0, 0.25], "wn3_mean_day1_temp": [10.0, 12.0]})
    with pytest.raises(ValueError, match="unit is probably wrong"):
        w.check_physical_range(built=frame, domain="solar", day=1)


def test_physical_range_rejects_temperature_left_in_kelvin() -> None:
    frame = pl.DataFrame({"wn3_mean_day1_ghi": [0.0, 800.0], "wn3_mean_day1_temp": [283.0, 285.0]})
    with pytest.raises(ValueError, match="outside"):
        w.check_physical_range(built=frame, domain="solar", day=1)


def test_physical_range_accepts_plausible_values() -> None:
    frame = pl.DataFrame({"wn3_mean_day1_ghi": [0.0, 800.0], "wn3_mean_day1_temp": [3.0, 25.0]})
    w.check_physical_range(built=frame, domain="solar", day=1)


def test_the_mean_vector_reference_is_refitted_at_the_sensitivity_setting_for_wind_only() -> None:
    """The planned reference is refitted even when no contrast is near the 5% line."""
    times = [datetime(2026, month, 10, 12, tzinfo=UTC) for month in (2, 3, 4, 6, 7, 8, 9)]
    for domain, expected in (("wind", True), ("solar", False)):
        arms = fa.wn3_arms(domain=domain, day=1)
        losses = pl.concat(
            [
                pl.DataFrame(
                    {
                        "arm": arm,
                        "site": "A",
                        "time": times,
                        "month": [f"{time:%Y-%m}" for time in times],
                        "seed": 0,
                        "setting": "primary",
                        fa.METRIC: 1.0 + 10.0 * position,
                        "device": "cuda",
                    },
                    schema_overrides={"time": pl.Datetime("us", "UTC")},
                )
                for position, arm in enumerate(arms)
            ]
        )

        refitted = fa.wn3_sensitivity_arms(losses=losses, domain=domain, day=1)

        assert ("ens_meanvec_day1" in refitted) is expected


def test_a_missing_wn3_run_stops_the_fit_and_names_its_date() -> None:
    frame = pl.DataFrame(
        {
            "time": [datetime(2026, 3, 5, 12, tzinfo=UTC), datetime(2026, 3, 6, 12, tzinfo=UTC)],
            "wn3_mean_day1_init_time": [datetime(2026, 3, 4, tzinfo=UTC), None],
        },
        schema_overrides={"wn3_mean_day1_init_time": pl.Datetime("us", "UTC")},
    )

    with pytest.raises(ValueError, match=r"wn3_mean_day1: .*2026-03-05"):
        fa.check_wn3_runs_present(frame=frame, domain="wind", arms=("wn3_mean_day1",))


def test_a_missing_solar_run_is_named_by_the_hour_start_not_the_hour_end() -> None:
    """The hour ending 00:00 on 6 March starts on 5 March, so day 1 reads the run of 4 March."""
    frame = pl.DataFrame(
        {
            "time": [datetime(2026, 3, 6, 0, tzinfo=UTC)],
            "wn3_mean_day1_init_time": [None],
        },
        schema_overrides={"wn3_mean_day1_init_time": pl.Datetime("us", "UTC")},
    )

    with pytest.raises(ValueError, match="2026-03-04"):
        fa.check_wn3_runs_present(frame=frame, domain="solar", arms=("wn3_mean_day1",))


def test_the_crop_weights_join_a_float32_point_one_degree_grid() -> None:
    """Float32 0.1 degree coordinates must still meet the Float64 H3 grid after rounding."""
    cell = h3.str_to_int(h3.latlng_to_cell(53.2, -0.5, 5))
    overlap = bfi.compute_h3_grid_weights(nwp_grid_size_degrees=0.1, h3_index=[cell])
    cells = (
        overlap.select(latitude=pl.col("nwp_lat"), longitude=pl.col("nwp_lon"))
        .unique()
        .with_row_index("lat_index")
        .with_columns(lon_index=pl.col("lat_index"))
        .with_columns(pl.col("latitude", "longitude").cast(pl.Float32))
    )

    weights = bfi._h3_crop_weights(site_cells={"A": cell}, grid_cells=cells, grid_degrees=0.1)

    assert weights["weight"].sum() == pytest.approx(1.0)


def _day0_frame(*, domain: str, hours: list[int]) -> pl.DataFrame:
    dataset = _dataset()
    return w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=_keys(times=[datetime(2026, 3, 2, hour) for hour in hours]),
        domain=domain,
        day=0,
    )


def test_wind_day_0_has_no_value_at_00_utc_and_reads_lead_1_at_01_utc() -> None:
    frame = _day0_frame(domain="wind", hours=[0, 1])
    assert frame["wn3_mean_day0_speed_100m"][0] is None
    assert frame["wn3_mean_day0_init_time"][0] is None
    assert frame["wn3_mean_day0_speed_100m"][1] == pytest.approx(5.0)
    assert frame["wn3_mean_day0_init_time"][1] == datetime(2026, 3, 2, tzinfo=UTC)


def test_solar_day_0_has_no_value_at_01_utc_and_reads_lead_2_at_02_utc() -> None:
    frame = _day0_frame(domain="solar", hours=[1, 2])
    assert frame["wn3_mean_day0_ghi"][0] is None
    assert frame["wn3_mean_day0_temp"][0] is None
    assert frame["wn3_mean_day0_init_time"][0] is None
    # The hour ending 02:00 reads lead 2 of the run of its own day (run 1 of the copy).
    assert frame["wn3_mean_day0_ghi"][1] == pytest.approx(RUN_STRIDE + 2, rel=1e-5)
    assert frame["wn3_mean_day0_temp"][1] == pytest.approx(RUN_STRIDE + 1.5, rel=1e-5)


def test_lean_inputs_reads_ens_at_days_4_and_5_from_their_native_steps_not_the_6_hourly_emulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = {"site": ["A"], "time": [datetime(2026, 3, 5, 12, tzinfo=UTC)]}
    pl.DataFrame(
        {
            **keys,
            "ens_mean6_day4_ghi": [111.0],
            "ens_mean6_day5_ghi": [112.0],
            "ens_mean6_day10_ghi": [10.0],
            "aifs_single_day4_ghi": [5.0],
        }
    ).write_parquet(tmp_path / "solar_aifs_inputs.parquet")
    pl.DataFrame(
        {**keys, "ens_mean_day5_ghi": [333.0], "ens_mean_day10_ghi": [10.0]}
    ).write_parquet(tmp_path / "solar_extra_lead_inputs.parquet")
    seen: dict[str, object] = {}

    def native(
        *,
        keys: pl.DataFrame,
        domain: str,
        mean_days: tuple[int, ...],
        control_days: tuple[int, ...],
        keep_init_time: bool = False,
    ) -> pl.DataFrame:
        seen.update(mean_days=mean_days, control_days=control_days, keep=keep_init_time)
        return keys.with_columns(ens_mean_day4_ghi=pl.lit(222.0), ens_mean_day5_ghi=pl.lit(333.0))

    monkeypatch.setattr(fa, "_ens_extra_frame", native)
    frame = fa.lean_inputs(aifs_dir=tmp_path, leads_day10_dir=tmp_path, domain="solar")
    # The run stamp must be requested, or `check_runs` cannot see the native run's date.
    assert seen == {"mean_days": (4, 5), "control_days": (), "keep": True}
    assert frame["ens_mean_day4_ghi"][0] == 222.0
    assert frame["ens_mean_day5_ghi"][0] == 333.0
    assert frame["ens_mean_day10_ghi"][0] == 10.0


def test_lean_inputs_stops_when_the_day_10_ens_read_differs_from_the_extra_lead_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = {"site": ["A"], "time": [datetime(2026, 3, 5, 12, tzinfo=UTC)]}
    pl.DataFrame({**keys, "ens_mean6_day10_ghi": [10.0]}).write_parquet(
        tmp_path / "solar_aifs_inputs.parquet"
    )
    pl.DataFrame({**keys, "ens_mean_day10_ghi": [11.0]}).write_parquet(
        tmp_path / "solar_extra_lead_inputs.parquet"
    )
    monkeypatch.setattr(fa, "_ens_extra_frame", lambda *, keys, **_: keys)
    with pytest.raises(ValueError, match="differ"):
        fa.lean_inputs(aifs_dir=tmp_path, leads_day10_dir=tmp_path, domain="solar")


def test_lean_inputs_stops_when_the_day_5_ens_read_differs_from_the_extra_lead_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = {"site": ["A"], "time": [datetime(2026, 3, 5, 12, tzinfo=UTC)]}
    pl.DataFrame({**keys, "ens_mean6_day5_ghi": [9.0]}).write_parquet(
        tmp_path / "solar_aifs_inputs.parquet"
    )
    pl.DataFrame({**keys, "ens_mean_day5_ghi": [11.0]}).write_parquet(
        tmp_path / "solar_extra_lead_inputs.parquet"
    )
    monkeypatch.setattr(
        fa,
        "_ens_extra_frame",
        lambda *, keys, **_: keys.with_columns(ens_mean_day5_ghi=pl.lit(12.0)),
    )
    with pytest.raises(ValueError, match="ens_mean_day5"):
        fa.lean_inputs(aifs_dir=tmp_path, leads_day10_dir=tmp_path, domain="solar")


def test_every_lean_day_has_exactly_one_source_of_its_ens_mean() -> None:
    for day in fa.LEAN_DAYS:
        sources = [
            day in bfi.ENS_DAYS,
            day in fa.LEAN_ENS_NATIVE_DAYS,
            day in fa.LEAN_ENS_BUILT_DAYS,
        ]
        assert sum(sources) == 1, day


def test_the_day_5_fit_days_have_exactly_one_source_of_their_ens_mean() -> None:
    assert driver.DAY5 == (5,)
    for day in driver.DAY5:
        sources = [
            day in bfi.ENS_DAYS,
            day in fa.LEAN_ENS_NATIVE_DAYS,
            day in fa.LEAN_ENS_BUILT_DAYS,
        ]
        assert sum(sources) == 1, day
        assert day in w.ENS_EXTRA_DAYS
        assert day not in bfi.ENS_DAYS


def test_run_lean_fits_each_row_set_at_each_day_and_stamps_the_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted: list[tuple[str, str, int, tuple[str, ...]]] = []
    dropped: dict[tuple[str, int], bool] = {}
    monkeypatch.setattr(fa, "lean_inputs", lambda **_: pl.DataFrame({"site": ["A"]}))

    def rows(
        *,
        domain: str,
        row_set: str,
        arms: tuple[str, ...],
        day: int,
        drop: pl.Expr | None = None,
        **_: object,
    ) -> pl.DataFrame:
        fitted.append((domain, row_set, day, arms))
        dropped[domain, day] = drop is not None
        return pl.DataFrame({"site": ["A"], "month": ["2026-03"]})

    monkeypatch.setattr(fa, "aifs_rows", rows)
    monkeypatch.setattr(fa, "build_stamp", lambda **_: {})
    monkeypatch.setattr(
        fa, "fit_jobs", lambda *, jobs, **_: pl.DataFrame({"arm": [arm for arm, _ in jobs]})
    )
    monkeypatch.setattr(fa, "predictions_from_losses", lambda **_: pl.DataFrame({"a": [1]}))
    monkeypatch.setattr(fa, "lean_stage_lines", lambda **_: [])
    assert (
        fa.run_lean(
            published_dir=tmp_path,
            output_dir=tmp_path,
            leads_day10_dir=tmp_path,
            workers=1,
            report_name="report_aifs.md",
        )
        == 0
    )
    assert (tmp_path / "report_aifs.md").exists()
    assert not (tmp_path / "report.md").exists()
    expected = {
        (domain, row_set, day, fa.lean_arms(row_set=row_set, day=day))
        for domain in fa.DOMAINS
        for row_set in fa.ROW_SETS
        for day in fa.LEAN_DAYS
    }
    assert set(fitted) == expected
    assert len(fitted) == len(expected)
    # Only solar day 0 drops rows (hours 1 to 6 UTC), so the arms are scored on the same rows.
    assert {key for key, value in dropped.items() if value} == {("solar", 0)}


def test_workers_argument_accepts_one_to_the_cap_and_refuses_the_rest() -> None:
    import argparse

    assert fa.workers_argument("1") == 1
    assert fa.workers_argument(str(fa.MAX_WORKERS)) == fa.MAX_WORKERS
    for bad in ("0", str(fa.MAX_WORKERS + 1), "-1", "two"):
        with pytest.raises(argparse.ArgumentTypeError):
            fa.workers_argument(bad)


def _steps(*, first_lead: float) -> object:
    return ens_members.Steps(
        keys=pl.DataFrame(),
        leads=np.array([first_lead, first_lead + 6.0]),
        widths=np.array([6, 6]),
        values={},
        ensemble_size=1,
    )


def test_a_solar_band_scoring_hours_before_its_first_step_is_refused() -> None:
    assert bfi.SOLAR_DAY0_FIRST_SCORED_LEAD == 7
    # Day 0 scores from the hour ending at lead 7, whose midpoint is 6.5, so a first step at lead 12
    # leaves the hours ending at leads 7 to 11 unread.
    with pytest.raises(ValueError, match="extrapolation"):
        bfi.check_first_step_reaches_targets(
            steps=_steps(first_lead=12.0), day=0, domain="solar", arm_prefix="aifs_single"
        )
    # A first step at lead 6 reaches the midpoint (6.5) of the first scored day-0 hour.
    bfi.check_first_step_reaches_targets(
        steps=_steps(first_lead=6.0), day=0, domain="solar", arm_prefix="aifs_single"
    )
    # A first step at lead 7 is after that midpoint.
    with pytest.raises(ValueError, match="extrapolation"):
        bfi.check_first_step_reaches_targets(
            steps=_steps(first_lead=7.0), day=0, domain="solar", arm_prefix="aifs_single"
        )
    # Day 1's first hour ends at lead 25, after a step at lead 24.
    bfi.check_first_step_reaches_targets(
        steps=_steps(first_lead=24.0), day=1, domain="solar", arm_prefix="aifs_single"
    )
    with pytest.raises(ValueError, match="extrapolation"):
        bfi.check_first_step_reaches_targets(
            steps=_steps(first_lead=30.0), day=1, domain="solar", arm_prefix="aifs_single"
        )
    # Wind reads at each hour's start, so a wind band is not checked here.
    bfi.check_first_step_reaches_targets(
        steps=_steps(first_lead=6.0), day=0, domain="wind", arm_prefix="aifs_single"
    )


def test_ens_member_arms_refuses_a_solar_day_0_band_whose_first_step_is_after_the_first_hour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bfi.efh, "clear_sky_table", lambda **_: pl.DataFrame())
    monkeypatch.setattr(bfi.efh, "band_steps", lambda **_: _steps(first_lead=12.0))
    with pytest.raises(ValueError, match="aifs_single day 0"):
        bfi.ens_member_arms(
            extract=pl.DataFrame(),
            domain="solar",
            days=(0,),
            method="linear",
            ensemble_size=1,
            arm_name=lambda way, day: f"aifs_single_day{day}",
            ways=("control",),
            fine_step_last_lead=0,
        )


def test_ens_member_arms_checks_six_hourly_steps_even_where_the_fine_steps_last_a_while(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ENS emulated on 6-hourly steps has fine_step_last_lead 144, so only `six_hourly` triggers it.
    monkeypatch.setattr(bfi.efh, "clear_sky_table", lambda **_: pl.DataFrame())
    monkeypatch.setattr(bfi.efh, "band_steps", lambda **_: _steps(first_lead=12.0))
    with pytest.raises(ValueError, match="ens_mean day 0"):
        bfi.ens_member_arms(
            extract=pl.DataFrame(),
            domain="solar",
            days=(0,),
            method="linear",
            ensemble_size=1,
            arm_name=lambda way, day: f"ens_{way}_day{day}",
            ways=("mean",),
            fine_step_last_lead=144,
            six_hourly=True,
        )


def test_wn3_rows_passes_the_day_0_drop_to_aifs_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[tuple[str, int], pl.Expr | None] = {}

    def rows(*, domain: str, day: int, drop: pl.Expr | None = None, **_: object) -> pl.DataFrame:
        seen[domain, day] = drop
        return pl.DataFrame()

    monkeypatch.setattr(fa, "aifs_rows", rows)
    for domain in fa.DOMAINS:
        for day in (0, 1):
            fa.wn3_rows(published_dir=tmp_path, inputs=pl.DataFrame(), domain=domain, day=day)
    assert {key for key, value in seen.items() if value is not None} == {
        ("solar", 0),
        ("wind", 0),
    }


def test_ens_members_fills_the_day_4_gap_with_the_supplement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def rows(leads: list[int]) -> pl.DataFrame:
        init = datetime(2026, 1, 1, tzinfo=UTC)
        return pl.DataFrame(
            [
                {
                    "site": "A",
                    "init_time": init,
                    "valid_time": init + timedelta(hours=lead),
                    "lead_hours": lead,
                    "ensemble_member": member,
                    "ghi_w_m2": 100.0,
                    "temp_c": 10.0,
                }
                for lead in leads
                for member in range(2)
            ]
        ).with_columns(pl.col("lead_hours").cast(pl.Int32))

    gap = {105, 108, 111}
    supplement = tmp_path / "supplement.parquet"
    rows(sorted(gap)).write_parquet(supplement)
    extract = rows([lead for lead in range(90, 127, 3) if lead not in gap])
    monkeypatch.setattr(
        bfi.efh,
        "members",
        lambda *, sites, source=None: (
            pl.read_parquet(source) if source is not None else extract
        ).filter(pl.col("site").is_in(sites)),
    )

    def day_4_leads(members: pl.DataFrame) -> list[float]:
        steps = bfi.efh.band_steps(
            members=members, day=4, domain="solar", ensemble_size=2, six_hourly=True
        )
        return steps.leads.tolist()

    monkeypatch.setattr(bfi, "ENS_DAY4_SUPPLEMENT_PATH", tmp_path / "absent.parquet")
    assert day_4_leads(bfi.ens_members(sites=["A"])) == [96.0, 102.0, 120.0, 126.0]
    monkeypatch.setattr(bfi, "ENS_DAY4_SUPPLEMENT_PATH", supplement)
    filled = bfi.ens_members(sites=["A"])
    assert filled.height == extract.height + 2 * len(gap)
    assert day_4_leads(filled) == [96.0, 102.0, 108.0, 114.0, 120.0, 126.0]


@pytest.mark.parametrize("six_hourly", [False, True])
def test_a_day_4_band_without_the_supplement_is_refused_and_with_it_accepted(
    monkeypatch: pytest.MonkeyPatch, six_hourly: bool
) -> None:
    def steps(leads: list[float]) -> object:
        return ens_members.Steps(
            keys=pl.DataFrame(),
            leads=np.array(leads),
            widths=np.full(len(leads), 6 if six_hourly else 3),
            values={},
            ensemble_size=1,
        )

    step = 6 if six_hourly else 3
    whole = [float(lead) for lead in range(96 if six_hourly else 90, 127, step)]
    holed = [lead for lead in whole if not 104 < lead < 112]
    bfi.check_no_step_gap(steps=steps(whole), day=4, arm_prefix="ens_mean")
    with pytest.raises(ValueError, match="fetch_ens_day4_supplement"):
        bfi.check_no_step_gap(steps=steps(holed), day=4, arm_prefix="ens_mean")

    monkeypatch.setattr(bfi.efh, "clear_sky_table", lambda **_: pl.DataFrame())
    monkeypatch.setattr(bfi.efh, "band_steps", lambda **_: steps(holed))
    with pytest.raises(ValueError, match="ens_mean day 4"):
        bfi.ens_member_arms(
            extract=pl.DataFrame(),
            domain="wind",
            days=(4,),
            method="linear",
            ensemble_size=1,
            arm_name=lambda way, day: f"ens_{way}_day{day}",
            ways=("mean",),
            six_hourly=six_hourly,
        )


def test_day_5_reads_the_run_five_days_before_at_lead_120_plus_the_hour() -> None:
    dataset = _dataset()
    # A wind hour at 12:00 on 6 March reads the 00 UTC run of 1 March at lead 5 * 24 + 12 = 132,
    # so a band one day off reads 108 or 156 instead.
    frame = w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=_keys(times=[datetime(2026, 3, 6, 12)]),
        domain="wind",
        day=5,
    )
    assert frame["wn3_mean_day5_init_time"][0] == datetime(2026, 3, 1, tzinfo=UTC)
    # The synthetic wind is the same at every lead, so the lead shows only in a solar read.
    solar = w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        keys=_keys(times=[datetime(2026, 3, 6, 12)]),
        domain="solar",
        day=5,
    )
    assert solar["wn3_mean_day5_ghi"][0] == pytest.approx(5 * 24 + 12, rel=1e-5)


def _holey_frame(*, domain: str, day: int, nan_cell: tuple[int, int] | None) -> pl.DataFrame:
    dataset = _dataset(nan_cell=nan_cell)
    return w.wn3_arm_frame(
        cubes=_cubes(dataset=dataset),
        runs=dataset[w.fetch.INIT_TIME].to_numpy().astype("datetime64[h]"),
        sites=["A"],
        # 6 March reads the run of 1 March at day 5; 9 March reads 4 March, which the copy lacks.
        keys=_keys(times=[datetime(2026, 3, 6, 12), datetime(2026, 3, 9, 12)]),
        domain=domain,
        day=day,
    )


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_a_nan_inside_a_stored_run_stops_the_build_and_names_the_band(domain: str) -> None:
    built = _holey_frame(domain=domain, day=5, nan_cell=(0, 0))

    with pytest.raises(ValueError, match=r"wn3_mean_day5: 1 rows .*day-5 band"):
        w.check_band_complete(built=built, domain=domain, day=5)


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_a_run_missing_from_the_copy_is_not_a_hole_inside_a_band(domain: str) -> None:
    built = _holey_frame(domain=domain, day=5, nan_cell=None)

    assert built["wn3_mean_day5_init_time"].null_count() == 1
    w.check_band_complete(built=built, domain=domain, day=5)


def test_the_day_5_fits_refuse_every_output_folder_but_their_own(tmp_path: Path) -> None:
    published = tmp_path / "nwp_forecast_comparison/original"
    for name in (
        "nwp_forecast_comparison/aifs_extra_days",
        "nwp_forecast_comparison/leads_day10",
    ):
        with pytest.raises(ValueError, match="writes only to"):
            driver.check_output_dir(output_dir=tmp_path / name, published_dir=published)
    driver.check_output_dir(output_dir=tmp_path / driver.OUTPUT_DIR.name, published_dir=published)


def test_the_joined_report_keeps_both_fits_under_one_title_and_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    (tmp_path / driver.AIFS_REPORT_NAME).write_text("# AIFS\n\n## Solar\n\ntext a")
    (tmp_path / driver.WN3_REPORT_NAME).write_text("# WN3\n\n## Wind\n\ntext b")

    path = driver.write_joined_report(output_dir=tmp_path)

    lines = path.read_text().splitlines()
    assert lines[0].startswith("# ")
    assert lines.count("## AIFS") == 1
    assert lines.count("### Solar") == 1
    assert lines.count("## WN3") == 1
    assert lines.index("## AIFS") < lines.index("## WN3")
    with pytest.raises(FileExistsError):
        driver.write_joined_report(output_dir=tmp_path)


@pytest.mark.parametrize(
    ("domain", "column"),
    [("solar", "temp"), ("solar", "ghi"), ("wind", "speed_10m"), ("wind", "speed_100m")],
)
def test_a_hole_in_any_one_value_column_stops_the_build(domain: str, column: str) -> None:
    built = _holey_frame(domain=domain, day=5, nan_cell=None).with_columns(
        pl.when(pl.col("time").dt.day() == 6)
        .then(None)
        .otherwise(pl.col(f"wn3_mean_day5_{column}"))
        .alias(f"wn3_mean_day5_{column}")
    )

    with pytest.raises(ValueError, match="day-5 band"):
        w.check_band_complete(built=built, domain=domain, day=5)


def test_the_day_5_wn3_build_refuses_any_output_folder_but_its_own(tmp_path: Path) -> None:
    published = tmp_path / "nwp_forecast_comparison/original"
    with pytest.raises(ValueError, match="day 5 builds only into"):
        w.build_domain(
            domain="solar",
            published_dir=published,
            output_dir=tmp_path / "nwp_forecast_comparison/wn3_extra_days",
            weather_dir=tmp_path,
            days=(3, 5),
        )
    assert driver.OUTPUT_DIR.name == "day5_aifs_wn3"


def _run_driver(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: list[str]
) -> list[str]:
    """Run `driver.main` with both fits stubbed, and return which fits ran."""
    out = tmp_path / driver.OUTPUT_DIR.name
    out.mkdir(exist_ok=True)
    for name in existing:
        (out / name).write_text("# done\n\n## x")
    ran: list[str] = []

    def lean(*, report_name: str, output_dir: Path, **_: object) -> int:
        ran.append("lean")
        (output_dir / report_name).write_text("# AIFS\n")
        return 0

    def wn3(*, report_name: str, output_dir: Path, **_: object) -> int:
        ran.append("wn3")
        (output_dir / report_name).write_text("# WN3\n")
        return 0

    monkeypatch.setattr(driver.fit_aifs, "check_gpu_visible", lambda: None)
    monkeypatch.setattr(driver.fit_aifs, "run_lean", lean)
    monkeypatch.setattr(driver.fit_aifs, "run_wn3", wn3)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--published-dir",
            str(tmp_path / "nwp_forecast_comparison/original"),
            "--output-dir",
            str(out),
            "--lookahead-cleared",
        ],
    )
    driver.main()
    return ran


def test_the_driver_runs_both_fits_writes_the_readme_once_and_joins_the_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = _run_driver(tmp_path=tmp_path, monkeypatch=monkeypatch, existing=[])

    out = tmp_path / driver.OUTPUT_DIR.name
    assert ran == ["lean", "wn3"]
    assert (out / driver.README_NAME).read_text() == driver.README_TEXT
    assert (out / driver.REPORT_NAME).exists()


def test_the_driver_skips_a_fit_whose_report_exists_and_keeps_an_existing_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / driver.OUTPUT_DIR.name
    out.mkdir(parents=True)
    (out / driver.README_NAME).write_text("mine")

    ran = _run_driver(
        tmp_path=tmp_path, monkeypatch=monkeypatch, existing=[driver.AIFS_REPORT_NAME]
    )

    assert ran == ["wn3"]
    assert (out / driver.README_NAME).read_text() == "mine"


def test_the_driver_refuses_before_any_fit_when_report_md_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileExistsError):
        _run_driver(tmp_path=tmp_path, monkeypatch=monkeypatch, existing=[driver.REPORT_NAME])
    assert not (tmp_path / driver.OUTPUT_DIR.name / driver.AIFS_REPORT_NAME).exists()


def _patch_day5_build(*, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ens: float) -> Path:
    """Stub everything `build_domain` reads but the keys, the WN3 copy and the ENS reference."""
    published = tmp_path / "nwp_forecast_comparison/original"
    published.mkdir(parents=True)
    _keys(times=[datetime(2026, 3, 6, 12)]).write_parquet(
        published / "wind_forecast_inputs.parquet"
    )
    reference = tmp_path / "nwp_forecast_comparison/leads_day10"
    reference.mkdir(parents=True)
    _keys(times=[datetime(2026, 3, 6, 12)]).with_columns(
        ens_mean_day5_speed_100m=pl.lit(7.0)
    ).write_parquet(reference / "wind_extra_lead_inputs.parquet")
    monkeypatch.setattr(w, "aifs_site_weights", lambda **_: _weights(cells=[("A", 0, 0, 1.0)]))
    monkeypatch.setattr(w, "open_local", lambda **_: _dataset())
    monkeypatch.setattr(w, "check_against_store", lambda **_: None)
    monkeypatch.setattr(
        w,
        "_ens_extra_frame",
        lambda *, keys, **_: keys.with_columns(ens_mean_day5_speed_100m=pl.lit(ens)),
    )
    monkeypatch.setattr(w, "ens_members", lambda **_: None)
    monkeypatch.setattr(
        w, "ens_vector_mean_frame", lambda *, extract, day: _keys(times=[datetime(2026, 3, 6, 12)])
    )
    return published


def test_the_day_5_wn3_build_stops_when_its_ens_mean_differs_from_the_extra_lead_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _patch_day5_build(monkeypatch=monkeypatch, tmp_path=tmp_path, ens=8.0)

    with pytest.raises(ValueError, match="ens_mean_day5"):
        w.build_domain(
            domain="wind",
            published_dir=published,
            output_dir=tmp_path / DAY5_FOLDER_NAME,
            weather_dir=tmp_path,
            days=(5,),
        )


def test_the_day_5_wn3_build_accepts_an_ens_mean_equal_to_the_extra_lead_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _patch_day5_build(monkeypatch=monkeypatch, tmp_path=tmp_path, ens=7.0)

    frame = w.build_domain(
        domain="wind",
        published_dir=published,
        output_dir=tmp_path / DAY5_FOLDER_NAME,
        weather_dir=tmp_path,
        days=(5,),
    )

    assert frame["ens_mean_day5_speed_100m"][0] == 7.0


def test_the_day_5_folder_may_be_reached_through_a_symbolic_link(tmp_path: Path) -> None:
    (tmp_path / DAY5_FOLDER_NAME).mkdir()
    link = tmp_path / "old_name"
    link.symlink_to(tmp_path / DAY5_FOLDER_NAME, target_is_directory=True)

    driver.check_output_dir(
        output_dir=link, published_dir=tmp_path / "nwp_forecast_comparison/original"
    )


def test_the_day_5_wn3_build_accepts_a_symbolic_link_to_its_own_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = _patch_day5_build(monkeypatch=monkeypatch, tmp_path=tmp_path, ens=7.0)
    real = tmp_path / DAY5_FOLDER_NAME
    real.mkdir(parents=True)
    link = tmp_path / "old_name"
    link.symlink_to(real, target_is_directory=True)

    frame = w.build_domain(
        domain="wind", published_dir=published, output_dir=link, weather_dir=tmp_path, days=(5,)
    )

    assert frame["ens_mean_day5_speed_100m"][0] == 7.0
