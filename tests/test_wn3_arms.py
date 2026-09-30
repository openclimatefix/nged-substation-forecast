"""Tests for the WeatherNext 3 build, verify and fit code under `studies/nwp_forecast_comparison/`.

Each test is written to fail on the defect it names. The store is a tiny synthetic copy whose
value at each run and lead is a known number, so a wrong run, lead, unit or site shows up as a
wrong number. The scripts are imported by path because `studies/` is not an importable package.
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

STUDIES_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "studies"
"""The `studies/` directory, whose scripts do bare imports of their siblings."""

RUNS: Final[np.ndarray] = np.array(["2026-03-01T00", "2026-03-02T00"], dtype="datetime64[h]")
"""The two 00 UTC runs of the synthetic copy."""

RUN_STRIDE: Final[float] = 1000.0
"""How far apart, in the synthetic values, consecutive runs are."""


def _load(*, name: str) -> ModuleType:
    """Import a study script by name, with the study directories on `sys.path` while it loads."""
    paths = [
        str(STUDIES_DIR / "nwp_forecast_comparison"),
        str(STUDIES_DIR / "beam_diffuse_split"),
        str(STUDIES_DIR / "weather_downloads"),
    ]
    sys.path[:0] = paths
    try:
        return importlib.import_module(name)
    finally:
        for path in paths:
            sys.path.remove(path)


efh = _load(name="ens_forecast_horizons")
w = _load(name="build_wn3_inputs")
fa = _load(name="fit_aifs")
vw = _load(name="verify_wn3_steps")
charts = _load(name="nwp_forecast_charts")
bfi = _load(name="build_forecast_inputs")


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
    assert set(rows["product"]) == {"WeatherNext 3 mean (7 months, out-of-sample)"}
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
