"""Tests for the WeatherNext 3 build, verify and fit code under `studies/nwp_forecast_comparison/`.

Each test is written to fail on the defect it names. The store is a tiny synthetic copy whose
value at each run and lead is a known number, so a wrong run, lead, unit or site shows up as a
wrong number. The scripts are imported by path because `studies/` is not an importable package.
"""

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Final

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


w = _load(name="build_wn3_inputs")
fa = _load(name="fit_aifs")
vw = _load(name="verify_wn3_steps")
charts = _load(name="nwp_forecast_charts")


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


def test_wn3_arms_add_the_mean_vector_reference_only_for_wind_at_day_one() -> None:
    assert "ens_meanvec_day1" in fa.wn3_arms(domain="wind", day=1)
    assert "ens_meanvec_day2" not in fa.wn3_arms(domain="wind", day=2)
    assert not any("meanvec" in arm for arm in fa.wn3_arms(domain="solar", day=1))


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
