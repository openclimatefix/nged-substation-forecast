"""Tests for the pure functions of `studies/beam_diffuse_split/cerra_past_solar.py`.

Every frame is synthetic, so no test needs `data/`. Each test is built to fail on the bug it exists
for: a wrong divisor from J m⁻² to W m⁻², a window label read as the window's start (which moves
every rebuilt hour by 3 hours without any value looking wrong), a row cut that lets one product's
missing hour into the row set, and a fold design that leaves a calendar month untrained.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.cross_validation import (
    calendar_month_coverage,
    cut_eras,
    uncovered_months,
)

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_DIR: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split"
DAY: Final[datetime] = datetime(2025, 6, 1, tzinfo=UTC)
UTC_US: Final[pl.Datetime] = pl.Datetime("us", "UTC")


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "cerra_past_solar", SCRIPT_DIR / "cerra_past_solar.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sun(*, hour_end: np.ndarray) -> np.ndarray:
    """Return a clear-sky curve, in W m⁻², peaking in the hour that ends at 12.

    The hour ending at `h` has its midpoint at `h - 0.5`, and the sun is up from 05:00 to 19:00.
    """
    midpoint = hour_end - 0.5
    return np.where(
        (midpoint > 5.0) & (midpoint < 19.0), 1000.0 * np.sin(np.pi * (midpoint - 5.0) / 14.0), 0.0
    )


def _clear_sky(*, site: str = "A", days: int = 1) -> pl.DataFrame:
    hours = pl.datetime_range(
        DAY + timedelta(hours=1),
        DAY + timedelta(days=days),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    hour_of_day = ((hours.dt.hour().to_numpy() - 1) % 24) + 1
    return pl.DataFrame(
        {"site": site, "time": hours, "clear_sky_w_m2": _sun(hour_end=hour_of_day.astype(float))}
    )


def _windows(*, clear_sky: pl.DataFrame, index_by_window_end: dict[int, float]) -> pl.DataFrame:
    """Return 3-hour window means: each window's clear-sky mean times its own clear-sky index."""
    hourly = clear_sky.with_columns(
        end=(pl.col("time") - pl.duration(hours=1)).dt.truncate("3h") + pl.duration(hours=3)
    )
    means = hourly.group_by("site", "end").agg(mean=pl.col("clear_sky_w_m2").mean()).sort("end")
    hour = means["end"].dt.hour().to_numpy()
    index = np.array([index_by_window_end[int(h)] for h in hour])
    return pl.DataFrame(
        {"site": means["site"], "time": means["end"], "ghi_cerra": means["mean"].to_numpy() * index}
    )


def _peak_hour(*, rebuilt: pl.DataFrame) -> int:
    return int(rebuilt.sort("ghi_cerra", descending=True)["time"].dt.hour()[0])


ASYMMETRIC_DAY: Final[dict[int, float]] = {
    3: 1.0,
    6: 1.0,
    9: 1.0,
    12: 1.0,
    15: 0.5,
    18: 0.5,
    21: 0.5,
    0: 0.5,
}
"""A day of clear sky until noon and half of it after, so the peak sits before the sun's peak."""


def test_ten_point_eight_million_joules_over_three_hours_is_a_thousand_watts() -> None:
    module = _load()
    accumulation = pl.DataFrame(
        {
            "site": ["A", "A"],
            "valid_time": [datetime(2025, 6, 1, 12), datetime(2025, 6, 1, 15)],
            "value": [10_800_000.0, 5_400_000.0],
        },
        schema_overrides={"valid_time": pl.Datetime("ns")},
    )

    windows = module.windows_from_accumulation(
        accumulation=accumulation, value_column="value", name="ghi"
    )

    assert windows["ghi"].to_list() == [1000.0, 500.0]
    assert windows["time"].to_list() == [DAY + timedelta(hours=12), DAY + timedelta(hours=15)]
    assert windows["time"].dtype == UTC_US


@pytest.mark.parametrize(
    ("valid_time", "dtype"),
    [
        (datetime(2025, 6, 1, 12), pl.Datetime("us")),
        (datetime(2025, 6, 1, 12, tzinfo=UTC), pl.Datetime("ns", "UTC")),
        (datetime(2025, 6, 1, 13), pl.Datetime("ns")),
        (datetime(2025, 6, 1, 12, 30), pl.Datetime("ns")),
    ],
)
def test_windows_refuse_a_time_column_of_the_wrong_kind_or_a_label_off_the_grid(
    valid_time: datetime, dtype: pl.Datetime
) -> None:
    module = _load()
    accumulation = pl.DataFrame({"site": ["A"], "valid_time": [valid_time], "value": [1.0]})
    accumulation = accumulation.cast({"valid_time": dtype})

    with pytest.raises(ValueError, match=r"valid_time|multiples"):
        module.windows_from_accumulation(
            accumulation=accumulation, value_column="value", name="ghi"
        )


def test_the_rebuilt_hours_peak_in_the_exact_hour_and_a_start_labelled_reading_moves_the_peak() -> (
    None
):
    module = _load()
    clear_sky = _clear_sky()
    windows = _windows(clear_sky=clear_sky, index_by_window_end=ASYMMETRIC_DAY)

    rebuilt = module.rebuild_hourly_from_windows(
        windows=windows, clear_sky=clear_sky, column="ghi_cerra"
    )
    # A reader who took each label as its window's start would move every window 3 hours later.
    misread = module.rebuild_hourly_from_windows(
        windows=windows.with_columns(time=pl.col("time") + pl.duration(hours=3)),
        clear_sky=_clear_sky(days=2).filter(pl.col("time") <= DAY + timedelta(hours=27)),
        column="ghi_cerra",
    )

    assert rebuilt.height == 24
    assert _peak_hour(rebuilt=rebuilt) == 11
    assert _peak_hour(rebuilt=misread) != 11


def test_a_constant_clear_sky_index_rebuilds_the_clear_sky_and_each_window_keeps_its_mean() -> None:
    module = _load()
    clear_sky = _clear_sky()
    windows = _windows(clear_sky=clear_sky, index_by_window_end=dict.fromkeys(range(0, 24, 3), 0.6))

    rebuilt = module.rebuild_hourly_from_windows(
        windows=windows, clear_sky=clear_sky, column="ghi_cerra"
    )
    joined = rebuilt.join(clear_sky, on=["site", "time"])
    gaps = module.window_mean_gaps(windows=windows, rebuilt=rebuilt, column="ghi_cerra")

    np.testing.assert_allclose(
        joined["ghi_cerra"].to_numpy(), 0.6 * joined["clear_sky_w_m2"].to_numpy(), atol=1e-9
    )
    assert gaps.len() > 0
    assert gaps.max() < 1e-9


def test_a_stepped_clear_sky_index_keeps_each_windows_mean_within_a_stated_tolerance() -> None:
    module = _load()
    clear_sky = _clear_sky()
    windows = _windows(clear_sky=clear_sky, index_by_window_end=ASYMMETRIC_DAY)

    rebuilt = module.rebuild_hourly_from_windows(
        windows=windows, clear_sky=clear_sky, column="ghi_cerra"
    )
    gaps = module.window_mean_gaps(windows=windows, rebuilt=rebuilt, column="ghi_cerra")

    assert gaps.max() < 0.2
    assert gaps.max() > 1e-3  # No rescaling is applied, so the gap is not zero.


def test_a_missing_window_leaves_its_three_hours_out_and_no_other_hour() -> None:
    module = _load()
    clear_sky = _clear_sky()
    windows = _windows(clear_sky=clear_sky, index_by_window_end=ASYMMETRIC_DAY)
    missing = DAY + timedelta(hours=12)

    rebuilt = module.rebuild_hourly_from_windows(
        windows=windows.filter(pl.col("time") != missing), clear_sky=clear_sky, column="ghi_cerra"
    )

    assert rebuilt.height == 21
    hours = set(rebuilt["time"].dt.hour().to_list())
    assert hours.isdisjoint({10, 11, 12})
    assert {9, 13} <= hours


def test_a_rebuild_with_no_clear_sky_for_an_hour_it_covers_raises() -> None:
    module = _load()
    clear_sky = _clear_sky()
    windows = _windows(clear_sky=clear_sky, index_by_window_end=ASYMMETRIC_DAY)

    with pytest.raises(ValueError, match="clear_sky misses an hour"):
        module.rebuild_hourly_from_windows(
            windows=windows, clear_sky=clear_sky.head(10), column="ghi_cerra"
        )


def _hourly(
    *, values: list[float | None], first_end: datetime, column: str = "ghi_w_m2"
) -> pl.DataFrame:
    times = [first_end + timedelta(hours=i) for i in range(len(values))]
    return pl.DataFrame(
        {"site": "A", "time": times, column: values},
        schema_overrides={"time": UTC_US, column: pl.Float64},
    )


def test_a_windowed_mean_averages_its_three_hours_and_drops_partial_windows() -> None:
    module = _load()
    # Hours ending 01:00 to 07:00: windows ending 03:00 (hours 1, 2, 3) and 06:00 (hours 4, 5, 6)
    # are whole, and the hour ending 07:00 opens a window that is not.
    hourly = _hourly(
        values=[10.0, 20.0, 60.0, 1.0, 2.0, 9.0, 100.0], first_end=DAY + timedelta(hours=1)
    )

    windows = module.windowed_mean(hourly=hourly, column="ghi_w_m2")

    assert windows["time"].to_list() == [DAY + timedelta(hours=3), DAY + timedelta(hours=6)]
    assert windows["ghi_w_m2"].to_list() == [30.0, 4.0]


def test_the_hour_ending_at_midnight_belongs_to_the_previous_days_last_window() -> None:
    module = _load()
    hourly = _hourly(values=[3.0, 6.0, 9.0], first_end=DAY + timedelta(hours=22))

    windows = module.windowed_mean(hourly=hourly, column="ghi_w_m2")

    assert windows["time"].to_list() == [DAY + timedelta(days=1)]
    assert windows["ghi_w_m2"].to_list() == [6.0]


def test_a_window_with_a_missing_hour_is_dropped_not_averaged_over_two() -> None:
    module = _load()
    hourly = _hourly(values=[10.0, None, 30.0, 1.0, 2.0, 3.0], first_end=DAY + timedelta(hours=1))

    windows = module.windowed_mean(hourly=hourly, column="ghi_w_m2")

    assert windows["time"].to_list() == [DAY + timedelta(hours=6)]


def _base(*, hours: int, first_end: datetime) -> pl.DataFrame:
    times = [first_end + timedelta(hours=i) for i in range(hours)]
    return pl.DataFrame(
        {
            "site": "A",
            "time": times,
            "ghi_era5": 100.0,
            "ghi_cams": 90.0,
            "era": "0",
            "era_code": 0,
            "fold": 0,
            "power_mw": 1.0,
        },
        schema_overrides={"time": UTC_US},
    )


def _product(*, base: pl.DataFrame, columns: dict[str, float | None]) -> pl.DataFrame:
    return base.select("site", "time").with_columns(
        **{name: pl.lit(value, dtype=pl.Float64) for name, value in columns.items()}
    )


def test_join_rows_drops_hours_after_the_window_end_and_hours_any_product_lacks() -> None:
    module = _load()
    end = datetime(2026, 7, 1, tzinfo=UTC)
    base = _base(
        hours=6, first_end=end - timedelta(hours=3)
    )  # Hours ending 3 hours before the end to 2 hours after it.
    cerra = _product(base=base, columns={"ghi_cerra": 1.0, "bhi_cerra": 0.5})
    era5_3h = _product(base=base, columns={"ghi_era5_3h": 1.0})
    cams_3h = _product(base=base, columns={"ghi_cams_3h": 1.0})
    # CERRA lacks the hour ending 2 hours before the end, and ERA5's copy the hour before that.
    cerra = cerra.filter(pl.col("time") != end - timedelta(hours=2))
    era5_3h = era5_3h.filter(pl.col("time") != end - timedelta(hours=3))
    cams_3h = cams_3h.with_columns(
        ghi_cams_3h=pl.when(pl.col("time") == end - timedelta(hours=1))
        .then(None)
        .otherwise(pl.col("ghi_cams_3h"))
    )

    joined, lost = module.join_rows(
        base=base, cerra=cerra, era5_3h=era5_3h, cams_3h=cams_3h, window_end=end
    )

    assert joined["time"].to_list() == [end]
    assert {"era", "era_code", "fold"}.isdisjoint(joined.columns)
    assert "main_fold" in joined.columns
    # The four hours ending at or before the end lose one row at each of three joins in turn: the
    # hour CERRA lacks, the hour ERA5's copy lacks, and the hour CAMS's copy holds a null for.
    assert lost == {"cerra": 1, "era5_3h": 1, "cams_3h": 0, "null_or_nan": 1}


def test_join_rows_raises_on_a_naive_or_wrongly_united_time_column() -> None:
    module = _load()
    base = _base(hours=2, first_end=DAY)
    cerra = _product(base=base, columns={"ghi_cerra": 1.0, "bhi_cerra": 0.5})
    era5_3h = _product(base=base, columns={"ghi_era5_3h": 1.0})
    cams_3h = _product(base=base, columns={"ghi_cams_3h": 1.0})

    for bad in (
        cerra.with_columns(time=pl.col("time").dt.replace_time_zone(None)),
        cerra.with_columns(time=pl.col("time").dt.cast_time_unit("ns")),
    ):
        with pytest.raises(ValueError, match="time must be"):
            module.join_rows(
                base=base,
                cerra=bad,
                era5_3h=era5_3h,
                cams_3h=cams_3h,
                window_end=DAY + timedelta(days=1),
            )


def test_diffuse_is_global_minus_direct_clipped_at_zero_and_the_clipped_hours_are_counted() -> None:
    module = _load()
    frame = pl.DataFrame(
        {
            "time": [DAY + timedelta(hours=h) for h in (10, 11, 12)],
            "solar_zenith_deg": [40.0, 30.0, 25.0],
            "ghi_cerra": [500.0, 600.0, 700.0],
            "bhi_cerra": [300.0, 650.0, 700.0],
        },
        schema_overrides={"time": UTC_US},
    )

    result, clipped = module.with_diffuse_and_erbs(frame=frame)

    assert result["dhi_cerra"].to_list() == [200.0, 0.0, 0.0]
    assert clipped == 1
    for column in ("erbs_bhi_cerra", "erbs_dhi_cerra"):
        assert result[column].min() >= 0.0
    assert (result["erbs_bhi_cerra"] <= result["ghi_cerra"]).all()


def test_the_four_planned_contrasts_are_exactly_the_four_the_plan_names() -> None:
    module = _load()

    assert set(module.PLANNED_CONTRASTS) == {
        ("cerra_global", "era5_global"),
        ("cerra_global", "cams_global"),
        ("cerra_global", "era5_3h"),
        ("cerra_split", "cerra_erbs"),
    }
    assert len(module.PLANNED_CONTRASTS) == 4


def test_every_arm_carries_the_same_number_of_distinct_columns_where_a_contrast_pairs_them() -> (
    None
):
    module = _load()

    features = module._arm_features()

    assert {arm: len(columns) for arm, columns in features.items()} == {
        "cerra_global": 8,
        "cerra_split": 10,
        "cerra_erbs": 10,
        "era5_global": 8,
        "cams_global": 8,
        "era5_3h": 8,
        "cams_3h": 8,
    }
    assert set(features) == set(module.ARM_ORDER)


def test_only_the_planned_contrasts_arms_are_fitted_at_the_second_setting() -> None:
    module = _load()

    settings = {(job[0], job[1]) for job in module.jobs()}

    assert settings == {
        *((arm, "pooled") for arm in module.ARM_ORDER),
        *((arm, "sensitivity") for arm in module.PLANNED_ARMS),
    }
    assert ("cams_3h", "sensitivity") not in settings
    assert set(module.PLANNED_ARMS) == set(module.ARM_ORDER) - {"cams_3h"}


def test_a_contrast_between_arms_of_different_widths_raises() -> None:
    module = _load()
    features = {"a": ("x", "y"), "b": ("x", "y", "z")}

    with pytest.raises(ValueError, match="equal counts are required"):
        module.check_column_counts(features=features, contrasts=(("a", "b"),))


def test_an_arm_that_repeats_a_column_or_a_contrast_naming_an_unknown_arm_raises() -> None:
    module = _load()

    with pytest.raises(ValueError, match="repeats a feature column"):
        module.check_column_counts(features={"a": ("x", "x")}, contrasts=())
    with pytest.raises(ValueError, match="has no feature columns"):
        module.check_column_counts(features={"a": ("x",)}, contrasts=(("a", "b"),))


def _two_era_frame() -> pl.DataFrame:
    """Return one generator's rows in February to June of 2025 and 2026.

    With every era's folds unrotated, each calendar month falls in the same fold in both years, so
    its held-out fold leaves no training row for the season.
    """
    months = [f"2025-{m:02d}" for m in range(2, 7)] + [f"2026-{m:02d}" for m in range(2, 7)]
    times = [datetime(int(m[:4]), int(m[5:]), 15, 12, tzinfo=UTC) for m in months]
    return pl.DataFrame(
        {"site": "A", "month": months, "time": times}, schema_overrides={"time": UTC_US}
    )


def test_the_chosen_fold_offsets_cover_every_month_where_all_zero_offsets_do_not() -> None:
    module = _load()
    frame = _two_era_frame()
    unrotated = cut_eras(frame=frame, first_months=module.FIRST_MONTHS, fold_offsets={0: 0, 1: 0})

    chosen, offsets = module.with_covering_folds(frame=frame)

    assert uncovered_months(coverage=calendar_month_coverage(frame=unrotated)).height == 5
    assert module.uncovered_share(frame=unrotated) == 1.0
    assert uncovered_months(coverage=calendar_month_coverage(frame=chosen)).is_empty()
    assert module.uncovered_share(frame=chosen) == 0.0
    assert offsets != {0: 0, 1: 0}


def test_folds_raise_where_no_rotation_covers_every_calendar_month() -> None:
    module = _load()
    # Ten months in one era give two months per fold, so the two Februaries that come first share
    # fold 0, and no rotation of the second era (which holds no rows) can separate them.
    months = ["2024-02", "2025-02", *(f"2025-{m:02d}" for m in range(3, 11))]
    times = [datetime(int(m[:4]), int(m[5:]), 15, 12, tzinfo=UTC) for m in months]
    frame = pl.DataFrame(
        {"site": "A", "month": months, "time": times}, schema_overrides={"time": UTC_US}
    )

    with pytest.raises(ValueError, match="no fold rotation"):
        module.choose_fold_offsets(frame=frame)


def _clear_day_hours(*, peak: int = 12) -> pl.DataFrame:
    """Return 3 days of each month of 2025-02 to 06 and 2026-02 to 06, hours ending 1 to 24.

    Every row carries a daylight-shaped `ghi_era5`, `ghi_cams`, the published `month`, and the
    columns the arms are shown, so the whole row-set tail can run on it.
    """
    months = [f"2025-{m:02d}" for m in range(2, 7)] + [f"2026-{m:02d}" for m in range(2, 7)]
    rows = [
        {
            "site": "A",
            "month": month,
            "time": datetime(int(month[:4]), int(month[5:]), day, tzinfo=UTC)
            + timedelta(hours=hour),
            "ghi_era5": (1.0 + day / 20.0) * max(0.0, 100.0 - 20.0 * abs(hour - peak)),
        }
        for month in months
        for day in (10, 11, 12)
        for hour in range(1, 25)
    ]
    return pl.DataFrame(rows, schema_overrides={"time": UTC_US}).with_columns(
        ghi_cams=pl.col("ghi_era5") * 0.9, power_mw=1.0, solar_zenith_deg=40.0
    )


def _tail_inputs(*, main_fold_all_zero: bool) -> dict[str, pl.DataFrame]:
    """Return the four frames `assemble_rows` takes, shaped like the study's own."""
    module = _load()
    plain = _clear_day_hours()
    for column in module.SOLAR.shared_features:
        if column not in plain.columns and column != "era_code":
            plain = plain.with_columns(pl.lit(1.0).alias(column))
    base, _ = module.with_covering_folds(frame=plain)
    if main_fold_all_zero:
        base = base.with_columns(fold=pl.lit(0))
    keys = base.select("site", "time")
    return {
        "base": base,
        "cerra": keys.with_columns(ghi_cerra=base["ghi_era5"], bhi_cerra=base["ghi_era5"] * 0.5),
        "era5_3h": keys.with_columns(ghi_era5_3h=base["ghi_era5"]),
        "cams_3h": keys.with_columns(ghi_cams_3h=base["ghi_cams"]),
    }


def test_the_row_set_tail_runs_end_to_end_and_checks_the_columns_after_the_folds_are_cut() -> None:
    module = _load()
    inputs = _tail_inputs(main_fold_all_zero=False)

    rows = module.assemble_rows(**inputs)

    assert rows.frame.height == inputs["base"].height
    assert {"era_code", "fold", "dhi_cerra", "erbs_bhi_cerra"} <= set(rows.frame.columns)
    assert "main_fold" not in rows.frame.columns
    assert uncovered_months(coverage=calendar_month_coverage(frame=rows.frame)).is_empty()
    assert rows.rows_lost == {"cerra": 0, "era5_3h": 0, "cams_3h": 0, "null_or_nan": 0}
    assert rows.months_dropped == []


def test_the_main_rows_uncovered_share_is_taken_on_the_published_fold_column() -> None:
    module = _load()

    covered = module.assemble_rows(**_tail_inputs(main_fold_all_zero=False))
    uncovered = module.assemble_rows(**_tail_inputs(main_fold_all_zero=True))

    # A recut of the shorter row set would give the same share whatever the published folds were.
    assert covered.uncovered_main_folds == 0.0
    assert uncovered.uncovered_main_folds == 1.0


def test_the_report_names_the_rows_each_input_lost_and_the_months_dropped() -> None:
    module = _load()
    inputs = _tail_inputs(main_fold_all_zero=False)
    inputs["cerra"] = inputs["cerra"].filter(pl.col("time").dt.strftime("%Y-%m") != "2025-03")

    rows = module.assemble_rows(**inputs)

    assert rows.rows_lost["cerra"] == 3 * 24
    assert rows.months_dropped == ["2025-03"]


def test_a_missing_value_in_an_arms_column_stops_the_tail_after_the_folds_are_cut() -> None:
    module = _load()
    inputs = _tail_inputs(main_fold_all_zero=False)
    inputs["base"] = inputs["base"].with_columns(temp_c=pl.lit(None, dtype=pl.Float64))

    with pytest.raises(ValueError, match="missing values in columns an arm is shown"):
        module.assemble_rows(**inputs)


def _profile_frame(*, cerra_peak: int, era5_peak: int) -> pl.DataFrame:
    rows = [
        {
            "site": "A",
            "time": DAY + timedelta(days=day, hours=hour),
            "ghi_cerra": (1.0 + day / 20.0) * max(0.0, 100.0 - 20.0 * abs(hour - cerra_peak)),
            "ghi_era5": (1.0 + day / 20.0) * max(0.0, 100.0 - 20.0 * abs(hour - era5_peak)),
        }
        for day in range(20)
        for hour in range(1, 25)
    ]
    return pl.DataFrame(rows, schema_overrides={"time": UTC_US})


def test_the_clear_day_peak_hour_is_the_exact_hour_of_the_mean_profile() -> None:
    module = _load()
    frame = _profile_frame(cerra_peak=11, era5_peak=11)

    assert (
        module.clear_day_peak_hour(frame=frame, column="ghi_cerra", ranking_column="ghi_era5") == 11
    )
    assert module.check_peak_hours_agree(frame=frame) == (11, 11)


def test_peak_hours_that_differ_by_a_shifted_window_convention_stop_the_run() -> None:
    module = _load()
    frame = _profile_frame(cerra_peak=14, era5_peak=11)

    with pytest.raises(ValueError, match="peak is at hour 14 UTC and ERA5's at hour 11"):
        module.check_peak_hours_agree(frame=frame)


def test_the_nearest_cell_is_by_distance_on_the_sphere_and_a_longitude_above_180_is_wrapped() -> (
    None
):
    module = _load()
    # At 53 N a degree of longitude is about 0.6 of a degree of latitude on the ground. The site is
    # 0.4 degrees east of cell 0 and 0.3 degrees north of cell 1: cell 0 is nearer on the sphere
    # (0.4 * 0.6 = 0.24 against 0.3) though it is farther in degrees.
    grid = pl.DataFrame(
        {
            "y_index": [10, 11],
            "x_index": [20, 21],
            "latitude": [53.0, 53.3],
            "longitude": [359.0, 0.0],
        }
    )
    sites = pl.DataFrame({"site": ["A"], "latitude": [53.0], "longitude": [-0.6]})

    nearest = module.derive_nearest_cells(grid=grid, sites=sites)

    assert nearest.select("site", "y_index", "x_index").rows() == [("A", 10, 20)]
    assert 20.0 < nearest["distance_km"][0] < 30.0


def test_cells_that_differ_from_the_saved_table_stop_the_run_without_naming_a_cell() -> None:
    module = _load()
    derived = pl.DataFrame({"site": ["A", "B"], "y_index": [10, 11], "x_index": [20, 21]})
    same = derived.clone()
    different = derived.with_columns(x_index=pl.Series([20, 99]))

    module.check_cells_match(derived=derived, saved=same)
    with pytest.raises(ValueError, match="1 of 2 generators") as error:
        module.check_cells_match(derived=derived, saved=different)

    assert "99" not in str(error.value)
    assert "21" not in str(error.value)


def _synthetic_losses(
    *, arms: tuple[str, ...], second_setting_arms: tuple[str, ...]
) -> pl.DataFrame:
    """Return capped fractional losses: all arms at `pooled`, some at `sensitivity`.

    Only `second_setting_arms` are fitted at `sensitivity`, as `jobs()` fits only the planned
    contrasts' arms there. There are 2 generators and 8 months.
    """
    months = [f"2025-{m:02d}" for m in range(1, 9)]
    return pl.DataFrame(
        {
            "arm": arm,
            "setting": setting,
            "site": site,
            "time": datetime(2025, int(month[5:]), 15, hour, tzinfo=UTC),
            "seed": seed,
            "month": month,
            "fold": index % 2,
            "absolute_error_capped_fraction_of_capacity": 0.02
            + 0.003 * arm_index
            + 0.001 * hour
            + 0.002 * seed
            + 0.004 * (index % 3)
            + (0.01 if setting == "sensitivity" else 0.0),
        }
        for arm_index, arm in enumerate(arms)
        for setting in ("pooled", "sensitivity")
        if setting == "pooled" or arm in second_setting_arms
        for site in ("A", "B")
        for index, month in enumerate(months)
        for hour in (10, 11)
        for seed in (0, 1)
    )


def test_the_report_the_script_writes_is_reproduced_by_the_leaderboard_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    leaderboard = _load_leaderboard()
    losses = _synthetic_losses(arms=module.ARM_ORDER, second_setting_arms=module.PLANNED_ARMS)
    frame = losses.filter(
        pl.col("arm") == "cerra_global", pl.col("setting") == "pooled", pl.col("seed") == 0
    ).select("site", "time")
    built = module.Built(
        frame=frame,
        candidates=0,
        within_window=0,
        clipped_diffuse_hours=0,
        fold_offsets={0: 0, 1: 0},
        uncovered_main_folds=0.0,
        rows_lost={},
        months_dropped=[],
        rebuilt_negative_shares={},
        window_gaps={},
        grid_cells=1,
        distance_range_km=(1.0, 2.0),
    )
    for name in (
        "_row_lines",
        "_cell_lines",
        "_rebuild_lines",
        "_hour_profile_lines",
        "_main_panel_lines",
        "geometry_lines",
    ):
        monkeypatch.setattr(module, name, lambda **_: [])
    report = module._report(
        built=built, losses=losses, sites=pl.DataFrame(), job_list=module.jobs()
    )
    path = tmp_path / "report.md"
    path.write_text(report)
    row_set = leaderboard.ROW_SETS[-1]

    result = leaderboard.score_row_set(
        row_set=row_set, losses=losses, report_text=report, report_path=path
    )

    assert result.site_hours == frame.height
    assert result.planned.height == 4
    planning = dict(zip(result.contrasts["arm"], result.contrasts["planning"], strict=True))
    assert planning["cerra_global"] == "planned"
    assert planning["cerra_split"] == "exploratory"
    assert planning["cams_3h"] == "exploratory"
    assert result.planned["second_difference"].null_count() == 0
    # The exploratory pairs with an arm not refitted at the second setting are named, not scored.
    assert (
        "Not fitted at the second setting: cerra_global − cams_3h, cams_global − cams_3h." in report
    )
    second = report.split("The same contrasts at the second hyperparameter setting")[1]
    assert "| sensitivity | era5_global − era5_3h |" in second
    assert "| sensitivity | cerra_global − cams_3h |" not in second


def _load_leaderboard() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "past_solar_leaderboard", SCRIPT_DIR / "past_solar_leaderboard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
