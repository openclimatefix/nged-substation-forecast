"""Tests for the pure functions of the four `ukv_ceda_*` scripts of study #1024.

Every frame is synthetic, so no test needs `data/`. Each test is built to fail on the bug it exists
for: a bias removed across calendar months instead of within one, a power-hour offset read at the
wrong sign, an hour built from one instant instead of the mean of two, a decision that lets a
statistically significant but small difference win, and a controls column that is shuffled across
months.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pytest
import ukv_ceda_station_scores as scores
import ukv_ceda_vs_era5_build as build
import ukv_ceda_vs_era5_charts as charts
import ukv_ceda_vs_era5_fit as fit
import ukv_ceda_vs_era5_verify as verify
from studies.ukv_ceda_profiles import DEFAULT_PROFILE, STATUS_COMPLETE, STATUS_PARTIAL
from studies.ukv_ceda_stores import make_stores

UTC_US: Final[pl.Datetime] = pl.Datetime("us", "UTC")
EPOCH: Final[datetime] = DEFAULT_PROFILE.slot_epoch


# --- build: the margin reading --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("difference", "lower", "upper", "expected"),
    [
        (-0.30, -0.40, -0.20, "ukv_clearly_better"),
        (0.30, 0.20, 0.40, "era5_clearly_better"),
        (-0.10, -0.15, -0.05, "small_not_clear"),
        (0.10, 0.05, 0.15, "small_not_clear"),
        (-0.30, -0.40, 0.10, "no_clear_difference"),
        (-0.16, -0.20, -0.12, "small_not_clear"),
    ],
)
def test_a_contrast_is_clear_only_if_its_interval_excludes_zero_and_its_estimate_passes_the_margin(
    difference: float, lower: float, upper: float, expected: str
):
    reading = build.contrast_reading(difference=difference, lower=lower, upper=upper, margin=0.16)

    assert reading == expected


def test_the_margins_are_the_frozen_values_in_points_of_capacity():
    assert (build.MARGIN_WIND_PP, build.MARGIN_SOLAR_PP, build.MARGIN_STATION_SHARE) == (
        0.16,
        0.06,
        0.05,
    )


# --- build: the arm columns -----------------------------------------------------------------------


def test_every_wind_arm_has_seven_columns_and_every_solar_arm_ten_whichever_product_it_reads():
    for shuffled in (False, True):
        assert {
            len(build.wind_arm_columns(product=p, shuffled=shuffled)) for p in build.PRODUCTS
        } == {7}
        assert {
            len(build.solar_arm_columns(product=p, shuffled=shuffled)) for p in build.PRODUCTS
        } == {10}
    build.check_arm_widths()


def test_the_two_solar_arms_differ_only_in_the_temperature_column():
    era5 = build.solar_arm_columns(product="era5")
    ukv = build.solar_arm_columns(product="ukv_ceda")

    differing = [(a, b) for a, b in zip(era5, ukv, strict=True) if a != b]

    assert differing == [("era5_temp", "ukv_ceda_temp")]


def test_the_shuffled_solar_arm_keeps_the_cams_irradiance_unshuffled():
    columns = build.solar_arm_columns(product="ukv_ceda", shuffled=True)

    assert "ukv_ceda_temp_shuffled" in columns
    assert set(build.SOLAR_IRRADIANCE_FEATURES) <= set(columns)
    assert not any(c.endswith("_shuffled") for c in build.SOLAR_IRRADIANCE_FEATURES)


def test_ukv_ceda_wind_has_no_100_m_column_and_reads_925_hpa_instead():
    assert not any("100m" in c for c in build.wind_columns(product="ukv_ceda"))
    assert "ukv_ceda_speed_925hpa" in build.wind_columns(product="ukv_ceda")
    assert "era5_speed_100m" in build.wind_columns(product="era5")


def test_an_unknown_product_raises():
    with pytest.raises(ValueError, match="unknown product"):
        build.wind_columns(product="icon")


# --- build: ERA5 wind -----------------------------------------------------------------------------


def test_a_wind_blowing_from_the_west_has_direction_270_and_from_the_north_360():
    frame = pl.DataFrame({"u": [5.0, 0.0, 0.0], "v": [0.0, -3.0, 4.0]})
    speed, direction = build.wind_from_components(u=pl.col("u"), v=pl.col("v"))

    result = frame.select(speed=speed, direction=direction)

    assert result["speed"].to_list() == pytest.approx([5.0, 3.0, 4.0])
    assert result["direction"].to_list() == pytest.approx([270.0, 0.0, 180.0])


def test_the_hours_of_a_december_month_run_to_the_first_hour_of_january():
    hours = build.month_hours(month="2024-12")

    assert hours[0] == datetime(2024, 12, 1, tzinfo=UTC)
    assert hours[-1] == datetime(2024, 12, 31, 23, tzinfo=UTC)


# --- build: UKV-CEDA at an hour -------------------------------------------------------------------


def _ukv_stores(*, partial_slots: tuple[int, ...] = ()) -> build.UkvStores:
    """One store of 8 slots, whose value at slot `s` and lead `l` is `100 s + l`."""
    values = np.array(
        [[[100.0 * slot + lead] for lead in range(6)] for slot in range(8)], dtype=np.float64
    )
    statuses = np.full(8, STATUS_COMPLETE, dtype=np.int8)
    for slot in partial_slots:
        statuses[slot] = STATUS_PARTIAL
    stores = make_stores(groups=[{"temperature_1p5m": values}], statuses=[statuses])
    return build.UkvStores(
        stores=stores, latitude=np.zeros(1), longitude=np.zeros(1), snapshot_ids=("test",)
    )


def _hours(*hours_after_epoch: int) -> pl.Series:
    return pl.Series([EPOCH + timedelta(hours=h) for h in hours_after_epoch], dtype=UTC_US)


def test_an_hour_built_from_two_instants_is_their_mean_not_their_sum_or_the_label_alone():
    # The hour labelled 06 UTC averages 05 UTC (run 0, lead 5, value 5) and 06 UTC (run 1, lead 0,
    # value 100).
    usable, means, leads = build.ukv_at(
        ukv=_ukv_stores(),
        hours=_hours(6),
        offsets_hours=(-1, 0),
        variables=["temperature_1p5m"],
        cells=np.array([0]),
        read_values=True,
    )

    assert usable.tolist() == [True]
    assert means["temperature_1p5m"][0, 0] == pytest.approx(52.5)
    assert leads.tolist() == [0]


def test_a_solar_hour_is_unusable_when_either_instants_run_is_partial():
    usable, _, _ = build.ukv_at(
        ukv=_ukv_stores(partial_slots=(0,)),
        hours=_hours(6, 7),
        offsets_hours=(-1, 0),
        variables=["temperature_1p5m"],
        cells=np.array([0]),
        read_values=True,
    )

    assert usable.tolist() == [False, True]


def test_the_coverage_check_decides_availability_without_reading_a_value():
    usable, means, _ = build.ukv_at(
        ukv=_ukv_stores(partial_slots=(1,)),
        hours=_hours(6, 12),
        offsets_hours=(0,),
        variables=["temperature_1p5m"],
        cells=np.array([0]),
        read_values=False,
    )

    assert usable.tolist() == [False, True]
    assert not means["temperature_1p5m"].any()


# --- build: the shuffled control ------------------------------------------------------------------


def test_a_shuffled_column_group_moves_together_and_stays_within_its_month_and_hour():
    times = [
        datetime(2024, month, 1, hour, tzinfo=UTC)
        for month in (1, 2)
        for hour in (3, 4)
        for _ in range(6)
    ]
    frame = pl.DataFrame(
        {
            "site": "W1",
            "time": pl.Series(times, dtype=UTC_US),
            "a": np.arange(len(times), dtype=float),
        }
    ).with_columns(
        b=pl.col("a") * 10,
        month=pl.col("time").dt.strftime("%Y-%m"),
        hour_of_day=pl.col("time").dt.hour(),
    )

    shuffled = build._shuffled(frame=frame, groups=[("a", "b")])

    assert (shuffled["b_shuffled"] == shuffled["a_shuffled"] * 10).all()
    assert shuffled["a_shuffled"].to_list() != shuffled["a"].to_list()
    same_cell = shuffled.group_by("month", "hour_of_day").agg(
        original=pl.col("a").sort(), moved=pl.col("a_shuffled").sort()
    )
    assert all(row["original"] == row["moved"] for row in same_cell.iter_rows(named=True))


# --- station scores -------------------------------------------------------------------------------


def _station_frame(*, months: int = 8) -> pl.DataFrame:
    """Hourly rows at one station whose ERA5 value is the station's plus a bias per month."""
    times = pl.datetime_range(
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, months, 28, tzinfo=UTC),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    frame = pl.DataFrame({"site": "S1", "time": times}).with_columns(
        month=pl.col("time").dt.strftime("%Y-%m"),
        lead_hours=pl.col("time").dt.hour() % 6,
        station=pl.lit(10.0),
    )
    return frame.with_columns(
        station_wind_m_s=pl.col("station"),
        era5_wind_m_s=pl.col("station") + pl.col("time").dt.month().cast(pl.Float64),
        ukv_wind_m_s=pl.col("station") + 0.5,
    ).drop("station")


def test_a_bias_is_removed_within_each_calendar_month_not_across_them():
    frame = pl.DataFrame(
        {
            "site": "S1",
            "calendar_month": [1, 1, 2, 2],
            "hour_of_day": [0, 0, 0, 0],
            "error": [1.0, 1.0, 5.0, 5.0],
        }
    )

    within = frame.select(scores.debiased(error="error", by=scores.BIAS_BY_HOUR))["error"]
    across = frame.select(scores.debiased(error="error", by=("site",)))["error"]

    assert within.to_list() == [0.0, 0.0, 0.0, 0.0]
    assert across.to_list() == [-2.0, -2.0, 2.0, 2.0]


def test_a_bias_is_removed_within_each_station():
    frame = pl.DataFrame(
        {
            "site": ["S1", "S1", "S2", "S2"],
            "calendar_month": [1, 1, 1, 1],
            "hour_of_day": [0, 0, 0, 0],
            "error": [1.0, 1.0, 3.0, 3.0],
        }
    )

    result = frame.select(scores.debiased(error="error", by=scores.BIAS_BY_MONTH))["error"]

    assert result.to_list() == [0.0, 0.0, 0.0, 0.0]


def test_a_station_hour_is_scored_only_if_the_station_and_both_products_have_a_value():
    frame = _station_frame().with_row_index("i")
    frame = frame.with_columns(
        ukv_wind_m_s=pl.when(pl.col("i") == 0).then(None).otherwise(pl.col("ukv_wind_m_s")),
        era5_wind_m_s=pl.when(pl.col("i") == 1)
        .then(float("nan"))
        .otherwise(pl.col("era5_wind_m_s")),
        station_wind_m_s=pl.when(pl.col("i") == 2).then(None).otherwise(pl.col("station_wind_m_s")),
    ).drop("i")

    rows = scores.scored_rows(frame=frame, variable=scores.VARIABLES[0])

    assert rows.height == frame.height - 3


def test_a_constant_bias_per_calendar_month_leaves_no_bias_removed_error_for_that_product():
    rows = scores.scored_rows(frame=_station_frame(), variable=scores.VARIABLES[0])

    assert rows["abs_era5_bias_removed_month"].max() == pytest.approx(0.0)
    assert rows["abs_ukv_bias_removed_month"].max() == pytest.approx(0.0)
    assert rows["abs_era5_raw"].max() == pytest.approx(8.0)


def test_a_scope_with_too_few_months_gets_a_point_estimate_and_no_reading():
    rows = scores.scored_rows(frame=_station_frame(months=3), variable=scores.VARIABLES[0])

    record = scores.interval_record(
        rows=rows,
        variable=scores.VARIABLES[0],
        label="P1",
        planned=True,
        score="raw",
        scope="all",
    )

    assert record["n_months"] == 3
    assert not record["enough_months"]
    assert record["reading"] == "no_interval"
    assert np.isnan(record["lower_95"])


def test_the_margin_is_five_percent_of_erass_bias_removed_error_on_the_same_rows():
    frame = pl.DataFrame(
        {
            "site": "S1",
            "time": pl.datetime_range(
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 12, 31, tzinfo=UTC),
                interval="1d",
                time_zone="UTC",
                eager=True,
            ),
        }
    ).with_columns(
        month=pl.col("time").dt.strftime("%Y-%m"),
        lead_hours=pl.lit(0),
        station_wind_m_s=pl.lit(5.0),
        era5_wind_m_s=5.0 + pl.col("time").dt.day().cast(pl.Float64) % 2 * 2.0 - 1.0,
        ukv_wind_m_s=pl.lit(5.0),
    )
    variable = scores.VARIABLES[0]
    rows = scores.scored_rows(frame=frame, variable=variable)

    record = scores.interval_record(
        rows=rows, variable=variable, label="P1", planned=True, score="raw", scope="all"
    )

    era5_error = float(np.mean(rows["abs_era5_bias_removed_hour"].to_numpy()))
    assert record["margin"] == pytest.approx(build.MARGIN_STATION_SHARE * era5_error)
    assert record["margin"] > 0.0


def test_p2_lead_splits_temperature_into_leads_0_to_2_and_3_to_5_and_wind_has_no_such_split():
    frame = _station_frame().with_columns(
        station_temp_c=pl.col("station_wind_m_s"),
        era5_temp_c=pl.col("era5_wind_m_s"),
        ukv_temp_c=pl.col("ukv_wind_m_s"),
        lead_hours=pl.col("time").dt.hour() % 6,
    )
    leads = frame.with_columns(lead_hours=pl.col("time").dt.hour() % 6)

    temperature = scores.variable_records(frame=leads, variable=scores.VARIABLES[1])
    wind = scores.variable_records(frame=leads, variable=scores.VARIABLES[0])

    lead_rows = {r["scope"]: r for r in temperature if r["label"] == "P2-lead"}
    assert set(lead_rows) == {"leads 0 to 2", "leads 3 to 5"}
    assert all(r["planned"] for r in lead_rows.values())
    assert sum(r["n_rows"] for r in lead_rows.values()) == temperature[0]["n_rows"]
    assert not [r for r in wind if r["label"] == "P2-lead"]


def test_the_early_window_ends_at_the_month_before_2021_01():
    frame = _station_frame()
    rows = scores.scored_rows(frame=frame, variable=scores.VARIABLES[0])
    shifted = rows.with_columns(
        time=pl.col("time").dt.offset_by("1y"), month=pl.col("month").str.replace("2020", "2021")
    )

    scopes = {
        scope: subset.height
        for _, scope, subset in scores.scope_rows(rows=pl.concat([rows, shifted]))
    }

    assert scopes["early window"] == rows.height
    assert scopes["late window"] == shifted.height


# --- verify ---------------------------------------------------------------------------------------


def _lag_frame(*, ukv_shift_hours: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    times = pl.datetime_range(
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 20, tzinfo=UTC),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    station = rng.normal(size=len(times))
    series = pl.DataFrame({"time": times, "station": station})
    # The product at time t equals the station at t - ukv_shift_hours.
    frame = series.with_columns(
        ukv=pl.col("station").shift(ukv_shift_hours), era5=pl.col("station")
    ).drop_nulls()
    return frame.select(
        site=pl.lit("S1"),
        time="time",
        station_wind_m_s="station",
        era5_wind_m_s="era5",
        ukv_wind_m_s="ukv",
    )


def test_a_product_that_lags_the_station_by_one_hour_is_lowest_at_a_lag_of_one_hour():
    frame = _lag_frame(ukv_shift_hours=1)

    scan = verify.lag_scan(frame=frame, variable=scores.VARIABLES[0])

    assert verify.lowest_lag(scan=scan, product="era5") == 0
    assert verify.lowest_lag(scan=scan, product="ukv") == 1


def test_the_lag_scan_compares_every_lag_on_the_same_station_hours():
    scan = verify.lag_scan(frame=_lag_frame(ukv_shift_hours=0), variable=scores.VARIABLES[0])

    assert scan["n_rows"].n_unique() == 1


def test_a_step_in_a_monthly_series_is_found_at_the_month_it_starts():
    series = np.concatenate([np.zeros(10), np.full(10, 5.0)]) + np.tile([0.1, -0.1], 10)

    position, size = verify.step_candidates(series=series)[0]

    assert position == 10
    assert size > 3.0


def test_a_series_shorter_than_two_windows_has_no_step_candidates():
    assert verify.step_candidates(series=np.zeros(11)) == []


def test_a_station_that_drops_out_part_way_does_not_enter_the_monthly_steps():
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    full = pl.DataFrame({"site": "S1", "month": months})
    partial = pl.DataFrame({"site": "S2", "month": months[:3]})
    frame = pl.concat([full, partial]).with_columns(
        station_wind_m_s=pl.lit(1.0),
        era5_wind_m_s=pl.when(pl.col("site") == "S1").then(2.0).otherwise(100.0),
        ukv_wind_m_s=pl.when(pl.col("site") == "S1").then(3.0).otherwise(100.0),
        station_temp_c=pl.lit(1.0),
        era5_temp_c=pl.lit(1.0),
        ukv_temp_c=pl.lit(1.0),
    )

    steps = verify.monthly_steps(frame=frame).filter(pl.col("variable") == "wind")

    assert steps["ukv_minus_era5"].to_list() == [1.0] * 6


# --- fit: the jobs --------------------------------------------------------------------------------


def test_the_fit_sets_are_60_with_three_wind_farms_and_six_solar_farms():
    wind = fit.domain_jobs(domain="wind")
    solar = fit.domain_jobs(domain="solar")

    assert fit.fit_set_count(jobs=wind, n_sites=3) + fit.fit_set_count(jobs=solar, n_sites=6) == 60
    assert fit.fit_set_count(jobs=fit.cpu_refit_jobs(), n_sites=3) == 3


def test_the_planned_arms_run_at_both_settings_and_the_controls_at_the_primary_setting_only():
    settings: dict[str, set[str]] = {}
    for domain in ("wind", "solar"):
        for arm, setting, _, _, _, _ in fit.domain_jobs(domain=domain):
            settings.setdefault(arm, set()).add(setting)

    assert settings["era5_wind"] == settings["ukv_ceda_wind"] == {"pooled", "sensitivity"}
    assert (
        settings["solar_era5_temp"] == settings["solar_ukv_ceda_temp"] == {"pooled", "sensitivity"}
    )
    assert settings["era5_wind_shuffled"] == {"pooled"}
    assert settings["solar_ukv_ceda_temp_shuffled"] == {"pooled"}


def test_the_hour_ending_arms_fit_the_hour_ending_power_and_every_other_arm_the_centred_power():
    targets = {arm: target for arm, _, target, _, _, _ in fit.domain_jobs(domain="wind")}

    assert (
        targets["era5_wind_hour_ending"]
        == targets["ukv_ceda_wind_hour_ending"]
        == "power_hour_ending_mw"
    )
    assert targets["era5_wind"] == targets["ukv_ceda_wind_shuffled"] == "power_mw"


def test_the_two_arms_of_every_contrast_carry_equal_columns_in_every_domain():
    for domain in ("wind", "solar", "wind_keep_zero"):
        jobs = fit.domain_jobs(domain=domain)
        widths = {arm: len(columns) for arm, _, _, columns, _, _ in jobs}
        for planned in fit.PLANNED_CONTRASTS:
            if planned.domain == domain.removesuffix("_keep_zero"):
                assert widths[planned.treatment] == widths[planned.reference]


def test_a_gpu_runs_two_fits_at_once_and_a_cpu_runs_the_shared_default():
    assert fit.workers_for(device="cuda") == fit.GPU_WORKERS
    assert fit.workers_for(device="cpu") > fit.GPU_WORKERS


def test_the_measured_power_is_read_from_each_jobs_own_target_column():
    frame = pl.DataFrame(
        {
            "site": ["W1", "W1"],
            "time": pl.Series([EPOCH, EPOCH + timedelta(hours=1)], dtype=UTC_US),
            "power_mw": [1.0, 2.0],
            "power_hour_ending_mw": [10.0, 20.0],
        }
    )
    fitted = pl.DataFrame(
        {
            "site": ["W1"] * 4,
            "time": pl.Series([EPOCH, EPOCH + timedelta(hours=1)] * 2, dtype=UTC_US),
            "target": ["power_mw", "power_mw", "power_hour_ending_mw", "power_hour_ending_mw"],
            "signed_error_mw": [0.5, -0.5, 1.0, 2.0],
        }
    )

    result = fit.with_actual_and_prediction(fitted=fitted, frame=frame).sort("target", "time")

    assert result["actual_mw"].to_list() == [10.0, 20.0, 1.0, 2.0]
    assert result["prediction_mw"].to_list() == [11.0, 22.0, 1.5, 1.5]


def test_saved_losses_are_refused_when_the_rows_or_jobs_have_changed(tmp_path: Path):
    frame = pl.DataFrame({"site": ["W1"], "time": pl.Series([EPOCH], dtype=UTC_US), "x": [1.0]})
    jobs = fit.cpu_refit_jobs()
    losses, fingerprint = fit._paths(directory=tmp_path, stem="losses_test")
    pl.DataFrame({"a": [1]}).write_parquet(losses)
    fingerprint.write_text(fit._fingerprint(frame=frame, job_list=jobs))

    assert (
        fit.load_losses(stem="losses_test", frame=frame, jobs=jobs, directory=tmp_path).height == 1
    )
    with pytest.raises(ValueError, match="different row set"):
        fit.load_losses(
            stem="losses_test",
            frame=frame.with_columns(x=pl.lit(2.0)),
            jobs=jobs,
            directory=tmp_path,
        )


# --- fit: intervals -------------------------------------------------------------------------------


def _losses(*, months: int, treatment_error: float, reference_error: float) -> pl.DataFrame:
    """Two arms' losses at one site over `months` months and three seeds."""
    times = [
        datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=30 * m + d)
        for m in range(months)
        for d in range(3)
    ]
    rows = [
        {
            "arm": arm,
            "site": "W1",
            "time": time,
            "seed": seed,
            "month": time.strftime("%Y-%m"),
            "fold": 0,
            "setting": "pooled",
            fit.METRIC: error * (1.0 + 0.1 * np.sin(i)),
        }
        for arm, error in (("t", treatment_error), ("r", reference_error))
        for i, time in enumerate(times)
        for seed in (0, 1, 2)
    ]
    return pl.DataFrame(rows).with_columns(pl.col("time").cast(UTC_US))


def test_a_scope_of_fewer_than_six_months_has_no_interval_and_the_domain_sets_the_margin():
    losses = _losses(months=3, treatment_error=0.05, reference_error=0.06)

    wind = fit.contrast_record(
        losses=losses,
        domain="wind",
        setting="pooled",
        label="P3",
        planned=True,
        kind="all",
        scope="all",
        treatment="t",
        reference="r",
    )
    solar = fit.contrast_record(
        losses=losses,
        domain="solar",
        setting="pooled",
        label="P4",
        planned=True,
        kind="all",
        scope="all",
        treatment="t",
        reference="r",
    )

    assert not wind["enough_months"]
    assert wind["reading"] == "no_interval"
    assert np.isnan(wind["lower_95_pp"])
    assert (wind["margin_pp"], solar["margin_pp"]) == (0.16, 0.06)
    assert wind["difference_pp"] == pytest.approx(-1.0, abs=0.3)


def test_a_wide_gap_over_many_months_reads_as_clearly_ukv_better_in_percentage_points():
    losses = _losses(months=12, treatment_error=0.04, reference_error=0.08)

    record = fit.contrast_record(
        losses=losses,
        domain="wind",
        setting="pooled",
        label="P3",
        planned=True,
        kind="all",
        scope="all",
        treatment="t",
        reference="r",
    )

    assert record["difference_pp"] == pytest.approx(-4.0, abs=0.5)
    assert record["reading"] == "ukv_clearly_better"
    assert record["n_months"] == losses["month"].n_unique() >= 6


def test_a_scope_is_cut_by_calendar_year_half_year_and_window():
    losses = _losses(months=26, treatment_error=0.05, reference_error=0.05)

    scopes = {scope for _, scope, _ in fit.scopes_of(losses=losses)}

    assert {
        "all",
        "year 2023",
        "year 2024",
        "year 2025",
        "October to March",
        "April to September",
        "early window",
        "late window",
    } <= scopes


# --- fit: the decision rule -----------------------------------------------------------------------

UKV_CLEAR = fit.Contrast(difference=-0.5, lower=-0.6, upper=-0.4, margin=0.16)
ERA5_CLEAR = fit.Contrast(difference=0.5, lower=0.4, upper=0.6, margin=0.16)
SMALL = fit.Contrast(difference=-0.05, lower=-0.08, upper=-0.02, margin=0.16)
NULL = fit.Contrast(difference=-0.2, lower=-0.5, upper=0.1, margin=0.16)


def test_wind_goes_to_ukv_ceda_only_if_p3_clearly_favours_it_at_both_settings():
    both = fit.decide_wind(primary=UKV_CLEAR, second=UKV_CLEAR, early=UKV_CLEAR, context=NULL)
    one = fit.decide_wind(primary=UKV_CLEAR, second=NULL, early=UKV_CLEAR, context=UKV_CLEAR)
    other = fit.decide_wind(primary=NULL, second=UKV_CLEAR, early=UKV_CLEAR, context=UKV_CLEAR)

    assert both.product == "ukv_ceda"
    assert one.product == other.product == "era5"


@pytest.mark.parametrize("p3", [ERA5_CLEAR, SMALL, NULL])
def test_wind_defaults_to_era5_when_p3_is_not_clearly_in_favour_of_ukv_ceda(p3: fit.Contrast):
    assert fit.decide_wind(primary=p3, second=p3, early=p3, context=UKV_CLEAR).product == "era5"


def test_p1_never_decides_wind_but_a_disagreement_with_p3_is_reported():
    decision = fit.decide_wind(primary=NULL, second=NULL, early=NULL, context=UKV_CLEAR)

    assert decision.product == "era5"
    assert any("P1 reads" in reason and "P3 decides" in reason for reason in decision.reasons)


def test_temperature_needs_p2_clear_overall_and_at_leads_3_to_5_and_no_p4_veto():
    ok = fit.decide_temperature(
        p2=UKV_CLEAR, p2_late_leads=UKV_CLEAR, p4_primary=NULL, p4_second=SMALL, early=UKV_CLEAR
    )
    late_leads_fail = fit.decide_temperature(
        p2=UKV_CLEAR, p2_late_leads=NULL, p4_primary=NULL, p4_second=NULL, early=UKV_CLEAR
    )
    veto_primary = fit.decide_temperature(
        p2=UKV_CLEAR,
        p2_late_leads=UKV_CLEAR,
        p4_primary=ERA5_CLEAR,
        p4_second=NULL,
        early=UKV_CLEAR,
    )
    veto_second = fit.decide_temperature(
        p2=UKV_CLEAR,
        p2_late_leads=UKV_CLEAR,
        p4_primary=NULL,
        p4_second=ERA5_CLEAR,
        early=UKV_CLEAR,
    )

    assert ok.product == "ukv_ceda"
    assert late_leads_fail.product == veto_primary.product == veto_second.product == "era5"
    assert any("leads 3 to 5" in reason for reason in late_leads_fail.reasons)
    assert any("vetoes" in reason for reason in veto_second.reasons)


def test_a_small_but_statistically_significant_temperature_gap_does_not_win():
    decision = fit.decide_temperature(
        p2=SMALL, p2_late_leads=SMALL, p4_primary=NULL, p4_second=NULL, early=SMALL
    )

    assert decision.product == "era5"


def test_training_history_from_2019_needs_a_negative_early_estimate_and_an_upper_bound_in_margin():
    negative_inside = fit.Contrast(difference=-0.1, lower=-0.3, upper=0.15, margin=0.16)
    upper_beyond = fit.Contrast(difference=-0.1, lower=-0.3, upper=0.17, margin=0.16)
    positive_estimate = fit.Contrast(difference=0.01, lower=-0.1, upper=0.1, margin=0.16)

    assert fit.early_years_test(early=negative_inside) == "from 2019"
    assert fit.early_years_test(early=upper_beyond) == "from 2021 only"
    assert fit.early_years_test(early=positive_estimate) == "from 2021 only"


def test_a_ukv_ceda_recommendation_carries_the_licence_and_names_the_2021_restriction():
    early_fails = fit.decide_wind(
        primary=UKV_CLEAR,
        second=UKV_CLEAR,
        early=fit.Contrast(difference=-0.1, lower=-0.3, upper=0.2, margin=0.16),
        context=UKV_CLEAR,
    )

    text = fit.decision_text(
        wind=early_fails, temperature=fit.Decision("temperature", "era5", "not applicable", ())
    )

    assert "Wind: UKV-CEDA" in text
    assert "from 2021 only" in text
    assert "era-mixed design is untested" in text
    assert fit.LICENCE_CONDITION in text
    assert text.count(fit.LICENCE_CONDITION) == 1


def _set_b_record(
    *, label: str, setting: str, scope: str, domain: str, contrast: fit.Contrast
) -> dict:
    return {
        "label": label,
        "setting": setting,
        "scope": scope,
        "domain": domain,
        "difference_pp": contrast.difference,
        "lower_95_pp": contrast.lower,
        "upper_95_pp": contrast.upper,
        "margin_pp": contrast.margin,
    }


def _set_a_record(
    *,
    variable: str,
    label: str,
    scope: str,
    contrast: fit.Contrast,
    score: str = "bias_removed_hour",
) -> dict:
    return {
        "variable": variable,
        "label": label,
        "scope": scope,
        "score": score,
        "difference": contrast.difference,
        "lower_95": contrast.lower,
        "upper_95": contrast.upper,
        "margin": contrast.margin,
    }


def test_the_rule_reads_each_contrast_from_its_own_record_and_the_primary_score_only():
    set_b = [
        _set_b_record(label="P3", setting=s, scope=sc, domain="wind", contrast=c)
        for s in ("pooled", "sensitivity")
        for sc, c in (("all", UKV_CLEAR), ("early window", UKV_CLEAR))
    ] + [
        _set_b_record(label="P4", setting=s, scope="all", domain="solar", contrast=NULL)
        for s in ("pooled", "sensitivity")
    ]
    set_a = [
        _set_a_record(variable="wind", label="P1", scope="all", contrast=UKV_CLEAR),
        _set_a_record(variable="wind", label="P1", scope="all", contrast=ERA5_CLEAR, score="raw"),
        _set_a_record(variable="temperature", label="P2", scope="all", contrast=UKV_CLEAR),
        _set_a_record(
            variable="temperature", label="P2", scope="all", contrast=ERA5_CLEAR, score="raw"
        ),
        _set_a_record(
            variable="temperature", label="P2-lead", scope="leads 3 to 5", contrast=UKV_CLEAR
        ),
        _set_a_record(
            variable="temperature", label="P2-lead", scope="leads 0 to 2", contrast=ERA5_CLEAR
        ),
        _set_a_record(
            variable="temperature",
            label="P2",
            scope="early window",
            contrast=UKV_CLEAR,
        ),
    ]

    wind, temperature = fit.decisions(set_b=set_b, set_a=set_a)

    assert (wind.product, temperature.product) == ("ukv_ceda", "ukv_ceda")


def test_a_missing_or_duplicated_record_raises_rather_than_guessing():
    record = _set_a_record(variable="wind", label="P1", scope="all", contrast=NULL)

    with pytest.raises(ValueError, match="0 records"):
        fit._contrast_from(records=[], label="P1")
    with pytest.raises(ValueError, match="2 records"):
        fit._contrast_from(records=[record, record], label="P1")


# --- fit: which rows are planned ------------------------------------------------------------------


def _domain_losses(
    *, domain: fit.DomainType, sites: tuple[str, ...], months: int = 14
) -> pl.DataFrame:
    """Losses of every arm and setting of a domain, a day in each of `months` months."""
    rng = np.random.default_rng(0)
    times = [datetime(2020 + (m // 12), m % 12 + 1, 3, tzinfo=UTC) for m in range(months)]
    arms = {(arm, setting) for arm, setting, *_ in fit.domain_jobs(domain=domain)}
    if domain == "wind":
        arms |= {(fit.CPU_REFIT_ARM, "pooled")}
    rows = [
        pl.DataFrame(
            {
                "arm": arm,
                "setting": setting,
                "site": site,
                "seed": seed,
                "time": pl.Series(times, dtype=UTC_US),
                fit.METRIC: np.abs(rng.normal(0.08, 0.01, months)),
            }
        )
        for arm, setting in sorted(arms)
        for site in sites
        for seed in (0, 1, 2)
    ]
    return pl.concat(rows).with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def test_only_the_deciding_contrasts_whole_and_window_rows_are_planned():
    records = fit.domain_records(
        domain="wind", losses=_domain_losses(domain="wind", sites=("W1", "W2"))
    )

    planned = {(r["label"], r["kind"], r["scope"], r["setting"]) for r in records if r["planned"]}

    assert planned == {
        ("P3", kind, scope, setting)
        for kind, scope in (
            ("all", "all"),
            ("window", "early window"),
            ("window", "late window"),
        )
        for setting in ("pooled", "sensitivity")
    }


def test_the_solar_contrast_p4_is_planned_on_the_whole_row_set_only():
    records = fit.domain_records(
        domain="solar", losses=_domain_losses(domain="solar", sites=("A", "B"))
    )

    assert {(r["kind"], r["scope"]) for r in records if r["planned"]} == {("all", "all")}


def test_the_controls_and_checks_are_exploratory_and_fitted_at_the_primary_setting_only():
    records = fit.domain_records(domain="wind", losses=_domain_losses(domain="wind", sites=("W1",)))

    controls = [
        r for r in records if r["label"] in {"control", "GPU against CPU", "hour-ending pair"}
    ]

    assert {r["label"] for r in controls} == {"control", "GPU against CPU", "hour-ending pair"}
    assert all(
        not r["planned"] and r["setting"] == "pooled" and r["scope"] == "all" for r in controls
    )


def test_the_keep_zero_hours_block_reports_the_whole_row_set_and_the_early_window_only():
    records = fit.domain_records(
        domain="wind_keep_zero", losses=_domain_losses(domain="wind_keep_zero", sites=("W1",))
    )

    contrasts = [r for r in records if r["label"] == "keep zero hours"]

    assert {r["scope"] for r in contrasts} == {"all", "early window"}


def test_the_dry_run_counts_the_fit_sets_of_every_domain(capsys: pytest.CaptureFixture[str]):
    frames: dict[fit.DomainType, pl.DataFrame] = {
        "wind": pl.DataFrame({"site": ["W1", "W2", "W3"]}),
        "solar": pl.DataFrame({"site": list("ABCDEF")}),
        "wind_keep_zero": pl.DataFrame({"site": ["W1", "W2", "W3"]}),
    }

    lines = fit.dry_run_lines(frames=frames)

    assert "Fit-sets: 66 on the GPU, plus 3 on the CPU." in " ".join(lines)


# --- charts ---------------------------------------------------------------------------------------


def _chart_rows(*differences: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label": [f"row {i}" for i in range(len(differences))],
            "difference": list(differences),
            "lower_95": [d - 0.1 for d in differences],
            "upper_95": [d + 0.1 for d in differences],
            "second_difference": [None] * len(differences),
            "margin": [0.2] * len(differences),
        },
        schema_overrides={"second_difference": pl.Float64},
    )


def test_a_figure_stops_unless_every_difference_it_draws_is_in_the_report():
    rows = _chart_rows(-0.123, 0.456)

    charts.check_against_report(rows=rows, report="| a | -0.123 | +0.456 |", name="report")
    with pytest.raises(ValueError, match=r"-0\.123"):
        charts.check_against_report(rows=rows, report="| a | +0.123 | +0.456 |", name="report")


def test_the_x_range_is_symmetric_and_holds_every_interval_and_the_margin():
    rows = _chart_rows(-0.5, 0.1).with_columns(margin=pl.lit(0.9))

    low, high = charts.x_domain_for(rows=rows)

    assert low == -high
    assert high >= 0.9
    assert high >= 0.6


def test_a_figure_row_is_read_from_exactly_one_record():
    records = [{"label": "P3", "scope": "all", "setting": "pooled"}] * 2

    with pytest.raises(ValueError, match="2 records"):
        charts._one(records=records, where={"label": "P3"})
    with pytest.raises(ValueError, match="0 records"):
        charts._one(records=records, where={"label": "P4"})


def test_the_three_weeks_are_picked_by_mean_output_and_spread_not_by_eye():
    days = pl.datetime_range(
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 1, 28, 23, tzinfo=UTC),
        interval="1h",
        time_zone="UTC",
        eager=True,
    )
    week = ((days - days.min()).dt.total_days() // 7).to_numpy()
    base = np.array([0.5, 0.9, 0.1, 0.5])[week]
    spread = np.array([0.0, 0.0, 0.0, 0.4])[week] * np.tile([1.0, -1.0], len(days) // 2)
    hourly = pl.DataFrame({"time": days, "fraction": base + spread})

    weeks = charts.pick_weeks(hourly=hourly)

    assert weeks["highest mean output"] == datetime(2023, 1, 8, tzinfo=UTC)
    assert weeks["lowest mean output"] == datetime(2023, 1, 15, tzinfo=UTC)
    assert weeks["largest spread"] == datetime(2023, 1, 22, tzinfo=UTC)


# --- extra cases the mutation pass asked for ------------------------------------------------------


def test_the_primary_score_removes_an_hour_of_day_bias_that_the_month_score_keeps():
    frame = _station_frame().with_columns(
        era5_wind_m_s=pl.col("station_wind_m_s") + pl.col("time").dt.hour().cast(pl.Float64)
    )

    rows = scores.scored_rows(frame=frame, variable=scores.VARIABLES[0])

    assert rows["abs_era5_bias_removed_hour"].max() == pytest.approx(0.0)
    assert float(np.max(rows["abs_era5_bias_removed_month"].to_numpy())) > 1.0


def test_the_half_years_split_october_to_march_from_april_to_september():
    times = [datetime(2020, month, 5, tzinfo=UTC) for month in range(1, 13)]
    rows = pl.DataFrame({"site": "S1", "time": pl.Series(times, dtype=UTC_US)}).with_columns(
        month=pl.col("time").dt.strftime("%Y-%m")
    )

    halves = {
        scope: subset["time"].dt.month().to_list()
        for _, scope, subset in scores.scope_rows(rows=rows)
        if scope in {"October to March", "April to September"}
    }

    assert sorted(halves["October to March"]) == [1, 2, 3, 10, 11, 12]
    assert sorted(halves["April to September"]) == [4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize(("months", "enough"), [(5, False), (6, True)])
def test_an_interval_is_read_from_six_months_and_not_from_five(months: int, enough: bool):
    losses = _losses(months=months, treatment_error=0.04, reference_error=0.08)
    losses = losses.filter(
        pl.col("month").is_in(sorted(losses["month"].unique().to_list())[:months])
    )

    record = fit.contrast_record(
        losses=losses,
        domain="wind",
        setting="pooled",
        label="P3",
        planned=True,
        kind="all",
        scope="all",
        treatment="t",
        reference="r",
    )

    assert record["n_months"] == months
    assert record["enough_months"] is enough


def test_a_downward_step_is_found_as_readily_as_an_upward_one():
    series = np.concatenate([np.full(10, 5.0), np.zeros(10)]) + np.tile([0.1, -0.1], 10)

    position, size = verify.step_candidates(series=series)[0]

    assert position == 10
    assert size < -3.0


def test_a_larger_step_is_listed_before_a_smaller_one_whatever_its_sign():
    rng = np.random.default_rng(0)
    series = rng.normal(0, 0.1, 40)
    series[20:] -= 5.0
    series[8:14] += 0.5

    found = verify.step_candidates(series=series)

    assert found[0][0] == 20


@pytest.mark.parametrize(("months", "enough"), [(5, False), (6, True)])
def test_a_station_interval_is_read_from_six_months_and_not_from_five(months: int, enough: bool):
    frame = _station_frame(months=months + 1)
    rows = scores.scored_rows(frame=frame, variable=scores.VARIABLES[0])
    first = sorted(rows["month"].unique().to_list())[:months]

    record = scores.interval_record(
        rows=rows.filter(pl.col("month").is_in(first)),
        variable=scores.VARIABLES[0],
        label="P1",
        planned=True,
        score="raw",
        scope="all",
    )

    assert record["n_months"] == months
    assert record["enough_months"] is enough


def test_every_shuffled_arm_reads_only_shuffled_product_columns_and_no_other_arm_does():
    for domain in ("wind", "solar"):
        for arm, _, _, columns, _, _ in fit.domain_jobs(domain=domain):
            product_columns = [c for c in columns if "era5" in c or "ukv_ceda" in c]
            if arm.endswith("_shuffled"):
                assert product_columns
                assert all(c.endswith("_shuffled") for c in product_columns)
            else:
                assert not any(c.endswith("_shuffled") for c in columns)


def test_the_readme_names_every_dropped_month_and_says_the_rule_was_planned():
    text = build.dropped_months_text(dropped_months={"2022-12": 0.56, "2020-03": 0.33})

    assert "2020-03 (33%), 2022-12 (56%)" in text
    assert "not a choice made after seeing one" in text
    assert "found 2 such months" in text


def test_a_lossy_month_is_dropped_from_every_arms_rows_and_the_eras_stay_three():
    months = pl.datetime_range(
        datetime(2019, 9, 17, tzinfo=UTC),
        datetime(2026, 4, 30, tzinfo=UTC),
        interval="1d",
        time_zone="UTC",
        eager=True,
    )
    frame = pl.DataFrame({"site": "W1", "time": months, "x": 1.0})

    kept = build._finish(
        frame=frame, shuffle_groups=[("x",)], drop_months=frozenset({"2022-12", "2023-05"})
    )

    assert not {"2022-12", "2023-05", "2019-12", "2026-01"} & set(kept["month"].to_list())
    assert set(kept["era_code"].to_list()) == {0, 1, 2}


# --- review 3 fixes -------------------------------------------------------------------------------


def _record(*, lower: float, upper: float, enough: bool = True) -> fit.IntervalRecord:
    losses = _losses(months=8, treatment_error=0.05, reference_error=0.05)
    record = fit.contrast_record(
        losses=losses,
        domain="wind",
        setting="pooled",
        label="control",
        planned=False,
        kind="all",
        scope="all",
        treatment="t",
        reference="r",
    )
    return {**record, "lower_95_pp": lower, "upper_95_pp": upper, "enough_months": enough}


@pytest.mark.parametrize(
    ("lower", "upper", "enough", "expected"),
    [
        (0.01, 0.51, True, True),
        (-0.51, -0.01, True, True),
        (-0.30, 0.30, True, False),
        (0.15, 0.65, True, False),
        (0.01, 0.51, False, False),
    ],
)
def test_a_control_is_near_the_line_when_a_bound_is_within_a_fifth_of_the_width_from_zero(
    lower: float, upper: float, enough: bool, expected: bool
):
    assert fit.near_the_line(record=_record(lower=lower, upper=upper, enough=enough)) is expected


def test_the_report_flags_a_control_near_the_line_and_a_non_control_is_not_flagged():
    near = _record(lower=0.01, upper=0.51)
    other = fit.IntervalRecord(**{**near, "label": "P3"})

    flagged = fit.near_line_lines(records=[near, other])

    assert sum("wind, control" in line for line in flagged) == 1
    assert not any("P3" in line for line in flagged)
    assert "None" in fit.near_line_lines(records=[other])[-1]


def test_fitted_losses_are_saved_in_a_fixed_order_whatever_order_the_fits_finish_in():
    losses = pl.DataFrame(
        {
            "arm": ["b", "a", "a", "a"],
            "setting": ["pooled"] * 4,
            "target": ["power_mw"] * 4,
            "site": ["W1"] * 4,
            "time": pl.Series([EPOCH] * 4, dtype=UTC_US),
            "seed": [0, 2, 1, 0],
        }
    )

    result = fit.in_stable_order(losses=losses)

    assert result["arm"].to_list() == ["a", "a", "a", "b"]
    assert result["seed"].to_list() == [0, 1, 2, 0]


def test_the_fit_rows_are_sorted_by_site_and_time_so_the_shuffle_and_subsampling_are_repeatable():
    days = pl.datetime_range(
        datetime(2019, 9, 17, tzinfo=UTC),
        datetime(2026, 4, 30, tzinfo=UTC),
        interval="1d",
        time_zone="UTC",
        eager=True,
    )
    frame = pl.concat(
        [pl.DataFrame({"site": site, "time": days, "x": 1.0}) for site in ("W2", "W1")]
    ).sample(fraction=1.0, shuffle=True, seed=3)

    kept = build._finish(frame=frame, shuffle_groups=[("x",)], drop_months=frozenset())

    assert kept.select("site", "time").equals(kept.select("site", "time").sort("site", "time"))


def test_the_monthly_steps_carry_era5_minus_station_and_a_deseasonalised_copy_of_every_series():
    months = [f"{year}-{m:02d}" for year in (2020, 2021) for m in range(1, 13)]
    frame = pl.DataFrame({"site": "S1", "month": months}).with_columns(
        station_wind_m_s=pl.lit(1.0),
        era5_wind_m_s=pl.lit(3.0),
        ukv_wind_m_s=pl.lit(2.0),
        station_temp_c=pl.lit(1.0),
        era5_temp_c=pl.lit(1.0),
        ukv_temp_c=pl.lit(1.0),
    )
    # A seasonal cycle that repeats every year disappears when each calendar month's mean goes.
    frame = frame.with_columns(
        ukv_wind_m_s=pl.col("ukv_wind_m_s") + pl.col("month").str.slice(5, 2).cast(pl.Float64)
    )

    steps = verify.deseasonalised(steps=verify.monthly_steps(frame=frame))
    wind = steps.filter(pl.col("variable") == "wind")

    assert wind["era5_minus_station"].to_list() == [2.0] * 24
    assert wind["ukv_minus_era5_deseasonalised"].abs().max() == pytest.approx(0.0)
    assert float(np.max(wind["ukv_minus_era5"].abs().to_numpy())) > 5.0
    assert set(verify.STEP_SERIES) <= set(steps.columns)
