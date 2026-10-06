"""Tests for the pure functions of the four `ukv_ceda_vs_openmeteo_*` scripts of study #1051.

Every frame is synthetic, so no test needs `data/`. Each test is built to fail on the bug it exists
for: a wind speed left in km/h, an irradiance snapshot left unscaled by the zenith-cosine ratio, a
transfer frame that copies CEDA's values instead of Open-Meteo's, one shared permutation for both
archives' control columns, a temperature taken at one instant instead of two, a contrast read
against the wrong margin, and a one-sided penalty read two-sided.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pytest
import ukv_ceda_vs_openmeteo_build as build
import ukv_ceda_vs_openmeteo_compare as compare
import ukv_ceda_vs_openmeteo_extra_reads as extra_reads
import ukv_ceda_vs_openmeteo_fit as fit
import ukv_ceda_vs_openmeteo_wind_steps as steps

UTC_US: Final[pl.Datetime] = pl.Datetime("us", "UTC")


# --- build: months, columns, Open-Meteo -----------------------------------------------------------


def test_the_study_months_are_the_23_whole_months_without_the_upgrade_month():
    assert len(build.STUDY_MONTHS) == 23
    assert build.STUDY_MONTHS[0] == "2024-09"
    assert build.STUDY_MONTHS[-1] == "2026-08"
    assert "2026-01" not in build.STUDY_MONTHS
    assert "2024-08" not in build.STUDY_MONTHS


def test_the_two_arms_of_each_domain_carry_equal_numbers_of_distinct_columns():
    build.check_arm_widths()

    for archive in build.ARCHIVES:
        assert len(build.wind_arm_columns(archive=archive)) == 6
        assert len(build.solar_arm_columns(archive=archive)) == 8
    assert set(build.wind_arm_columns(archive="ceda")) != set(build.wind_arm_columns(archive="om"))
    shuffled = build.solar_arm_columns(archive="om", shuffled=True)
    assert all(name.endswith(build.SHUFFLED_SUFFIX) for name in shuffled[-2:])


def _open_meteo_file(path: Path, *, first: datetime, hours: int) -> Path:
    times = [first + timedelta(hours=hour) for hour in range(hours)]
    pl.DataFrame(
        {
            "site": "A",
            "time": times,
            "temperature_2m": 10.0,
            "wind_speed_10m": 36.0,
            "wind_direction_10m": 90.0,
            "shortwave_radiation": 100.0,
        }
    ).write_parquet(path)
    return path


def test_open_meteo_wind_is_converted_from_kilometres_per_hour_to_metres_per_second(tmp_path: Path):
    path = _open_meteo_file(tmp_path / "om.parquet", first=build.OPEN_METEO_FIRST_HOUR, hours=3)

    frame = build.read_open_meteo(path=path)

    assert frame["om_speed_10m_m_s"].to_list() == pytest.approx([10.0] * 3)


def test_the_backfill_and_the_unverified_hours_never_reach_an_arm(tmp_path: Path):
    first = build.OPEN_METEO_FIRST_HOUR - timedelta(hours=5)
    path = _open_meteo_file(tmp_path / "om.parquet", first=first, hours=24 * 120)

    frame = build.read_open_meteo(path=path)

    low, high = build.UNVERIFIED_HOURS
    assert frame["time"].min() == build.OPEN_METEO_FIRST_HOUR
    assert frame.filter(pl.col("time").is_between(low, high)).is_empty()
    # The 94 hours are inclusive at both ends.
    assert frame.filter(pl.col("time") == low - timedelta(hours=1)).height == 1
    assert frame.filter(pl.col("time") == high + timedelta(hours=1)).height == 1
    build.check_no_backfill(frame=frame)


def test_a_row_before_the_downloader_started_raises():
    early = pl.DataFrame({"time": [build.OPEN_METEO_FIRST_HOUR - timedelta(hours=1)]}).cast(
        {"time": UTC_US}
    )

    with pytest.raises(ValueError, match="backfill"):
        build.check_no_backfill(frame=early)


def test_open_meteo_temperature_is_the_mean_of_the_instants_at_the_hours_two_ends():
    start = datetime(2025, 6, 1, tzinfo=UTC)
    source = pl.DataFrame(
        {
            "site": "A",
            "time": [start + timedelta(hours=h) for h in range(4)],
            "om_temp_c": [10.0, 14.0, 20.0, 30.0],
        }
    )
    frame = pl.DataFrame({"site": "A", "time": [start + timedelta(hours=h) for h in (0, 2, 3)]})

    result = build.open_meteo_temperature_two_end_mean(frame=frame, open_meteo=source)

    # The hour before hour 0 is missing from the source, so its mean is null rather than the
    # instant at the label alone.
    assert result["om_temp"].to_list() == [None, 17.0, 25.0]


# --- build: the guards -------------------------------------------------------------------------


def _model_free(*, speed_ratio: float = 1.0, ghi_ratio: float = 1.0, era: int = 0) -> pl.DataFrame:
    hours = [6, 12, 18]
    rows = []
    for index, hour in enumerate(hours * 20):
        rows.append(
            {
                "site": "A",
                "time": datetime(2025, 6, 1, tzinfo=UTC) + timedelta(days=index),
                "lead_hours": 0,
                "hour_of_day": hour,
                "era_code": era,
                "ceda_temp_c": 10.0,
                "om_temp_c": 10.0,
                "ceda_speed_10m_m_s": 8.0,
                "om_speed_10m_m_s": 8.0 * speed_ratio,
                "om_ghi": 300.0,
                "ceda_ghi": 300.0 * ghi_ratio,
            }
        )
    return pl.DataFrame(rows)


def test_the_unit_guard_fails_on_a_factor_of_3_6_and_passes_on_matched_units():
    assert build.unit_guard_failures(frame=_model_free()) == []
    assert build.unit_guard_failures(frame=_model_free(speed_ratio=3.6))
    assert build.unit_guard_failures(frame=_model_free(speed_ratio=1 / 3.6))


def test_the_unit_guard_fails_on_a_temperature_in_kelvin():
    frame = _model_free().with_columns(ceda_temp_c=pl.col("ceda_temp_c") + 273.15)

    assert build.unit_guard_failures(frame=frame)


def test_the_irradiance_guard_fails_where_the_rebuilt_value_misses_by_the_zenith_ratio():
    assert build.irradiance_guard_failures(frame=_model_free()) == []
    # A snapshot that skipped the zenith-cosine ratio reads 30% high, as it does at 06 UTC.
    assert build.irradiance_guard_failures(frame=_model_free(ghi_ratio=1.3))


def test_the_irradiance_guard_binds_the_era_before_the_upgrade_and_only_notes_the_era_after():
    after = _model_free(ghi_ratio=1.3, era=1).with_columns(era_code=pl.lit(1, dtype=pl.Int64))
    both = pl.concat([_model_free(), after])

    assert build.irradiance_guard_failures(frame=both) == []
    assert build.irradiance_mismatch_notes(frame=both)


# --- build: the transfer frame and the rows ----------------------------------------------------


def _wind_rows_for(*, n_months: int = 23) -> pl.DataFrame:
    rows = []
    for index, month in enumerate(build.STUDY_MONTHS[:n_months]):
        year, number = (int(part) for part in month.split("-"))
        # Eight rows to a month, so two permutations of a month are unlikely to coincide.
        for day in range(1, 9):
            time = datetime(year, number, day, 12, tzinfo=UTC)
            rows.append(
                {
                    "site": "W1",
                    "time": time,
                    "month": month,
                    "fold": index % 5,
                    "power_mw": 1.0,
                    "ukv_ceda_speed_10m": 5.0 + index + day / 100.0,
                    "ukv_ceda_sin_10m": 0.1,
                    "ukv_ceda_cos_10m": 0.9,
                    "om_speed_10m": 50.0 + index + day / 100.0,
                    "om_speed_10m_rescaled": 5.0 + index + day / 100.0,
                    "om_sin_10m": 0.2,
                    "om_cos_10m": 0.8,
                    "hour_of_day": 12,
                    "day_of_year": time.timetuple().tm_yday,
                    "era_code": 0,
                }
            )
    return pl.DataFrame(rows)


def test_the_transfer_frame_holds_open_meteo_values_under_the_ceda_arms_column_names():
    rows = _wind_rows_for()

    swapped = build.transfer_frame(frame=rows, domain="wind")

    speed, sine, cosine = build.wind_columns(archive="ceda")
    assert swapped[speed].to_list() == rows["om_speed_10m"].to_list()
    assert swapped[sine].to_list() == rows["om_sin_10m"].to_list()
    assert swapped[cosine].to_list() == rows["om_cos_10m"].to_list()
    assert swapped["hour_of_day"].to_list() == rows["hour_of_day"].to_list()
    assert set(build.wind_arm_columns(archive="ceda")) <= set(swapped.columns)


def test_a_transfer_frame_for_an_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown domain"):
        build.transfer_frame(frame=_wind_rows_for(), domain="hydro")


def _wind_base() -> pl.DataFrame:
    rows = _wind_rows_for()
    return rows.select(
        "site",
        "time",
        "month",
        "power_mw",
        "hour_of_day",
        "day_of_year",
        "ukv_ceda_speed_10m",
        "ukv_ceda_sin_10m",
        "ukv_ceda_cos_10m",
        effective_capacity_mw=pl.lit(2.0),
        constrained=pl.lit(value=False),
        cap_mw=pl.lit(None, dtype=pl.Float64),
    )


def _open_meteo_for(*, base: pl.DataFrame) -> pl.DataFrame:
    return base.select(
        "site",
        "time",
        om_temp_c=pl.col("ukv_ceda_speed_10m") * 0.0 + 10.0,
        om_speed_10m_m_s=pl.col("ukv_ceda_speed_10m"),
        om_direction_10m_deg=pl.lit(90.0),
        om_ghi=pl.lit(100.0),
    )


def _wind_model_free(*, base: pl.DataFrame) -> pl.DataFrame:
    """Lead-0 instants at the base rows' hours, with Open-Meteo 3% below CEDA's speed."""
    return base.select(
        "site",
        "month",
        lead_hours=pl.lit(0),
        om_wind_step=pl.lit(value=False),
        ceda_speed_10m_m_s=pl.col("ukv_ceda_speed_10m"),
        om_speed_10m_m_s=pl.col("ukv_ceda_speed_10m") * 0.97,
    )


def test_each_archives_control_columns_get_their_own_permutation():
    base = _wind_base()
    # Identical values in both archives: a shared permutation would shuffle them identically, and
    # the control would then differ by zero by construction.
    open_meteo = _open_meteo_for(base=base)

    rows = build.wind_rows(base=base, open_meteo=open_meteo, model_free=_wind_model_free(base=base))

    ceda = rows["ukv_ceda_speed_10m_shuffled"].to_list()
    om = rows["om_speed_10m_shuffled"].to_list()
    assert rows["ukv_ceda_speed_10m"].to_list() == pytest.approx(rows["om_speed_10m"].to_list())
    assert ceda != om


def test_the_wind_rows_drop_hours_with_a_null_open_meteo_direction_and_carry_two_eras():
    base = _wind_base()
    open_meteo = _open_meteo_for(base=base).with_columns(
        om_direction_10m_deg=pl.when(pl.col("time") == base["time"][0])
        .then(None)
        .otherwise(pl.col("om_direction_10m_deg"))
    )

    rows = build.wind_rows(base=base, open_meteo=open_meteo, model_free=_wind_model_free(base=base))

    # One null direction, the two days of 2024-11 inside the first step span (the month keeps its
    # other hours, being under the loss limit), and all of 2025-02, which the second span empties.
    assert rows.height == base.height - 1 - 2 - 8
    assert base["time"][0] not in rows["time"].to_list()
    assert "2025-02" not in rows["month"].to_list()
    assert "2024-11" in rows["month"].to_list()
    assert sorted(rows["era_code"].unique().to_list()) == [0, 1]
    assert rows.filter(pl.col("month") >= "2026-02")["era_code"].min() == 1


def test_the_loss_by_month_counts_the_share_of_base_rows_lost_and_flags_the_lossy_months():
    base = pl.DataFrame({"month": ["2025-01"] * 4 + ["2025-02"] * 4})
    kept = pl.DataFrame({"month": ["2025-01"] * 4 + ["2025-02"] * 2})

    shares = build.loss_by_month(base=base, kept=kept)

    assert shares == {"2025-01": 0.0, "2025-02": 0.5}
    assert build.lossy_months(shares=shares) == ["2025-02"]
    assert build.lossy_months(shares={"2025-03": 0.25}) == []


# --- fit: readings and arms --------------------------------------------------------------------


def test_the_margins_are_the_frozen_values_in_points_of_capacity():
    """A tripwire: the margins were fixed before any result, so a change here is a post hoc edit."""
    assert build.MARGIN_WIND_PP == 0.16
    assert build.MARGIN_SOLAR_PP == 0.06
    assert fit.MARGINS_PP == {"wind": 0.16, "solar": 0.06, "solar_era0": 0.06}


@pytest.mark.parametrize(
    ("difference", "lower", "upper", "expected"),
    [
        (0.0, -0.10, 0.10, "interchangeable"),
        (0.05, 0.00, 0.10, "interchangeable"),
        (0.30, 0.20, 0.40, "differ"),
        (-0.30, -0.40, -0.20, "differ"),
        (0.10, 0.05, 0.12, "interchangeable"),
        (0.10, 0.02, 0.20, "unresolved"),
        (0.30, -0.10, 0.60, "unresolved"),
        (-0.30, -0.45, -0.17, "differ"),
    ],
)
def test_a_two_sided_contrast_reads_against_its_margin(
    difference: float, lower: float, upper: float, expected: str
):
    reading = fit.two_sided_reading(difference=difference, lower=lower, upper=upper, margin=0.16)

    assert reading == expected


@pytest.mark.parametrize(
    ("difference", "lower", "upper", "expected"),
    [
        (0.02, -0.05, 0.10, "no_penalty"),
        (-0.50, -0.80, -0.20, "no_penalty"),
        (0.40, 0.20, 0.60, "penalty"),
        (0.10, 0.02, 0.30, "unresolved"),
        (0.40, -0.10, 0.60, "unresolved"),
    ],
)
def test_a_transfer_penalty_is_read_one_sided(
    difference: float, lower: float, upper: float, expected: str
):
    reading = fit.penalty_reading(difference=difference, lower=lower, upper=upper, margin=0.16)

    assert reading == expected


def test_a_large_negative_penalty_is_never_a_two_sided_differ():
    # CEDA-trained models scoring better on Open-Meteo's values is not a penalty to act on.
    assert fit.penalty_reading(difference=-0.5, lower=-0.8, upper=-0.2, margin=0.16) == "no_penalty"
    assert fit.two_sided_reading(difference=-0.5, lower=-0.8, upper=-0.2, margin=0.16) == "differ"


def test_a_verdict_stands_only_if_both_settings_agree():
    assert fit.combine_settings(readings=["penalty", "penalty"]) == "penalty"
    assert fit.combine_settings(readings=["penalty", "unresolved"]) == "unresolved"
    assert fit.combine_settings(readings=["no_penalty", "penalty"]) == "unresolved"


def test_an_unresolved_transfer_penalty_recommends_not_mixing_the_archives():
    assert "do not mix" in fit.RECOMMENDATIONS["unresolved"]
    assert "do not mix" not in fit.RECOMMENDATIONS["no_penalty"]


def test_each_arm_is_fitted_on_columns_of_its_own_archive_and_the_ceda_arm_is_scored_four_ways():
    wind_ceda, wind_om = fit.arm_names(domain="wind")
    ordinary = {job[0]: job for job in fit.ordinary_jobs(domain="wind")}
    transfer = fit.transfer_jobs(domain="wind")

    assert wind_om in ordinary
    assert wind_ceda not in ordinary
    assert all(job[0] == wind_ceda for job in transfer)
    assert {job[1] for job in transfer} == {fit.PRIMARY_SETTING, fit.SECOND_SETTING}
    assert all(set(job[3]) == set(build.wind_arm_columns(archive="ceda")) for job in transfer)
    assert fit.scoring_names(domain="solar") == (
        "ceda",
        "om",
        "om_temp",
        "om_ghi",
        "om_temp_offset_removed",
    )


def test_the_dry_run_counts_the_78_planned_fit_sets():
    frames: dict[fit.DomainType, pl.DataFrame] = {
        "wind": pl.DataFrame({"site": ["W1", "W2", "W3"] * 2, "x": range(6)}),
        "solar": pl.DataFrame({"site": list("ABCDEF") * 2, "x": range(12)}),
        "solar_era0": pl.DataFrame({"site": list("ABCDEF") * 2, "x": range(12)}),
    }

    jobs = {domain: len(fit.all_jobs(domain=domain)) for domain in frames}

    # Wind and solar have 2 + 2 controls + 2 CEDA-trained jobs. The era-0 sensitivity has 2 + 2.
    assert jobs == {"wind": 6, "solar": 6, "solar_era0": 4}
    assert jobs["wind"] * 3 + jobs["solar"] * 6 + jobs["solar_era0"] * 6 == 78
    assert "Total 78 fit-sets" in "\n".join(fit.dry_run_lines(frames=frames))


def test_the_era_0_sensitivity_has_no_controls_and_reads_the_solar_arms():
    arms = {job[0] for job in fit.all_jobs(domain="solar_era0")}

    assert arms == {"om_ghi_temp", "ceda_ghi_temp"}
    assert fit.arm_names(domain="solar_era0") == ("ceda_ghi_temp", "om_ghi_temp")
    assert fit.scoring_names(domain="solar_era0") == fit.scoring_names(domain="solar")
    assert fit.PLANNED_SCOPE["solar_era0"] == "all"


def test_a_scored_arm_is_named_for_its_scoring_frame():
    assert fit.scored_arm_name(arm="ceda_wind_10m", scoring="ceda") == "ceda_wind_10m"
    assert fit.scored_arm_name(arm="ceda_wind_10m", scoring="om") == "ceda_wind_10m_scored_on_om"


def _solar_site_rows() -> pl.DataFrame:
    times = [datetime(2025, 6, 1, h, tzinfo=UTC) for h in range(24)]
    columns = dict.fromkeys(build.SOLAR_SHARED_FEATURES, 1.0)
    return pl.DataFrame(
        {
            **columns,
            "time": times,
            "fold": 0,
            "hour_of_day": [t.hour for t in times],
            "ceda_ghi": 100.0,
            "om_ghi": 200.0,
            "ceda_temp": 10.0,
            "om_temp": [11.0 if t.hour % 6 == 0 else 15.0 for t in times],
            "om_temp_offset_removed": [t.hour * 0.5 for t in times],
        }
    )


def test_each_partial_swap_replaces_only_its_own_columns_of_the_ceda_arm():
    site_rows = _solar_site_rows()

    frames = fit.scoring_frames(site_rows=site_rows, domain="solar")

    assert frames["ceda"]["ceda_ghi"].unique().to_list() == [100.0]
    assert frames["om"]["ceda_ghi"].unique().to_list() == [200.0]
    assert frames["om_temp"]["ceda_ghi"].unique().to_list() == [100.0]
    assert frames["om_temp"]["ceda_temp"].to_list() == site_rows["om_temp"].to_list()
    assert frames["om_ghi"]["ceda_temp"].unique().to_list() == [10.0]
    assert frames["om_ghi"]["ceda_ghi"].unique().to_list() == [200.0]
    # The offset-removed temperature is the build's own column, which the build learned on the
    # training folds, so the fit adds nothing to it.
    assert frames["om_temp_offset_removed"]["ceda_temp"].to_list() == (
        site_rows["om_temp_offset_removed"].to_list()
    )


def test_the_wind_partial_swap_replaces_the_direction_and_keeps_the_ceda_speed():
    rows = _wind_rows_for()

    frames = fit.scoring_frames(site_rows=rows, domain="wind")

    speed, sine, cosine = build.wind_columns(archive="ceda")
    assert frames["om_direction"][speed].to_list() == rows[speed].to_list()
    assert frames["om_direction"][sine].to_list() == rows["om_sin_10m"].to_list()
    assert frames["om_direction"][cosine].to_list() == rows["om_cos_10m"].to_list()


def test_the_scopes_of_a_contrast_split_the_eras_the_lead_zero_hours_and_each_site():
    losses = pl.DataFrame(
        {
            "time": [
                datetime(2025, 6, 1, 6, tzinfo=UTC),
                datetime(2025, 6, 1, 7, tzinfo=UTC),
                datetime(2026, 3, 1, 12, tzinfo=UTC),
            ],
            "month": ["2025-06", "2025-06", "2026-03"],
            "site": ["W1", "W1", "W2"],
        }
    )

    scopes = dict(fit.scopes_of(losses=losses))
    count = {name: losses.filter(cond).height for name, cond in scopes.items() if cond is not None}

    assert count["era 0"] == 2
    assert count["era 1"] == 1
    assert count["lead 0 only"] == 2
    assert count["site W1"] == 2
    assert count["April to September"] == 2
    assert count["October to March"] == 1


# --- compare -----------------------------------------------------------------------------------


def test_a_direction_difference_wraps_at_north():
    frame = pl.DataFrame({"a": [359.0, 1.0, 90.0], "b": [1.0, 359.0, 270.0]})

    result = frame.select(d=compare.circular_difference_degrees(ceda=pl.col("a"), om=pl.col("b")))

    # 359 against 1 is -2 degrees, 1 against 359 is +2, and opposite directions are +-180.
    assert result["d"].to_list() == [-2.0, 2.0, -180.0]


def test_the_nearest_cell_wins_where_it_has_the_smallest_error():
    ceda = np.array([[1.0, 5.0, 9.0], [4.0, 1.0, 9.0], [2.0, 2.0, 2.0]])
    om = np.array([1.2, 1.2, 2.0])

    # Hour 1's best cell is the second, and hour 2 ties, which goes to the nearest cell.
    assert compare.nearest_cell_wins(ceda=ceda, om=om) == pytest.approx(2 / 3)


def test_the_temperature_offsets_are_read_at_lead_zero_only_and_by_site():
    frame = pl.DataFrame(
        {
            "site": ["A", "A", "A", "B"],
            "lead_hours": [0, 0, 3, 0],
            "ceda_temp_c": [10.0, 12.0, 10.0, 5.0],
            "om_temp_c": [9.0, 11.0, 0.0, 5.5],
        }
    )

    offsets = compare.temperature_offsets(frame=frame)

    assert offsets["mean_offset_k"].to_list() == pytest.approx([1.0, -0.5])
    assert offsets["n"].to_list() == [2, 1]


def test_a_variable_keeps_only_rows_where_both_archives_hold_a_value_and_differences_them():
    frame = pl.DataFrame(
        {
            "ceda_temp_c": [10.0, None, 12.0],
            "om_temp_c": [9.0, 9.0, None],
        }
    )

    rows = compare.with_difference(frame=frame, variable=compare.VARIABLES[0])

    assert rows["difference"].to_list() == [1.0]


def test_irradiance_rows_leave_out_the_night_hours_where_both_archives_read_zero():
    frame = pl.DataFrame({"ceda_ghi": [0.0, 0.0, 5.0], "om_ghi": [0.0, 3.0, 0.0]})
    variable = next(v for v in compare.VARIABLES if v.name == "global irradiance (rebuilt)")

    rows = compare.with_difference(frame=frame, variable=variable)

    assert rows["difference"].to_list() == [-3.0, 5.0]


# --- fit: the transfer run ------------------------------------------------------------------------


def test_the_transfer_run_scores_one_ceda_trained_model_on_every_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    few_rounds = {**fit.PRIMARY_HYPER_PARAMETERS, "num_boost_round": 5, "min_child_weight": 1.0}
    monkeypatch.setattr(fit, "PRIMARY_HYPER_PARAMETERS", few_rounds)
    monkeypatch.setattr(fit, "SENSITIVITY_HYPER_PARAMETERS", few_rounds)
    rows = _wind_rows_for().with_columns(
        # Open-Meteo reads far above anything CEDA's model trained on, so scoring on it must move
        # the predictions, and the target rises with CEDA's speed.
        power_mw=pl.col("ukv_ceda_speed_10m"),
        effective_capacity_mw=pl.lit(2.0),
        cap_mw=pl.lit(None, dtype=pl.Float64),
        constrained=pl.lit(value=False),
        fold=pl.col("fold").cast(pl.Int32),
    )

    losses = fit.run_transfer(domain="wind", frame=rows, device="cpu", max_workers=1)

    arms = set(losses["arm"].unique().to_list())
    assert arms == {
        "ceda_wind_10m",
        "ceda_wind_10m_scored_on_om",
        "ceda_wind_10m_scored_on_om_direction",
        "ceda_wind_10m_scored_on_om_speed_rescaled",
    }

    def mean_error(arm: str) -> float:
        rows = losses.filter(pl.col("arm") == arm)
        return float(np.mean(rows["absolute_error_mw"].to_numpy()))

    own_rows = losses.filter(pl.col("arm") == "ceda_wind_10m")
    swapped_rows = losses.filter(pl.col("arm") == "ceda_wind_10m_scored_on_om")
    assert own_rows.height == swapped_rows.height == rows.height * 3 * 2
    assert set(own_rows["setting"].unique().to_list()) == {
        fit.PRIMARY_SETTING,
        fit.SECOND_SETTING,
    }
    own = mean_error("ceda_wind_10m")
    swapped = mean_error("ceda_wind_10m_scored_on_om")
    assert swapped != own
    # The direction swap changes columns the model barely uses here, so it scores like the own frame
    # far more closely than the full swap does.
    partial = mean_error("ceda_wind_10m_scored_on_om_direction")
    assert abs(partial - own) < abs(swapped - own)


# --- the solar contrasts are planned on era 0 only ----------------------------------------------


def _ratio_frame(*, era_1_scale: float) -> pl.DataFrame:
    """Lead-0 rows with CEDA at 06 and 18 UTC scaled against Open-Meteo, by era."""
    rows = []
    for era in (0, 1):
        for hour, factor in ((6, 1.0), (12, 1.0), (18, 1.0)):
            for index in range(5):
                scale = era_1_scale if era == 1 and hour == 6 else 1.0
                scale = 1.0 / era_1_scale if era == 1 and hour == 18 else scale
                rows.append(
                    {
                        "site": "A",
                        "time": datetime(2025, 6, 1, tzinfo=UTC) + timedelta(days=index),
                        "lead_hours": 0,
                        "hour_of_day": hour,
                        "era_code": era,
                        "om_ghi": 300.0,
                        "ceda_ghi": 300.0 * factor * scale,
                        "ceda_ghi_snapshot": 450.0,
                    }
                )
    return pl.DataFrame(rows)


def test_the_era_1_note_names_the_hours_whose_rebuilt_ratio_misses():
    note = build.era_1_irradiance_note(frame=_ratio_frame(era_1_scale=1.25))

    assert note == (
        "irradiance construction differs after PS47 (ratio 1.25 at 06 UTC, 0.80 at 18 UTC)"
    )


def test_the_era_1_note_is_empty_where_every_hour_matches():
    assert build.era_1_irradiance_note(frame=_ratio_frame(era_1_scale=1.0)) == ""


def test_the_ratio_table_is_by_era_and_hour_and_reports_the_raw_snapshot_too():
    table = build.irradiance_ratios_by_era_hour(frame=_ratio_frame(era_1_scale=1.25))

    assert table.height == 6
    row = table.filter((pl.col("era_code") == 1) & (pl.col("hour_of_day") == 6)).row(0, named=True)
    assert row["rebuilt_ratio"] == pytest.approx(1.25)
    assert row["raw_ratio"] == pytest.approx(1.5)


def _contrast_losses() -> pl.DataFrame:
    """Solar losses over 10 months of era 0 and 7 of era 1, three seeds, both settings."""
    months = [f"2025-{m:02d}" for m in range(3, 13)] + [f"2026-{m:02d}" for m in range(2, 9)]
    rows = []
    for month in months:
        year, number = (int(part) for part in month.split("-"))
        time = datetime(year, number, 15, 12, tzinfo=UTC)
        for setting in (fit.PRIMARY_SETTING, fit.SECOND_SETTING):
            for seed in range(3):
                for arm, error in (
                    ("ceda_ghi_temp", 0.10),
                    ("om_ghi_temp", 0.10),
                    ("ceda_ghi_temp_scored_on_om", 0.10),
                    ("ceda_ghi_temp_shuffled", 0.30),
                    ("om_ghi_temp_shuffled", 0.30),
                ):
                    rows.append(
                        {
                            "site": "A",
                            "time": time,
                            "month": month,
                            "seed": seed,
                            "setting": setting,
                            "arm": arm,
                            fit.METRIC: error
                            + (0.5 if (arm == "om_ghi_temp" and year == 2026) else 0.0),
                        }
                    )
    return pl.DataFrame(rows)


def test_a_solar_contrast_is_planned_only_on_era_0_and_other_scopes_carry_the_era_1_note():
    note = "irradiance construction differs after PS47 (ratio 1.11 at 06 UTC, 0.86 at 18 UTC)"

    records = fit.domain_records(domain="solar", losses=_contrast_losses(), era_1_note=note)

    p2 = [r for r in records if r["label"] == "P2"]
    planned = {r["scope"] for r in p2 if r["planned"]}
    assert planned == {"era 0"}
    assert all(r["note"] == "" for r in p2 if r["scope"] == "era 0")
    assert all(r["note"] == note for r in p2 if r["scope"] not in fit.ERA_0_SCOPES)
    # The era-1 months differ by 0.5 per row, so the exploratory era-1 row sees a large difference
    # while the planned era-0 row sees none.
    era_0 = next(r for r in p2 if r["scope"] == "era 0" and r["setting"] == fit.PRIMARY_SETTING)
    era_1 = next(r for r in p2 if r["scope"] == "era 1" and r["setting"] == fit.PRIMARY_SETTING)
    assert era_0["difference_pp"] == pytest.approx(0.0)
    assert era_1["difference_pp"] < -1.0


def test_the_solar_controls_are_read_on_the_planned_era_only():
    records = fit.domain_records(domain="solar", losses=_contrast_losses())

    controls = [r for r in records if r["label"] == "control"]

    assert controls
    assert {r["scope"] for r in controls} == {"era 0"}
    assert not any(r["planned"] for r in controls)


def test_the_solar_verdicts_read_era_0_and_the_wind_verdicts_read_all_rows():
    assert fit.PLANNED_SCOPE == {"wind": "all", "solar": "era 0", "solar_era0": "all"}
    records = fit.domain_records(domain="solar", losses=_contrast_losses())

    found = [v for v in fit.verdicts(records=records) if v.domain == "solar"]

    assert {v.label for v in found} == {"P2", "P3"}
    assert all(v.primary["scope"] == "era 0" for v in found)
    # Era 0 shows no difference, so a verdict read on all rows would not be "interchangeable".
    assert all(v.primary["difference_pp"] == pytest.approx(0.0) for v in found)


# --- the planned solar verdict combines two fits, and the build and fit gates --------------------


def _two_fit_records(*, era_0_trained_error: float) -> list[fit.IntervalRecord]:
    """Interval records for the all-rows solar fit and the era-0-trained fit."""
    all_rows = _contrast_losses()
    era_0_only = all_rows.filter(pl.col("month") < "2026-02").with_columns(
        pl.when(pl.col("arm") == "om_ghi_temp")
        .then(pl.col(fit.METRIC) + era_0_trained_error)
        .otherwise(pl.col(fit.METRIC))
        .alias(fit.METRIC)
    )
    return [
        *fit.domain_records(domain="solar", losses=all_rows),
        *fit.domain_records(domain="solar_era0", losses=era_0_only),
    ]


def test_a_solar_verdict_stands_only_if_the_all_rows_and_era_0_fits_agree():
    agree = fit.verdicts(records=_two_fit_records(era_0_trained_error=0.0))
    disagree = fit.verdicts(records=_two_fit_records(era_0_trained_error=0.5))

    assert {v.label: v.reading for v in agree}["P2"] == "interchangeable"
    # The era-0-trained fit finds a large difference, which the all-rows fit does not, so the
    # planned read is unresolved.
    assert {v.label: v.reading for v in disagree}["P2"] == "unresolved"
    sensitivity = next(v for v in disagree if v.label == "P2").sensitivity
    assert len(sensitivity) == 2
    assert {r["domain"] for r in sensitivity} == {"solar_era0"}
    assert all(r["planned"] for r in sensitivity)


def test_the_era_0_sensitivity_is_read_on_all_its_rows_and_at_each_lead_only():
    losses = _contrast_losses().filter(pl.col("month") < "2026-02")

    scopes = [name for name, _ in fit.scopes_of(losses=losses, domain="solar_era0")]

    assert scopes == ["all", *(f"lead {lead} only" for lead in range(6))]


def _scope_height(*, losses: pl.DataFrame, name: str) -> int:
    condition = dict(fit.scopes_of(losses=losses, domain="solar"))[name]
    assert condition is not None
    return losses.filter(condition).height


def test_a_sun_above_5_degrees_scope_reads_the_geometry_when_the_zenith_is_there():
    losses = _contrast_losses().with_columns(solar_zenith_deg=pl.lit(80.0))

    assert _scope_height(losses=losses, name="sun above 5 degrees") == losses.height
    assert _scope_height(losses=losses, name="era 0, sun above 5 degrees") == (
        _scope_height(losses=losses, name="era 0")
    )
    low_sun = losses.with_columns(solar_zenith_deg=pl.lit(88.0))
    assert _scope_height(losses=low_sun, name="sun above 5 degrees") == 0
    assert "sun above 5 degrees" not in dict(fit.scopes_of(losses=_contrast_losses()))


def _stamped_folder(tmp_path: Path, *, guards_passed: bool = True) -> Path:
    stamp: dict[str, object] = {"guards_passed": guards_passed, "row_file_hashes": {}}
    for name in fit.ROW_NAMES.values():
        (tmp_path / name).write_bytes(name.encode())
        stamp["row_file_hashes"][name] = hashlib.sha256(name.encode()).hexdigest()
    (tmp_path / build.STAMP_NAME).write_text(json.dumps(stamp))
    (tmp_path / "direct_report.md").write_text("report")
    return tmp_path


def test_the_verified_gate_passes_on_a_stamped_folder_whose_guards_passed(tmp_path: Path):
    fit.check_verified(directory=_stamped_folder(tmp_path))


def test_the_verified_gate_refuses_a_stamp_where_a_guard_failed(tmp_path: Path):
    folder = _stamped_folder(tmp_path, guards_passed=False)

    with pytest.raises(ValueError, match="guard"):
        fit.check_verified(directory=folder)


def test_the_verified_gate_refuses_a_row_file_that_no_longer_hashes_to_its_stamp(tmp_path: Path):
    folder = _stamped_folder(tmp_path)
    (folder / build.SOLAR_ERA_0_ROWS_NAME).write_bytes(b"edited after the build")

    with pytest.raises(ValueError, match="hash"):
        fit.check_verified(directory=folder)


def test_the_verified_gate_refuses_without_the_compare_report(tmp_path: Path):
    folder = _stamped_folder(tmp_path)
    (folder / "direct_report.md").unlink()

    with pytest.raises(ValueError, match="compare"):
        fit.check_verified(directory=folder)


def test_a_stale_output_of_any_domain_is_found_before_the_run_writes_anything(tmp_path: Path):
    expected = {path.name for path in fit.planned_outputs(directory=tmp_path)}

    for domain in fit.ROW_NAMES:
        assert f"losses_{domain}.parquet" in expected
        assert f"losses_{domain}.fingerprint" in expected
    assert {
        fit.STAMP_NAME,
        f"{fit.CPU_REFIT_STEM}.parquet",
        fit.REPORT_NAME,
        fit.DECISION_NAME,
        fit.INTERVALS_NAME,
    } <= expected


def test_the_rows_must_be_sorted_by_site_and_time_with_no_repeated_key():
    times = [datetime(2025, 1, 1, h, tzinfo=UTC) for h in range(3)]
    good = pl.DataFrame({"site": ["A", "A", "B"], "time": [times[0], times[1], times[0]]})

    build.check_sorted_and_unique(frame=good, name="rows")
    with pytest.raises(ValueError, match="not sorted"):
        build.check_sorted_and_unique(frame=good.reverse(), name="rows")
    with pytest.raises(ValueError, match="repeated"):
        build.check_sorted_and_unique(
            frame=pl.concat([good, good.head(1)]).sort("site", "time"), name="rows"
        )


# --- the temperature offset, the era-0 rows, and the elevation ratios -----------------------------


def _offset_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    months = ["2025-03", "2025-04", "2025-05"]
    frame = pl.DataFrame(
        {
            "site": "A",
            "month": months,
            "fold": [0, 1, 2],
            "om_temp": [20.0, 20.0, 20.0],
        }
    )
    # Open-Meteo minus CEDA is 1, 2, and 3 K at lead-0 instants of the three months, and 50 K at
    # an instant that is not lead 0, which the offset must ignore.
    model_free = pl.DataFrame(
        {
            "site": "A",
            "month": [*months, "2025-03"],
            "lead_hours": [0, 0, 0, 3],
            "om_temp_c": [11.0, 12.0, 13.0, 60.0],
            "ceda_temp_c": [10.0, 10.0, 10.0, 10.0],
        }
    )
    return frame, model_free


def test_the_temperature_offset_is_learned_on_the_other_folds_at_lead_zero_instants():
    frame, model_free = _offset_inputs()

    result = build.with_offset_removed_temperature(frame=frame, model_free=model_free)

    # Fold 0 learns from months 2 and 3 (offsets 2 and 3), fold 1 from months 1 and 3, fold 2 from
    # months 1 and 2, so a row's own month never enters its offset.
    assert result["om_temp_offset_removed"].to_list() == pytest.approx(
        [20.0 - 2.5, 20.0 - 2.0, 20.0 - 1.5]
    )


def _solar_for_era_0() -> pl.DataFrame:
    rows = []
    for month in build.STUDY_MONTHS:
        year, number = (int(part) for part in month.split("-"))
        rows += [
            {
                "site": "A",
                "time": datetime(year, number, day, 12, tzinfo=UTC),
                "month": month,
                "om_temp": 10.0,
                "fold": 4,
                "era": "x",
                "era_code": 1 if month >= "2026-02" else 0,
                "om_temp_offset_removed": 0.0,
            }
            for day in range(1, 5)
        ]
    return pl.DataFrame(rows)


def test_the_era_0_rows_keep_only_the_months_before_the_upgrade_with_their_own_folds():
    solar = _solar_for_era_0()
    months = [m for m in build.STUDY_MONTHS if m < "2026-02"]
    model_free = pl.DataFrame(
        {
            "site": "A",
            "month": months,
            "lead_hours": 0,
            "om_temp_c": 10.0,
            "ceda_temp_c": 9.0,
        }
    )

    rows = build.era_0_solar_rows(solar=solar, model_free=model_free)

    assert sorted(rows["month"].unique().to_list()) == months
    assert rows["era_code"].unique().to_list() == [0]
    assert sorted(rows["fold"].unique().to_list()) == [0, 1, 2, 3, 4]
    assert rows["om_temp_offset_removed"].to_list() == pytest.approx([9.0] * rows.height)


def test_the_elevation_ratio_bins_the_sun_and_keeps_the_rows_without_a_light_cut():
    frame = pl.DataFrame(
        {
            "lead_hours": 0,
            "era_code": 0,
            "sun_elevation_deg": [1.0, 3.0, 7.0, 15.0, 30.0, 60.0, 60.0, 2.0],
            "om_ghi": [7.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 7.0],
            "ceda_ghi": [0.0, 7.0, 9.0, 10.0, 10.0, 10.0, 50.0, 0.0],
        }
    )

    table = build.irradiance_ratios_by_elevation(frame=frame)

    assert table["bin"].to_list() == [
        "up to 2 degrees",
        "2 to 5 degrees",
        "5 to 10 degrees",
        "10 to 20 degrees",
        "20 to 40 degrees",
        "above 40 degrees",
    ]
    assert table["rebuilt_ratio"].to_list() == pytest.approx([0.0, 0.7, 0.9, 1.0, 1.0, 1.0])
    # The zero-irradiance row is left out, and a row exactly at an edge belongs to the lower bin.
    assert table["n"].to_list() == [2, 1, 1, 1, 1, 1]


# --- the wind step spans, the speed rescale, and the by-lead and signed-error tables --------------


def test_the_step_spans_include_both_end_days_and_nothing_beside_them():
    times = [
        datetime(2024, 11, 6, 23, tzinfo=UTC),
        datetime(2024, 11, 7, 0, tzinfo=UTC),
        datetime(2024, 11, 30, 23, tzinfo=UTC),
        datetime(2024, 12, 1, 0, tzinfo=UTC),
        datetime(2025, 1, 16, 0, tzinfo=UTC),
        datetime(2025, 2, 18, 23, tzinfo=UTC),
        datetime(2025, 2, 19, 0, tzinfo=UTC),
    ]

    flags = pl.DataFrame({"time": times}).select(step=build.in_wind_step_days())

    assert flags["step"].to_list() == [False, True, True, False, True, True, False]


def test_a_month_losing_over_a_quarter_of_its_rows_to_the_step_spans_is_dropped_whole():
    days = [datetime(2024, 11, day, 12, tzinfo=UTC) for day in range(1, 31)]
    december = [datetime(2024, 12, day, 12, tzinfo=UTC) for day in (1, 8, 15, 22)]
    base = pl.DataFrame({"time": [*days, *december], "month": ["2024-11"] * 30 + ["2024-12"] * 4})

    months = build.wind_step_months(base=base)

    # 24 of November's 30 days lie in the first span, and none of December's rows do.
    assert months == {"2024-11": pytest.approx(24 / 30)}


def _rescale_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    months = ["2025-03", "2025-04", "2025-05"]
    frame = pl.DataFrame(
        {"site": "W1", "month": months, "fold": [0, 1, 2], "om_speed_10m": [10.0, 10.0, 10.0]}
    )
    # CEDA over Open-Meteo is 1.1, 1.2, and 1.3 at lead-0 instants of the three months. A lead-3
    # instant at 9.0 and a calm instant must not count.
    model_free = pl.DataFrame(
        {
            "site": "W1",
            "month": [*months, "2025-03", "2025-04", "2025-03"],
            "lead_hours": [0, 0, 0, 3, 0, 0],
            "om_wind_step": False,
            "ceda_speed_10m_m_s": [11.0, 12.0, 13.0, 90.0, 0.5, 11.0],
            "om_speed_10m_m_s": [10.0, 10.0, 10.0, 10.0, 0.1, 10.0],
        }
    )
    return frame, model_free


def test_the_speed_rescale_uses_the_median_lead_zero_ratio_of_the_other_folds_only():
    frame, model_free = _rescale_inputs()

    result = build.with_rescaled_speed(frame=frame, model_free=model_free)

    # Fold 0 learns from months 2 and 3 (ratios 1.2 and 1.3, median 1.25), fold 1 from months 1 and
    # 3 (1.1 twice in month 1 and 1.3 in month 3, median 1.1), and fold 2 from months 1 and 2 (1.1
    # twice and 1.2, median 1.1). A row's own month
    # never enters its scale, and the lead-3 and calm instants are left out.
    assert result["om_speed_10m_rescaled"].to_list() == pytest.approx([12.5, 11.0, 11.0])


def test_the_speed_rescale_ignores_instants_inside_a_step_span():
    frame, model_free = _rescale_inputs()
    stepped = model_free.with_columns(om_wind_step=pl.col("ceda_speed_10m_m_s") == 13.0)

    result = build.with_rescaled_speed(frame=frame, model_free=stepped)

    # Month 3's instant is in a span, so fold 0 learns from month 2 alone and fold 1 from month 1.
    assert result["om_speed_10m_rescaled"].to_list()[:2] == pytest.approx([12.0, 11.0])


def test_the_settings_are_named_primary_and_second():
    assert (fit.PRIMARY_SETTING, fit.SECOND_SETTING) == ("primary", "second")


def test_every_ceda_lead_has_a_scope_over_all_rows_and_over_era_0():
    losses = pl.DataFrame(
        {
            "time": [datetime(2025, 6, 1, h, tzinfo=UTC) for h in range(6)]
            + [datetime(2026, 3, 1, h, tzinfo=UTC) for h in range(6)],
            "month": ["2025-06"] * 6 + ["2026-03"] * 6,
            "site": "A",
        }
    )

    scopes = dict(fit.scopes_of(losses=losses))

    for lead in range(6):
        everywhere = scopes[f"lead {lead} only"]
        era_0 = scopes[f"era 0, lead {lead} only"]
        assert everywhere is not None
        assert era_0 is not None
        assert losses.filter(everywhere)["time"].dt.hour().to_list() == [lead, lead]
        assert losses.filter(era_0).height == 1
    without = scopes["without 2025-01"]
    assert without is not None
    assert losses.filter(without).height == 12
    january = losses.with_columns(month=pl.lit("2025-01"))
    january_scope = dict(fit.scopes_of(losses=january))["without 2025-01"]
    assert january_scope is not None
    assert january.filter(january_scope).is_empty()


def test_the_mean_signed_error_is_prediction_minus_measured_over_capacity():
    losses = pl.DataFrame(
        {
            "arm": "a",
            "setting": fit.PRIMARY_SETTING,
            "signed_error_capped_mw": [-1.0, -3.0],
            "effective_capacity_mw": [10.0, 20.0],
        }
    )

    value = fit.mean_signed_error_pp(losses=losses, arm="a", setting=fit.PRIMARY_SETTING)

    # (-0.1 + -0.15) / 2 = -0.125 of capacity, so an under-prediction of 12.5 points.
    assert value == pytest.approx(-12.5)


def test_the_planned_scope_losses_cut_the_solar_fit_on_both_eras_to_era_0_and_leave_the_rest():
    losses = _contrast_losses()

    solar = fit.planned_scope_losses(domain="solar", losses=losses)
    era_0 = fit.planned_scope_losses(domain="solar_era0", losses=losses)
    wind = fit.planned_scope_losses(domain="wind", losses=losses)

    assert max(solar["month"].to_list()) < "2026-02"
    assert era_0.height == losses.height
    assert wind.height == losses.height


def test_an_unresolved_verdict_prints_its_bound_and_the_default_wording():
    # The transfer penalty is a clear penalty at the primary setting and none at the second, so the
    # two settings disagree and the verdict is unresolved.
    losses = _contrast_losses().with_columns(
        pl.when(
            (pl.col("arm") == "ceda_ghi_temp_scored_on_om")
            & (pl.col("setting") == fit.PRIMARY_SETTING)
        )
        .then(pl.col(fit.METRIC) + 0.01)
        .otherwise(pl.col(fit.METRIC))
        .alias(fit.METRIC)
    )
    records = fit.domain_records(domain="solar", losses=losses)
    records += fit.domain_records(
        domain="solar_era0", losses=losses.filter(pl.col("month") < "2026-02")
    )
    found = [v for v in fit.verdicts(records=records) if v.label == "P3"]

    text = fit.decision_text(found=found)

    assert found[0].reading == "unresolved"
    assert "default for an unresolved reading" in text
    assert "The largest upper bound" in text
    assert fit.verdict_bound(verdict=found[0]) == pytest.approx(1.0)
    penalty = fit.decision_text(found=[v for v in fit.verdicts(records=records) if v.label == "P2"])
    assert "The largest upper bound" not in penalty


def test_the_scoring_frame_of_the_wind_calibrator_swaps_in_the_rescaled_speed():
    rows = _wind_rows_for()

    frames = fit.scoring_frames(site_rows=rows, domain="wind")

    speed = build.wind_columns(archive="ceda")[0]
    assert frames["om_speed_rescaled"][speed].to_list() == rows["om_speed_10m_rescaled"].to_list()
    # The direction columns stay Open-Meteo's.
    assert frames["om_speed_rescaled"][build.wind_columns(archive="ceda")[1]].to_list() == (
        rows["om_sin_10m"].to_list()
    )


def test_a_split_with_no_interval_prints_a_label_not_nan():
    assert compare._interval(lower=float("nan"), upper=float("nan"), sign=True) == "(no interval)"
    assert compare._interval(lower=-0.1, upper=0.25, sign=True) == "[-0.100, +0.250]"


def test_wind_speed_excludes_the_step_spans_and_the_step_variable_keeps_only_them():
    frame = pl.DataFrame(
        {
            "ceda_speed_10m_m_s": [5.0, 5.0, 5.0],
            "om_speed_10m_m_s": [5.0, 5.2, 4.8],
            "om_wind_step": [False, True, False],
        }
    )
    speed = next(v for v in compare.VARIABLES if v.name == "10 m wind speed")
    inside = next(v for v in compare.VARIABLES if "step spans" in v.name)

    assert compare.with_difference(frame=frame, variable=speed).height == 2
    assert compare.with_difference(frame=frame, variable=inside)["difference"].to_list() == (
        pytest.approx([-0.2])
    )


def test_wind_direction_leaves_out_calm_hours():
    frame = pl.DataFrame(
        {
            "ceda_speed_10m_m_s": [1.0, 3.0],
            "ceda_direction_10m_deg": [10.0, 10.0],
            "om_direction_10m_deg": [200.0, 12.0],
            "om_wind_step": False,
        }
    )
    direction = next(v for v in compare.VARIABLES if v.name.startswith("10 m wind direction"))

    rows = compare.with_difference(frame=frame, variable=direction)

    assert rows["difference"].to_list() == [-2.0]


def test_the_lead_zero_levels_are_the_means_both_archives_hold_at_lead_zero():
    frame = pl.DataFrame(
        {
            "lead_hours": [0, 0, 3],
            "ceda_temp_c": [10.0, 12.0, 99.0],
            "om_temp_c": [9.0, 11.0, 0.0],
            "om_wind_step": False,
        }
    )

    levels = compare.lead_zero_levels(frame=frame)

    assert levels["air temperature"] == pytest.approx((11.0, 10.0))


def test_the_by_lead_table_prints_each_lead_of_a_contrast_and_a_dash_where_one_is_missing():
    losses = _contrast_losses().with_columns(
        time=pl.col("time").dt.replace(hour=15)  # CEDA lead 3 only
    )
    records = fit.domain_records(domain="solar", losses=losses)

    lines = fit.by_lead_lines(records=records)

    row = next(line for line in lines if line.startswith("| P2, ceda_ghi_temp | solar | primary"))
    cells = [cell.strip() for cell in row.strip("|").split("|")[3:]]
    assert len(cells) == 6
    assert cells[3].startswith("+0.000 [")
    assert [cells[i] for i in (0, 1, 2, 4, 5)] == ["-"] * 5


def test_the_hours_outside_a_span_of_a_dropped_month_are_dropped_with_the_month():
    base = _wind_base()
    # 2025-02-20 lies after the second span ends, but the span empties 2025-02 (days 1 to 8 of the
    # base are inside it), so the month goes whole, and a row on day 20 goes with it.
    extra = base.filter(pl.col("time") == datetime(2025, 2, 8, 12, tzinfo=UTC)).with_columns(
        time=pl.lit(datetime(2025, 2, 20, 12, tzinfo=UTC))
    )
    with_day_20 = pl.concat([base, extra]).sort("site", "time")

    rows = build.wind_rows(
        base=with_day_20,
        open_meteo=_open_meteo_for(base=with_day_20),
        model_free=_wind_model_free(base=with_day_20),
    )

    assert "2025-02" not in rows["month"].to_list()
    assert datetime(2025, 2, 20, 12, tzinfo=UTC) not in rows["time"].to_list()


# --- the wind-step evidence ---------------------------------------------------------------------


def _step_frame() -> pl.DataFrame:
    rows = []
    for day, step, ratio in (
        (datetime(2024, 10, 20, 12, tzinfo=UTC), False, 0.97),
        (datetime(2024, 11, 10, 12, tzinfo=UTC), True, 1.06),
        (datetime(2024, 11, 11, 12, tzinfo=UTC), True, 1.08),
    ):
        rows.append(
            {
                "site": "A",
                "time": day,
                "month": day.strftime("%Y-%m"),
                "lead_hours": 0,
                "om_wind_step": step,
                "ceda_speed_10m_m_s": 5.0,
                "om_speed_10m_m_s": 5.0 * ratio,
                "om_temp_c": 10.0,
                "ceda_temp_c": 10.0,
                "om_direction_10m_deg": 100.0,
                "ceda_direction_10m_deg": 99.0,
                "ceda_ghi": 100.0,
                "om_ghi": 100.0,
                "sun_elevation_deg": 30.0,
            }
        )
    rows.append({**rows[0], "ceda_speed_10m_m_s": 0.5, "om_speed_10m_m_s": 50.0})
    return pl.DataFrame(rows)


def test_the_month_ratio_is_the_median_lead_zero_speed_ratio_and_leaves_out_calm_hours():
    ratios = steps.month_ratios(frame=_step_frame())

    by_month = dict(zip(ratios["month"].to_list(), ratios["ratio"].to_list(), strict=True))
    # The calm hour's ratio of 100 would pull October's median to 1.0 or more if it were kept.
    assert by_month["2024-10"] == pytest.approx(0.97)
    assert by_month["2024-11"] == pytest.approx(1.07)


def test_the_span_table_separates_the_hours_inside_and_outside_the_spans():
    table = steps.span_ratios(frame=_step_frame())

    inside = table.filter(pl.col("om_wind_step")).row(0, named=True)
    outside = table.filter(~pl.col("om_wind_step")).row(0, named=True)
    assert inside["speed_ratio"] == pytest.approx(1.07)
    assert outside["speed_ratio"] == pytest.approx(0.97)
    assert inside["direction_difference_deg"] == pytest.approx(1.0)


def test_the_daily_ratio_marks_the_days_of_the_spans():
    daily = steps.daily_ratio(frame=_step_frame())

    flags = dict(zip(daily["day"].to_list(), daily["in_span"].to_list(), strict=True))
    assert flags[datetime(2024, 10, 20).date()] is False
    assert flags[datetime(2024, 11, 10).date()] is True


def test_the_day_flag_includes_the_first_and_last_day_of_each_span():
    days = [datetime(2024, 11, d).date() for d in (6, 7, 30)] + [datetime(2024, 12, 1).date()]

    flags = pl.DataFrame({"day": days}).select(flag=steps.in_wind_step_days_of_day())

    assert flags["flag"].to_list() == [False, True, True, False]


# --- extra reads: the first run and the UTC hours -------------------------------------------------


def test_the_first_runs_wind_rows_carry_the_current_setting_names(tmp_path: Path):
    def record(*, label: str, scope: str, setting: str, domain: str = "wind") -> dict[str, object]:
        return {
            "domain": domain,
            "label": label,
            "scope": scope,
            "setting": setting,
            "difference_pp": 0.162,
            "lower_95_pp": 0.08,
            "upper_95_pp": 0.241,
            "reading": "differ",
            "n_months": 23,
        }

    pl.DataFrame(
        [
            record(label="P1", scope="all", setting="sensitivity"),
            record(label="P1", scope="all", setting="pooled"),
            record(label="P1", scope="era 0", setting="pooled"),
            record(label="P2", scope="all", setting="pooled"),
            record(label="P1", scope="all", setting="pooled", domain="solar"),
        ]
    ).write_parquet(tmp_path / "intervals.parquet")

    lines = extra_reads.first_run_lines(first_run_dir=tmp_path)

    rows = [line for line in lines if line.startswith("| P")]
    assert [row.split(" | ")[:3] for row in rows] == [
        ["| P1", "all", "primary"],
        ["| P1", "all", "second"],
    ]


def test_the_by_hour_read_gives_each_utc_hour_its_run_and_lead():
    generator = np.random.default_rng(3)
    records = [
        {
            "arm": arm,
            "site": "A",
            "time": datetime(2025, month, 1, hour, tzinfo=UTC),
            "seed": seed,
            "month": f"2025-{month:02d}",
            "setting": "primary",
            extra_reads.METRIC: float(generator.uniform()),
        }
        for month in range(1, 8)
        for hour in range(24)
        for seed in (0, 1)
        for arm in ("ceda_wind_10m", "ceda_wind_10m_scored_on_om", "om_wind_10m")
    ]
    lines = extra_reads.by_hour_lines(losses=pl.DataFrame(records))

    rows = [line.split(" | ") for line in lines if line.startswith("| ") and "UTC |" in line]
    assert len(rows) == 24
    assert rows[17][:3] == ["| 17", "12 UTC", "5"]
    assert rows[6][:3] == ["| 06", "06 UTC", "0"]
