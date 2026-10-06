"""Tests for the pure functions of the four `ukv_ceda_vs_openmeteo_*` scripts of study #1051.

Every frame is synthetic, so no test needs `data/`. Each test is built to fail on the bug it exists
for: a wind speed left in km/h, an irradiance snapshot left unscaled by the zenith-cosine ratio, a
transfer frame that copies CEDA's values instead of Open-Meteo's, one shared permutation for both
archives' control columns, a temperature taken at one instant instead of two, a contrast read
against the wrong margin, and a one-sided penalty read two-sided.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pytest
import ukv_ceda_vs_openmeteo_build as build
import ukv_ceda_vs_openmeteo_compare as compare
import ukv_ceda_vs_openmeteo_fit as fit

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


def test_each_archives_control_columns_get_their_own_permutation():
    base = _wind_base()
    # Identical values in both archives: a shared permutation would shuffle them identically, and
    # the control would then differ by zero by construction.
    open_meteo = _open_meteo_for(base=base)

    rows = build.wind_rows(base=base, open_meteo=open_meteo)

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

    rows = build.wind_rows(base=base, open_meteo=open_meteo)

    assert rows.height == base.height - 1
    assert base["time"][0] not in rows["time"].to_list()
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
    assert fit.MARGINS_PP == {"wind": 0.16, "solar": 0.06}


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


def test_the_dry_run_counts_the_54_planned_fit_sets():
    frames: dict[fit.DomainType, pl.DataFrame] = {
        "wind": pl.DataFrame({"site": ["W1", "W2", "W3"] * 2, "x": range(6)}),
        "solar": pl.DataFrame({"site": list("ABCDEF") * 2, "x": range(12)}),
    }

    jobs = {domain: len(fit.all_jobs(domain=domain)) for domain in ("wind", "solar")}

    assert jobs["wind"] * 3 + jobs["solar"] * 6 == 54
    assert "Total 54 fit-sets" in "\n".join(fit.dry_run_lines(frames=frames))


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
        }
    )


def test_the_lead_zero_temperature_offset_reads_only_the_hours_at_lead_zero():
    # Open-Meteo is 1 K warmer at the lead-0 hours and 5 K warmer elsewhere.
    assert fit.lead_zero_temperature_offset(site_rows=_solar_site_rows()) == pytest.approx(1.0)


def test_each_partial_swap_replaces_only_its_own_columns_of_the_ceda_arm():
    site_rows = _solar_site_rows()

    frames = fit.scoring_frames(site_rows=site_rows, domain="solar")

    assert frames["ceda"]["ceda_ghi"].unique().to_list() == [100.0]
    assert frames["om"]["ceda_ghi"].unique().to_list() == [200.0]
    assert frames["om_temp"]["ceda_ghi"].unique().to_list() == [100.0]
    assert frames["om_temp"]["ceda_temp"].to_list() == site_rows["om_temp"].to_list()
    assert frames["om_ghi"]["ceda_temp"].unique().to_list() == [10.0]
    assert frames["om_ghi"]["ceda_ghi"].unique().to_list() == [200.0]
    # The offset removed is the lead-0 mean offset of 1 K, subtracted from every hour.
    assert frames["om_temp_offset_removed"]["ceda_temp"].to_list() == pytest.approx(
        [value - 1.0 for value in site_rows["om_temp"].to_list()]
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
    }
    own = losses.filter(pl.col("arm") == "ceda_wind_10m")
    swapped = losses.filter(pl.col("arm") == "ceda_wind_10m_scored_on_om")
    assert own.height == swapped.height == rows.height * 3 * 2
    assert set(own["setting"].unique().to_list()) == {fit.PRIMARY_SETTING, fit.SECOND_SETTING}
    assert swapped["absolute_error_mw"].mean() != own["absolute_error_mw"].mean()
    # The direction swap changes columns the model barely uses here, so it scores like the own frame
    # far more closely than the full swap does.
    partial = losses.filter(pl.col("arm") == "ceda_wind_10m_scored_on_om_direction")
    assert abs(partial["absolute_error_mw"].mean() - own["absolute_error_mw"].mean()) < abs(
        swapped["absolute_error_mw"].mean() - own["absolute_error_mw"].mean()
    )
