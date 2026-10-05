import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_STUDIES_DIR = Path(__file__).resolve().parents[3] / "studies"
sys.path.insert(0, str(_STUDIES_DIR / "ukv_ceda_blends"))
sys.path.insert(0, str(_STUDIES_DIR / "nwp_forecast_comparison"))
sys.path.insert(0, str(_STUDIES_DIR / "beam_diffuse_split"))
sys.path.insert(0, str(_STUDIES_DIR / "weather_downloads"))

import build_ukv_ceda_inputs as build  # noqa: E402
import check_arm_columns_unchanged as unchanged  # noqa: E402
import verify_ukv_ceda_inputs as verify  # noqa: E402
from nwp_forecast_comparison import DomainType  # noqa: E402
from studies.baselines import haurwitz_w_m2  # noqa: E402

NAN = float("nan")


# --- the independent recomputation ----------------------------------------------------------------


@pytest.mark.parametrize("zenith", [0.0, 20.0, 60.0, 85.0, 89.9, 90.0, 120.0])
def test_the_plain_haurwitz_agrees_with_the_array_version(zenith: float):
    expected = float(haurwitz_w_m2(apparent_zenith_deg=np.array([zenith]))[0])

    assert verify.haurwitz(zenith_deg=zenith) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("domain", "label", "day", "init", "lead"),
    [
        ("wind", datetime(2026, 3, 10, 0), 2, datetime(2026, 3, 8, 3), 45),
        ("wind", datetime(2026, 3, 10, 2), 2, datetime(2026, 3, 8, 3), 47),
        ("wind", datetime(2026, 3, 10, 23), 4, datetime(2026, 3, 6, 3), 116),
        ("solar", datetime(2026, 3, 10, 1), 1, datetime(2026, 3, 9, 3), 22),
        ("solar", datetime(2026, 3, 11, 0), 4, datetime(2026, 3, 6, 3), 117),
    ],
)
def test_the_row_to_run_rule_is_restated_independently_of_the_build(
    domain: DomainType, label: datetime, day: int, init: datetime, lead: int
):
    got_init, slot, got_lead = verify.run_of_row(
        time=label.replace(tzinfo=UTC),
        day=day,
        domain=domain,
    )

    assert got_init == init.replace(tzinfo=UTC)
    assert got_lead == lead
    assert build.slot_init_time(slot=slot) == got_init


def test_the_row_to_run_rule_agrees_with_the_builds_own_for_every_hour_of_a_day():
    hours = [datetime(2026, 3, 10, hour, tzinfo=UTC) for hour in range(24)]
    for domain in ("solar", "wind"):
        built = build.with_run(
            frame=pl.DataFrame({"site": ["A"] * 24, "time": hours}), day=3, domain=domain
        )
        for row in built.iter_rows(named=True):
            init, slot, lead = verify.run_of_row(time=row["time"], day=3, domain=domain)
            assert (init, slot, lead) == (row["init_time"], row["slot"], row["lead_hours"])


def test_native_leads_and_the_anchors_bracketing_a_rebuilt_lead():
    assert all(verify.is_native(lead=lead) for lead in (48, 51, 120))
    assert not any(verify.is_native(lead=lead) for lead in (49, 118))
    assert verify.bracket(lead=49) == (48, 51)
    assert verify.bracket(lead=52) == (51, 54)
    assert verify.bracket(lead=119) == (117, 120)


def test_a_rebuilt_radiation_is_the_clear_sky_index_between_its_anchors_times_its_own_clear_sky():
    init = datetime(2026, 6, 21, 3, tzinfo=UTC)
    latitude, longitude = 52.0, -1.0
    series = [NAN] * 121
    fraction = {54: 0.4, 57: 0.6}
    for anchor, share in fraction.items():
        series[anchor] = share * verify.clear_sky_at(
            init=init, lead=anchor, latitude=latitude, longitude=longitude
        )

    value = verify.radiation_snapshot(
        series=series, init=init, lead=55, latitude=latitude, longitude=longitude
    )

    clear = verify.clear_sky_at(init=init, lead=55, latitude=latitude, longitude=longitude)
    assert clear > verify.DAYLIGHT_FLOOR_W_M2
    assert value == pytest.approx((2 / 3 * 0.4 + 1 / 3 * 0.6) * clear)


def test_a_native_lead_is_returned_as_stored_and_a_night_anchor_is_not_recomputed():
    init = datetime(2026, 6, 21, 3, tzinfo=UTC)
    series = [100.0] * 121

    assert (
        verify.radiation_snapshot(series=series, init=init, lead=30, latitude=52.0, longitude=-1.0)
        == 100.0
    )
    # Lead 64 sits between the anchors at 63 and 66 (18 and 21 UTC on 2026-06-23): the sun has set
    # at the second anchor, so the clear-sky index there is not defined.
    assert (
        verify.radiation_snapshot(series=series, init=init, lead=64, latitude=52.0, longitude=-1.0)
        is None
    )


def test_a_wind_rebuilt_through_north_turns_the_short_way_and_loses_speed():
    speed = [10.0] * 121
    direction = [350.0 if lead <= 51 else 10.0 for lead in range(121)]

    got = verify.wind_at(speed=speed, direction=direction, lead=52)

    assert got is not None
    assert min(got[1], 360.0 - got[1]) < 10.0
    assert 9.7 < got[0] < 10.0
    assert verify.wind_at(speed=speed, direction=direction, lead=30) == (10.0, 350.0)


def test_a_wind_with_a_missing_anchor_cannot_be_recomputed():
    speed = [10.0] * 121
    speed[54] = NAN

    assert verify.wind_at(speed=speed, direction=[0.0] * 121, lead=52) is None


def test_a_built_value_and_its_recomputation_agree_only_within_the_tolerances():
    assert verify.agree(built=1.0, recomputed=1.0 + 1e-9)
    assert not verify.agree(built=1.0, recomputed=1.001)
    assert verify.agree(built=0.0, recomputed=1e-9)


def test_strata_split_at_the_end_of_the_hourly_leads_and_at_the_first_rebuilt_leads():
    assert verify.stratum_of(lead=30) == "native"
    assert verify.stratum_of(lead=48) == "native"
    assert verify.stratum_of(lead=49) == "first rebuilt"
    assert verify.stratum_of(lead=54) == "first rebuilt"
    assert verify.stratum_of(lead=57) == "late"


def _built(*, domain: DomainType) -> pl.DataFrame:
    times = [datetime(2026, 3, day, hour, tzinfo=UTC) for day in range(5, 25) for hour in range(24)]
    fields = build.WEATHER_FIELDS[domain]
    return pl.DataFrame({"site": ["A"] * len(times), "time": times}).with_columns(
        **{f"ukv_ceda_day{day}_{field}": pl.lit(1.0) for day in (1, 2, 3, 4) for field in fields}
    )


def test_the_sample_covers_every_stratum_and_for_wind_the_early_hours():
    strata = verify.sample_rows(built=_built(domain="wind"), domain="wind", day=2)

    assert set(strata) == {"native", "first rebuilt", "late", "wind hours 0 to 2"}
    assert all(row["time"].hour <= 2 for row in strata["wind hours 0 to 2"])
    for name, rows in strata.items():
        assert rows, name
    assert "wind hours 0 to 2" not in verify.sample_rows(
        built=_built(domain="solar"), domain="solar", day=2
    )


def test_the_sample_is_the_same_on_every_run():
    first = verify.sample_rows(built=_built(domain="wind"), domain="wind", day=3)
    again = verify.sample_rows(built=_built(domain="wind"), domain="wind", day=3)

    assert first == again


# --- the gates ------------------------------------------------------------------------------------


def test_the_skill_check_fails_when_day_1_is_far_below_open_meteo_or_the_correlation_rises():
    assert (
        verify.skill_gate(
            ukv_ceda=[0.90, 0.85, 0.80, 0.70], open_meteo_day1=0.91, day1_ceda_same_rows=0.89
        )
        == []
    )
    low = verify.skill_gate(
        ukv_ceda=[0.80, 0.75, 0.70, 0.60], open_meteo_day1=0.91, day1_ceda_same_rows=0.80
    )
    assert len(low) == 1
    assert "below Open-Meteo UKV" in low[0]
    rising = verify.skill_gate(
        ukv_ceda=[0.90, 0.85, 0.86, 0.70], open_meteo_day1=0.91, day1_ceda_same_rows=0.90
    )
    assert len(rising) == 1
    assert "rises with the lead day" in rising[0]


def test_each_lead_days_correlation_is_taken_on_the_rows_all_four_days_hold():
    truth = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Day 1 alone also holds a last row that is perfectly aligned; the shared rows are noisy, and
    # the same for every day.
    noisy = [1.0, 3.0, 2.0, 5.0, 4.0, None]
    joined = pl.DataFrame(
        {
            "ghi_cams": truth,
            "ukv_day1_ghi": truth,
            "ukv_ceda_day1_ghi": [*noisy[:5], 6.0],
            "ukv_ceda_day2_ghi": noisy,
            "ukv_ceda_day3_ghi": noisy,
            "ukv_ceda_day4_ghi": noisy,
        }
    )

    lines, _ = verify.skill_lines(domain="solar", joined=joined)

    shared_correlation = verify.correlation(first=pl.Series(noisy[:5]), second=pl.Series(truth[:5]))
    assert f"[{round(shared_correlation, 3)}, {round(shared_correlation, 3)}" in lines[0]


def test_the_radiation_alignment_gate_is_the_median_over_generators_in_two_parts():
    assert verify.alignment_gate(raw_peaks=[10, 10, 15, 10, 10, 10], rebuilt_peaks=[-20] * 6) == []
    # One generator off does not fail a median.
    assert verify.alignment_gate(raw_peaks=[10, 10, 40, 10, 10, 10], rebuilt_peaks=[-20] * 6) == []
    # The raw snapshots are an instant: a median 20 minutes from 0 is a lead or slot error.
    raw = verify.alignment_gate(raw_peaks=[20] * 6, rebuilt_peaks=[-10] * 6)
    assert len(raw) == 1
    assert "raw day-1 snapshots" in raw[0]
    # A rebuilt column that peaks with the snapshots is not averaging L - 1 and L.
    mean = verify.alignment_gate(raw_peaks=[10] * 6, rebuilt_peaks=[10] * 6)
    assert len(mean) == 1
    assert "not averaging" in mean[0]
    # The rebuilt column 41 minutes before the raw snapshots is outside 30 plus or minus 10.
    assert len(verify.alignment_gate(raw_peaks=[10] * 6, rebuilt_peaks=[-31] * 6)) == 1
    assert verify.alignment_gate(raw_peaks=[10] * 6, rebuilt_peaks=[-30] * 6) == []


def test_the_verify_stamp_records_the_result_and_the_hash_of_every_inputs_file(tmp_path: Path):
    for domain in ("solar", "wind"):
        pl.DataFrame({"a": [1, 2]}).write_parquet(tmp_path / f"{domain}_ukv_ceda_inputs.parquet")

    verify.write_verify_stamp(output_dir=tmp_path, passed=True)

    stamp = json.loads((tmp_path / "verify.json").read_text())
    assert stamp["passed"] is True
    assert stamp["inputs_sha256"] == {
        domain: build.sha256_of(path=tmp_path / f"{domain}_ukv_ceda_inputs.parquet")
        for domain in ("solar", "wind")
    }
    verify.write_verify_stamp(output_dir=tmp_path, passed=False)
    assert json.loads((tmp_path / "verify.json").read_text())["passed"] is False


def test_a_correlation_that_stays_level_does_not_count_as_rising():
    assert verify.non_increasing(values=[0.9, 0.9, 0.8])
    assert not verify.non_increasing(values=[0.9, 0.91])


def test_the_correlation_ignores_rows_where_either_series_is_missing():
    first = pl.Series([1.0, 2.0, 3.0, None, 100.0, NAN])
    second = pl.Series([2.0, 4.0, 6.0, 9.0, None, 1.0])

    assert verify.correlation(first=first, second=second) == pytest.approx(1.0)


def test_a_month_whose_ratio_to_ens_jumps_by_fifteen_percent_is_flagged():
    months = [(2026, 3), (2026, 4), (2026, 5)]
    times = [datetime(y, m, 10, 12, tzinfo=UTC) for y, m in months]
    joined = pl.DataFrame(
        {
            "site": ["A"] * 3,
            "time": times,
            "ukv_ceda_day1_ghi": [100.0, 100.0, 130.0],
            "ens_mean_day1_ghi": [100.0, 100.0, 100.0],
        }
    )

    lines = verify.step_lines(joined=joined, domain="solar")

    assert lines == ["solar A 2026-05: ratio 1.000 -> 1.300"]
    quiet = joined.with_columns(ukv_ceda_day1_ghi=pl.lit(100.0))
    assert "no month-to-month change" in verify.step_lines(joined=quiet, domain="solar")[0]


# --- the unchanged-columns check ------------------------------------------------------------------


def _stamp(*, folder: Path, name: str, columns: dict[str, list[str]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps({"columns": json.dumps(columns)}))
    return path


def test_a_stamp_whose_columns_still_resolve_passes_and_one_that_differs_is_named(tmp_path: Path):
    import fit_aifs

    good = {
        "ukv_day1": list(fit_aifs.arm_features(arm="ukv_day1", domain="wind")),
        "blend_ukv_day1": list(fit_aifs.arm_features(arm="blend_ukv_day1", domain="wind")),
    }
    path = _stamp(
        folder=tmp_path / "nwp_forecast_comparison_x", name="wind_a_losses.json", columns=good
    )
    assert unchanged.check_stamp(path=path) == (2, [])

    bad = {**good, "ukv_day1": [*good["ukv_day1"][:-1], "ukv_day1_speed_925hpa"]}
    path = _stamp(
        folder=tmp_path / "nwp_forecast_comparison_y", name="wind_b_losses.json", columns=bad
    )
    compared, problems = unchanged.check_stamp(path=path)

    assert compared == 2
    assert problems == ["nwp_forecast_comparison_y/wind_b_losses.json: ukv_day1"]


def test_the_check_reads_every_earlier_studys_stamps_and_fails_on_none(tmp_path: Path):
    import fit_aifs

    columns = {"ens_mean_day1": list(fit_aifs.arm_features(arm="ens_mean_day1", domain="solar"))}
    _stamp(
        folder=tmp_path / "nwp_forecast_comparison_a", name="solar_s_losses.json", columns=columns
    )
    _stamp(
        folder=tmp_path / "nwp_forecast_comparison_b", name="solar_t_losses.json", columns=columns
    )
    _stamp(folder=tmp_path / "other_study", name="solar_u_losses.json", columns={"nope": []})

    stamps, arms, problems = unchanged.check_all(studies_dir=tmp_path)

    assert (stamps, arms, problems) == (2, 2, [])
    assert unchanged.check_all(studies_dir=tmp_path / "empty")[0] == 0


def test_a_stamp_must_name_its_technology_in_its_file_name(tmp_path: Path):
    assert unchanged.stamp_domain(path=tmp_path / "solar_single_day1_losses.json") == "solar"
    assert unchanged.stamp_domain(path=tmp_path / "wind_ens_losses.json") == "wind"
    with pytest.raises(ValueError, match="solar or wind"):
        unchanged.stamp_domain(path=tmp_path / "p4_losses.json")


# --- the post hoc older run -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "label", "day", "init", "lead"),
    [
        ("wind", datetime(2026, 3, 10, 12), 1, datetime(2026, 3, 8, 15), 45),
        ("wind", datetime(2026, 3, 10, 0), 3, datetime(2026, 3, 6, 15), 81),
        ("solar", datetime(2026, 3, 10, 13), 2, datetime(2026, 3, 7, 15), 70),
        ("solar", datetime(2026, 3, 11, 0), 1, datetime(2026, 3, 8, 15), 57),
    ],
)
def test_the_older_run_rule_is_restated_independently_of_the_build(
    domain: DomainType, label: datetime, day: int, init: datetime, lead: int
):
    got_init, slot, got_lead = verify.run_of_row(
        time=label.replace(tzinfo=UTC), day=day, domain=domain, spec=build.OLDER_RUN
    )

    assert got_init == init.replace(tzinfo=UTC)
    assert got_lead == lead
    assert build.slot_init_time(slot=slot) == got_init


def test_the_older_run_rule_agrees_with_the_builds_own_for_every_hour_of_a_day():
    hours = [datetime(2026, 3, 10, hour, tzinfo=UTC) for hour in range(24)]
    for domain in ("solar", "wind"):
        for day in build.OLDER_RUN.lead_days:
            built = build.with_run(
                frame=pl.DataFrame({"site": ["A"] * 24, "time": hours}),
                day=day,
                domain=domain,
                spec=build.OLDER_RUN,
            )
            for row in built.iter_rows(named=True):
                got = verify.run_of_row(
                    time=row["time"], day=day, domain=domain, spec=build.OLDER_RUN
                )
                assert got == (row["init_time"], row["slot"], row["lead_hours"])


def test_the_older_runs_sample_reads_its_own_columns_and_leads():
    times = [datetime(2026, 3, day, hour, tzinfo=UTC) for day in range(5, 25) for hour in range(24)]
    fields = build.WEATHER_FIELDS["wind"]
    built = pl.DataFrame({"site": ["A"] * len(times), "time": times}).with_columns(
        **{f"ukv_ceda_run15_day{d}_{field}": pl.lit(1.0) for d in (1, 2, 3) for field in fields}
    )

    strata = verify.sample_rows(built=built, domain="wind", day=1, spec=build.OLDER_RUN)

    # A wind hour at lead day 1 of the older run has lead 33 + h: native up to h of 15 (lead 48),
    # first rebuilt for h of 16 to 21 (leads 49 to 54), and late for h of 22 and 23.
    assert {row["time"].hour for row in strata["native"]} <= set(range(16))
    assert {row["time"].hour for row in strata["first rebuilt"]} <= set(range(16, 22))
    assert {row["time"].hour for row in strata["late"]} <= {22, 23}
    assert all(strata[name] for name in ("native", "first rebuilt", "late"))
    assert f"ukv_ceda_run15_day1_{fields[0]}" in strata["native"][0]
    assert verify.OLDER_DAY1_SNAPSHOT_LEADS.start == 33


def test_the_older_skill_gate_fails_when_the_correlation_rises_or_beats_the_planned_run():
    assert verify.older_skill_gate(older=[0.90, 0.88, 0.86], planned=[0.92, 0.90, 0.88]) == []
    rising = verify.older_skill_gate(older=[0.88, 0.89, 0.86], planned=[0.92, 0.91, 0.88])
    assert len(rising) == 1
    assert "rises with the lead day" in rising[0]
    better = verify.older_skill_gate(older=[0.935, 0.90, 0.86], planned=[0.92, 0.91, 0.88])
    assert len(better) == 1
    assert "day-1 correlation 0.935 of the older run is above the planned run's 0.920" in better[0]
    # Within the tolerance of the planned run's correlation is allowed.
    assert verify.older_skill_gate(older=[0.925, 0.90, 0.86], planned=[0.92, 0.91, 0.88]) == []


def test_the_older_skill_lines_compare_both_runs_on_the_rows_both_hold():
    rows = 50
    truth = np.linspace(0.0, 1.0, rows)
    older_noise = np.sin(np.arange(rows) * 7.0)
    planned_noise = np.sin(np.arange(rows) * 3.0)
    data = {"speed_10m_era5": truth}
    for day, scale in zip((1, 2, 3), (0.1, 0.2, 0.4), strict=True):
        data[f"ukv_ceda_run15_day{day}_speed_10m"] = truth + scale * older_noise
        data[f"ukv_ceda_day{day}_speed_10m"] = truth + 0.05 * planned_noise
    joined = pl.DataFrame(data)

    lines, ok = verify.older_skill_lines(domain="wind", joined=joined)

    assert ok
    assert f"on the {rows} rows both runs hold at every lead day" in lines[0]
    # A row missing the older run at one lead day leaves the shared rows.
    holed = joined.with_columns(
        ukv_ceda_run15_day2_speed_10m=pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("ukv_ceda_run15_day2_speed_10m"))
    )
    holed_lines, _ = verify.older_skill_lines(domain="wind", joined=holed)
    assert f"on the {rows - 1} rows both runs hold" in holed_lines[0]


def test_the_older_verify_stamp_names_the_older_inputs_files(tmp_path: Path):
    for domain in ("solar", "wind"):
        pl.DataFrame({"a": [1]}).write_parquet(tmp_path / f"{domain}_ukv_ceda_run15_inputs.parquet")

    verify.write_verify_stamp(output_dir=tmp_path, passed=True, spec=build.OLDER_RUN)

    stamp = json.loads((tmp_path / "verify.json").read_text())
    assert stamp["inputs_sha256"] == {
        domain: build.sha256_of(path=tmp_path / f"{domain}_ukv_ceda_run15_inputs.parquet")
        for domain in ("solar", "wind")
    }
