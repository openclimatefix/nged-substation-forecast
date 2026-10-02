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
