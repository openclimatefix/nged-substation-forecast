import re
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

import fit_aifs  # noqa: E402
import nwp_forecast_charts as charts  # noqa: E402
from build_forecast_inputs import (  # noqa: E402
    AIFS_VALUE_COLUMNS,
    aifs_members_frame,
    build_aifs,
)
from fit_aifs import (  # noqa: E402
    BLEND_DAYS,
    ENS_STAMPED,
    LONG_DAYS,
    NO_DOY_SUFFIX,
    NO_SKILL,
    ROW_SETS,
    SKILL,
    Contrast,
    add_shuffled_columns,
    arm_features,
    arm_prefixes,
    blend_arm_name,
    blend_arms,
    blend_contrasts,
    blend_inputs,
    build_stamp,
    check_columns_equal,
    check_runs,
    check_saved_losses,
    contrast_losses,
    day14_reading,
    drop_runs_outside_era,
    ens_control_prefix,
    expected_column_count,
    fit_jobs,
    lead_verdict,
    refuse_read_only_folders,
    shuffled_prefix,
    smoothing_reading,
    stage_deciding_arms,
    summed_arm,
    weather_spread,
)
from nwp_forecast_comparison import (  # noqa: E402
    BLEND_ARMS,
    METRIC,
    DomainType,
    add_blend_guard_columns,
)
from nwp_forecast_comparison import jobs as published_jobs  # noqa: E402
from studies.bootstrap import BootstrapInterval  # noqa: E402


def _interval(*, difference: float, lower: float, upper: float) -> BootstrapInterval:
    return {
        "difference": difference,
        "lower_95": lower,
        "upper_95": upper,
        "seed_spread": 0.0,
        "n_rows": 10,
        "n_months": 6,
    }


# --- check_runs ---------------------------------------------------------------------------------


def _run_frame(
    *, day: int, arm_day: int | None = None, prefix: str = "aifs_single"
) -> pl.DataFrame:
    """Two rows on 2025-10-15 whose `<prefix>_day<day>` run is `arm_day` days back."""
    times = [datetime(2025, 10, 15, 10, tzinfo=UTC), datetime(2025, 10, 15, 11, tzinfo=UTC)]
    run = datetime(2025, 10, 15, tzinfo=UTC) - timedelta(days=arm_day or day)
    return pl.DataFrame(
        {
            "time": times,
            "era_code": [1, 1],
            f"{prefix}_day{day}_init_time": [run, run],
        }
    )


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_check_runs_reads_day_14_from_the_prefix_not_its_last_digit(domain: DomainType):
    frame = _run_frame(day=14)

    check_runs(frame=frame, domain=domain, row_set="single", arms=("aifs_single_day14",))

    # `int(prefix[-1])` reads day 4, so the day-4 run would pass and the day-14 run would fail.
    wrong = _run_frame(day=14, arm_day=4)
    with pytest.raises(ValueError, match="wrong_day"):
        check_runs(frame=wrong, domain=domain, row_set="single", arms=("aifs_single_day14",))


def test_check_runs_rejects_a_run_thirteen_days_back_at_day_14():
    frame = _run_frame(day=14, arm_day=13)

    with pytest.raises(ValueError, match="aifs_single_day14"):
        check_runs(frame=frame, domain="wind", row_set="single", arms=("aifs_single_day14",))


@pytest.mark.parametrize("day", [7, 14])
def test_check_runs_inspects_the_long_lead_arms_the_row_set_lists(day: int):
    frame = _run_frame(day=day, arm_day=day + 1)

    with pytest.raises(ValueError, match=f"aifs_single_day{day}"):
        check_runs(
            frame=frame, domain="wind", row_set="single", arms=(f"aifs_single_day{day}", "x")
        )


def test_check_runs_finds_the_aifs_prefix_inside_a_blend_but_not_inside_its_control():
    good = _run_frame(day=7)
    bad = _run_frame(day=7, arm_day=6)
    blend = blend_arm_name(product="aifs_single", day=7)
    control = blend_arm_name(product="aifs_single", day=7, role="_control")

    check_runs(frame=good, domain="wind", row_set="single", arms=(blend,))
    with pytest.raises(ValueError, match="aifs_single_day7"):
        check_runs(frame=bad, domain="wind", row_set="single", arms=(blend,))
    # A control shows shuffled AIFS columns, so it needs no run stamp of its own.
    check_runs(
        frame=pl.DataFrame({"time": good["time"], "era_code": good["era_code"]}),
        domain="wind",
        row_set="single",
        arms=(control,),
    )


def test_check_runs_raises_when_an_aifs_arm_has_no_run_stamp():
    frame = pl.DataFrame({"time": [datetime(2025, 10, 15, tzinfo=UTC)], "era_code": [1]})

    with pytest.raises(ValueError, match="aifs_single_day7_init_time is missing"):
        check_runs(frame=frame, domain="wind", row_set="single", arms=("aifs_single_day7",))


def test_check_runs_checks_an_ens_stamp_where_present_and_skips_an_unstamped_ens_arm():
    frame = _run_frame(day=7, arm_day=6, prefix="ens_control")

    check_runs(frame=frame, domain="wind", row_set="single", arms=("ens_mean_day1",))
    with pytest.raises(ValueError, match="ens_control_day7"):
        check_runs(frame=frame, domain="wind", row_set="single", arms=("ens_control_day7",))


def test_check_runs_rejects_a_run_outside_its_era():
    early = datetime(2025, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "time": [datetime(2025, 1, 8, 10, tzinfo=UTC)],
            "era_code": [0],
            "aifs_single_day7_init_time": [early],
        }
    )

    with pytest.raises(ValueError, match="outside_era"):
        check_runs(frame=frame, domain="wind", row_set="single", arms=("aifs_single_day7",))


def test_check_runs_treats_a_solar_hour_by_the_hour_it_ends():
    # 2025-10-16 00:00 labels the solar hour that ends then, which lies on 2025-10-15.
    frame = pl.DataFrame(
        {
            "time": [datetime(2025, 10, 16, 0, tzinfo=UTC)],
            "era_code": [1],
            "aifs_single_day7_init_time": [datetime(2025, 10, 8, tzinfo=UTC)],
        }
    )

    check_runs(frame=frame, domain="solar", row_set="single", arms=("aifs_single_day7",))
    with pytest.raises(ValueError, match="wrong_day"):
        check_runs(frame=frame, domain="wind", row_set="single", arms=("aifs_single_day7",))


# --- the row-level era rule -----------------------------------------------------------------------


def _hours(*, first: datetime, n_days: int) -> pl.DataFrame:
    times = [first + timedelta(hours=h) for h in range(24 * n_days)]
    return pl.DataFrame({"time": times}).with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def test_the_era_rule_drops_only_hours_whose_run_is_before_the_first_usable_run():
    frame = _hours(first=datetime(2025, 3, 1, tzinfo=UTC), n_days=10)

    kept = drop_runs_outside_era(frame=frame, domain="wind", row_set="single", day=7)

    # The first usable run is 2025-02-26, so the first kept target day is 2025-03-05.
    assert kept["time"].min() == datetime(2025, 3, 5, tzinfo=UTC)
    assert set(kept["month"]) == {"2025-03"}
    assert kept.height == 24 * 6


def test_the_era_rule_drops_nothing_at_day_1_and_keeps_every_month_hour():
    frame = _hours(first=datetime(2025, 3, 1, tzinfo=UTC), n_days=3)

    kept = drop_runs_outside_era(frame=frame, domain="wind", row_set="single", day=1)

    assert kept.height == frame.height


def test_the_era_rule_uses_the_v1_1_start_for_the_second_era_at_day_14():
    frame = _hours(first=datetime(2025, 9, 1, tzinfo=UTC), n_days=20)

    kept = drop_runs_outside_era(frame=frame, domain="wind", row_set="single", day=14)

    # Era 1's first run is 2025-08-28, so the first target day is 2025-09-11.
    assert kept["time"].min() == datetime(2025, 9, 11, tzinfo=UTC)


# --- column counts and arm names ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm", "domain", "count"),
    [
        ("aifs_single_day7", "solar", 7),
        ("aifs_single_day7", "wind", 7),
        ("blend_aifs_single_day7", "solar", 9),
        ("blend_aifs_single_day7_control", "solar", 9),
        ("blend_aifs_single_day7_mirror", "wind", 11),
        ("blend_aifs_ens_day14", "wind", 11),
        (f"aifs_single_day7{NO_DOY_SUFFIX}", "solar", 6),
    ],
)
def test_an_arm_holds_the_column_count_its_kind_promises(
    arm: str, domain: DomainType, count: int
) -> None:
    assert expected_column_count(arm=arm, domain=domain) == count
    assert len(arm_features(arm=arm, domain=domain)) == count


def test_a_blend_control_shows_the_shuffled_aifs_columns_and_the_mirror_the_shuffled_ens_columns():
    assert arm_prefixes(arm="blend_aifs_single_day7") == ("ens_mean_day7", "aifs_single_day7")
    assert arm_prefixes(arm="blend_aifs_single_day7_control") == (
        "ens_mean_day7",
        "aifs_single_day7_permuted",
    )
    assert arm_prefixes(arm="blend_aifs_single_day7_mirror") == (
        "ens_mean_day7_permuted",
        "aifs_single_day7",
    )
    assert arm_prefixes(arm="blend_aifs_ens_day2_control") == (
        "ens_mean_day2",
        "aifs_ens_mean_day2_permuted",
    )


def test_each_stage_holds_the_arms_the_plan_counts():
    single = {day: blend_arms(row_set="single", day=day) for day in BLEND_DAYS}
    ens = {day: blend_arms(row_set="ens", day=day) for day in BLEND_DAYS}

    assert sum(len(arms) for arms in single.values()) == 30
    assert sum(len(arms) for arms in ens.values()) == 16
    assert {"ifs025_day7", "ifs_single_day7"} <= set(single[7])
    assert not {"ifs025_day7", "ifs_single_day7"} & set(single[14])
    assert "aifs_single_day14_permuted_b" in single[14]
    assert "aifs_single_day2_permuted_b" not in single[2]


@pytest.mark.parametrize("row_set", ["single", "ens"])
@pytest.mark.parametrize("day", BLEND_DAYS)
def test_every_listed_contrast_names_arms_that_the_stage_fits(row_set: str, day: int):
    arms = set(blend_arms(row_set=row_set, day=day))

    for contrast in blend_contrasts(row_set=row_set, day=day):
        assert {contrast.treatment, contrast.reference} <= arms


def test_the_deciding_contrasts_are_exactly_h7_h14_b7_b14_and_their_guards_on_single():
    deciding = [
        (day, c.label)
        for row_set in ROW_SETS
        for day in BLEND_DAYS
        for c in blend_contrasts(row_set=row_set, day=day)
        if c.label.startswith("deciding")
    ]

    assert sorted(deciding) == sorted(
        [
            (7, "deciding (H7)"),
            (7, "deciding (B7)"),
            (7, "deciding (B7 guard)"),
            (14, "deciding (H14)"),
            (14, "deciding (B14)"),
            (14, "deciding (B14 guard)"),
        ]
    )
    assert all(not stage_deciding_arms(row_set="ens", day=day) for day in BLEND_DAYS)
    assert stage_deciding_arms(row_set="single", day=1) == ()


def test_the_control_member_is_six_hourly_at_days_1_and_2_and_native_beyond():
    assert ens_control_prefix(day=1) == "ens_control6_day1"
    assert ens_control_prefix(day=7) == "ens_control_day7"


# --- fit_jobs -------------------------------------------------------------------------------------


def _frame_with(*, arms: list[str], domain: DomainType) -> pl.DataFrame:
    columns = {column for arm in arms for column in arm_features(arm=arm, domain=domain)}
    return pl.DataFrame({"site": ["A", "A"], **{column: [1.0, 2.0] for column in columns}})


@pytest.fixture
def stub_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(**kwargs: object) -> pl.DataFrame:
        return pl.DataFrame({"site": ["A"], "time": [1], "seed": [0], "value": [1.0]})

    monkeypatch.setattr(fit_aifs, "out_of_fold_losses", fake)


@pytest.mark.parametrize(("domain", "blend"), [("solar", 9), ("wind", 11)])
def test_fit_jobs_accepts_a_blend_and_its_controls_at_their_own_column_count(
    stub_fit: None, domain: DomainType, blend: int
):
    arms = [
        blend_arm_name(product="aifs_single", day=7, role=role)
        for role in ("", "_control", "_mirror")
    ]
    frame = _frame_with(arms=arms, domain=domain)

    losses = fit_jobs(
        frame=frame, domain=domain, jobs=[(arm, "primary") for arm in arms], workers=1
    )

    assert set(losses["arm"]) == set(arms)
    assert all(len(arm_features(arm=arm, domain=domain)) == blend for arm in arms)


def test_fit_jobs_still_rejects_an_arm_whose_columns_differ_from_its_kind(
    stub_fit: None, monkeypatch: pytest.MonkeyPatch
):
    arm = "blend_aifs_single_day7"
    frame = _frame_with(arms=[arm], domain="solar")
    real = fit_aifs.arm_features
    monkeypatch.setattr(fit_aifs, "arm_features", lambda **kw: real(**kw)[:-1])

    with pytest.raises(ValueError, match="promises 9"):
        fit_jobs(frame=frame, domain="solar", jobs=[(arm, "primary")], workers=1)


def test_fit_jobs_rejects_a_single_product_with_a_blends_column_count(
    stub_fit: None, monkeypatch: pytest.MonkeyPatch
):
    frame = _frame_with(arms=["blend_aifs_single_day7"], domain="wind")
    real = fit_aifs.arm_features
    monkeypatch.setattr(
        fit_aifs,
        "arm_features",
        lambda *, arm, domain: real(arm="blend_aifs_single_day7", domain=domain),
    )

    with pytest.raises(ValueError, match="promises 7"):
        fit_jobs(frame=frame, domain="wind", jobs=[("aifs_single_day7", "primary")], workers=1)


# --- shuffles -------------------------------------------------------------------------------------


def _shuffle_frame(*, domain: str, day: int = 7) -> pl.DataFrame:
    """Two sites, two months, 30 days: every value encodes the group it belongs to."""
    records = []
    for site in ("A", "B"):
        for month in ("2025-10", "2025-11"):
            for _ in range(30):
                for hour in (10, 11):
                    records.append(
                        {
                            "site": site,
                            "month": month,
                            "hour_of_day": hour,
                            "code": f"{site}|{month}|{hour}",
                            "value": float(len(records)),
                        }
                    )
    frame = pl.DataFrame(records)
    source = f"aifs_single_day{day}"
    fields = (
        ("ghi", "temp")
        if domain == "solar"
        else ("speed_100m", "sin_100m", "cos_100m", "speed_10m")
    )
    return frame.with_columns(pl.col("value").alias(f"{source}_{field}") for field in fields)


def test_a_shuffle_never_crosses_a_site_a_year_month_or_an_hour():
    frame = _shuffle_frame(domain="solar")
    source = "aifs_single_day7"
    lookup = dict(zip(frame["value"], frame["code"], strict=True))

    shuffled = add_shuffled_columns(frame=frame, domain="solar", shuffles={source: ("", "_b")})

    for variant in ("_permuted", "_permuted_b"):
        moved = shuffled[f"{source}{variant}_ghi"]
        assert [lookup[value] for value in moved] == frame["code"].to_list()


def test_a_shuffle_is_deterministic_and_the_two_seeds_move_values_differently():
    frame = _shuffle_frame(domain="solar")
    source = "aifs_single_day7"

    first = add_shuffled_columns(frame=frame, domain="solar", shuffles={source: ("", "_b")})
    again = add_shuffled_columns(frame=frame, domain="solar", shuffles={source: ("", "_b")})

    assert first.equals(again)
    seed_0 = first[f"{source}_permuted_ghi"].to_numpy()
    seed_1000 = first[f"{source}_permuted_b_ghi"].to_numpy()
    assert not np.array_equal(seed_0, seed_1000)
    real = first[f"{source}_ghi"].to_numpy()
    assert (seed_0 != real).mean() >= 0.9
    assert (seed_1000 != real).mean() >= 0.9


def test_a_shuffle_keeps_each_group_multiset_of_values():
    frame = _shuffle_frame(domain="solar")
    source = "aifs_single_day7"

    shuffled = add_shuffled_columns(frame=frame, domain="solar", shuffles={source: ("",)})

    for _, group in shuffled.group_by("site", "month", "hour_of_day"):
        assert sorted(group[f"{source}_ghi"]) == sorted(group[f"{source}_permuted_ghi"])


def test_the_mirror_and_the_blend_control_shuffle_ens_and_aifs_by_the_same_seed():
    frame = _shuffle_frame(domain="solar").with_columns(
        ens_mean_day7_ghi=pl.col("aifs_single_day7_ghi"),
        ens_mean_day7_temp=pl.col("aifs_single_day7_temp"),
    )

    shuffled = add_shuffled_columns(
        frame=frame, domain="solar", shuffles={"aifs_single_day7": ("",), "ens_mean_day7": ("",)}
    )

    assert shuffled["aifs_single_day7_permuted_ghi"].equals(shuffled["ens_mean_day7_permuted_ghi"])


def test_a_wind_shuffle_moves_a_direction_sine_and_cosine_together():
    frame = (
        _shuffle_frame(domain="wind")
        .with_columns(
            angle=pl.col("value") * 0.37,
        )
        .with_columns(
            aifs_single_day7_sin_100m=pl.col("angle").sin(),
            aifs_single_day7_cos_100m=pl.col("angle").cos(),
        )
    )

    shuffled = add_shuffled_columns(
        frame=frame, domain="wind", shuffles={"aifs_single_day7": ("",)}
    )

    norm = (
        shuffled["aifs_single_day7_permuted_sin_100m"] ** 2
        + shuffled["aifs_single_day7_permuted_cos_100m"] ** 2
    )
    assert np.allclose(norm.to_numpy(), 1.0)


# --- verdicts and readings ------------------------------------------------------------------------


@pytest.mark.parametrize("day", [1, 2, 7, 14])
def test_the_lead_verdict_names_the_lead_and_never_says_day_ahead(day: int):
    lowers = lead_verdict(
        day=day,
        versus_ens=_interval(difference=-0.5, lower=-0.9, upper=-0.1),
        versus_control=_interval(difference=-0.4, lower=-0.8, upper=-0.05),
    )

    assert lowers["verdict"] == f"lowers the error at day {day}"
    assert "day-ahead" not in lowers["verdict"]
    assert lowers["largest_gain_not_excluded"] is None


def test_the_lead_verdict_needs_both_the_blend_and_its_guard_to_be_below_zero():
    verdict = lead_verdict(
        day=7,
        versus_ens=_interval(difference=-0.5, lower=-0.9, upper=-0.1),
        versus_control=_interval(difference=-0.2, lower=-0.6, upper=0.1),
    )

    assert verdict["verdict"] == "no detectable difference"
    assert verdict["largest_gain_not_excluded"] == pytest.approx(0.9)


def test_a_blend_significantly_worse_than_ens_raises_the_error_not_no_difference():
    verdict = lead_verdict(
        day=14,
        versus_ens=_interval(difference=0.3, lower=0.1, upper=0.5),
        versus_control=_interval(difference=0.1, lower=-0.2, upper=0.4),
    )

    assert verdict["verdict"] == "raises the error at day 14"
    assert verdict["largest_gain_not_excluded"] is None


def test_the_day_14_rule_needs_two_shuffle_seeds_to_agree():
    below = _interval(difference=-0.5, lower=-0.9, upper=-0.1)
    across = _interval(difference=-0.2, lower=-0.6, upper=0.2)

    assert day14_reading(aifs_single=[below, below], ens_control=[across, across]) == SKILL
    assert day14_reading(aifs_single=[below, across], ens_control=[across, below]) == NO_SKILL
    assert day14_reading(aifs_single=[across, across], ens_control=[across, across]) == NO_SKILL


def test_a_lower_error_than_the_control_but_not_the_mean_reads_as_smoothing():
    lower = _interval(difference=-0.5, lower=-0.9, upper=-0.1)
    across = _interval(difference=-0.1, lower=-0.4, upper=0.2)

    text = smoothing_reading(versus_control=lower, versus_mean=across)

    assert "consistent with smoothing" in text
    assert "not described as the better weather forecast" in text
    assert "both" in smoothing_reading(versus_control=lower, versus_mean=lower)
    assert "does not have a significantly lower" in smoothing_reading(
        versus_control=across, versus_mean=lower
    )


# --- summed arms, spread and restricted rows ---------------------------------------------------


def test_a_summed_arm_adds_each_arms_loss_on_the_shared_rows():
    def arm(name: str, values: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "arm": name,
                "site": "A",
                "time": [1, 2, 3],
                "seed": 0,
                "month": "2025-10",
                METRIC: values,
            }
        )

    losses = pl.concat([arm("a", [1.0, 2.0, 3.0]), arm("b", [10.0, 20.0, 30.0])])

    total = summed_arm(losses=losses, arms=("a", "b"), name="both")

    assert total.sort("time")[METRIC].to_list() == [11.0, 22.0, 33.0]
    assert set(total["arm"]) == {"both"}


def test_the_weather_spread_is_each_columns_standard_deviation():
    frame = pl.DataFrame(
        {
            "aifs_single_day7_ghi": [0.0, 2.0, 4.0],
            "aifs_single_day7_temp": [5.0, 5.0, 5.0],
        }
    )

    spread = weather_spread(frame=frame, domain="solar", prefixes=("aifs_single_day7",))

    assert spread.sort("column")["std"].to_list() == pytest.approx([2.0, 0.0])
    assert set(spread["n_rows"]) == {3}


def test_a_contrast_with_ifs_hres_runs_on_the_rows_where_hres_has_values():
    frame = pl.DataFrame(
        {
            "site": ["A"] * 3,
            "time": [1, 2, 3],
            "ifs_single_day7_ghi": [1.0, None, 3.0],
            "ifs_single_day7_temp": [1.0, 2.0, 3.0],
        }
    )
    losses = pl.DataFrame({"site": ["A"] * 3, "time": [1, 2, 3], "arm": "x"})

    restricted = contrast_losses(
        losses=losses,
        frame=frame,
        domain="solar",
        contrast=Contrast("aifs_single_day7", "ifs_single_day7", "x"),
    )
    untouched = contrast_losses(
        losses=losses,
        frame=frame,
        domain="solar",
        contrast=Contrast("aifs_single_day7", "ens_mean_day7", "x"),
    )

    assert restricted["time"].to_list() == [1, 3]
    assert untouched.height == 3


# --- inputs ---------------------------------------------------------------------------------------


def test_columns_that_differ_beyond_float32_rounding_raise_and_rounded_ones_pass():
    keys = {"site": ["A", "A"], "time": [1, 2]}
    built = pl.DataFrame({**keys, "c": [100.0, 200.0]})
    rounded = pl.DataFrame({**keys, "c": pl.Series([100.0, 200.0], dtype=pl.Float32)})
    off = pl.DataFrame({**keys, "c": [100.0, 200.5]})

    check_columns_equal(built=built, reference=rounded, columns=["c"], label="x")
    with pytest.raises(ValueError, match="differ"):
        check_columns_equal(built=built, reference=off, columns=["c"], label="x")


def test_a_null_on_one_side_only_and_a_missing_row_both_raise():
    keys = {"site": ["A", "A"], "time": [1, 2]}
    built = pl.DataFrame({**keys, "c": [1.0, 2.0]})
    with pytest.raises(ValueError, match="differ"):
        check_columns_equal(
            built=built,
            reference=pl.DataFrame({**keys, "c": [1.0, None]}),
            columns=["c"],
            label="x",
        )
    with pytest.raises(ValueError, match="missing"):
        check_columns_equal(
            built=built,
            reference=pl.DataFrame({"site": ["A"], "time": [1], "c": [1.0]}),
            columns=["c"],
            label="x",
        )


def _stamped_build(*, times: list[int]) -> pl.DataFrame:
    """A solar AIFS build holding every ENS prefix at days 1, 2, 7 and 14, with run stamps."""
    frame = pl.DataFrame({"site": ["A"] * len(times), "time": times})
    run = datetime(2025, 10, 1, tzinfo=UTC)
    columns = {}
    for day in BLEND_DAYS:
        for way in ("mean", "control"):
            prefix = f"ens_{way}6_day{day}"
            columns[f"{prefix}_ghi"] = [float(day)] * len(times)
            columns[f"{prefix}_temp"] = [1.0] * len(times)
            columns[f"{prefix}_init_time"] = [run] * len(times)
    return frame.with_columns(**{name: pl.Series(values) for name, values in columns.items()})


def _write_inputs(tmp_path: Path, *, drift: bool = False, stamped: bool = True) -> dict[str, Path]:
    times = [1, 2, 3]
    build = _stamped_build(times=times)
    if not stamped:
        build = build.drop("ens_mean6_day7_init_time")
    (tmp_path / "blends").mkdir()
    build.write_parquet(tmp_path / "blends" / "solar_aifs_inputs.parquet")
    keys = pl.DataFrame({"site": ["A"] * len(times), "time": times})
    folders = {
        "leads_day10": {
            "ens_mean_day14": 14.0,
            "ifs025_day7": 5.0,
        },
        "leads_day10b": {
            "ens_mean_day7": 7.5 if drift else 7.0,
            "ens_control_day7": 7.0,
            "ens_control_day14": 14.0,
        },
        "leads_day10d": {"ifs_single_day7": 6.0},
    }
    paths = {}
    for name, prefixes in folders.items():
        folder = tmp_path / name
        folder.mkdir()
        extra = keys
        for prefix, value in prefixes.items():
            extra = extra.with_columns(
                pl.lit(value).alias(f"{prefix}_ghi"), pl.lit(1.0).alias(f"{prefix}_temp")
            )
        extra.write_parquet(folder / "solar_extra_lead_inputs.parquet")
        paths[name] = folder
    return paths


def test_blend_inputs_renames_the_native_ens_columns_and_joins_the_ifs_references(tmp_path: Path):
    extra_dirs = _write_inputs(tmp_path)

    inputs = blend_inputs(aifs_dir=tmp_path / "blends", extra_dirs=extra_dirs, domain="solar")

    assert {"ens_mean_day7_ghi", "ens_control_day14_init_time"} <= set(inputs.columns)
    assert "ens_mean6_day7_ghi" not in inputs.columns
    assert "ens_control6_day2_ghi" in inputs.columns
    assert inputs["ifs025_day7_ghi"].to_list() == [5.0] * 3
    assert inputs["ifs_single_day7_ghi"].to_list() == [6.0] * 3


def test_blend_inputs_raises_when_the_build_disagrees_with_an_extra_lead_folder(tmp_path: Path):
    extra_dirs = _write_inputs(tmp_path, drift=True)

    with pytest.raises(ValueError, match="ens_mean_day7_ghi"):
        blend_inputs(aifs_dir=tmp_path / "blends", extra_dirs=extra_dirs, domain="solar")


def test_blend_inputs_raises_when_a_required_ens_run_stamp_is_absent(tmp_path: Path):
    extra_dirs = _write_inputs(tmp_path, stamped=False)

    with pytest.raises(ValueError, match="ens_mean_day7"):
        blend_inputs(aifs_dir=tmp_path / "blends", extra_dirs=extra_dirs, domain="solar")


def test_every_ens_prefix_the_blends_frames_check_is_stamped():
    assert ens_control_prefix(day=1) in ENS_STAMPED
    assert ens_control_prefix(day=14) in ENS_STAMPED
    assert "ens_mean_day7" in ENS_STAMPED
    assert "ens_mean_day1" not in ENS_STAMPED


# --- the AIFS build's lead filter ------------------------------------------------------------


def _store(tmp_path: Path) -> Path:
    leads = list(range(0, 361, 6))
    frame = pl.DataFrame(
        {
            "init_time": [datetime(2025, 10, 1)] * len(leads),
            "lead_time": [timedelta(hours=lead) for lead in leads],
            "lat_index": [0] * len(leads),
            "lon_index": [0] * len(leads),
            **{column: [1.0] * len(leads) for column in AIFS_VALUE_COLUMNS},
        }
    )
    path = tmp_path / "store.parquet"
    frame.write_parquet(path)
    return path


@pytest.mark.parametrize(
    ("days", "kept"),
    [
        ((1, 2), set(range(18, 55, 6)) | set(range(42, 79, 6))),
        ((7, 14), set(range(162, 199, 6)) | set(range(330, 361, 6))),
    ],
)
def test_the_aifs_read_keeps_only_the_leads_of_the_requested_bands(
    tmp_path: Path, days: tuple[int, ...], kept: set[int]
):
    weights = pl.DataFrame({"site": ["A"], "lat_index": [0], "lon_index": [0], "weight": [1.0]})

    frame = aifs_members_frame(
        store=_store(tmp_path),
        weights=weights,
        ensemble=False,
        first_init=datetime(2025, 9, 1, tzinfo=UTC),
        days=days,
    )

    assert set(frame["lead_hours"]) == kept


def test_the_build_refuses_the_published_and_the_existing_aifs_folders(tmp_path: Path):
    studies = tmp_path / "studies"
    published = studies / "nwp_forecast_comparison"
    published.mkdir(parents=True)

    for output in (published, studies / "nwp_forecast_comparison_aifs"):
        with pytest.raises(ValueError, match="must not be"):
            build_aifs(
                domain="wind",
                published_dir=published,
                output_dir=output,
                weather_dir=tmp_path,
                days=(7,),
            )


def test_the_build_refuses_no_days(tmp_path: Path):
    with pytest.raises(ValueError, match="days"):
        build_aifs(
            domain="wind",
            published_dir=tmp_path / "a",
            output_dir=tmp_path / "b",
            weather_dir=tmp_path,
            days=(),
        )


# --- stamps and folders ---------------------------------------------------------------------------


def test_the_stamp_names_the_gpu_the_seeds_and_every_extra_input(tmp_path: Path):
    extra_dirs = _write_inputs(tmp_path)
    (tmp_path / "published").mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "published" / "solar_forecast_inputs.parquet")
    arms = ["aifs_single_day7", "blend_aifs_single_day7"]

    stamp = build_stamp(
        published_dir=tmp_path / "published",
        aifs_dir=tmp_path / "blends",
        domain="solar",
        arms=arms,
        extra_dirs=extra_dirs,
    )

    assert stamp["device"] == "cuda"
    assert set(__import__("json").loads(stamp["extra_sha256"])) == set(extra_dirs)
    assert set(__import__("json").loads(stamp["columns"])) == set(arms)
    assert "seeds" in stamp


def test_saved_losses_are_refused_when_the_stamp_or_the_arms_differ(tmp_path: Path):
    frame = pl.DataFrame({"site": ["A"], "time": [1], "fold": [0]})
    losses = pl.DataFrame(
        {"site": ["A"], "time": [1], "fold": [0], "arm": ["a"], "setting": ["primary"]}
    )
    stamp_file = tmp_path / "stamp.json"
    stamp_file.write_text('{"device": "cuda"}')

    check_saved_losses(
        losses=losses,
        frame=frame,
        stage="s",
        arms=["a"],
        stamp_file=stamp_file,
        stamp={"device": "cuda"},
    )
    with pytest.raises(ValueError, match="another build or device"):
        check_saved_losses(
            losses=losses,
            frame=frame,
            stage="s",
            arms=["a"],
            stamp_file=stamp_file,
            stamp={"device": "cpu"},
        )
    with pytest.raises(ValueError, match="other arms"):
        check_saved_losses(
            losses=losses,
            frame=frame,
            stage="s",
            arms=["a", "b"],
            stamp_file=stamp_file,
            stamp={"device": "cuda"},
        )


def test_no_output_may_land_in_a_folder_the_blends_fit_reads(tmp_path: Path):
    published = tmp_path / "nwp_forecast_comparison"
    extra = tmp_path / "nwp_forecast_comparison_leads_day10"
    new = tmp_path / "nwp_forecast_comparison_aifs_blends"

    refuse_read_only_folders(output_dir=new, read_only=[published, extra])
    for folder in (published, extra):
        with pytest.raises(ValueError, match="never writes"):
            refuse_read_only_folders(output_dir=folder, read_only=[published, extra])


def test_the_old_era_dates_are_unchanged():
    assert ROW_SETS["single"].era_runs[0] == (date(2025, 2, 26), date(2025, 7, 31))


# --- the report, on synthetic fits ---------------------------------------------------------------


def _synthetic_stage(
    *, row_set: str, day: int, domain: DomainType, rng: np.random.Generator
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One stage's rows and losses: 2 sites, 12 months, 6 days a month, arms at both settings."""
    months = [f"2025-{m:02d}" for m in range(3, 13)] + ["2026-02", "2026-03"]
    records = [
        {
            "site": site,
            "month": month,
            "time": datetime(2025, 3, 1, tzinfo=UTC) + timedelta(days=30 * m + d, hours=10),
            "era_code": 0 if m < 4 else 1,
        }
        for site in ("A", "B")
        for m, month in enumerate(months)
        for d in range(6)
    ]
    frame = pl.DataFrame(records)
    prefixes = {p for arm in blend_arms(row_set=row_set, day=day) for p in arm_prefixes(arm=arm)}
    weather = {
        column: rng.normal(size=frame.height)
        for prefix in prefixes
        if "_permuted" not in prefix
        for column in arm_features(arm=prefix, domain=domain)
        if column.startswith(f"{prefix}_")
    }
    frame = frame.with_columns(pl.Series(name, values) for name, values in weather.items())
    arms = [
        *blend_arms(row_set=row_set, day=day),
        *(
            f"{a}{NO_DOY_SUFFIX}"
            for a in (f"aifs_single_day{day}", ens_control_prefix(day=day))
            if row_set == "single" and day in LONG_DAYS
        ),
    ]
    second = {*stage_deciding_arms(row_set=row_set, day=day), "ens_mean_day1"} & set(arms)
    losses = [
        frame.select("site", "month", "time").with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            seed=pl.lit(seed),
            fold=pl.lit(0),
            signed_error_capped_mw=pl.lit(0.0),
            **{METRIC: pl.Series(np.abs(rng.normal(0.1, 0.03, frame.height)))},
        )
        for setting, chosen in (("primary", arms), ("sensitivity", sorted(second)))
        for arm in chosen
        for seed in range(3)
    ]
    return frame, pl.concat(losses)


def test_the_report_prints_every_deciding_check_the_reading_rules_and_the_spread(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 30)
    rng = np.random.default_rng(0)
    frames, losses = {}, {}
    for day in BLEND_DAYS:
        frames[day], losses[day] = _synthetic_stage(
            row_set="single", day=day, domain="wind", rng=rng
        )

    report = "\n".join(
        fit_aifs.blend_set_lines(domain="wind", row_set="single", frames=frames, losses=losses)
    )

    for heading in (
        "### Deciding contrast, wind: aifs_single_day7 − ens_control_day7",
        "### Deciding contrast B14, wind:",
        "### Day-14 reading rule, wind",
        "### Smoothing reading beside H7 and H14",
        "### Gap at day 14 minus gap at day 7",
        "### Spread of each forecast's weather columns over the scored rows",
        "Positive control, ens_control_day7 − ens_mean_day7",
        "inside each version era",
    ):
        assert heading in report
    assert "lowers the error at day" in report or "no detectable difference" in report
    assert "| Error of the first arm (%) | Error of the second arm (%) |" in report
    assert report.count("Absolute error at the primary setting (percent of capacity):") >= 5
    assert "day-ahead" not in report
    assert report.count("| 7 | ens_mean_day7 |") >= 1
    for arm in blend_arms(row_set="single", day=14):
        assert f"| {arm} |" in report


def test_the_report_of_the_ens_row_set_is_descriptive_and_has_no_deciding_section(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 30)
    rng = np.random.default_rng(1)
    frames, losses = {}, {}
    for day in BLEND_DAYS:
        frames[day], losses[day] = _synthetic_stage(row_set="ens", day=day, domain="solar", rng=rng)

    report = "\n".join(
        fit_aifs.blend_set_lines(domain="solar", row_set="ens", frames=frames, losses=losses)
    )

    assert "Deciding contrast" not in report
    assert "descriptive, no deciding label" in report
    assert "deciding (" not in report


# --- the P4 second-seed control refit ------------------------------------------------------------


def _published_like_frame(*, domain: DomainType) -> pl.DataFrame:
    """Rows carrying every column the published P4 blends read, each value encoding its group."""
    records = [
        {
            "site": site,
            "month": month,
            "hour_of_day": hour,
            "code": f"{site}|{month}|{hour}",
            "value": float(index),
        }
        for index, (site, month, hour, _) in enumerate(
            (site, month, hour, day)
            for site in ("A", "B")
            for month in ("2025-10", "2025-11")
            for hour in (10, 11)
            for day in range(30)
        )
    ]
    calendar = {
        "day_of_year": 1,
        "era_code": 0,
        "solar_elevation_deg": 1.0,
        "solar_azimuth_deg": 1.0,
    }
    frame = pl.DataFrame(records).with_columns(**{k: pl.lit(v) for k, v in calendar.items()})
    fields = (
        ("ghi", "temp")
        if domain == "solar"
        else ("speed_100m", "sin_100m", "cos_100m", "speed_10m")
    )
    prefixes = {p for blend in fit_aifs.P4_BLENDS for p in BLEND_ARMS[blend]}
    return frame.with_columns(
        (pl.col("value") * (1 + index / 10) + index % 7).alias(f"{prefix}_{field}")
        for index, (prefix, field) in enumerate((p, f) for p in sorted(prefixes) for f in fields)
    )


@pytest.mark.parametrize(("domain", "count"), [("solar", 11), ("wind", 15)])
def test_a_published_blend_and_both_controls_hold_three_products_columns(
    domain: DomainType, count: int
):
    for arm in fit_aifs.p4_arms()[1:]:
        assert expected_column_count(arm=arm, domain=domain) == count
        assert len(arm_features(arm=arm, domain=domain)) == count


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_the_first_controls_columns_are_the_published_controls_own(domain: DomainType):
    frame = add_blend_guard_columns(frame=_published_like_frame(domain=domain), domain=domain)
    published = {
        arm: columns
        for arm, setting, columns, _ in published_jobs(domain=domain, frame=frame)
        if setting == "primary"
    }

    for arm in ("blend_p4a", "blend_p4a_control", "blend_p4b", "blend_p4b_control"):
        assert arm_features(arm=arm, domain=domain) == published[arm]


def test_the_second_control_shows_the_second_seeds_columns_and_no_others():
    second = arm_features(arm="blend_p4a_control_b", domain="solar")
    first = arm_features(arm="blend_p4a_control", domain="solar")

    assert "icon_eu_day1_permuted_b_ghi" in second
    assert "ifs025_day1_permuted_b_temp" in second
    assert not any(column.startswith("icon_eu_day1_permuted_ghi") for column in second)
    assert [c for c in first if "permuted" not in c] == [c for c in second if "permuted" not in c]


@pytest.mark.parametrize("domain", ["solar", "wind"])
def test_the_second_seed_never_crosses_a_site_a_year_month_or_an_hour(domain: DomainType):
    frame = _published_like_frame(domain=domain)
    lookup = dict(zip(frame["value"], frame["code"], strict=True))
    guarded = add_blend_guard_columns(frame=frame, domain=domain)

    shuffled = fit_aifs.add_second_seed_guard_columns(frame=guarded, domain=domain)

    field = "ghi" if domain == "solar" else "speed_100m"
    for prefix in ("icon_eu_day1", "ifs025_day1", "icon_eu_day2", "ifs025_day2"):
        original = frame[f"{prefix}_{field}"].to_list()
        moved = shuffled[f"{prefix}_permuted_b_{field}"].to_list()
        by_value = dict(zip(original, frame["code"], strict=True))
        assert [by_value[value] for value in moved] == frame["code"].to_list()
    assert lookup  # every value encodes its group


def test_the_second_seed_shuffles_differently_from_the_first_and_is_deterministic():
    frame = add_blend_guard_columns(frame=_published_like_frame(domain="solar"), domain="solar")

    first = fit_aifs.add_second_seed_guard_columns(frame=frame, domain="solar")
    again = fit_aifs.add_second_seed_guard_columns(frame=frame, domain="solar")

    assert first.equals(again)
    one = first["icon_eu_day1_permuted_ghi"].to_numpy()
    two = first["icon_eu_day1_permuted_b_ghi"].to_numpy()
    assert (one != two).mean() >= 0.9


def test_adding_the_second_seed_leaves_the_published_controls_columns_unchanged():
    frame = add_blend_guard_columns(frame=_published_like_frame(domain="wind"), domain="wind")

    after = fit_aifs.add_second_seed_guard_columns(frame=frame, domain="wind")

    assert after.select(frame.columns).equals(frame)


def test_a_wind_second_seed_moves_a_direction_sine_and_cosine_together():
    frame = _published_like_frame(domain="wind")
    angle = pl.col("value") * 0.37
    frame = frame.with_columns(
        [
            expression.alias(f"{prefix}_{name}")
            for prefix in ("icon_eu_day1", "ifs025_day1", "icon_eu_day2", "ifs025_day2")
            for name, expression in (("sin_100m", angle.sin()), ("cos_100m", angle.cos()))
        ]
    )

    shuffled = fit_aifs.add_second_seed_guard_columns(
        frame=add_blend_guard_columns(frame=frame, domain="wind"), domain="wind"
    )

    norm = shuffled["ifs025_day1_permuted_b_sin_100m"] ** 2 + (
        shuffled["ifs025_day1_permuted_b_cos_100m"] ** 2
    )
    assert np.allclose(norm.to_numpy(), 1.0)


def test_the_two_seeds_verdicts_must_agree_or_the_claim_is_unresolved():
    assert fit_aifs.seed_agreement_verdict(
        first="lowers the day-ahead error", second="lowers the day-ahead error"
    ) == ("lowers the day-ahead error")
    assert (
        fit_aifs.seed_agreement_verdict(
            first="lowers the day-ahead error", second="no detectable difference"
        )
        == fit_aifs.UNRESOLVED_ACROSS_SEEDS
    )


def _p4_losses(*, means: dict[str, float]) -> pl.DataFrame:
    """Per-row losses of every P4 arm at both settings, each arm's loss near its own mean."""
    rng = np.random.default_rng(3)
    months = [f"2025-{m:02d}" for m in range(1, 9)]
    keys = pl.DataFrame(
        [
            {
                "site": "A",
                "month": month,
                "time": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=30 * m + d),
            }
            for m, month in enumerate(months)
            for d in range(10)
        ]
    )
    return pl.concat(
        [
            keys.with_columns(
                arm=pl.lit(arm),
                setting=pl.lit(setting),
                seed=pl.lit(seed),
                fold=pl.lit(0),
                signed_error_capped_mw=pl.lit(0.0),
                **{METRIC: pl.Series(means[arm] + rng.normal(0.0, 0.002, keys.height))},
            )
            for setting in ("primary", "sensitivity")
            for arm in fit_aifs.p4_arms()
            for seed in range(3)
        ]
    )


def _p4_means(*, second_control: float) -> dict[str, float]:
    return {
        "ens_mean_day1": 0.12,
        "blend_p4a": 0.08,
        "blend_p4b": 0.08,
        "blend_p4a_control": 0.12,
        "blend_p4b_control": 0.12,
        "blend_p4a_control_b": second_control,
        "blend_p4b_control_b": second_control,
    }


def test_p4_blends_that_beat_both_controls_lower_the_error_and_the_seeds_agree(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 50)
    losses = _p4_losses(means=_p4_means(second_control=0.12)).filter(pl.col("setting") == "primary")

    verdicts = fit_aifs.p4_verdicts(losses=losses)

    assert verdicts["first"] == verdicts["second"] == verdicts["agreed"]
    assert verdicts["agreed"] == "lowers the day-ahead error"


def test_a_second_control_as_good_as_the_blend_makes_the_seeds_disagree_and_the_claim_unresolved(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 50)
    losses = _p4_losses(means=_p4_means(second_control=0.08)).filter(pl.col("setting") == "primary")

    verdicts = fit_aifs.p4_verdicts(losses=losses)

    assert verdicts["first"] == "lowers the day-ahead error"
    assert verdicts["second"] == "no detectable difference"
    assert verdicts["agreed"] == fit_aifs.UNRESOLVED_ACROSS_SEEDS


def test_the_p4_report_recomputes_both_guards_at_both_settings_with_errors_beside_each_contrast(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 50)
    losses = _p4_losses(means=_p4_means(second_control=0.08))
    frame = losses.filter(pl.col("setting") == "primary").select("site", "month").unique()

    report = "\n".join(fit_aifs.p4_lines(domain="solar", frame=frame, losses=losses))

    for setting in ("primary", "sensitivity"):
        assert f"### Contrasts at the {setting} setting" in report
    for blend in ("blend_p4a", "blend_p4b"):
        assert f"| {blend} − {blend}_control |" in report
        assert f"| {blend} − {blend}_control_b |" in report
        assert f"| {blend}_control − {blend}_control_b |" in report
    assert report.count("seed-to-seed gap") == 4
    assert "| 8.018 | 12.000 | 80 | 8 |" in report
    assert fit_aifs.UNRESOLVED_ACROSS_SEEDS in report
    assert "the page must call the blend claim unresolved" in report


def test_the_p4_stamp_names_the_gpu_and_both_seeds(tmp_path: Path):
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "solar_forecast_inputs.parquet")

    stamp = fit_aifs.p4_stamp(published_dir=tmp_path, domain="solar")

    assert stamp["device"] == "cuda"
    assert '"second": 1000' in stamp["seeds"]
    assert set(__import__("json").loads(stamp["columns"])) == set(fit_aifs.p4_arms())


def test_the_p4_refit_never_writes_beside_the_blends_or_the_published_folders(tmp_path: Path):
    read_only = [
        tmp_path / "nwp_forecast_comparison",
        tmp_path / fit_aifs.BLENDS_DIR_NAME,
        tmp_path / fit_aifs.EXISTING_AIFS_DIR_NAME,
    ]

    refuse_read_only_folders(output_dir=tmp_path / fit_aifs.P4_DIR_NAME, read_only=read_only)
    for folder in read_only:
        with pytest.raises(ValueError, match="never writes"):
            refuse_read_only_folders(output_dir=folder, read_only=read_only)


# --- the AIFS lead chart -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm", "label"),
    [
        ("aifs_single_day7", "AIFS Single day 7"),
        ("ens_control6_day2", "ENS control member day 2"),
        ("ens_control_day14", "ENS control member day 14"),
        ("aifs_single_day14_permuted_b", "AIFS Single day 14, shuffled again"),
        (
            "blend_aifs_single_day7_control",
            "Blend of ENS mean and AIFS Single day 7, AIFS shuffled",
        ),
        (
            "blend_aifs_single_day7_mirror",
            "Blend of ENS mean and AIFS Single day 7, ENS mean shuffled",
        ),
        ("ifs025_day7", "IFS 0.25° day 7"),
    ],
)
def test_the_lead_chart_names_every_arm_of_the_blends_fit(arm: str, label: str):
    assert charts.aifs_lead_label(arm=arm) == label


def test_the_lead_chart_refuses_an_arm_it_cannot_name():
    with pytest.raises(ValueError, match="not an arm of the blends fit"):
        charts.aifs_lead_label(arm="climatology")


def test_the_lead_chart_draws_the_deciding_and_matching_exploratory_contrasts_only():
    single = charts.aifs_lead_contrasts(row_set="single")

    deciding = [c.label for c, status in single if status == "Deciding"]
    assert sorted(deciding) == sorted(
        [
            "deciding (H7)",
            "deciding (B7)",
            "deciding (B7 guard)",
            "deciding (H14)",
            "deciding (B14)",
            "deciding (B14 guard)",
        ]
    )
    assert {status for _, status in single} == {"Deciding", "Exploratory"}
    assert all(re.search(r"_day(7|14)", c.treatment) for c, _ in single)
    assert {status for _, status in charts.aifs_lead_contrasts(row_set="ens")} == {"Descriptive"}


@pytest.mark.parametrize(
    ("lower", "upper", "reading"),
    [
        (-0.9, -0.1, "has a lower error than"),
        (0.1, 0.9, "has a higher error than"),
        (-0.4, 0.2, "cannot be told apart from"),
    ],
)
def test_the_lead_chart_reads_an_interval_by_whether_it_excludes_zero(
    lower: float, upper: float, reading: str
):
    assert (
        charts.lead_reading(interval=_interval(difference=0.0, lower=lower, upper=upper)) == reading
    )


def test_the_lead_chart_draws_from_synthetic_fits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 30)
    rng = np.random.default_rng(2)
    by_set = {}
    for row_set in ROW_SETS:
        parts = [
            _synthetic_stage(row_set=row_set, day=day, domain="solar", rng=rng)[1]
            for day in BLEND_DAYS
        ]
        by_set[row_set] = pl.concat(parts, how="diagonal_relaxed").with_columns(
            time=pl.col("time").dt.replace_time_zone("UTC")
        )

    chart, title = charts.aifs_leads(losses_by_set=by_set, domain="solar")

    assert " ".join(chart.to_dict()["title"]["text"]) == f"Figure 15: {title}"
    subtitle = " ".join(chart.to_dict()["title"]["subtitle"])
    assert "ENS control member series is 6-hourly" in subtitle


def test_the_lead_chart_refuses_a_site_label_that_is_not_anonymised(tmp_path: Path):
    for row_set in ROW_SETS:
        for day in BLEND_DAYS:
            pl.DataFrame({"site": ["Real Farm"], "arm": ["x"]}).write_parquet(
                tmp_path / f"solar_{row_set}_day{day}_losses.parquet"
            )

    with pytest.raises(ValueError, match="not anonymised"):
        charts.load_aifs_leads(blends_dir=tmp_path, domain="solar")


# --- constructed losses: verdict text, chart titles and the code review's fixes ---------------


def _levels(
    *, aifs: float, control: float, mean: float, blend: float, blend_control: float
) -> dict[str, float]:
    return {
        "aifs": aifs,
        "control": control,
        "mean": mean,
        "blend": blend,
        "blend_control": blend_control,
    }


def _single_losses(
    *,
    day7: dict[str, float],
    day14: dict[str, float],
    permuted_14: float,
    sensitivity_aifs_7: float | None = None,
) -> pl.DataFrame:
    """Per-row losses of the days 7 and 14 arms of `single`, each near its own level."""
    rng = np.random.default_rng(5)
    months = [f"2025-{m:02d}" for m in range(1, 9)]
    keys = pl.DataFrame(
        [
            {
                "site": "A",
                "month": month,
                "time": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=30 * m + d),
            }
            for m, month in enumerate(months)
            for d in range(10)
        ]
    )
    frames = []
    for day, level in ((7, day7), (14, day14)):
        control = ens_control_prefix(day=day)
        blend = blend_arm_name(product="aifs_single", day=day)
        means = {
            f"aifs_single_day{day}": level["aifs"],
            control: level["control"],
            f"ens_mean_day{day}": level["mean"],
            blend: level["blend"],
            f"{blend}_control": level["blend_control"],
            f"aifs_single_day{day}{NO_DOY_SUFFIX}": level["aifs"],
            f"{control}{NO_DOY_SUFFIX}": level["control"],
        }
        if day == 14:
            means[shuffled_prefix(source="aifs_single_day14")] = permuted_14
            means[shuffled_prefix(source="aifs_single_day14", variant="_b")] = permuted_14
        for setting in ("primary", "sensitivity"):
            for arm, mean in means.items():
                shifted = (
                    sensitivity_aifs_7
                    if setting == "sensitivity"
                    and sensitivity_aifs_7 is not None
                    and arm == "aifs_single_day7"
                    else mean
                )
                if setting == "sensitivity" and arm.endswith(NO_DOY_SUFFIX):
                    continue
                frames.extend(
                    keys.with_columns(
                        arm=pl.lit(arm),
                        setting=pl.lit(setting),
                        seed=pl.lit(seed),
                        fold=pl.lit(0),
                        signed_error_capped_mw=pl.lit(0.0),
                        **{METRIC: pl.Series(shifted + rng.normal(0.0, 0.002, keys.height))},
                    )
                    for seed in range(3)
                )
    return pl.concat(frames)


_WIN = _levels(aifs=0.08, control=0.12, mean=0.10, blend=0.07, blend_control=0.11)
_DAY14_WIN = _levels(aifs=0.10, control=0.14, mean=0.13, blend=0.09, blend_control=0.12)


@pytest.fixture
def few_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 50)


def test_the_title_states_a_lower_error_and_a_blend_that_lowers_it(few_resamples: None):
    losses = _single_losses(day7=_WIN, day14=_DAY14_WIN, permuted_14=0.16)

    title = charts.aifs_leads_title(losses=losses, domain="solar")

    assert title == (
        "For the six solar farms, at day 7 AIFS Single has a lower error than ENS's control member "
        "and at day 14 AIFS Single has a lower error than ENS's control member. For a blend of "
        "ENS's mean and AIFS Single, at day 7 the blend lowers the error and at day 14 the blend "
        "lowers the error. The day-14 reading rule finds skill to compare at day 14."
    )


def test_the_title_replaces_the_day_14_reading_when_the_rule_finds_no_skill(few_resamples: None):
    losses = _single_losses(day7=_WIN, day14=_DAY14_WIN, permuted_14=0.10)

    title = charts.aifs_leads_title(losses=losses, domain="wind")

    assert "at day 14 there is no skill to compare" in title
    assert "at day 14 AIFS Single" not in title
    assert title.endswith("The day-14 reading rule finds no skill to compare at day 14.")


def test_the_title_says_higher_raises_and_cannot_be_told_apart(few_resamples: None):
    higher = _levels(aifs=0.14, control=0.12, mean=0.10, blend=0.14, blend_control=0.13)
    same = _levels(aifs=0.12, control=0.12, mean=0.12, blend=0.12, blend_control=0.12)
    losses = _single_losses(day7=higher, day14=same, permuted_14=0.30)

    title = charts.aifs_leads_title(losses=losses, domain="solar")

    assert "at day 7 AIFS Single has a higher error than ENS's control member" in title
    assert "at day 7 the blend raises the error" in title
    assert "at day 14 AIFS Single cannot be told apart from ENS's control member" in title
    assert "at day 14 the blend shows no detectable difference" in title


def test_the_title_calls_a_lower_error_than_the_control_but_not_the_mean_smoothing(
    few_resamples: None,
):
    smooth = _levels(aifs=0.10, control=0.12, mean=0.10, blend=0.10, blend_control=0.10)
    losses = _single_losses(day7=smooth, day14=_DAY14_WIN, permuted_14=0.16)

    title = charts.aifs_leads_title(losses=losses, domain="solar")

    assert (
        "at day 7 AIFS Single has a lower error than ENS's control member, which is consistent "
        "with smoothing: it is not lower than the ENS mean's"
    ) in title


def test_the_title_calls_a_reading_between_the_settings_unresolved(few_resamples: None):
    losses = _single_losses(day7=_WIN, day14=_DAY14_WIN, permuted_14=0.16, sensitivity_aifs_7=0.12)

    title = charts.aifs_leads_title(losses=losses, domain="solar")

    assert "at day 7 AIFS Single is unresolved against ENS's control member" in title


def test_the_title_marks_a_reading_the_claim_rule_refuses(
    few_resamples: None, monkeypatch: pytest.MonkeyPatch
):
    losses = _single_losses(day7=_WIN, day14=_DAY14_WIN, permuted_14=0.16)
    monkeypatch.setattr(charts, "leave_one_month_out", lambda **_: (-0.04, -0.05, 0.01, False))

    title = charts.aifs_leads_title(losses=losses, domain="solar")

    assert (
        "at day 7 AIFS Single has a lower error than ENS's control member (not claimable)" in title
    )


def test_a_stage_with_nothing_to_refit_at_the_second_setting_returns_the_primary_fits(
    stub_fit: None, monkeypatch: pytest.MonkeyPatch
):
    arms = list(blend_arms(row_set="ens", day=1))
    frame = _frame_with(arms=arms, domain="solar")
    monkeypatch.setattr(fit_aifs, "blend_sensitivity_arms", lambda **_: [])

    losses = fit_aifs.fit_blend_stage(frame=frame, domain="solar", row_set="ens", day=1, workers=1)

    assert set(losses["setting"]) == {"primary"}
    assert set(losses["arm"]) == set(arms)
    assert set(losses["device"]) == {"cuda"}


def test_a_blend_whose_sign_flips_when_one_month_is_dropped_is_not_claimable(
    few_resamples: None, monkeypatch: pytest.MonkeyPatch
):
    losses = _single_losses(day7=_WIN, day14=_DAY14_WIN, permuted_14=0.16)
    monkeypatch.setattr(fit_aifs, "leave_one_month_out", lambda **_: (-0.03, -0.04, 0.01, False))

    text = "\n".join(fit_aifs.deciding_blend_lines(losses=losses, domain="solar", day=7))

    assert "Verdict: not claimable: dropping one month changes the sign of" in text
    assert "Verdict: lowers" not in text


def test_a_blend_that_lowers_the_error_with_stable_months_is_claimed(few_resamples: None):
    losses = _single_losses(day7=_WIN, day14=_DAY14_WIN, permuted_14=0.16)

    text = "\n".join(fit_aifs.deciding_blend_lines(losses=losses, domain="solar", day=7))

    assert "Verdict: lowers the error at day 7" in text


def test_the_report_gates_h14_by_the_reading_rule(few_resamples: None):
    rng = np.random.default_rng(0)
    frames, losses = {}, {}
    for day in BLEND_DAYS:
        frames[day], losses[day] = _synthetic_stage(
            row_set="single", day=day, domain="solar", rng=rng
        )

    report = "\n".join(
        fit_aifs.blend_set_lines(domain="solar", row_set="single", frames=frames, losses=losses)
    )

    assert "Rule: skill exists at day 14 only if" in report
    assert (
        (
            "H14 verdict under the reading rule: no skill to compare at day 14; the deciding "
            "verdict above is not read."
        )
        in report
        or "H14 verdict under the reading rule: the deciding verdict above stands." in report
    )


def test_the_p4_report_lists_the_controls_significantly_worse_than_ens(few_resamples: None):
    losses = _p4_losses(means=_p4_means(second_control=0.14))
    frame = losses.filter(pl.col("setting") == "primary").select("site", "month").unique()

    report = "\n".join(fit_aifs.p4_lines(domain="solar", frame=frame, losses=losses))

    assert "(their guards are uninformative): blend_p4a_control_b at primary" in report
    assert "Settings agree within each seed:" in report

    clean = _p4_losses(means=_p4_means(second_control=0.08))
    frame = clean.filter(pl.col("setting") == "primary").select("site", "month").unique()
    text = "\n".join(fit_aifs.p4_lines(domain="solar", frame=frame, losses=clean))
    assert "(their guards are uninformative): none at either setting." in text


def test_the_day_1_and_day_2_columns_must_equal_the_existing_build(tmp_path: Path):
    stamp = datetime(2025, 3, 1, tzinfo=UTC)
    base = pl.DataFrame(
        {
            "site": ["A", "A"],
            "time": [1, 2],
            "aifs_single_day1_ghi": [1.0, 2.0],
            "aifs_single_day1_init_time": [stamp, stamp],
        }
    )
    base.write_parquet(tmp_path / "solar_aifs_inputs.parquet")

    fit_aifs.check_equal_to_existing(built=base, existing_dir=tmp_path, domain="solar")
    with pytest.raises(ValueError, match="differ"):
        fit_aifs.check_equal_to_existing(
            built=base.with_columns(aifs_single_day1_ghi=pl.Series([1.0, 2.5])),
            existing_dir=tmp_path,
            domain="solar",
        )
    with pytest.raises(ValueError, match="run stamps differ"):
        fit_aifs.check_equal_to_existing(
            built=base.with_columns(
                aifs_single_day1_init_time=pl.Series([stamp, stamp + timedelta(days=1)])
            ),
            existing_dir=tmp_path,
            domain="solar",
        )


def test_the_environment_stamp_names_the_gpu_without_its_serial_and_the_library_versions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        fit_aifs.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(
            [], 0, stdout="GPU 0: NVIDIA RTX A6000 (UUID: GPU-abc)\n"
        ),
    )

    stamp = fit_aifs.environment_stamp()

    assert stamp["gpu"] == "GPU 0: NVIDIA RTX A6000"
    assert stamp["xgboost"] == xgboost.__version__
    assert stamp["polars"] == pl.__version__

    def missing(*_a: object, **_k: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(fit_aifs.subprocess, "run", missing)
    assert fit_aifs.environment_stamp()["gpu"] == "unavailable"


def test_the_printed_interval_count_is_the_plans_contrast_lists_count():
    # 38 listed contrasts per technology across both row sets and four days.
    assert fit_aifs.count_listed_intervals() == 2 * 38
