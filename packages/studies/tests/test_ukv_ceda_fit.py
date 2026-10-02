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
import fit_aifs  # noqa: E402
import fit_ukv_ceda_blends as fit  # noqa: E402
from fit_ukv_ceda_blends import Stage  # noqa: E402
from nwp_forecast_comparison import METRIC, DomainType  # noqa: E402
from studies.bootstrap import BootstrapInterval  # noqa: E402
from studies.ifs_single_runs import served_init_time  # noqa: E402

PRIMARY = fit.PRIMARY
SENSITIVITY = fit.SENSITIVITY


def _interval(*, difference: float, lower: float, upper: float) -> BootstrapInterval:
    return {
        "difference": difference,
        "lower_95": lower,
        "upper_95": upper,
        "seed_spread": 0.0,
        "n_rows": 10,
        "n_months": 8,
    }


# --- arms and jobs --------------------------------------------------------------------------------


def test_a_stage_fits_the_padded_reference_the_blend_and_two_controls_at_both_settings():
    arms = fit.stage_arms(day=3)

    assert arms == (
        "blend_ukv_ceda_day3_pad",
        "blend_ukv_ceda_day3",
        "blend_ukv_ceda_day3_control",
        "blend_ukv_ceda_day3_control_b",
    )
    assert fit.planned_jobs(day=3) == [(arm, s) for arm in arms for s in (PRIMARY, SENSITIVITY)]


def test_there_is_a_stage_for_each_technology_and_lead_day_one_to_four():
    assert fit.stages() == [
        Stage(domain, day) for domain in ("solar", "wind") for day in (1, 2, 3, 4)
    ]


def test_the_reading_rule_corrects_p1_across_eight_intervals():
    assert fit.N_P1_INTERVALS == 8
    assert pytest.approx(99.375) == fit.BONFERRONI_LEVEL


# --- rows -----------------------------------------------------------------------------------------


def _months() -> list[str]:
    months = []
    year, month = 2024, 12
    while (year, month) <= (2026, 9):
        if (year, month) != (2026, 1):
            months.append(f"{year}-{month:02d}")
        year, month = (year, month + 1) if month < 12 else (year + 1, 1)
    return months


def _candidates(*, domain: DomainType) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    times = [
        datetime(int(m[:4]), int(m[5:]), day, hour, tzinfo=UTC)
        for m in _months()
        for day in (3, 6, 9, 12, 15, 18, 21, 24)
        for hour in (9, 11, 13, 15)
    ]
    sites = ["A", "B"] if domain == "solar" else ["W1", "W2"]
    frame = pl.DataFrame({"site": [s for s in sites for _ in times], "time": times * len(sites)})
    n = frame.height
    columns: dict[str, pl.Series] = {
        "month": frame["time"].dt.strftime("%Y-%m"),
        "power_mw": pl.Series(rng.uniform(0, 10, n), dtype=pl.Float32),
        "cap_mw": pl.Series([10.0] * n),
        "constrained": pl.Series([False] * n),
        "effective_capacity_mw": pl.Series([10.0] * n, dtype=pl.Float32),
        "hour_of_day": frame["time"].dt.hour().cast(pl.Int8),
        "day_of_year": frame["time"].dt.ordinal_day().cast(pl.Int16),
    }
    if domain == "solar":
        columns["solar_elevation_deg"] = pl.Series(rng.uniform(5, 60, n))
        columns["solar_azimuth_deg"] = pl.Series(rng.uniform(90, 270, n))
    for day in build.LEAD_DAYS:
        for column in build.ens_columns(domain=domain, day=day):
            columns[column] = pl.Series(rng.uniform(0, 1, n))
    return frame.with_columns(**columns)


def _inputs(*, candidates: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    rng = np.random.default_rng(4)
    n = candidates.height
    columns: dict[str, pl.Series] = {}
    for day in build.LEAD_DAYS:
        for field in build.WEATHER_FIELDS[domain]:
            columns[f"ukv_ceda_day{day}_{field}"] = pl.Series(rng.uniform(0, 1, n))
    frame = candidates.select("site", "time").with_columns(**columns)
    return frame.with_columns(
        **{
            f"ukv_ceda_day{day}_init_time": served_init_time(
                time=pl.col("time"), day=day, domain=domain, run_hour=3
            )
            for day in build.LEAD_DAYS
        }
    )


@pytest.fixture(scope="module")
def wind_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    candidates = _candidates(domain="wind")
    return candidates, _inputs(candidates=candidates, domain="wind")


def test_a_stage_frame_carries_folds_copies_and_both_shuffles_of_ukv_ceda(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data

    frame, offsets = fit.stage_frame(stage=Stage("wind", 2), candidates=candidates, inputs=inputs)

    assert set(offsets) == {0, 1, 2}
    assert {"era_code", "fold", "month"} <= set(frame.columns)
    assert frame.height == candidates.height
    for field in ("speed_100m", "sin_100m", "cos_100m", "speed_10m"):
        assert frame[f"ens_mean_day2_copy_{field}"].equals(frame[f"ens_mean_day2_{field}"])
    for variant in ("_permuted", "_permuted_b"):
        for field in build.WEATHER_FIELDS["wind"]:
            assert f"ukv_ceda_day2{variant}_{field}" in frame.columns
    for arm in fit.stage_arms(day=2):
        assert set(fit_aifs.arm_features(arm=arm, domain="wind")) <= set(frame.columns)


def test_a_row_without_ukv_ceda_columns_is_dropped_from_every_arm(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data
    holed = inputs.with_columns(
        ukv_ceda_day1_speed_10m=pl.when(pl.col("time").dt.day() == 3)
        .then(None)
        .otherwise(pl.col("ukv_ceda_day1_speed_10m")),
        ukv_ceda_day1_sin_10m=pl.when(pl.col("time").dt.day() == 6)
        .then(float("nan"))
        .otherwise(pl.col("ukv_ceda_day1_sin_10m")),
    )

    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=holed)

    days = set(frame["time"].dt.day().to_list())
    assert 3 not in days
    assert 6 not in days
    assert {9, 12} <= days
    other, _ = fit.stage_frame(stage=Stage("wind", 2), candidates=candidates, inputs=holed)
    assert 3 in set(other["time"].dt.day().to_list())


def test_a_row_without_the_target_or_ens_is_dropped(wind_data: tuple[pl.DataFrame, pl.DataFrame]):
    candidates, inputs = wind_data
    holed = candidates.with_columns(
        power_mw=pl.when(pl.col("time").dt.day() == 3).then(None).otherwise(pl.col("power_mw")),
        ens_mean_day1_speed_10m=pl.when(pl.col("time").dt.day() == 6)
        .then(None)
        .otherwise(pl.col("ens_mean_day1_speed_10m")),
    )

    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=holed, inputs=inputs)

    assert not {3, 6} & set(frame["time"].dt.day().to_list())


def test_a_candidate_row_missing_from_the_inputs_stops_the_stage(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data

    with pytest.raises(ValueError, match="missing from the UKV-CEDA inputs"):
        fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs.head(10))


def test_a_row_that_read_another_run_stops_the_stage(wind_data: tuple[pl.DataFrame, pl.DataFrame]):
    candidates, inputs = wind_data
    wrong = inputs.with_columns(
        ukv_ceda_day1_init_time=pl.col("ukv_ceda_day1_init_time") - pl.duration(days=1)
    )

    with pytest.raises(ValueError, match="read a run other than the 03 UTC run"):
        fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=wrong)


def test_the_stamp_must_be_the_03_utc_run_and_not_the_00_utc_run(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data
    midnight = inputs.with_columns(
        ukv_ceda_day1_init_time=pl.col("ukv_ceda_day1_init_time") - pl.duration(hours=3)
    )

    with pytest.raises(ValueError, match="03 UTC"):
        fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=midnight)


def test_the_shared_fold_design_is_used_when_it_covers_every_calendar_month(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, _ = wind_data

    _, offsets = fit.cut_folds(frame=candidates)

    assert offsets == {0: 0, 1: 0, 2: 3}


def test_a_stage_whose_rows_leave_a_month_uncovered_takes_the_first_searched_rotation(
    wind_data: tuple[pl.DataFrame, pl.DataFrame], monkeypatch: pytest.MonkeyPatch
):
    candidates, _ = wind_data
    calls = iter([pl.DataFrame({"x": [1]}), pl.DataFrame({"x": [1]})])
    monkeypatch.setattr(fit, "uncovered_months", lambda **_: next(calls))
    monkeypatch.setattr(
        fit, "search_fold_offsets", lambda **_: [{0: 0, 1: 2, 2: 4}, {0: 0, 1: 1, 2: 1}]
    )

    cut, offsets = fit.cut_folds(frame=candidates)

    assert offsets == {0: 0, 1: 2, 2: 4}
    assert "fold" in cut.columns


def test_a_stage_with_no_covering_rotation_stops(
    wind_data: tuple[pl.DataFrame, pl.DataFrame], monkeypatch: pytest.MonkeyPatch
):
    candidates, _ = wind_data
    monkeypatch.setattr(fit, "uncovered_months", lambda **_: pl.DataFrame({"x": [1]}))
    monkeypatch.setattr(fit, "search_fold_offsets", lambda **_: [])

    with pytest.raises(ValueError, match="no fold rotation"):
        fit.cut_folds(frame=candidates)


def test_the_copy_columns_equal_ens_for_solar_and_wind():
    solar = _candidates(domain="solar").head(5)

    padded = fit.add_copy_columns(frame=solar, domain="solar", day=3)

    assert padded["ens_mean_day3_copy_ghi"].equals(solar["ens_mean_day3_ghi"])
    assert padded["ens_mean_day3_copy_temp"].equals(solar["ens_mean_day3_temp"])
    assert padded.width == solar.width + 2


def test_an_arm_with_a_missing_column_or_the_wrong_count_is_refused(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data
    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)

    fit.check_arm_columns(frame=frame, domain="wind", arms=fit.stage_arms(day=1))
    with pytest.raises(ValueError, match="has no columns"):
        fit.check_arm_columns(
            frame=frame.drop("ens_mean_day1_copy_sin_100m"),
            domain="wind",
            arms=fit.stage_arms(day=1),
        )


# --- saved files ----------------------------------------------------------------------------------


def _losses(*, arms: list[str], settings: list[str], frame: pl.DataFrame) -> pl.DataFrame:
    keys = frame.select("site", "time", "month", "fold").head(6)
    return pl.concat(
        keys.with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            seed=pl.lit(seed),
            signed_error_capped_mw=pl.lit(0.5),
            absolute_error_mw=pl.lit(0.5),
            **{METRIC: pl.lit(0.05)},
        )
        for arm in arms
        for setting in settings
        for seed in (0, 1, 2)
    )


@pytest.fixture
def tiny_stage(wind_data: tuple[pl.DataFrame, pl.DataFrame]) -> fit.PlannedStage:
    candidates, inputs = wind_data
    frame, offsets = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)
    tiny = frame.sort("site", "time").head(6)
    return fit.PlannedStage(
        stage=Stage("wind", 1), frame=tiny, offsets=offsets, stamp={"build": "1", "device": "cuda"}
    )


def test_a_stage_with_no_saved_file_fits_every_pair_and_a_complete_one_fits_none(
    tmp_path: Path, tiny_stage: fit.PlannedStage
):
    stage = tiny_stage.stage

    assert fit.jobs_to_fit(output_dir=tmp_path, stage=stage, only_missing=False) == (
        "planned",
        fit.planned_jobs(day=1),
    )
    all_arms = list(fit.stage_arms(day=1))
    _losses(arms=all_arms, settings=[PRIMARY, SENSITIVITY], frame=tiny_stage.frame).write_parquet(
        tmp_path / "wind_day1_planned_losses.parquet"
    )
    assert fit.jobs_to_fit(output_dir=tmp_path, stage=stage, only_missing=False) == (
        "added_1",
        [],
    )


def test_a_partly_saved_stage_needs_only_missing_and_then_fits_just_the_absent_pairs(
    tmp_path: Path, tiny_stage: fit.PlannedStage
):
    stage = tiny_stage.stage
    arms = list(fit.stage_arms(day=1))
    _losses(arms=arms[:3], settings=[PRIMARY, SENSITIVITY], frame=tiny_stage.frame).write_parquet(
        tmp_path / "wind_day1_planned_losses.parquet"
    )

    with pytest.raises(ValueError, match="--only-missing"):
        fit.jobs_to_fit(output_dir=tmp_path, stage=stage, only_missing=False)
    group, jobs = fit.jobs_to_fit(output_dir=tmp_path, stage=stage, only_missing=True)

    assert group == "added_1"
    assert jobs == [(arms[3], PRIMARY), (arms[3], SENSITIVITY)]


def test_the_cpu_refit_file_is_not_counted_as_a_saved_fit(
    tmp_path: Path, tiny_stage: fit.PlannedStage
):
    _losses(arms=["blend_ukv_ceda_day1"], settings=[PRIMARY], frame=tiny_stage.frame).write_parquet(
        tmp_path / "wind_day1_cpu_losses.parquet"
    )

    assert fit.group_files(output_dir=tmp_path, stage=tiny_stage.stage) == []
    assert fit.saved_pairs(output_dir=tmp_path, stage=tiny_stage.stage) == set()


@pytest.fixture
def stub_fit(
    monkeypatch: pytest.MonkeyPatch, tiny_stage: fit.PlannedStage
) -> list[list[fit_aifs.Job]]:
    calls: list[list[fit_aifs.Job]] = []

    def fake(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        calls.append(jobs)
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake)
    return calls


def test_a_stage_is_fitted_once_written_atomically_and_reread_on_a_rerun(
    tmp_path: Path, tiny_stage: fit.PlannedStage, stub_fit: list[list[fit_aifs.Job]]
):
    first = fit.fit_stage(planned=tiny_stage, output_dir=tmp_path, workers=1, only_missing=False)
    second = fit.fit_stage(planned=tiny_stage, output_dir=tmp_path, workers=1, only_missing=False)

    assert len(stub_fit) == 1
    assert first.sort("arm", "setting", "site", "time", "seed").equals(
        second.sort("arm", "setting", "site", "time", "seed")
    )
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [
        "wind_day1_planned_losses.json",
        "wind_day1_planned_losses.parquet",
        "wind_day1_planned_predictions.parquet",
    ]
    predictions = pl.read_parquet(tmp_path / "wind_day1_planned_predictions.parquet")
    assert {"actual_mw", "prediction_capped_mw", "fold", "site", "arm", "setting"} <= set(
        predictions.columns
    )
    assert predictions.height == first.height
    assert set(predictions["site"]) <= {"W1", "W2"}


def test_a_saved_stage_from_another_build_is_refused(
    tmp_path: Path, tiny_stage: fit.PlannedStage, stub_fit: list[list[fit_aifs.Job]]
):
    fit.fit_stage(planned=tiny_stage, output_dir=tmp_path, workers=1, only_missing=False)
    other = tiny_stage._replace(stamp={**tiny_stage.stamp, "build": "2"})

    with pytest.raises(ValueError, match="another build or device"):
        fit.fit_stage(planned=other, output_dir=tmp_path, workers=1, only_missing=False)


def test_a_saved_stage_that_scores_other_rows_is_refused(
    tmp_path: Path, tiny_stage: fit.PlannedStage, stub_fit: list[list[fit_aifs.Job]]
):
    fit.fit_stage(planned=tiny_stage, output_dir=tmp_path, workers=1, only_missing=False)
    shifted = tiny_stage._replace(frame=tiny_stage.frame.head(5))

    with pytest.raises(ValueError, match="other rows or folds"):
        fit.fit_stage(planned=shifted, output_dir=tmp_path, workers=1, only_missing=False)


def test_only_missing_adds_a_second_file_and_never_touches_the_first(
    tmp_path: Path, tiny_stage: fit.PlannedStage, stub_fit: list[list[fit_aifs.Job]]
):
    arms = list(fit.stage_arms(day=1))
    first_file = tmp_path / "wind_day1_planned_losses.parquet"
    _losses(arms=arms[:3], settings=[PRIMARY, SENSITIVITY], frame=tiny_stage.frame).with_columns(
        device=pl.lit("cuda")
    ).write_parquet(first_file)
    first_file.with_suffix(".json").write_text(
        json.dumps(
            {
                **tiny_stage.stamp,
                "columns": json.dumps(
                    {a: fit_aifs.arm_features(arm=a, domain="wind") for a in sorted(arms[:3])}
                ),
            }
        )
    )
    before = first_file.read_bytes()

    merged = fit.fit_stage(planned=tiny_stage, output_dir=tmp_path, workers=1, only_missing=True)

    assert first_file.read_bytes() == before
    assert (tmp_path / "wind_day1_added_1_losses.parquet").exists()
    assert stub_fit == [[(arms[3], PRIMARY), (arms[3], SENSITIVITY)]]
    assert set(merged["arm"]) == set(arms)


def test_the_cpu_refit_is_written_once_and_read_back_after_a_stamp_check(
    tmp_path: Path, tiny_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def fake(*, site_rows: pl.DataFrame, device: str, **_: object) -> pl.DataFrame:
        calls.append(device)
        return _losses(arms=["x"], settings=["y"], frame=site_rows).drop("arm", "setting")

    monkeypatch.setattr(fit, "out_of_fold_losses", fake)
    gpu = _losses(arms=["blend_ukv_ceda_day1"], settings=[PRIMARY], frame=tiny_stage.frame)

    first = fit.cpu_noise_floor(planned=tiny_stage, output_dir=tmp_path, gpu=gpu)
    again = fit.cpu_noise_floor(planned=tiny_stage, output_dir=tmp_path, gpu=gpu)

    assert calls == ["cpu"]
    assert first.equals(again)
    assert set(first["arm"]) == {"blend_ukv_ceda_day1"}
    stamp = json.loads((tmp_path / "wind_day1_cpu_losses.json").read_text())
    assert stamp["device"] == "cpu"


def test_the_noise_floor_sentence_names_the_gap_between_the_cpu_and_gpu_rows(
    tiny_stage: fit.PlannedStage,
):
    gpu = _losses(arms=["blend_ukv_ceda_day1"], settings=[PRIMARY], frame=tiny_stage.frame)
    cpu = gpu.with_columns(device=pl.lit("cpu"), **{METRIC: pl.col(METRIC) + 0.001}).filter(
        pl.col("site") == fit.CPU_SITE
    )

    line = fit.noise_floor_line(cpu=cpu, gpu=gpu)

    assert "0.1000 points of capacity per row on average" in line
    assert "0.1000 at most" in line


def test_padded_and_unpadded_ens_are_called_identical_only_when_every_row_matches(
    tiny_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    def fake_same(
        *, frame: pl.DataFrame, domain: str, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake_same)
    assert fit.padded_matches_unpadded(frame=tiny_stage.frame, day=1) == (True, 0.0)

    def fake_different(
        *, frame: pl.DataFrame, domain: str, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        losses = fake_same(frame=frame, domain=domain, jobs=jobs, workers=workers)
        return losses.with_columns(
            absolute_error_mw=pl.when(pl.col("arm").str.ends_with("_pad"))
            .then(pl.col("absolute_error_mw") + 0.25)
            .otherwise(pl.col("absolute_error_mw"))
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake_different)
    identical, gap = fit.padded_matches_unpadded(frame=tiny_stage.frame, day=1)
    assert not identical
    assert gap == pytest.approx(0.25)


# --- the build gate -------------------------------------------------------------------------------


def _build_folder(*, folder: Path, passed: bool = True) -> dict[str, object]:
    folder.mkdir(exist_ok=True)
    stamp: dict[str, object] = {
        "snapshot_id": "SNAP",
        "coverage_guard_passed": passed,
        "inputs_sha256": {},
    }
    for domain in ("solar", "wind"):
        path = folder / f"{domain}_ukv_ceda_inputs.parquet"
        pl.DataFrame({"a": [1, 2]}).write_parquet(path)
        stamp["inputs_sha256"][domain] = fit_aifs.sha256_of(path=path)
    (folder / "build.json").write_text(json.dumps(stamp))
    return stamp


def test_a_stage_runs_only_after_a_build_that_passed_its_coverage_guard(tmp_path: Path):
    _build_folder(folder=tmp_path, passed=True)
    assert fit.read_build_stamp(output_dir=tmp_path)["snapshot_id"] == "SNAP"

    _build_folder(folder=tmp_path, passed=False)
    with pytest.raises(ValueError, match="did not pass its coverage guard"):
        fit.read_build_stamp(output_dir=tmp_path)


def test_an_inputs_file_changed_after_the_build_is_refused(tmp_path: Path):
    _build_folder(folder=tmp_path)
    pl.DataFrame({"a": [3]}).write_parquet(tmp_path / "wind_ukv_ceda_inputs.parquet")

    with pytest.raises(ValueError, match=r"not the one build\.json recorded"):
        fit.read_build_stamp(output_dir=tmp_path)


def test_no_fit_may_write_into_a_folder_it_reads(tmp_path: Path):
    reads = [tmp_path / "published", tmp_path / "day4"]

    fit.check_output_dir(output_dir=tmp_path / "ukv_ceda_blends", read_only=reads)
    with pytest.raises(ValueError, match="writes only to a folder named"):
        fit.check_output_dir(output_dir=reads[0], read_only=reads)


def test_more_than_two_workers_is_refused_under_the_load_rule():
    import argparse

    assert fit.workers_argument("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="at most 2"):
        fit.workers_argument("3")


# --- the reading rule -----------------------------------------------------------------------------

LOWER = _interval(difference=-0.5, lower=-0.8, upper=-0.2)
SAME = _interval(difference=-0.1, lower=-0.4, upper=0.2)
HIGHER = _interval(difference=0.5, lower=0.2, upper=0.8)


def test_the_blend_lowers_the_error_only_if_p1_and_both_p2_seeds_are_below_zero():
    assert fit.setting_verdict(day=2, p1=LOWER, p2=[LOWER, LOWER])["verdict"] == (
        "lowers the error at day 2"
    )
    assert fit.setting_verdict(day=2, p1=LOWER, p2=[LOWER, SAME])["verdict"] == (
        "no detectable difference"
    )
    assert fit.setting_verdict(day=2, p1=SAME, p2=[LOWER, LOWER])["verdict"] == (
        "no detectable difference"
    )


def test_a_blend_significantly_worse_than_padded_ens_raises_the_error():
    verdict = fit.setting_verdict(day=4, p1=HIGHER, p2=[LOWER, LOWER])

    assert verdict["verdict"] == "raises the error at day 4"
    assert verdict["largest_gain_not_excluded"] is None


def test_an_undetected_difference_names_the_largest_gain_p1_leaves_open():
    verdict = fit.setting_verdict(day=1, p1=SAME, p2=[SAME, SAME])

    assert verdict["largest_gain_not_excluded"] == pytest.approx(0.4)


def test_a_verdict_stands_only_if_both_settings_give_it():
    both = {PRIMARY: LOWER, SENSITIVITY: LOWER}
    split = {PRIMARY: LOWER, SENSITIVITY: SAME}
    p2 = {PRIMARY: [LOWER, LOWER], SENSITIVITY: [LOWER, LOWER]}

    assert fit.reading(day=3, p1=both, p2=p2) == "lowers the error at day 3"
    assert fit.reading(day=3, p1=split, p2=p2) == "no detectable difference"


def test_a_null_names_the_gain_the_interval_does_not_exclude_in_points_of_capacity():
    sentence = fit.null_reading(interval=_interval(difference=0.0, lower=-0.0021, upper=0.001))

    assert "an effect as large as 0.210 points of capacity is not excluded" in sentence
    assert "larger than 0.210 points is excluded" in sentence
    assert "below zero" in fit.null_reading(interval=LOWER)
    assert "above zero" in fit.null_reading(interval=HIGHER)


# --- report ---------------------------------------------------------------------------------------


@pytest.fixture
def few_resamples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 40)


def _report_stage() -> tuple[fit.PlannedStage, pl.DataFrame]:
    rng = np.random.default_rng(9)
    months = [f"2025-{m:02d}" for m in range(1, 11)]
    frame = pl.DataFrame(
        [
            {
                "site": site,
                "month": month,
                "time": datetime(2025, 1 + m, 1 + d, 12, tzinfo=UTC),
                "era_code": 0 if m < 5 else 1 if m < 7 else 2,
                "fold": m % 5,
            }
            for site in ("W1", "W2")
            for m, month in enumerate(months)
            for d in range(6)
        ]
    ).with_columns(era_code=pl.col("era_code").cast(pl.Int8), power_mw=pl.lit(1.0))
    stage = fit.PlannedStage(Stage("wind", 2), frame, {0: 0, 1: 0, 2: 3}, {})
    levels = {
        "blend_ukv_ceda_day2_pad": 0.10,
        "blend_ukv_ceda_day2": 0.09,
        "blend_ukv_ceda_day2_control": 0.11,
        "blend_ukv_ceda_day2_control_b": 0.11,
    }
    losses = pl.concat(
        frame.with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            seed=pl.lit(seed),
            signed_error_capped_mw=pl.lit(0.0),
            **{METRIC: pl.Series(level + rng.normal(0.0, 0.002, frame.height))},
        )
        for arm, level in levels.items()
        for setting in (PRIMARY, SENSITIVITY)
        for seed in range(3)
    )
    return stage, losses


def test_the_stage_section_reads_both_settings_and_prints_every_planned_and_exploratory_interval(
    few_resamples: None,
):
    stage, losses = _report_stage()
    records: list[fit.IntervalRecord] = []

    lines, reading = fit.stage_lines(planned=stage, losses=losses, records=records)

    text = "\n".join(lines)
    assert reading == "lowers the error at day 2"
    assert "### Wind, lead day 2" in text
    assert "P1: blend minus padded ENS" in text
    assert "P2: blend minus control" in text
    assert "P2: blend minus second-seed control" in text
    assert "Bonferroni 99.375%" in text
    assert "E4 before the upgrade (eras 0 and 1)" in text
    assert "E4 era 1" in text
    assert "E5 generator W1" in text
    assert "E5 generator W2" in text
    contrasts = {(r["contrast"], r["setting"], r["scope"]) for r in records}
    assert ("p1", PRIMARY, "all rows") in contrasts
    assert ("p2b", SENSITIVITY, "all rows") in contrasts
    assert ("p1", PRIMARY, "E5 generator W1") in contrasts
    assert all(r["domain"] == "wind" and r["day"] == 2 for r in records)


def test_an_era_of_fewer_than_six_months_gets_a_point_estimate_and_no_interval(few_resamples: None):
    stage, losses = _report_stage()
    records: list[fit.IntervalRecord] = []

    lines, _ = fit.stage_lines(planned=stage, losses=losses, records=records)

    era_two = next(line for line in lines if line.startswith("| E4 era 2"))
    assert "no interval: 3 months" in era_two
    assert "no interval" in next(line for line in lines if line.startswith("| E4 era 1"))
    assert "no interval" not in next(
        line for line in lines if line.startswith("| E4 before the upgrade")
    )
    assert not any(r["scope"] == "E4 era 2" for r in records)
    assert any(r["scope"] == "E4 before the upgrade (eras 0 and 1)" for r in records)


def test_the_report_and_interval_files_have_a_default_name_and_a_new_name_for_a_rerun(
    tmp_path: Path,
):
    assert fit.report_paths(output_dir=tmp_path, name="report") == (
        tmp_path / "report.md",
        tmp_path / "intervals.parquet",
    )
    assert fit.report_paths(output_dir=tmp_path, name="report_2") == (
        tmp_path / "report_2.md",
        tmp_path / "report_2_intervals.parquet",
    )


def test_the_report_lists_every_arms_columns_at_both_technologies():
    lines = "\n".join(fit.columns_lines())

    assert "`blend_ukv_ceda_day1_pad` (9 columns)" in lines
    assert "`blend_ukv_ceda_day1_control_b` (11 columns)" in lines
    assert "ukv_ceda_day1_speed_925hpa" in lines
    assert "ens_mean_day1_copy_ghi" in lines


def test_the_report_text_states_the_reading_rule_and_the_lead_gap():
    text = fit.report_text(
        sections=[("Wind", ["### Wind, lead day 1"])],
        summary=fit.summary_lines(readings={("wind", 1): "no detectable difference"}),
        cpu_line="CPU line.",
    )

    assert "| wind | 1 | no detectable difference |" in text
    assert "upper 95% bound of P1 and of both P2 contrasts is below zero at both" in text
    assert "3 hours fresher than ENS's at every hour, which favours the blend" in text
    assert "native 10 m and 925 hPa winds" in text
    assert "CPU line." in text


def test_check_init_times_accepts_the_03_utc_run_for_solar_and_wind():
    for domain in ("solar", "wind"):
        frame = _candidates(domain=domain).head(20)
        stamped = frame.select("site", "time").with_columns(
            ukv_ceda_day2_init_time=served_init_time(
                time=pl.col("time"), day=2, domain=domain, run_hour=3
            )
        )

        fit.check_init_times(frame=stamped, domain=domain, day=2)


def test_the_report_is_built_from_saved_losses_alone_and_its_intervals_are_saved(
    tmp_path: Path, few_resamples: None, monkeypatch: pytest.MonkeyPatch
):
    stage, losses = _report_stage()
    calls: list[int] = []

    def fake(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        calls.append(len(jobs))
        wanted = {(arm, setting) for arm, setting in jobs}
        return losses.filter(
            pl.struct("arm", "setting").map_elements(
                lambda row: (row["arm"], row["setting"]) in wanted, return_dtype=pl.Boolean
            )
        ).drop("era_code")

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake)
    planned = stage._replace(stamp={"build": "1"})
    fit.fit_stage(planned=planned, output_dir=tmp_path, workers=1, only_missing=False)

    text, intervals = fit.build_report(planned=[planned], output_dir=tmp_path)

    assert calls == [8]
    assert "## Wind" in text
    assert "### Wind, lead day 2" in text
    assert "| wind | 2 | lowers the error at day 2 |" in text
    assert "No CPU refit is saved." in text
    assert set(intervals.columns) >= {"domain", "day", "setting", "contrast", "lower", "upper"}
    assert intervals.filter(pl.col("contrast") == "p1", pl.col("scope") == "all rows").height == 2


def test_each_arms_error_is_listed_per_generator_in_percent_of_capacity():
    _, losses = _report_stage()
    primary = losses.filter(pl.col("setting") == PRIMARY)

    lines = fit.generator_error_lines(losses=primary, arms=fit.stage_arms(day=2))

    assert lines[2].startswith("| Generator | `blend_ukv_ceda_day2_pad` | `blend_ukv_ceda_day2` |")
    rows = [line for line in lines if line.startswith(("| W1", "| W2"))]
    assert len(rows) == 2
    pad_error = float(rows[0].split("|")[2])
    assert pad_error == pytest.approx(10.0, abs=0.5)
    blend_error = float(rows[0].split("|")[3])
    assert blend_error < pad_error


def test_a_predictions_file_left_without_its_losses_is_never_overwritten(
    tmp_path: Path, tiny_stage: fit.PlannedStage, stub_fit: list[list[fit_aifs.Job]]
):
    leftover = tmp_path / "wind_day1_planned_predictions.parquet"
    leftover.write_bytes(b"evidence")

    with pytest.raises(FileExistsError):
        fit.fit_stage(planned=tiny_stage, output_dir=tmp_path, workers=1, only_missing=False)

    assert leftover.read_bytes() == b"evidence"
    assert stub_fit == []
