import itertools
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


def test_the_noise_floor_sentence_gives_the_gap_between_two_gpu_fitting_seeds(
    tiny_stage: fit.PlannedStage,
):
    gpu = _losses(arms=["blend_ukv_ceda_day1"], settings=[PRIMARY], frame=tiny_stage.frame)
    gpu = gpu.with_columns(**{METRIC: pl.col(METRIC) + 0.01 * pl.col("seed")})
    cpu = gpu.with_columns(device=pl.lit("cpu")).filter(pl.col("site") == fit.CPU_SITE)

    line = fit.noise_floor_line(cpu=cpu, gpu=gpu)

    # Seeds score 5, 6, and 7 points: the pair gaps are 1, 2, and 1.
    assert "differ by 1.3333 points of capacity per row on average and 2.0000 at most" in line
    assert "span 2.0000 points (5.0000 to 7.0000)" in line


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
    assert fit.padded_matches_unpadded(frame=tiny_stage.frame, domain="wind", day=1) == (True, 0.0)

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
    identical, gap = fit.padded_matches_unpadded(frame=tiny_stage.frame, domain="wind", day=1)
    assert not identical
    assert gap == pytest.approx(0.25)


# --- the build gate -------------------------------------------------------------------------------


def _build_folder(*, folder: Path, passed: bool = True) -> dict[str, object]:
    folder.mkdir(exist_ok=True)
    stamp: dict[str, object] = {
        "snapshot_id": "SNAP",
        "run_hour": 3,
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
        fit.UNRESOLVED_LOWER
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

    lines, stage_reading = fit.stage_lines(planned=stage, losses=losses, records=records)

    text = "\n".join(lines)
    assert stage_reading.reading == "lowers the error at day 2"
    assert stage_reading.survives_bonferroni
    assert "### Wind, lead day 2" in text
    assert "P1: blend minus padded ENS" in text
    assert "P2: blend minus control" in text
    assert "P2: blend minus second-seed control" in text
    assert "Primary, Bonferroni 99.375%" in text
    assert "Sensitivity, Bonferroni 99.375%" in text
    assert "P1 survives the Bonferroni correction at both settings: yes." in text
    assert "the gap between the two shuffled controls" in text
    assert "P1 with each calendar month dropped in turn" in text
    assert "E4 before the upgrade (eras 0 and 1)" in text
    assert "E4 era 1" in text
    assert "E5 generator W1" in text
    assert "E5 generator W2" in text
    contrasts = {(r["contrast"], r["setting"], r["scope"]) for r in records}
    assert ("p1", PRIMARY, "all rows") in contrasts
    assert ("p2b", SENSITIVITY, "all rows") in contrasts
    assert ("p1", SENSITIVITY, "Bonferroni") in contrasts
    assert ("control_gap", PRIMARY, "control gap") in contrasts
    assert ("p1", SENSITIVITY, "E6 most influential month dropped") in contrasts
    errors = {r["scope"] for r in records if r["contrast"] == "error"}
    assert errors == set(fit.stage_arms(day=2))
    assert {r["level"] for r in records if r["scope"] == "Bonferroni"} == {fit.BONFERRONI_LEVEL}
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
        summary=fit.summary_lines(
            readings={
                ("wind", 1): fit.StageReading(
                    reading="no detectable difference",
                    survives_bonferroni=False,
                    left_open="primary 0.170, sensitivity 0.120",
                )
            }
        ),
        stale_summary=[],
        cpu_line="CPU line.",
        padding_line="Padding line.",
    )

    assert "| wind | 1 | no detectable difference | no | primary 0.170, sensitivity 0.120 |" in text
    assert "never means no gain" in text
    assert "Padding line." in text
    assert fit.UNRESOLVED_LOWER in text
    assert "upper 95% bound of P1 and of both P2 contrasts is below zero at both" in text
    assert "3 hours fresher than ENS's at every hour, which favours the blend" in text
    assert "native 10 m and 925 hPa winds" in text
    assert "CPU line." in text


def test_check_init_times_reads_the_03_utc_run_of_the_rows_own_day_minus_the_lead_day():
    # A wind label is an instant: 14:00 on 10 March reads the run of 8 March at lead day 2.
    wind = pl.DataFrame(
        {
            "time": [datetime(2026, 3, 10, 14, tzinfo=UTC)],
            "ukv_ceda_day2_init_time": [datetime(2026, 3, 8, 3, tzinfo=UTC)],
        }
    )
    fit.check_init_times(frame=wind, domain="wind", day=2)
    # A solar label names the hour ending at it, so 00:00 on 10 March belongs to 9 March.
    solar = pl.DataFrame(
        {
            "time": [datetime(2026, 3, 10, 0, tzinfo=UTC)],
            "ukv_ceda_day1_init_time": [datetime(2026, 3, 8, 3, tzinfo=UTC)],
        }
    )
    fit.check_init_times(frame=solar, domain="solar", day=1)

    for domain, frame, day in (("wind", wind, 2), ("solar", solar, 1)):
        shifted = frame.with_columns(pl.col(f"ukv_ceda_day{day}_init_time") + pl.duration(days=1))
        with pytest.raises(ValueError, match="read a run other than"):
            fit.check_init_times(frame=shifted, domain=domain, day=day)


def test_a_fit_needs_a_passing_verify_stamp_for_the_inputs_the_build_recorded(tmp_path: Path):
    stamp = _build_folder(folder=tmp_path)
    hashes = stamp["inputs_sha256"]
    with pytest.raises(ValueError, match="absent"):
        fit.check_verified(output_dir=tmp_path, stamp=stamp)

    (tmp_path / "verify.json").write_text(json.dumps({"passed": False, "inputs_sha256": hashes}))
    with pytest.raises(ValueError, match="did not pass"):
        fit.check_verified(output_dir=tmp_path, stamp=stamp)

    (tmp_path / "verify.json").write_text(
        json.dumps({"passed": True, "inputs_sha256": {"solar": "x", "wind": "y"}})
    )
    with pytest.raises(ValueError, match="other inputs"):
        fit.check_verified(output_dir=tmp_path, stamp=stamp)

    (tmp_path / "verify.json").write_text(json.dumps({"passed": True, "inputs_sha256": hashes}))
    fit.check_verified(output_dir=tmp_path, stamp=stamp)


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
    assert "| wind | 2 | lowers the error at day 2 | yes | primary below zero" in text
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


# --- the unresolved reading, the Bonferroni survival, and the bounds left open --------------------


def test_p1_below_zero_with_a_control_bound_touching_zero_is_unresolved_not_no_difference():
    # The solar day-1 pattern: P1 is below zero at both settings, and the second-seed control's
    # upper bound at the primary setting is +0.029.
    p1 = {
        PRIMARY: _interval(difference=-0.105, lower=-0.177, upper=-0.017),
        SENSITIVITY: _interval(difference=-0.091, lower=-0.155, upper=-0.017),
    }
    touching = _interval(difference=-0.064, lower=-0.142, upper=0.029)
    p2 = {
        PRIMARY: [LOWER, touching],
        SENSITIVITY: [LOWER, LOWER],
    }

    assert fit.setting_verdict(day=1, p1=p1[PRIMARY], p2=p2[PRIMARY])["verdict"] == (
        fit.UNRESOLVED_LOWER
    )
    assert fit.reading(day=1, p1=p1, p2=p2) == fit.UNRESOLVED_LOWER
    assert fit.UNRESOLVED_LOWER == "unresolved: lower than padded ENS, control test not passed"


def test_two_settings_that_both_put_p1_below_zero_but_differ_on_the_controls_are_unresolved():
    p1 = {PRIMARY: LOWER, SENSITIVITY: LOWER}
    p2 = {PRIMARY: [LOWER, LOWER], SENSITIVITY: [LOWER, SAME]}

    assert fit.reading(day=2, p1=p1, p2=p2) == fit.UNRESOLVED_LOWER


def test_no_detectable_difference_is_kept_for_a_p1_that_includes_zero_at_a_setting():
    p2 = {PRIMARY: [LOWER, LOWER], SENSITIVITY: [LOWER, LOWER]}

    assert fit.reading(day=4, p1={PRIMARY: LOWER, SENSITIVITY: SAME}, p2=p2) == (
        "no detectable difference"
    )
    assert fit.reading(day=4, p1={PRIMARY: SAME, SENSITIVITY: SAME}, p2=p2) == (
        "no detectable difference"
    )
    verdict = fit.setting_verdict(day=3, p1=LOWER, p2=[SAME, SAME])
    assert verdict["largest_gain_not_excluded"] is None


def test_p1_survives_the_bonferroni_correction_only_if_both_settings_are_below_zero():
    assert fit.survives_bonferroni(wide={PRIMARY: (-0.3, -0.1), SENSITIVITY: (-0.2, -0.01)})
    # Wind day 3 in the first science review: below zero at the primary setting only.
    assert not fit.survives_bonferroni(wide={PRIMARY: (-0.6, -0.02), SENSITIVITY: (-0.46, 0.035)})
    assert not fit.survives_bonferroni(wide={PRIMARY: (-0.3, 0.001), SENSITIVITY: (-0.2, -0.01)})


def test_the_bound_p1_leaves_open_is_stated_at_both_settings_and_never_as_no_gain():
    text = fit.left_open_text(
        p1={
            PRIMARY: _interval(difference=-0.05, lower=-0.0172, upper=0.04),
            SENSITIVITY: _interval(difference=-0.05, lower=-0.0108, upper=0.04),
        }
    )

    assert text == "primary 1.720, sensitivity 1.080"
    assert fit.left_open_text(p1={PRIMARY: LOWER, SENSITIVITY: SAME}) == (
        "primary below zero, sensitivity 40.000"
    )


# --- leave-one-month-out, the control gap, and the saved padding check ----------------------------


def _month_gain_losses(*, gain_month: str, gain: float) -> tuple[fit.PlannedStage, pl.DataFrame]:
    stage, losses = _report_stage()
    month = pl.col("month")
    return stage, losses.with_columns(
        **{
            METRIC: pl.when((pl.col("arm") == "blend_ukv_ceda_day2") & (month == gain_month))
            .then(pl.col(METRIC) - gain)
            .otherwise(pl.col(METRIC))
        }
    )


def test_dropping_the_one_month_that_carries_the_gain_moves_the_estimate_the_most(
    few_resamples: None,
):
    _, losses = _month_gain_losses(gain_month="2025-03", gain=0.5)
    primary = losses.filter(pl.col("setting") == PRIMARY)

    drops = fit.month_drops(
        losses=primary, treatment="blend_ukv_ceda_day2", reference="blend_ukv_ceda_day2_pad"
    )

    assert [drop.month for drop in drops] == sorted(losses["month"].unique().to_list())
    by_month = {drop.month: drop.interval["difference"] for drop in drops}
    assert by_month["2025-03"] == max(by_month.values())
    assert all(value < 0.0 for month, value in by_month.items() if month != "2025-03")


def test_the_leave_one_month_out_table_names_the_month_and_saves_the_influential_drop(
    few_resamples: None,
):
    stage, losses = _month_gain_losses(gain_month="2025-03", gain=0.5)
    per_setting = {s: losses.filter(pl.col("setting") == s) for s in (PRIMARY, SENSITIVITY)}
    full = {
        s: fit.difference(
            losses=per_setting[s],
            treatment="blend_ukv_ceda_day2",
            reference="blend_ukv_ceda_day2_pad",
        )
        for s in per_setting
    }
    records: list[fit.IntervalRecord] = []

    lines = fit.leave_one_month_out_lines(
        per_setting=per_setting,
        full=full,
        treatment="blend_ukv_ceda_day2",
        reference="blend_ukv_ceda_day2_pad",
        stage=stage.stage,
        records=records,
    )

    rows = [line for line in lines if line.startswith(("| primary", "| sensitivity"))]
    assert len(rows) == 2
    assert all("| 2025-03 |" in row for row in rows)
    for row, setting in zip(rows, (PRIMARY, SENSITIVITY), strict=True):
        dropped = {
            drop.month: drop.interval
            for drop in fit.month_drops(
                losses=per_setting[setting],
                treatment="blend_ukv_ceda_day2",
                reference="blend_ukv_ceda_day2_pad",
            )
        }
        loosest = max(dropped, key=lambda month: dropped[month]["upper_95"])
        tightest = min(dropped, key=lambda month: dropped[month]["upper_95"])
        assert loosest != tightest
        highest_upper = f"{dropped[loosest]['upper_95'] * fit.PERCENTAGE_POINTS:+.3f} ({loosest})"
        assert row.endswith(f"| {highest_upper} |")
        saved = next(r for r in records if r["setting"] == setting)
        assert saved["difference"] == dropped["2025-03"]["difference"]
        assert saved["upper"] == dropped["2025-03"]["upper_95"]
    assert {(r["setting"], r["scope"]) for r in records} == {
        (PRIMARY, "E6 most influential month dropped"),
        (SENSITIVITY, "E6 most influential month dropped"),
    }


def test_the_gap_between_the_two_controls_is_printed_at_both_settings_and_flagged_if_significant(
    few_resamples: None,
):
    stage, losses = _report_stage()
    shifted = losses.with_columns(
        **{
            METRIC: pl.when(pl.col("arm") == "blend_ukv_ceda_day2_control_b")
            .then(pl.col(METRIC) - 0.03)
            .otherwise(pl.col(METRIC))
        }
    )
    per_setting = {s: shifted.filter(pl.col("setting") == s) for s in (PRIMARY, SENSITIVITY)}
    records: list[fit.IntervalRecord] = []

    lines = fit.control_gap_lines(
        per_setting=per_setting,
        control="blend_ukv_ceda_day2_control",
        control_b="blend_ukv_ceda_day2_control_b",
        stage=stage.stage,
        records=records,
    )

    rows = [line for line in lines if line.startswith(("| primary", "| sensitivity"))]
    assert len(rows) == 2
    assert all(row.endswith("| yes |") for row in rows)
    assert all("| +3." in row or "| +2.9" in row for row in rows)
    assert {r["contrast"] for r in records} == {"control_gap"}
    assert all(r["difference"] == pytest.approx(0.03, abs=0.002) for r in records)


def test_a_gap_of_zero_between_the_controls_is_not_flagged(few_resamples: None):
    stage, losses = _report_stage()
    per_setting = {s: losses.filter(pl.col("setting") == s) for s in (PRIMARY, SENSITIVITY)}

    lines = fit.control_gap_lines(
        per_setting=per_setting,
        control="blend_ukv_ceda_day2_control",
        control_b="blend_ukv_ceda_day2_control_b",
        stage=stage.stage,
        records=[],
    )

    rows = [line for line in lines if line.startswith(("| primary", "| sensitivity"))]
    assert all(row.endswith("| no |") for row in rows)


def test_the_padding_check_runs_for_wind_and_solar_day_1_and_is_saved_once(
    tmp_path: Path, tiny_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    domains: list[str] = []

    def fake_fit(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        domains.append(domain)
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake_fit)
    monkeypatch.setattr(fit_aifs, "time_two_fits", lambda **_: (True, 2.0))
    solar = tiny_stage._replace(stage=Stage("solar", 1))

    assert fit.run_check(planned=[tiny_stage, solar], output_dir=tmp_path) == 0

    saved = json.loads((tmp_path / fit.PADDING_CHECK_NAME).read_text())
    assert set(saved) == {"wind_day1", "solar_day1"}
    assert saved["solar_day1"] == {
        "identical": True,
        "largest_gap_mw": 0.0,
        "setting": PRIMARY,
    }
    assert domains == ["wind", "solar"]
    with pytest.raises(FileExistsError):
        fit.run_check(planned=[tiny_stage, solar], output_dir=tmp_path)


def test_the_report_prints_the_saved_padding_check_or_says_none_is_saved(tmp_path: Path):
    assert fit.padding_check_line(output_dir=tmp_path) == "No padding check is saved."
    (tmp_path / fit.PADDING_CHECK_NAME).write_text(
        json.dumps(
            {
                "wind_day1": {"identical": True, "largest_gap_mw": 0.0, "setting": PRIMARY},
                "solar_day1": {"identical": False, "largest_gap_mw": 0.25, "setting": PRIMARY},
            }
        )
    )

    line = fit.padding_check_line(output_dir=tmp_path)

    assert "wind day1: identical (largest per-row gap 0 MW)" in line
    assert "solar day1: not identical (largest per-row gap 0.25 MW)" in line


# --- the post hoc stale blend ---------------------------------------------------------------------


def test_the_stale_blend_reads_ukv_ceda_one_day_older_and_keeps_the_planned_column_count():
    pad, blend, control, control_b = fit.stale_arms(day=2)

    assert (pad, blend, control, control_b) == (
        "blend_ukv_ceda_stale_day2_pad",
        "blend_ukv_ceda_stale_day2",
        "blend_ukv_ceda_stale_day2_control",
        "blend_ukv_ceda_stale_day2_control_b",
    )
    wind = fit_aifs.arm_features(arm=blend, domain="wind")
    assert wind[-4:] == (
        "ukv_ceda_day3_speed_10m",
        "ukv_ceda_day3_sin_10m",
        "ukv_ceda_day3_cos_10m",
        "ukv_ceda_day3_speed_925hpa",
    )
    assert wind[3:7] == tuple(
        f"ens_mean_day2_{f}" for f in ("speed_100m", "sin_100m", "cos_100m", "speed_10m")
    )
    assert fit_aifs.arm_features(arm=control, domain="wind")[-1] == (
        "ukv_ceda_day3_permuted_speed_925hpa"
    )
    solar = fit_aifs.arm_features(arm=blend, domain="solar")
    assert solar[-2:] == ("ukv_ceda_day3_ghi", "ukv_ceda_day3_temp")
    for domain, count in (("solar", 9), ("wind", 11)):
        for arm in fit.stale_arms(day=2):
            assert len(fit_aifs.arm_features(arm=arm, domain=domain)) == count
            assert fit_aifs.expected_column_count(arm=arm, domain=domain) == count
    # The planned arm of the same lead day is untouched.
    assert fit_aifs.arm_features(arm="blend_ukv_ceda_day2", domain="solar")[-2:] == (
        "ukv_ceda_day2_ghi",
        "ukv_ceda_day2_temp",
    )


def test_the_stale_blend_has_lead_days_one_to_three_and_both_settings():
    assert fit.STALE_DAYS == (1, 2, 3)
    assert fit.stale_jobs(day=4) == []
    assert fit.stale_jobs(day=1) == [
        (arm, setting) for arm in fit.stale_arms(day=1) for setting in (PRIMARY, SENSITIVITY)
    ]
    assert fit.is_stale_arm(arm="blend_ukv_ceda_stale_day3_control")
    assert not fit.is_stale_arm(arm="blend_ukv_ceda_day3_control")


def test_the_stale_frame_keeps_the_stage_folds_and_drops_rows_without_the_next_day(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data
    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)
    first_times = sorted(frame["time"].unique().to_list())[:3]
    missing = frame.filter(pl.col("time").is_in(first_times)).select("site", "time")
    holed = inputs.with_columns(
        ukv_ceda_day2_speed_10m=pl.when(pl.col("time").is_in(first_times))
        .then(None)
        .otherwise(pl.col("ukv_ceda_day2_speed_10m"))
    )

    stale = fit.stale_stage_frame(stage=Stage("wind", 1), frame=frame, inputs=holed)

    assert stale.height == frame.height - missing.height
    assert stale.join(missing, on=["site", "time"], how="inner").is_empty()
    folds = stale.select("site", "time", "fold").join(
        frame.select("site", "time", "fold"), on=["site", "time"], suffix="_stage"
    )
    assert folds["fold"].equals(folds["fold_stage"])
    for field in build.WEATHER_FIELDS["wind"]:
        assert f"ukv_ceda_day2_{field}" in stale.columns
        assert f"ukv_ceda_day2_permuted_{field}" in stale.columns
        assert f"ukv_ceda_day2_permuted_b_{field}" in stale.columns
    for arm in fit.stale_arms(day=1):
        assert set(fit_aifs.arm_features(arm=arm, domain="wind")) <= set(stale.columns)


def test_the_stale_frame_refuses_a_row_that_read_the_wrong_run_for_the_next_day(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data
    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)
    shifted = inputs.with_columns(pl.col("ukv_ceda_day2_init_time") + pl.duration(days=1))

    with pytest.raises(ValueError, match="read a run other than the 03 UTC run of day D-2"):
        fit.stale_stage_frame(stage=Stage("wind", 1), frame=frame, inputs=shifted)


def test_the_stale_frame_refuses_a_stage_where_no_row_holds_the_next_day(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = wind_data
    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)
    empty = inputs.with_columns(ukv_ceda_day2_speed_925hpa=pl.lit(None, dtype=pl.Float64))

    with pytest.raises(ValueError, match="no row holds UKV-CEDA's day 2 columns"):
        fit.stale_stage_frame(stage=Stage("wind", 1), frame=frame, inputs=empty)


@pytest.fixture
def stale_stage(
    wind_data: tuple[pl.DataFrame, pl.DataFrame], tiny_stage: fit.PlannedStage
) -> fit.PlannedStage:
    _, inputs = wind_data
    lacking = tiny_stage.frame.sort("site", "time").row(0, named=True)
    holed = inputs.with_columns(
        ukv_ceda_day2_speed_10m=pl.when(
            (pl.col("site") == lacking["site"]) & (pl.col("time") == lacking["time"])
        )
        .then(None)
        .otherwise(pl.col("ukv_ceda_day2_speed_10m"))
    )
    stale = fit.stale_stage_frame(stage=tiny_stage.stage, frame=tiny_stage.frame, inputs=holed)
    assert stale.height == tiny_stage.frame.height - 1
    return tiny_stage._replace(stale_frame=stale)


def _save_planned(*, folder: Path, stage: fit.PlannedStage) -> None:
    arms = list(fit.stage_arms(day=1))
    path = folder / "wind_day1_planned_losses.parquet"
    _losses(arms=arms, settings=[PRIMARY, SENSITIVITY], frame=stage.frame).with_columns(
        device=pl.lit("cuda")
    ).write_parquet(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                **stage.stamp,
                "columns": json.dumps(
                    {a: fit_aifs.arm_features(arm=a, domain="wind") for a in sorted(arms)}
                ),
            }
        )
    )


def test_the_stale_fits_wait_for_every_planned_pair_and_then_list_the_eight_stale_pairs(
    tmp_path: Path, stale_stage: fit.PlannedStage
):
    stage = stale_stage.stage
    with pytest.raises(ValueError, match="fit the planned pairs before the stale ones"):
        fit.jobs_to_fit(output_dir=tmp_path, stage=stage, only_missing=False, post_hoc="stale")

    _save_planned(folder=tmp_path, stage=stale_stage)
    group, jobs = fit.jobs_to_fit(
        output_dir=tmp_path, stage=stage, only_missing=False, post_hoc="stale"
    )

    assert group == "added_1"
    assert jobs == fit.stale_jobs(day=1)
    # The planned jobs are still complete: the stale pairs are never planned ones.
    assert fit.jobs_to_fit(output_dir=tmp_path, stage=stage, only_missing=False) == ("added_1", [])


def test_a_post_hoc_fit_scores_the_stale_rows_into_a_new_file_and_never_refits(
    tmp_path: Path, stale_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    _save_planned(folder=tmp_path, stage=stale_stage)
    planned_file = tmp_path / "wind_day1_planned_losses.parquet"
    before = planned_file.read_bytes()
    calls: list[tuple[list[fit_aifs.Job], int]] = []

    def fake(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        calls.append((jobs, frame.height))
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake)

    merged = fit.fit_stage(
        planned=stale_stage, output_dir=tmp_path, workers=1, only_missing=False, post_hoc="stale"
    )
    again = fit.fit_stage(
        planned=stale_stage, output_dir=tmp_path, workers=1, only_missing=False, post_hoc="stale"
    )

    assert calls == [(fit.stale_jobs(day=1), stale_stage.frame.height - 1)]
    assert planned_file.read_bytes() == before
    assert (tmp_path / "wind_day1_added_1_losses.parquet").exists()
    assert (tmp_path / "wind_day1_added_1_predictions.parquet").exists()
    assert set(merged["arm"]) == {*fit.stage_arms(day=1), *fit.stale_arms(day=1)}
    assert merged.height == again.height


def test_a_saved_stale_arm_that_scores_other_rows_than_the_stale_rows_is_refused(
    tmp_path: Path, stale_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    _save_planned(folder=tmp_path, stage=stale_stage)

    def fake(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake)
    fit.fit_stage(
        planned=stale_stage, output_dir=tmp_path, workers=1, only_missing=False, post_hoc="stale"
    )
    assert stale_stage.stale_frame is not None
    narrower = stale_stage._replace(stale_frame=stale_stage.stale_frame.head(3))

    with pytest.raises(ValueError, match="other rows or folds than its rows"):
        fit.verified_losses(planned=narrower, output_dir=tmp_path)
    no_stale = stale_stage._replace(stale_frame=None)
    with pytest.raises(ValueError, match="no stale rows"):
        fit.verified_losses(planned=no_stale, output_dir=tmp_path)


def test_a_post_hoc_fit_at_a_stage_with_no_stale_blend_fits_nothing(
    tmp_path: Path, tiny_stage: fit.PlannedStage
):
    day4 = Stage("wind", 4)
    arms = list(fit.stage_arms(day=4))
    losses = _losses(arms=arms, settings=[PRIMARY, SENSITIVITY], frame=tiny_stage.frame)
    losses.write_parquet(tmp_path / "wind_day4_planned_losses.parquet")

    group, jobs = fit.jobs_to_fit(
        output_dir=tmp_path, stage=day4, only_missing=False, post_hoc="stale"
    )

    assert (group, jobs) == ("added_1", [])


def _stale_losses(
    *, stage: fit.PlannedStage, levels: dict[str, float]
) -> tuple[fit.PlannedStage, pl.DataFrame]:
    """Planned arms on every row, stale arms on the rows that are not in the last month."""
    rng = np.random.default_rng(5)
    frame = stage.frame
    stale_rows = frame.filter(pl.col("month") != "2025-10")
    losses = pl.concat(
        rows.with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            seed=pl.lit(seed),
            signed_error_capped_mw=pl.lit(0.0),
            **{METRIC: pl.Series(level + rng.normal(0.0, 0.002, rows.height))},
        )
        for arm, level in levels.items()
        for rows in ([stale_rows] if fit.is_stale_arm(arm=arm) else [frame])
        for setting in (PRIMARY, SENSITIVITY)
        for seed in range(3)
    )
    return stage._replace(stale_frame=stale_rows), losses


def test_the_stale_section_scores_every_contrast_on_the_stale_rows_and_reads_both_controls(
    few_resamples: None,
):
    base, _ = _report_stage()
    stage, losses = _stale_losses(
        stage=base,
        levels={
            "blend_ukv_ceda_day2_pad": 0.10,
            "blend_ukv_ceda_day2": 0.09,
            "blend_ukv_ceda_day2_control": 0.11,
            "blend_ukv_ceda_day2_control_b": 0.11,
            "blend_ukv_ceda_stale_day2_pad": 0.105,
            "blend_ukv_ceda_stale_day2": 0.095,
            "blend_ukv_ceda_stale_day2_control": 0.11,
            "blend_ukv_ceda_stale_day2_control_b": 0.11,
        },
    )
    records: list[fit.IntervalRecord] = []

    lines, stale_reading = fit.stale_lines(planned=stage, losses=losses, records=records)

    text = "\n".join(lines)
    assert "Post hoc: ENS day 2 plus UKV-CEDA day 3, wind" in text
    assert "21 hours staler than ENS's" in text
    assert stale_reading == "lowers the error at day 2"
    stale_rows = stage.stale_frame
    assert stale_rows is not None
    assert stale_rows.height < stage.frame.height
    by_code = {(r["contrast"], r["setting"]): r for r in records if r["scope"] == fit.STALE_SCOPE}
    assert set(by_code) == {
        (code, setting) for code in fit.STALE_CONTRASTS for setting in (PRIMARY, SENSITIVITY)
    }
    assert all(r["n_rows"] == stale_rows.height for r in by_code.values())
    # P1 is read against the padded reference trained on the stale rows, not the planned one.
    assert by_code[("stale_p1", PRIMARY)]["difference"] == pytest.approx(-0.01, abs=0.001)
    assert by_code[("training_rows", PRIMARY)]["difference"] == pytest.approx(-0.005, abs=0.001)
    assert by_code[("stale_vs_fresh", PRIMARY)]["difference"] == pytest.approx(0.005, abs=0.002)
    assert by_code[("fresh_p1_same_rows", PRIMARY)]["difference"] == pytest.approx(-0.01, abs=0.002)


def test_a_second_stale_control_that_matches_the_stale_blend_blocks_the_stale_reading(
    few_resamples: None,
):
    base, _ = _report_stage()
    stage, losses = _stale_losses(
        stage=base,
        levels={
            "blend_ukv_ceda_day2_pad": 0.10,
            "blend_ukv_ceda_day2": 0.09,
            "blend_ukv_ceda_day2_control": 0.11,
            "blend_ukv_ceda_day2_control_b": 0.11,
            "blend_ukv_ceda_stale_day2_pad": 0.10,
            "blend_ukv_ceda_stale_day2": 0.09,
            "blend_ukv_ceda_stale_day2_control": 0.11,
            "blend_ukv_ceda_stale_day2_control_b": 0.09,
        },
    )
    records: list[fit.IntervalRecord] = []

    _, stale_reading = fit.stale_lines(planned=stage, losses=losses, records=records)

    assert stale_reading != "lowers the error at day 2"
    by_code = {(r["contrast"], r["setting"]): r for r in records if r["scope"] == fit.STALE_SCOPE}
    assert by_code[("stale_p2b", PRIMARY)]["difference"] == pytest.approx(0.0, abs=0.002)
    assert by_code[("stale_p2", PRIMARY)]["difference"] == pytest.approx(-0.02, abs=0.002)


def test_the_stale_section_is_absent_until_every_stale_pair_is_saved(few_resamples: None):
    base, losses = _report_stage()
    with_rows = base._replace(stale_frame=base.frame.head(60))

    assert fit.stale_lines(planned=with_rows, losses=losses, records=[]) == ([], None)
    assert fit.stale_lines(planned=base, losses=losses, records=[]) == ([], None)


def test_a_stale_blend_that_does_not_lower_the_error_is_read_as_no_detectable_difference(
    few_resamples: None,
):
    base, _ = _report_stage()
    stage, losses = _stale_losses(
        stage=base,
        levels={
            "blend_ukv_ceda_day2_pad": 0.10,
            "blend_ukv_ceda_day2": 0.09,
            "blend_ukv_ceda_day2_control": 0.11,
            "blend_ukv_ceda_day2_control_b": 0.11,
            "blend_ukv_ceda_stale_day2_pad": 0.10,
            "blend_ukv_ceda_stale_day2": 0.10,
            "blend_ukv_ceda_stale_day2_control": 0.10,
            "blend_ukv_ceda_stale_day2_control_b": 0.10,
        },
    )

    _, stale_reading = fit.stale_lines(planned=stage, losses=losses, records=[])

    assert stale_reading == "no detectable difference"


def test_the_report_adds_the_stale_readings_table_only_when_a_stale_fit_is_saved():
    text = fit.report_text(
        sections=[],
        summary=[],
        stale_summary=fit.stale_summary_lines(readings={("solar", 2): "lowers the error at day 2"}),
        cpu_line="",
        padding_line="",
    )

    assert "| solar | 2 | lowers the error at day 2 |" in text
    assert "UKV-CEDA 21 hours staler than ENS" in text
    assert fit.stale_summary_lines(readings={}) == []
    plain = fit.report_text(sections=[], summary=[], stale_summary=[], cpu_line="", padding_line="")
    assert "Post hoc: ENS day N" not in plain


def test_the_dry_run_lists_the_stale_fits_and_the_rows_that_lack_the_next_day(
    tmp_path: Path, stale_stage: fit.PlannedStage, capsys: pytest.CaptureFixture[str]
):
    _save_planned(folder=tmp_path, stage=stale_stage)

    fit.print_plan(planned=[stale_stage], output_dir=tmp_path, only_missing=False, post_hoc="stale")

    out = capsys.readouterr().out
    assert "wind day 1: 5 rows (1 stage rows lack the next day)" in out
    assert "8 (arm, setting) fits" in out
    assert "8 (arm, site) fits in all" in out


# --- the post hoc permutation test ----------------------------------------------------------------


@pytest.fixture(scope="module")
def solar_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    candidates = _candidates(domain="solar")
    return candidates, _inputs(candidates=candidates, domain="solar")


def test_the_permutation_test_has_fifteen_extra_controls_under_seeds_no_planned_shuffle_uses():
    arms = fit.permutation_arms(day=2)

    assert len(arms) == 15
    assert arms[0] == "blend_ukv_ceda_day2_control_s2010"
    assert arms[-1] == "blend_ukv_ceda_day2_control_s2150"
    assert len(set(fit.PERMUTATION_SEEDS)) == 15
    # A two-column shuffle uses its seed and the next, so no two draws may be one apart or equal.
    seeds = sorted({*fit.PERMUTATION_SEEDS, *fit_aifs.SHUFFLE_SEEDS.values()})
    assert all(later - earlier >= 2 for earlier, later in itertools.pairwise(seeds))
    assert fit.PERMUTATION_DRAWS == 17
    assert all(fit.is_permutation_arm(arm=arm) for arm in arms)
    for planned_arm in fit.stage_arms(day=2):
        assert not fit.is_permutation_arm(arm=planned_arm)
    assert not fit.is_permutation_arm(arm="blend_ukv_ceda_stale_day2_control")


def test_the_permutation_fits_are_the_primary_setting_only_and_solar_only():
    assert fit.permutation_jobs(stage=Stage("solar", 4)) == [
        (arm, PRIMARY) for arm in fit.permutation_arms(day=4)
    ]
    assert len(fit.permutation_jobs(stage=Stage("solar", 1))) == 15
    assert fit.permutation_jobs(stage=Stage("wind", 1)) == []
    assert fit.post_hoc_jobs(stage=Stage("solar", 2), kind="permutation") == fit.permutation_jobs(
        stage=Stage("solar", 2)
    )


def test_the_extra_shuffles_add_columns_and_change_no_planned_column(
    solar_data: tuple[pl.DataFrame, pl.DataFrame],
):
    candidates, inputs = solar_data
    stage = Stage("solar", 1)

    plain, plain_offsets = fit.stage_frame(stage=stage, candidates=candidates, inputs=inputs)
    extra, extra_offsets = fit.stage_frame(
        stage=stage, candidates=candidates, inputs=inputs, permutation=True
    )

    assert plain_offsets == extra_offsets
    for column in plain.columns:
        assert extra[column].equals(plain[column]), column
    added = set(extra.columns) - set(plain.columns)
    assert added == {
        f"ukv_ceda_day1_permuted_s{seed}_{field}"
        for seed in fit.PERMUTATION_SEEDS
        for field in ("ghi", "temp")
    }
    for arm in fit.permutation_arms(day=1):
        assert set(fit_aifs.arm_features(arm=arm, domain="solar")) <= set(extra.columns)
        assert len(fit_aifs.arm_features(arm=arm, domain="solar")) == 9
    # A shuffle keeps each (generator, year-month, hour) group's values.
    one = f"ukv_ceda_day1_permuted_s{fit.PERMUTATION_SEEDS[0]}_ghi"
    for _, group in extra.group_by("site", "month", "hour_of_day"):
        assert sorted(group["ukv_ceda_day1_ghi"]) == sorted(group[one])


def test_the_permutation_summary_ranks_the_planned_gain_among_the_draws_and_counts_ties():
    summary = fit.permutation_summary(planned=-0.10, draws=[-0.20, -0.05, 0.0, -0.10, 0.10])

    # Draws at or below the planned value: -0.20 and the tie at -0.10. Draws strictly below: one.
    assert summary.p_value == pytest.approx((1 + 2) / 6)
    assert summary.rank == 2
    assert summary.planned == -0.10
    assert summary.draws == (-0.20, -0.05, 0.0, -0.10, 0.10)


def test_a_planned_gain_below_every_draw_has_rank_one_and_the_smallest_p_value():
    draws = [-0.01 * k for k in range(17)]

    summary = fit.permutation_summary(planned=-0.5, draws=draws)

    assert summary.rank == 1
    assert summary.p_value == pytest.approx(1 / 18)


def test_a_planned_gain_above_every_draw_has_the_largest_rank_and_a_p_value_of_one():
    summary = fit.permutation_summary(planned=0.3, draws=[0.0, 0.1, 0.2])

    assert summary.rank == 4
    assert summary.p_value == pytest.approx(1.0)


def _permutation_losses(*, stage: fit.PlannedStage, levels: dict[str, float]) -> pl.DataFrame:
    """Per-arm constant losses on every row of the stage, so every difference is exact."""
    return pl.concat(
        stage.frame.with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            seed=pl.lit(seed),
            signed_error_capped_mw=pl.lit(0.0),
            **{METRIC: pl.lit(level)},
        )
        for arm, level in levels.items()
        for setting in (PRIMARY, SENSITIVITY)
        if setting == PRIMARY or not fit.is_permutation_arm(arm=arm)
        for seed in range(3)
    )


def _solar_stage(*, day: int = 2) -> fit.PlannedStage:
    base, _ = _report_stage()
    return base._replace(stage=Stage("solar", day))


def _permutation_levels(*, day: int = 2) -> dict[str, float]:
    levels = {
        f"blend_ukv_ceda_day{day}_pad": 0.10,
        f"blend_ukv_ceda_day{day}": 0.09,
        f"blend_ukv_ceda_day{day}_control": 0.10,
        f"blend_ukv_ceda_day{day}_control_b": 0.095,
    }
    for index, arm in enumerate(fit.permutation_arms(day=day)):
        levels[arm] = {0: 0.08, 1: 0.085}.get(index, 0.10 + 0.001 * index)
    return levels


def test_the_permutation_section_places_p1_among_the_seventeen_controls():
    stage = _solar_stage()
    losses = _permutation_losses(stage=stage, levels=_permutation_levels())
    records: list[fit.IntervalRecord] = []

    lines, summary = fit.permutation_lines(planned=stage, losses=losses, records=records)

    assert summary is not None
    text = "\n".join(lines)
    # P1 is 0.09 - 0.10; two extra controls (0.08, 0.085) are lower, so rank 3 of 18.
    assert summary.planned == pytest.approx(-0.01)
    assert summary.rank == 3
    assert summary.p_value == pytest.approx(3 / 18)
    assert len(summary.draws) == 17
    assert min(summary.draws) == pytest.approx(-0.02)
    assert "Post hoc: permutation test, solar lead day 2" in text
    assert "| 2010 | -2.000 |" in text
    assert "| 0 | +0.000 |" in text
    assert "| 1000 | -0.500 |" in text
    assert "| planned blend (P1) | -1.000 |" in text
    assert "P1 ranks 3 from the lowest" in text
    assert "The one-sided permutation p-value is 0.167" in text
    assert "smallest value it can take is 0.056" in text
    by_contrast = {r["contrast"]: r for r in records if r["contrast"] != "permutation_draw"}
    assert by_contrast["permutation_p"]["difference"] == pytest.approx(3 / 18)
    assert by_contrast["permutation_rank"]["difference"] == 3.0
    assert by_contrast["permutation_p1"]["difference"] == pytest.approx(-0.01)
    draws = [r for r in records if r["contrast"] == "permutation_draw"]
    assert len(draws) == 17
    assert {r["scope"] for r in draws} == {
        f"{fit.PERMUTATION_SCOPE}: seed {seed}" for seed in (0, 1000, *fit.PERMUTATION_SEEDS)
    }
    assert {r["setting"] for r in records} == {PRIMARY}
    assert all(r["domain"] == "solar" and r["day"] == 2 for r in records)


def test_the_permutation_section_waits_for_every_extra_control_and_skips_wind():
    stage = _solar_stage()
    levels = _permutation_levels()
    last = fit.permutation_arms(day=2)[-1]
    without_last = _permutation_losses(
        stage=stage, levels={arm: level for arm, level in levels.items() if arm != last}
    )

    assert fit.permutation_lines(planned=stage, losses=without_last, records=[]) == ([], None)
    wind = stage._replace(stage=Stage("wind", 2))
    assert fit.permutation_lines(
        planned=wind, losses=_permutation_losses(stage=stage, levels=levels), records=[]
    ) == ([], None)


def test_the_permutation_summary_table_lists_each_stage_and_is_empty_without_one():
    summary = fit.permutation_summary(planned=-0.01, draws=[0.0, -0.02, 0.01])

    lines = fit.permutation_summary_lines(summaries={("solar", 3): summary})

    assert lines[2] == ("| solar | 3 | -1.000 | -2.000 | +0.000 | +1.000 | 2 of 4 | 0.500 |")
    assert fit.permutation_summary_lines(summaries={}) == []


def _save_group(
    *,
    folder: Path,
    planned: fit.PlannedStage,
    group: str,
    jobs: list[fit_aifs.Job],
    frame: pl.DataFrame | None = None,
) -> None:
    """Write a losses file and its stamp for `jobs`, scored on `frame` (default: the stage's)."""
    arms = list(dict.fromkeys(arm for arm, _ in jobs))
    path = folder / f"{fit.stem(stage=planned.stage, group=group)}_losses.parquet"
    pl.concat(
        _losses(arms=[arm], settings=[setting], frame=frame if frame is not None else planned.frame)
        for arm, setting in jobs
    ).write_parquet(path)
    path.with_suffix(".json").write_text(json.dumps(fit.file_stamp(planned=planned, arms=arms)))


@pytest.fixture
def tiny_solar_stage(solar_data: tuple[pl.DataFrame, pl.DataFrame]) -> fit.PlannedStage:
    candidates, inputs = solar_data
    frame, offsets = fit.stage_frame(
        stage=Stage("solar", 1), candidates=candidates, inputs=inputs, permutation=True
    )
    return fit.PlannedStage(
        stage=Stage("solar", 1),
        frame=frame.sort("site", "time").head(6),
        offsets=offsets,
        stamp={"build": "1", "device": "cuda"},
    )


def test_the_permutation_fits_wait_for_the_planned_pairs_then_list_fifteen_primary_pairs(
    tmp_path: Path, tiny_solar_stage: fit.PlannedStage
):
    stage = tiny_solar_stage.stage
    with pytest.raises(ValueError, match="fit the planned pairs before the permutation ones"):
        fit.jobs_to_fit(
            output_dir=tmp_path, stage=stage, only_missing=False, post_hoc="permutation"
        )

    _save_group(
        folder=tmp_path, planned=tiny_solar_stage, group="planned", jobs=fit.planned_jobs(day=1)
    )
    group, jobs = fit.jobs_to_fit(
        output_dir=tmp_path, stage=stage, only_missing=False, post_hoc="permutation"
    )

    assert group == "added_1"
    assert jobs == fit.permutation_jobs(stage=stage)
    assert len(jobs) == 15


def test_the_permutation_fit_scores_the_whole_stage_into_a_new_file_and_stamps_its_seeds(
    tmp_path: Path, tiny_solar_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    _save_group(
        folder=tmp_path, planned=tiny_solar_stage, group="planned", jobs=fit.planned_jobs(day=1)
    )
    planned_file = tmp_path / "solar_day1_planned_losses.parquet"
    before = (planned_file.read_bytes(), planned_file.with_suffix(".json").read_bytes())
    calls: list[tuple[list[fit_aifs.Job], int]] = []

    def fake(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        calls.append((jobs, frame.height))
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake)

    merged = fit.fit_stage(
        planned=tiny_solar_stage,
        output_dir=tmp_path,
        workers=1,
        only_missing=False,
        post_hoc="permutation",
    )
    again = fit.fit_stage(
        planned=tiny_solar_stage,
        output_dir=tmp_path,
        workers=1,
        only_missing=False,
        post_hoc="permutation",
    )

    assert calls == [(fit.permutation_jobs(stage=tiny_solar_stage.stage), 6)]
    assert (planned_file.read_bytes(), planned_file.with_suffix(".json").read_bytes()) == before
    added = tmp_path / "solar_day1_added_1_losses.parquet"
    assert added.exists()
    stamp = json.loads(added.with_suffix(".json").read_text())
    assert json.loads(stamp["permutation_seeds"]) == list(fit.PERMUTATION_SEEDS)
    assert "older_run_inputs_sha256" not in stamp
    assert set(merged["arm"]) == {*fit.stage_arms(day=1), *fit.permutation_arms(day=1)}
    assert set(merged.filter(pl.col("arm").str.contains("_control_s"))["setting"]) == {PRIMARY}
    assert merged.height == again.height
    planned_stamp = json.loads(planned_file.with_suffix(".json").read_text())
    assert set(planned_stamp) == {*tiny_solar_stage.stamp, "columns"}


def test_a_file_of_planned_or_stale_arms_keeps_its_stamp_and_only_new_arms_add_keys(
    tiny_solar_stage: fit.PlannedStage,
):
    planned_arms = fit.stage_arms(day=1)

    stamp = fit.file_stamp(planned=tiny_solar_stage, arms=planned_arms)

    assert set(stamp) == {*tiny_solar_stage.stamp, "columns"}
    assert (
        fit.file_stamp(planned=tiny_solar_stage, arms=fit.stale_arms(day=1)).keys() == stamp.keys()
    )
    seeded = fit.file_stamp(planned=tiny_solar_stage, arms=fit.permutation_arms(day=1))
    assert set(seeded) == {*stamp, "permutation_seeds"}
    with pytest.raises(ValueError, match="no older-run inputs were read"):
        fit.file_stamp(planned=tiny_solar_stage, arms=fit.older_arms(day=1))
    with_older = tiny_solar_stage._replace(
        older_stamp={"older_run_inputs_sha256": "abc", "older_run_snapshot": "SNAP"}
    )
    older_stamp = fit.file_stamp(planned=with_older, arms=fit.older_arms(day=1))
    assert older_stamp["older_run_inputs_sha256"] == "abc"
    assert older_stamp["older_run_snapshot"] == "SNAP"
    assert "permutation_seeds" not in older_stamp


# --- the post hoc older run -----------------------------------------------------------------------


def _older_inputs(*, candidates: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """UKV-CEDA's 15 UTC run of the day before ENS's run, with the run restated by hand."""
    rng = np.random.default_rng(6)
    instant = pl.col("time") - pl.duration(hours=1) if domain == "solar" else pl.col("time")
    frame = candidates.select("site", "time")
    n = frame.height
    columns: dict[str, pl.Series | pl.Expr] = {}
    for day in build.OLDER_RUN.lead_days:
        for field in build.WEATHER_FIELDS[domain]:
            columns[f"ukv_ceda_run15_day{day}_{field}"] = pl.Series(rng.uniform(0, 1, n))
        columns[f"ukv_ceda_run15_day{day}_init_time"] = (
            instant.dt.truncate("1d") - pl.duration(days=day + 1) + pl.duration(hours=15)
        )
    return frame.with_columns(**columns)


@pytest.fixture(scope="module")
def wind_older(wind_data: tuple[pl.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    candidates, _ = wind_data
    return _older_inputs(candidates=candidates, domain="wind")


def test_the_older_run_blend_has_the_stale_blends_lead_days_and_the_planned_column_counts():
    pad, blend, control = fit.older_arms(day=2)

    assert (pad, blend, control) == (
        "blend_ukv_ceda_run15_day2_pad",
        "blend_ukv_ceda_run15_day2",
        "blend_ukv_ceda_run15_day2_control",
    )
    assert fit.OLDER_DAYS == (1, 2, 3)
    assert fit.older_jobs(day=4) == []
    assert fit.older_jobs(day=1) == [
        (arm, setting) for arm in fit.older_arms(day=1) for setting in (PRIMARY, SENSITIVITY)
    ]
    assert fit.post_hoc_jobs(stage=Stage("wind", 3), kind="older run") == fit.older_jobs(day=3)
    wind = fit_aifs.arm_features(arm=blend, domain="wind")
    assert wind[3:7] == tuple(
        f"ens_mean_day2_{f}" for f in ("speed_100m", "sin_100m", "cos_100m", "speed_10m")
    )
    assert wind[-4:] == (
        "ukv_ceda_run15_day2_speed_10m",
        "ukv_ceda_run15_day2_sin_10m",
        "ukv_ceda_run15_day2_cos_10m",
        "ukv_ceda_run15_day2_speed_925hpa",
    )
    assert fit_aifs.arm_features(arm=control, domain="solar")[-2:] == (
        "ukv_ceda_run15_day2_permuted_ghi",
        "ukv_ceda_run15_day2_permuted_temp",
    )
    for domain, count in (("solar", 9), ("wind", 11)):
        for arm in (*fit.older_arms(day=2), *fit.stage_arms(day=2)):
            assert len(fit_aifs.arm_features(arm=arm, domain=domain)) == count
    assert fit.is_older_arm(arm=blend)
    assert not fit.is_older_arm(arm="blend_ukv_ceda_day2")
    assert not fit.is_older_arm(arm="blend_ukv_ceda_stale_day2")
    assert not fit.is_stale_arm(arm=blend)


def test_the_stamp_check_restates_the_older_run_as_15_utc_of_the_day_before_ens_runs_day():
    spec = build.OLDER_RUN
    wind = pl.DataFrame(
        {
            "time": [datetime(2026, 3, 10, 0, tzinfo=UTC), datetime(2026, 3, 10, 23, tzinfo=UTC)],
            "ukv_ceda_run15_day2_init_time": [datetime(2026, 3, 7, 15, tzinfo=UTC)] * 2,
        }
    )
    fit.check_init_times(frame=wind, domain="wind", day=2, spec=spec)

    solar = wind.with_columns(
        time=pl.Series(
            [datetime(2026, 3, 11, 0, tzinfo=UTC), datetime(2026, 3, 10, 24 - 1, tzinfo=UTC)]
        )
    )
    # A solar label of 00:00 on 11 March is the hour that started at 23:00 on 10 March.
    fit.check_init_times(frame=solar, domain="solar", day=2, spec=spec)
    for wrong in (
        datetime(2026, 3, 7, 3, tzinfo=UTC),  # the 03 UTC run of the same day
        datetime(2026, 3, 8, 15, tzinfo=UTC),  # the 15 UTC run one day too late
        datetime(2026, 3, 6, 15, tzinfo=UTC),  # one day too early
    ):
        bad = wind.with_columns(ukv_ceda_run15_day2_init_time=pl.lit(wrong))
        with pytest.raises(ValueError, match="other than the 15 UTC run of day D-3"):
            fit.check_init_times(frame=bad, domain="wind", day=2, spec=spec)


def test_the_older_frame_keeps_the_stage_folds_and_drops_rows_without_the_older_run(
    wind_data: tuple[pl.DataFrame, pl.DataFrame], wind_older: pl.DataFrame
):
    candidates, inputs = wind_data
    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)
    first_times = sorted(frame["time"].unique().to_list())[:3]
    missing = frame.filter(pl.col("time").is_in(first_times)).select("site", "time")
    holed = wind_older.with_columns(
        ukv_ceda_run15_day1_speed_925hpa=pl.when(pl.col("time").is_in(first_times))
        .then(float("nan"))
        .otherwise(pl.col("ukv_ceda_run15_day1_speed_925hpa"))
    )

    older = fit.older_stage_frame(stage=Stage("wind", 1), frame=frame, inputs=holed)

    assert older.height == frame.height - missing.height
    assert older.join(missing, on=["site", "time"], how="inner").is_empty()
    folds = older.select("site", "time", "fold").join(
        frame.select("site", "time", "fold"), on=["site", "time"], suffix="_stage"
    )
    assert folds["fold"].equals(folds["fold_stage"])
    for field in build.WEATHER_FIELDS["wind"]:
        assert f"ukv_ceda_run15_day1_{field}" in older.columns
        assert f"ukv_ceda_run15_day1_permuted_{field}" in older.columns
        assert f"ukv_ceda_run15_day1_permuted_b_{field}" not in older.columns
    for arm in fit.older_arms(day=1):
        assert set(fit_aifs.arm_features(arm=arm, domain="wind")) <= set(older.columns)


def test_the_older_frame_refuses_the_wrong_run_missing_rows_and_an_empty_older_run(
    wind_data: tuple[pl.DataFrame, pl.DataFrame], wind_older: pl.DataFrame
):
    candidates, inputs = wind_data
    frame, _ = fit.stage_frame(stage=Stage("wind", 1), candidates=candidates, inputs=inputs)
    stage = Stage("wind", 1)

    planned_run = wind_older.with_columns(
        ukv_ceda_run15_day1_init_time=pl.col("ukv_ceda_run15_day1_init_time")
        - pl.duration(hours=12)
    )
    with pytest.raises(ValueError, match="other than the 15 UTC run of day D-2"):
        fit.older_stage_frame(stage=stage, frame=frame, inputs=planned_run)
    with pytest.raises(ValueError, match="missing from the older-run inputs"):
        fit.older_stage_frame(stage=stage, frame=frame, inputs=wind_older.head(10))
    empty = wind_older.with_columns(ukv_ceda_run15_day1_speed_10m=pl.lit(None, dtype=pl.Float64))
    with pytest.raises(ValueError, match="no row holds the older run's columns"):
        fit.older_stage_frame(stage=stage, frame=frame, inputs=empty)


@pytest.fixture
def older_stage(
    wind_data: tuple[pl.DataFrame, pl.DataFrame],
    wind_older: pl.DataFrame,
    tiny_stage: fit.PlannedStage,
) -> fit.PlannedStage:
    lacking = tiny_stage.frame.sort("site", "time").row(0, named=True)
    holed = wind_older.with_columns(
        ukv_ceda_run15_day1_speed_10m=pl.when(
            (pl.col("site") == lacking["site"]) & (pl.col("time") == lacking["time"])
        )
        .then(None)
        .otherwise(pl.col("ukv_ceda_run15_day1_speed_10m"))
    )
    older = fit.older_stage_frame(stage=tiny_stage.stage, frame=tiny_stage.frame, inputs=holed)
    assert older.height == tiny_stage.frame.height - 1
    return tiny_stage._replace(
        older_frame=older,
        older_stamp={"older_run_inputs_sha256": "abc", "older_run_snapshot": "SNAP"},
    )


def test_an_older_run_fit_scores_the_older_rows_stamps_the_older_inputs_and_never_refits(
    tmp_path: Path, older_stage: fit.PlannedStage, monkeypatch: pytest.MonkeyPatch
):
    _save_group(folder=tmp_path, planned=older_stage, group="planned", jobs=fit.planned_jobs(day=1))
    calls: list[tuple[list[fit_aifs.Job], int]] = []

    def fake(
        *, frame: pl.DataFrame, domain: DomainType, jobs: list[fit_aifs.Job], workers: int
    ) -> pl.DataFrame:
        calls.append((jobs, frame.height))
        return pl.concat(
            _losses(arms=[arm], settings=[setting], frame=frame) for arm, setting in jobs
        )

    monkeypatch.setattr(fit_aifs, "fit_jobs", fake)

    merged = fit.fit_stage(
        planned=older_stage,
        output_dir=tmp_path,
        workers=1,
        only_missing=False,
        post_hoc="older run",
    )
    fit.fit_stage(
        planned=older_stage,
        output_dir=tmp_path,
        workers=1,
        only_missing=False,
        post_hoc="older run",
    )

    assert calls == [(fit.older_jobs(day=1), older_stage.frame.height - 1)]
    stamp = json.loads((tmp_path / "wind_day1_added_1_losses.json").read_text())
    assert stamp["older_run_inputs_sha256"] == "abc"
    assert stamp["older_run_snapshot"] == "SNAP"
    assert set(merged["arm"]) == {*fit.stage_arms(day=1), *fit.older_arms(day=1)}
    # A different older build is refused when the saved file is read again.
    changed = older_stage._replace(
        older_stamp={"older_run_inputs_sha256": "other", "older_run_snapshot": "SNAP"}
    )
    with pytest.raises(ValueError, match="names another build or device"):
        fit.verified_losses(planned=changed, output_dir=tmp_path)
    narrower = older_stage._replace(older_frame=older_stage.older_frame.head(3))  # ty: ignore[unresolved-attribute]
    with pytest.raises(ValueError, match="other rows or folds than its rows"):
        fit.verified_losses(planned=narrower, output_dir=tmp_path)
    no_older = older_stage._replace(older_frame=None)
    with pytest.raises(ValueError, match="older-run arm, but the stage has no older-run rows"):
        fit.verified_losses(planned=no_older, output_dir=tmp_path)


def test_the_older_fits_wait_for_every_planned_pair(tmp_path: Path, older_stage: fit.PlannedStage):
    with pytest.raises(ValueError, match="fit the planned pairs before the older run ones"):
        fit.jobs_to_fit(
            output_dir=tmp_path, stage=older_stage.stage, only_missing=False, post_hoc="older run"
        )
    _save_group(folder=tmp_path, planned=older_stage, group="planned", jobs=fit.planned_jobs(day=1))
    group, jobs = fit.jobs_to_fit(
        output_dir=tmp_path, stage=older_stage.stage, only_missing=False, post_hoc="older run"
    )
    assert (group, jobs) == ("added_1", fit.older_jobs(day=1))


def _older_losses(
    *, stage: fit.PlannedStage, levels: dict[str, float]
) -> tuple[fit.PlannedStage, pl.DataFrame]:
    """Planned arms on every row, older-run arms on the rows outside 2025-10; exact losses."""
    frame = stage.frame
    older_rows = frame.filter(pl.col("month") != "2025-10")
    losses = pl.concat(
        rows.with_columns(
            arm=pl.lit(arm),
            setting=pl.lit(setting),
            seed=pl.lit(seed),
            signed_error_capped_mw=pl.lit(0.0),
            **{METRIC: pl.lit(level)},
        )
        for arm, level in levels.items()
        for rows in ([older_rows] if fit.is_older_arm(arm=arm) else [frame])
        for setting in (PRIMARY, SENSITIVITY)
        for seed in range(3)
    )
    return stage._replace(older_frame=older_rows), losses


_OLDER_LEVELS = {
    "blend_ukv_ceda_day2_pad": 0.10,
    "blend_ukv_ceda_day2": 0.09,
    "blend_ukv_ceda_day2_control": 0.11,
    "blend_ukv_ceda_day2_control_b": 0.11,
    "blend_ukv_ceda_run15_day2_pad": 0.105,
    "blend_ukv_ceda_run15_day2": 0.097,
    "blend_ukv_ceda_run15_day2_control": 0.11,
}


def test_the_older_section_scores_every_contrast_on_the_older_rows_against_its_own_reference(
    few_resamples: None,
):
    base, _ = _report_stage()
    stage, losses = _older_losses(stage=base, levels=_OLDER_LEVELS)
    records: list[fit.IntervalRecord] = []

    lines, older_reading = fit.older_lines(planned=stage, losses=losses, records=records)

    text = "\n".join(lines)
    assert "Post hoc: ENS day 2 plus UKV-CEDA's older run, wind" in text
    assert "starts 9 hours before ENS's 00 UTC run" in text
    assert "its lead is 12 hours longer" in text
    assert (
        "cannot separate the effect of the longer lead from the effect of the earlier start" in text
    )
    assert older_reading is not None
    assert older_reading.reading == "lowers the error at day 2"
    older_rows = stage.older_frame
    assert older_rows is not None
    assert older_rows.height < stage.frame.height
    by_code = {(r["contrast"], r["setting"]): r for r in records if r["scope"] == fit.OLDER_SCOPE}
    assert set(by_code) == {
        (code, setting) for code in fit.OLDER_CONTRASTS for setting in (PRIMARY, SENSITIVITY)
    }
    assert all(r["n_rows"] == older_rows.height for r in by_code.values())
    for setting in (PRIMARY, SENSITIVITY):
        # Older-run blend 0.097 against its own padded reference 0.105 and its control 0.11.
        assert by_code[("older_p1", setting)]["difference"] == pytest.approx(-0.008)
        assert by_code[("older_p2", setting)]["difference"] == pytest.approx(-0.013)
        # Against the planned blend (0.09), and the planned gain on the same rows.
        assert by_code[("older_vs_fresh", setting)]["difference"] == pytest.approx(0.007)
        assert by_code[("fresh_p1_same_rows", setting)]["difference"] == pytest.approx(-0.01)
        assert by_code[("training_rows", setting)]["difference"] == pytest.approx(-0.005)
    errors = {r["scope"] for r in records if r["contrast"] == "error"}
    assert errors == {
        f"{fit.OLDER_SCOPE} rows: {arm}"
        for arm in (
            "blend_ukv_ceda_day2_pad",
            "blend_ukv_ceda_day2",
            *fit.older_arms(day=2),
        )
    }


def test_an_older_run_blend_whose_control_matches_it_does_not_lower_the_error(few_resamples: None):
    base, _ = _report_stage()
    stage, losses = _older_losses(
        stage=base,
        levels={
            **_OLDER_LEVELS,
            "blend_ukv_ceda_run15_day2_control": 0.097,
        },
    )

    _, older_reading = fit.older_lines(planned=stage, losses=losses, records=[])

    assert older_reading is not None
    assert older_reading.reading != "lowers the error at day 2"
    assert older_reading.p2[PRIMARY]["difference"] == pytest.approx(0.0, abs=1e-9)
    assert older_reading.p1[PRIMARY]["difference"] == pytest.approx(-0.008)


def test_the_older_section_is_absent_until_every_older_pair_is_saved(few_resamples: None):
    base, losses = _report_stage()
    with_rows = base._replace(older_frame=base.frame.head(60))

    assert fit.older_lines(planned=with_rows, losses=losses, records=[]) == ([], None)
    assert fit.older_lines(planned=base, losses=losses, records=[]) == ([], None)


def test_the_older_readings_table_gives_both_settings_of_each_contrast():
    p1 = {
        PRIMARY: _interval(difference=-0.002, lower=-0.003, upper=-0.001),
        SENSITIVITY: _interval(difference=-0.001, lower=-0.002, upper=0.0005),
    }
    p2 = {
        PRIMARY: _interval(difference=-0.004, lower=-0.005, upper=-0.003),
        SENSITIVITY: _interval(difference=-0.003, lower=-0.004, upper=-0.002),
    }
    fresh = {
        PRIMARY: _interval(difference=-0.01, lower=-0.012, upper=-0.008),
        SENSITIVITY: _interval(difference=-0.009, lower=-0.011, upper=-0.007),
    }

    lines = fit.older_summary_lines(
        readings={("wind", 1): fit.OlderReading(reading="x", p1=p1, p2=p2, fresh_p1=fresh)}
    )

    assert lines[2] == (
        "| wind | 1 | x | -1.000 [-1.200, -0.800] / -0.900 [-1.100, -0.700] "
        "| -0.200 [-0.300, -0.100] / -0.100 [-0.200, +0.050] "
        "| -0.400 [-0.500, -0.300] / -0.300 [-0.400, -0.200] |"
    )
    assert fit.older_summary_lines(readings={}) == []


def test_the_report_adds_the_permutation_and_older_tables_only_when_they_are_given():
    text = fit.report_text(
        sections=[],
        summary=[],
        stale_summary=[],
        cpu_line="",
        padding_line="",
        permutation_summary=["| permutation row |"],
        older_summary=["| older row |"],
    )

    assert "| permutation row |" in text
    assert "15 extra, all at the primary setting" in text
    assert "| older row |" in text
    assert "tests how the gain falls as the UKV-CEDA run gets older" in text
    assert (
        "cannot separate the effect of the longer lead from the effect of the earlier start" in text
    )
    plain = fit.report_text(sections=[], summary=[], stale_summary=[], cpu_line="", padding_line="")
    assert "permutation test" not in plain
    assert "15 UTC run" not in plain


def test_the_dry_run_lists_the_older_run_fits_and_the_rows_that_lack_the_older_run(
    tmp_path: Path, older_stage: fit.PlannedStage, capsys: pytest.CaptureFixture[str]
):
    _save_group(folder=tmp_path, planned=older_stage, group="planned", jobs=fit.planned_jobs(day=1))

    fit.print_plan(
        planned=[older_stage], output_dir=tmp_path, only_missing=False, post_hoc="older run"
    )

    out = capsys.readouterr().out
    assert "wind day 1: 5 rows (1 stage rows lack the older run)" in out
    assert "6 (arm, setting) fits" in out
    assert "6 (arm, site) fits in all" in out


def test_the_dry_run_of_the_permutation_test_lists_solar_fits_and_no_wind_fits(
    tmp_path: Path,
    tiny_solar_stage: fit.PlannedStage,
    tiny_stage: fit.PlannedStage,
    capsys: pytest.CaptureFixture[str],
):
    _save_group(
        folder=tmp_path, planned=tiny_solar_stage, group="planned", jobs=fit.planned_jobs(day=1)
    )
    _save_group(folder=tmp_path, planned=tiny_stage, group="planned", jobs=fit.planned_jobs(day=1))

    fit.print_plan(
        planned=[tiny_solar_stage, tiny_stage],
        output_dir=tmp_path,
        only_missing=False,
        post_hoc="permutation",
    )

    out = capsys.readouterr().out
    assert "solar day 1: 6 rows, 1 sites, " in out
    assert "15 (arm, setting) fits" in out
    assert "wind day 1: no permutation fits" in out
    assert "15 (arm, site) fits in all" in out


def test_a_build_that_read_another_run_than_the_one_asked_for_is_refused(tmp_path: Path):
    _build_folder(folder=tmp_path)

    with pytest.raises(ValueError, match="read the 3 UTC run, not the 15 UTC run"):
        fit.read_build_stamp(output_dir=tmp_path, spec=build.OLDER_RUN)
    stamp = json.loads((tmp_path / "build.json").read_text())
    older = tmp_path / "older"
    older.mkdir()
    for domain in ("solar", "wind"):
        path = older / f"{domain}_ukv_ceda_run15_inputs.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(path)
        stamp["inputs_sha256"][domain] = fit_aifs.sha256_of(path=path)
    stamp["run_hour"] = 15
    (older / "build.json").write_text(json.dumps(stamp))
    assert fit.read_build_stamp(output_dir=older, spec=build.OLDER_RUN)["run_hour"] == 15
    with pytest.raises(ValueError, match="read the 15 UTC run, not the 3 UTC run"):
        fit.read_build_stamp(output_dir=older)
