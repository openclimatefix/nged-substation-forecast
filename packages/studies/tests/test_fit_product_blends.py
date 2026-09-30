import json
import re
import sys
from pathlib import Path

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

import fit_aifs  # noqa: E402
import fit_product_blends as fpb  # noqa: E402
from fit_aifs import PRIMARY, SENSITIVITY, arm_features  # noqa: E402
from fit_product_blends import (  # noqa: E402
    OUTPUT_DIR_NAME,
    SAME_BUILD_KEYS,
    PlannedStage,
    Stage,
    blend_verdict_at_both_settings,
    check_keys_match,
    check_output_dir,
    check_same_build,
    combine_losses,
    fit_single_stage,
    missing_jobs,
    product_contrasts,
    ranking_verdict,
    report_text,
    reused_arms,
    stage_jobs,
    stage_losses,
)
from nwp_forecast_comparison import METRIC, TARGET, DomainType  # noqa: E402
from studies.bootstrap import NO_DETECTABLE_DIFFERENCE, BootstrapInterval  # noqa: E402

SITES = ("A", "B")
SEEDS = (0, 1, 2)
CAPACITY_MW = {"A": 10.0, "B": 25.0}


def _interval(*, lower: float, upper: float) -> BootstrapInterval:
    return {
        "difference": (lower + upper) / 2,
        "lower_95": lower,
        "upper_95": upper,
        "seed_spread": 0.0,
        "n_rows": 10,
        "n_months": 12,
    }


# --- the fit list --------------------------------------------------------------------------------


def _saved(*, pairs: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"arm": [a for a, _ in pairs], "setting": [s for _, s in pairs]})


def test_a_missing_pair_is_every_arm_and_setting_the_saved_losses_lack():
    arms = reused_arms(day=2)
    saved = _saved(pairs=[(arm, PRIMARY) for arm in arms] + [(arms[0], SENSITIVITY)])

    assert missing_jobs(saved=saved, arms=arms) == [
        (arms[1], SENSITIVITY),
        (arms[2], SENSITIVITY),
    ]
    assert missing_jobs(saved=_saved(pairs=[]), arms=arms[:1]) == [
        (arms[0], PRIMARY),
        (arms[0], SENSITIVITY),
    ]


def test_a_single_stage_fits_each_new_arm_at_both_settings_and_refits_only_what_is_missing():
    arms = reused_arms(day=1)
    saved = _saved(
        pairs=[(arm, setting) for arm in arms for setting in (PRIMARY, SENSITIVITY)][:-1]
    )  # ENS's mean at the sensitivity setting is the one missing pair.

    jobs = stage_jobs(stage=Stage("solar", "single", 1), saved=saved)

    assert jobs[-1] == ("ens_mean_day1", SENSITIVITY)
    new = [job for job in jobs if job[0].startswith("blend_") and "aifs" not in job[0]]
    assert len(new) == 12 == len(jobs) - 1
    assert {setting for _, setting in new} == {PRIMARY, SENSITIVITY}


def test_a_day_with_nothing_new_and_nothing_missing_has_no_fits():
    arms = reused_arms(day=7)
    saved = _saved(pairs=[(arm, s) for arm in arms for s in (PRIMARY, SENSITIVITY)])

    assert stage_jobs(stage=Stage("wind", "single", 7), saved=saved) == []


def test_a_wn3_stage_fits_ens_mean_the_blend_and_its_control_at_the_primary_setting_only():
    assert stage_jobs(stage=Stage("wind", "wn3", 14), saved=None) == [
        ("ens_mean_day14", PRIMARY),
        ("blend_wn3_day14", PRIMARY),
        ("blend_wn3_day14_control", PRIMARY),
    ]


# --- the stamp and row checks --------------------------------------------------------------------

_STAMP = {
    "inputs_sha256": "a",
    "published_sha256": "b",
    "extra_sha256": "c",
    "device": "cuda",
    "gpu": "GPU 0: NVIDIA RTX A6000",
    "xgboost": "3.4.1",
    "settings": "s",
    "seeds": "{}",
    "columns": "new arms",
    "polars": "1.44.2",
}


def test_the_same_build_passes_even_where_the_arms_and_the_polars_version_differ():
    check_same_build(
        stamp=_STAMP, reused_stamp={**_STAMP, "columns": "old arms", "polars": "1.50.0"}
    )


@pytest.mark.parametrize("key", SAME_BUILD_KEYS)
def test_a_build_that_differs_in_any_shared_entry_is_refused(key: str):
    with pytest.raises(ValueError, match=key):
        check_same_build(stamp=_STAMP, reused_stamp={**_STAMP, key: "other"})


def test_the_inputs_gpu_and_xgboost_version_are_among_the_entries_compared():
    assert {"inputs_sha256", "gpu", "xgboost"} <= set(SAME_BUILD_KEYS)


def _keys(
    *, months: int = 4, sites: tuple[str, ...] = SITES, seeds: tuple[int, ...] = SEEDS
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"site": site, "time": month * 100 + hour, "seed": seed, "fold": month % 2}
            for site in sites
            for month in range(months)
            for hour in range(2)
            for seed in seeds
        ]
    )


def _saved_reference(*, frame: pl.DataFrame, arm: str = "ens_mean_day1") -> pl.DataFrame:
    return frame.with_columns(arm=pl.lit(arm), setting=pl.lit(PRIMARY))


def _new(*, frame: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [
            frame.with_columns(arm=pl.lit(arm), setting=pl.lit(setting))
            for arm, setting in [
                ("blend_icon_eu_day1", PRIMARY),
                ("blend_icon_eu_day1", SENSITIVITY),
            ]
        ]
    )


def test_new_arms_holding_the_saved_ens_means_rows_pass():
    keys = _keys()

    check_keys_match(
        losses=_new(frame=keys),
        saved=_saved_reference(frame=keys),
        reference_arm="ens_mean_day1",
        label="s",
    )


@pytest.mark.parametrize(
    "mutated",
    [
        _keys().slice(1),  # a row fewer
        pl.concat([_keys(), _keys().head(1)]),  # a duplicated row
        _keys(months=5),  # a month more
        _keys(sites=("A",)),  # a site fewer
        _keys(seeds=(0, 1)),  # a seed fewer
        _keys(seeds=(0, 1, 5)),  # the same count of seeds, one of them another seed
        _keys().with_columns(fold=pl.col("fold") + 1),  # the same rows in other folds
        _keys().with_columns(time=pl.col("time") + 1),  # the same hours shifted
    ],
    ids=["row", "duplicate", "month", "site", "seed", "other-seed", "fold", "time"],
)
def test_new_arms_whose_rows_differ_from_the_saved_ens_means_are_refused(mutated: pl.DataFrame):
    with pytest.raises(ValueError, match="blend_icon_eu_day1 at"):
        check_keys_match(
            losses=_new(frame=mutated),
            saved=_saved_reference(frame=_keys()),
            reference_arm="ens_mean_day1",
            label="s",
        )


def test_the_keys_are_compared_with_the_named_reference_arm_not_another():
    keys = _keys()
    saved = pl.concat(
        [
            _saved_reference(frame=keys, arm="ens_mean_day1"),
            _saved_reference(frame=keys.slice(1), arm="ens_mean_day2"),
        ]
    )

    check_keys_match(losses=_new(frame=keys), saved=saved, reference_arm="ens_mean_day1", label="s")
    with pytest.raises(ValueError, match="ens_mean_day2"):
        check_keys_match(
            losses=_new(frame=keys), saved=saved, reference_arm="ens_mean_day2", label="s"
        )


# --- a stage, with the fit stubbed ---------------------------------------------------------------


def _frame(*, arms: list[str], domain: DomainType, months: int = 4) -> pl.DataFrame:
    columns = {column for arm in arms for column in arm_features(arm=arm, domain=domain)}
    base = _keys(months=months, seeds=(0,)).drop("seed")
    return base.with_columns(
        month=pl.col("time") // 100,
        effective_capacity_mw=pl.col("site").replace_strict(CAPACITY_MW, return_dtype=pl.Float64),
        **dict.fromkeys(sorted(columns), 1.0),
        **{TARGET: pl.lit(2.0)},
    )


@pytest.fixture
def stub_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(**kwargs: pl.DataFrame) -> pl.DataFrame:
        rows = kwargs["site_rows"]
        return (
            rows.select("site", "time", "month", "fold", "effective_capacity_mw")
            .join(pl.DataFrame({"seed": list(SEEDS)}), how="cross")
            .with_columns(**{METRIC: pl.lit(0.1)}, signed_error_capped_mw=pl.lit(0.5))
        )

    monkeypatch.setattr(fit_aifs, "out_of_fold_losses", fake)


def _saved_stage(*, day: int, frame: pl.DataFrame, pairs: list[tuple[str, str]]) -> pl.DataFrame:
    keys = frame.select("site", "time", "fold").join(
        pl.DataFrame({"seed": list(SEEDS)}), how="cross"
    )
    return pl.concat(
        [keys.with_columns(arm=pl.lit(arm), setting=pl.lit(setting)) for arm, setting in pairs]
    )


def _stage_inputs(*, day: int) -> tuple[Stage, pl.DataFrame, pl.DataFrame]:
    stage = Stage("solar", "single", day)
    arms = [*fpb.product_blend_arms(row_set="single", day=day), *reused_arms(day=day)]
    frame = _frame(arms=arms, domain="solar")
    pairs = [(arm, PRIMARY) for arm in reused_arms(day=day)]
    return stage, frame, _saved_stage(day=day, frame=frame, pairs=pairs)


def test_a_stage_fits_its_jobs_and_keeps_every_new_arm_on_the_saved_ens_means_rows(
    stub_fit: None,
):
    stage, frame, saved = _stage_inputs(day=2)

    losses = fit_single_stage(frame=frame, stage=stage, saved=saved, workers=1)

    assert set(losses["arm"]) == {arm for arm, _ in stage_jobs(stage=stage, saved=saved)}
    assert set(losses["device"]) == {"cuda"}
    assert set(losses["setting"]) == {PRIMARY, SENSITIVITY}


def test_a_stage_refuses_a_frame_and_a_saved_ens_mean_that_hold_different_rows(stub_fit: None):
    stage, frame, saved = _stage_inputs(day=2)
    one_row_fewer = saved.filter(
        ~((pl.col("arm") == "ens_mean_day2") & (pl.col("time") == 0) & (pl.col("site") == "A"))
    )

    with pytest.raises(ValueError, match="does not hold ens_mean_day2's"):
        fit_single_stage(frame=frame, stage=stage, saved=one_row_fewer, workers=1)
    with pytest.raises(ValueError, match="does not hold ens_mean_day2's"):
        fit_single_stage(frame=frame.slice(1), stage=stage, saved=saved, workers=1)


def test_a_stage_compares_against_the_ens_mean_of_its_own_day(stub_fit: None):
    stage, frame, saved = _stage_inputs(day=2)
    day1_only = saved.filter(pl.col("arm") != "ens_mean_day2").with_columns(
        arm=pl.when(pl.col("arm") == "blend_aifs_single_day2")
        .then(pl.lit("ens_mean_day1"))
        .otherwise(pl.col("arm"))
    )

    with pytest.raises(ValueError, match="ens_mean_day2"):
        fit_single_stage(frame=frame, stage=stage, saved=day1_only, workers=1)


def test_a_stage_writes_its_outputs_once_and_a_rerun_checks_them(stub_fit: None, tmp_path: Path):
    stage, frame, saved = _stage_inputs(day=2)
    stamp = {"device": "cuda"}
    planned = PlannedStage(stage, frame, stage_jobs(stage=stage, saved=saved), stamp, saved)

    first = stage_losses(output_dir=tmp_path, planned=planned, workers=1)
    again = stage_losses(output_dir=tmp_path, planned=planned, workers=1)

    assert first.equals(again)
    assert json.loads((tmp_path / "solar_single_day2_losses.json").read_text()) == stamp
    with pytest.raises(ValueError, match="another build or device"):
        stage_losses(
            output_dir=tmp_path,
            planned=planned._replace(stamp={"device": "cpu"}),
            workers=1,
        )


def test_losses_already_saved_and_refitted_are_refused_rather_than_counted_twice():
    saved = pl.DataFrame({"arm": ["a", "a"], "setting": [PRIMARY, SENSITIVITY], "x": [1, 2]})
    new = pl.DataFrame({"arm": ["a"], "setting": [SENSITIVITY], "x": [3]})

    with pytest.raises(ValueError, match="both saved and refitted"):
        combine_losses(saved=saved, new=new, arms=("a",))
    kept = combine_losses(saved=saved.filter(pl.col("setting") == PRIMARY), new=new, arms=("a",))
    assert kept["x"].to_list() == [1, 3]
    assert combine_losses(saved=saved, new=None, arms=("b",)).is_empty()


def test_the_output_folder_must_be_the_named_one_and_not_a_folder_the_script_reads(
    tmp_path: Path,
):
    reused = tmp_path / fit_aifs.BLENDS_DIR_NAME

    check_output_dir(output_dir=tmp_path / OUTPUT_DIR_NAME, read_only=[reused])
    with pytest.raises(ValueError, match="writes only"):
        check_output_dir(output_dir=reused, read_only=[reused])
    with pytest.raises(ValueError, match="writes only"):
        check_output_dir(output_dir=tmp_path / "another", read_only=[reused])


# --- the contrasts, the verdicts, and the ranking rule -------------------------------------------


def test_the_planned_contrasts_name_the_arms_the_plan_states():
    listed = {(c.code, c.day, c.treatment, c.reference, c.planned) for c in product_contrasts()}

    assert ("C1", 2, "blend_icon_eu_conservative_day2", "ens_mean_day2", True) in listed
    assert ("C1", 2, "blend_icon_eu_day2", "ens_mean_day2", False) in listed
    assert ("C2", 7, "blend_aifs_single_day7", "ens_mean_day7", True) in listed
    assert ("C3", 1, "blend_ukv_day1", "ens_mean_day1", True) in listed
    assert ("C4", 1, "blend_ukv_day1", "blend_ukv_day1_control", True) in listed
    assert ("C5", 2, "blend_aifs_single_day2", "blend_icon_eu_conservative_day2", True) in listed
    assert ("C5", 2, "blend_aifs_single_day2", "blend_icon_eu_day2", True) in listed
    assert {c.day for c in product_contrasts() if c.code in ("C1", "C3", "C5")} == {1, 2}
    assert not [c for c in product_contrasts() if "ukv" in c.treatment and c.day != 1]
    assert {c.day for c in product_contrasts() if c.code == "C2"} == {1, 2, 7}


_LOWER = _interval(lower=-0.02, upper=-0.01)
_SPANS = _interval(lower=-0.02, upper=0.01)
_HIGHER = _interval(lower=0.01, upper=0.02)


def test_a_blend_is_recommended_only_if_both_settings_give_the_verdict():
    both = {PRIMARY: _LOWER, SENSITIVITY: _LOWER}

    lowers = blend_verdict_at_both_settings(day=2, versus_ens=both, versus_control=both)
    assert lowers == "lowers the error at day 2"
    assert (
        blend_verdict_at_both_settings(
            day=2, versus_ens={PRIMARY: _LOWER, SENSITIVITY: _SPANS}, versus_control=both
        )
        == NO_DETECTABLE_DIFFERENCE
    )
    # The control guard matters: a blend no better than its shuffled control is not recommended.
    assert (
        blend_verdict_at_both_settings(
            day=2, versus_ens=both, versus_control={PRIMARY: _LOWER, SENSITIVITY: _SPANS}
        )
        == NO_DETECTABLE_DIFFERENCE
    )


def _ranking(*, optimistic: BootstrapInterval, conservative: BootstrapInterval) -> str:
    return ranking_verdict(
        versus_optimistic={PRIMARY: optimistic, SENSITIVITY: optimistic},
        versus_conservative={PRIMARY: conservative, SENSITIVITY: conservative},
    )


def test_aifs_single_ranks_first_only_if_it_beats_the_optimistic_icon_eu_blend():
    assert _ranking(optimistic=_LOWER, conservative=_LOWER) == "AIFS Single ranks above ICON-EU"
    # Beating only the handicapped (conservative) ICON-EU blend settles nothing.
    assert _ranking(optimistic=_SPANS, conservative=_LOWER) == "not separable"


def test_icon_eu_ranks_first_only_if_it_beats_aifs_single_with_the_conservative_lead():
    assert _ranking(optimistic=_HIGHER, conservative=_HIGHER) == "ICON-EU ranks above AIFS Single"
    # Beating AIFS Single only with the fresher (optimistic) lead settles nothing.
    assert _ranking(optimistic=_HIGHER, conservative=_SPANS) == "not separable"


def test_a_ranking_needs_both_settings_to_agree():
    assert (
        ranking_verdict(
            versus_optimistic={PRIMARY: _LOWER, SENSITIVITY: _SPANS},
            versus_conservative={PRIMARY: _LOWER, SENSITIVITY: _LOWER},
        )
        == "not separable"
    )
    assert (
        ranking_verdict(
            versus_optimistic={PRIMARY: _SPANS, SENSITIVITY: _SPANS},
            versus_conservative={PRIMARY: _HIGHER, SENSITIVITY: _SPANS},
        )
        == "not separable"
    )


# --- the report ----------------------------------------------------------------------------------

ERRORS = {
    "ens_mean": 0.10,
    "blend_aifs_single": 0.08,
    "blend_aifs_single_control": 0.11,
    "blend_icon_eu": 0.085,
    "blend_icon_eu_control": 0.11,
    "blend_icon_eu_conservative": 0.09,
    "blend_icon_eu_conservative_control": 0.11,
    "blend_ukv": 0.095,
    "blend_ukv_control": 0.11,
}


def _error(*, arm: str) -> float:
    return ERRORS[re.sub(r"_day\d+", "", arm)]


def _arm_rows(*, arm: str, setting: str, error: float) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"arm": arm, "site": site, "time": month * 100 + hour, "seed": seed, "month": month}
            for site in SITES
            for month in range(12)
            for hour in range(2)
            for seed in SEEDS
        ]
    ).with_columns(setting=pl.lit(setting), **{METRIC: pl.lit(error)})


def _combined(*, day: int) -> pl.DataFrame:
    arms = {
        name for c in product_contrasts() if c.day == day for name in (c.treatment, c.reference)
    }
    return pl.concat(
        [
            _arm_rows(arm=arm, setting=setting, error=_error(arm=arm))
            for arm in sorted(arms)
            for setting in (PRIMARY, SENSITIVITY)
        ]
    )


def test_the_report_prints_the_contrasts_at_both_settings_the_verdicts_and_the_ranking(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("studies.bootstrap.N_BOOTSTRAP_RESAMPLES", 50)
    singles: dict[DomainType, dict[int, pl.DataFrame]] = {
        domain: {day: _combined(day=day) for day in fpb.single_days()}
        for domain in ("solar", "wind")
    }
    wn3: dict[DomainType, dict[int, pl.DataFrame]] = {
        domain: {
            day: pl.concat(
                [
                    _arm_rows(arm=arm, setting=PRIMARY, error=error)
                    for arm, error in (
                        (f"ens_mean_day{day}", 0.10),
                        (f"blend_wn3_day{day}", 0.09),
                        (f"blend_wn3_day{day}_control", 0.12),
                    )
                ]
            )
            for day in fpb.WN3_DAYS
        }
        for domain in ("solar", "wind")
    }

    text = report_text(singles=singles, wn3=wn3, refits=["- solar day 1: `x` at sensitivity"])

    assert "- solar day 1: `x` at sensitivity" in text
    assert "`blend_aifs_single_day7`: lowers the error at day 7" in text
    assert "`blend_ukv_day1`: lowers the error at day 1" in text
    assert "- day 2: AIFS Single ranks above ICON-EU" in text
    assert "| C5 | 2 | `blend_aifs_single_day2` minus `blend_icon_eu_conservative_day2` |" in text
    assert "`blend_icon_eu_conservative_day1_control` (9 columns)" in text
    assert "`blend_wn3_day14` minus `blend_wn3_day14_control`" in text
    assert "uncorrected for multiplicity" in text
