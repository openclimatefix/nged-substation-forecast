from datetime import UTC, datetime

import polars as pl
import pytest
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, HyperParameters
from studies.sources import STUDY_INPUTS_DIR

from studies import arm_runner


def test_time_features_are_the_hour_the_day_of_year_and_the_month_label():
    dataset = pl.DataFrame({"time": [datetime(2024, 3, 1, 13, tzinfo=UTC)]})

    result = arm_runner.add_time_features(dataset=dataset)

    assert result.select("hour_of_day", "day_of_year", "month").row(0) == (13, 61, "2024-03")


def test_the_dataset_path_is_named_for_the_source():
    assert arm_runner.dataset_path_for(source="cams") == (
        STUDY_INPUTS_DIR / "beam_diffuse_dataset_cams.parquet"
    )


def test_run_all_fits_every_job_on_each_sites_own_rows_with_the_jobs_own_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_losses(
        *,
        site_rows: pl.DataFrame,
        features: tuple[str, ...],
        target: str,
        hyper_parameters: HyperParameters,
        with_quantiles: bool,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "site": [site_rows["site"][0]],
                "n_rows": [site_rows.height],
                "features": [",".join(features)],
                "fitted_target": [target],
                "max_depth": [hyper_parameters["max_depth"]],
                "quantiles": [with_quantiles],
            }
        )

    monkeypatch.setattr(arm_runner, "out_of_fold_losses", fake_losses)
    dataset = pl.DataFrame({"site": ["A", "B", "B", "C", "C", "C"]})
    other_settings: HyperParameters = {**PRIMARY_HYPER_PARAMETERS, "max_depth": 3}
    jobs: list[arm_runner.Job] = [
        ("arm_one", "primary", "power", ("x",), PRIMARY_HYPER_PARAMETERS, False),
        ("arm_two", "sensitivity", "synthetic", ("x", "y"), other_settings, True),
    ]

    result = arm_runner.run_all(dataset=dataset, jobs=jobs, max_workers=2)

    assert sorted(result.rows()) == sorted(
        (site, n_rows, features, target, max_depth, quantiles, arm, setting, target)
        for site, n_rows in (("A", 1), ("B", 2), ("C", 3))
        for arm, setting, target, features, max_depth, quantiles in (
            ("arm_one", "primary", "power", "x", PRIMARY_HYPER_PARAMETERS["max_depth"], False),
            ("arm_two", "sensitivity", "synthetic", "x,y", 3, True),
        )
    )
