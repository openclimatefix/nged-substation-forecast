from datetime import UTC, datetime

import polars as pl
import pytest
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS
from studies.sources import STUDY_DATA_DIR

from studies import arm_runner


def test_time_features_are_the_hour_the_day_of_year_and_the_month_label():
    dataset = pl.DataFrame({"time": [datetime(2024, 3, 1, 13, tzinfo=UTC)]})

    result = arm_runner.add_time_features(dataset=dataset)

    assert result.select("hour_of_day", "day_of_year", "month").row(0) == (13, 61, "2024-03")


def test_the_dataset_path_is_named_for_the_source():
    assert arm_runner.dataset_path_for(source="cams") == (
        STUDY_DATA_DIR / "beam_diffuse_dataset_cams.parquet"
    )


def test_run_all_fits_every_job_at_every_site_and_labels_the_losses(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_losses(
        *, site_rows: pl.DataFrame, features: tuple[str, ...], **_: object
    ) -> pl.DataFrame:
        return pl.DataFrame({"n_rows": [site_rows.height], "n_features": [len(features)]})

    monkeypatch.setattr(arm_runner, "out_of_fold_losses", fake_losses)
    dataset = pl.DataFrame({"site": ["A", "A", "B"]})
    jobs: list[arm_runner.Job] = [
        ("arm_one", "primary", "power", ("x",), PRIMARY_HYPER_PARAMETERS, False),
        ("arm_two", "primary", "power", ("x", "y"), PRIMARY_HYPER_PARAMETERS, False),
    ]

    result = arm_runner.run_all(dataset=dataset, jobs=jobs, max_workers=2)

    assert sorted(result.rows()) == [
        (1, 1, "arm_one", "primary", "power"),
        (1, 2, "arm_two", "primary", "power"),
        (2, 1, "arm_one", "primary", "power"),
        (2, 2, "arm_two", "primary", "power"),
    ]
