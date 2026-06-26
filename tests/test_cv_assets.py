"""Integration tests for the CV Dagster assets.

These materialise ``eligible_time_series`` in-process against synthetic power data written to a
temporary Delta table, exercising the real asset wiring (Dagster + Delta). The pure eligibility
logic itself is unit-tested in ``packages/ml_core/tests/test_cv_helpers.py``.
"""

from datetime import datetime, timezone

import polars as pl
import pytest
from dagster import materialize

from nged_substation_forecast.defs.cv_assets import eligible_time_series


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _write_synthetic_power(power_delta_path: str) -> None:
    """Write a tiny synthetic power Delta with hand-picked coverage per time series.

    Only the per-series min/max observation times matter for eligibility, so two rows per series
    (first + last) suffice. Coverage is chosen so the eligible set differs between folds 2022 and
    2023 (min_training_months=6):

    - ts1: 2021-01-01 .. 2024-01-01 -> eligible for both 2022 and 2023.
    - ts2: 2021-01-01 .. 2023-06-01 -> eligible 2022 (reaches val_end), not 2023 (stops mid-year).
    - ts3: 2021-01-01 .. 2022-06-01 -> eligible for neither (stops before 2022 val_end).
    """
    coverage = {
        1: (_utc(2021, 1, 1), _utc(2024, 1, 1)),
        2: (_utc(2021, 1, 1), _utc(2023, 6, 1)),
        3: (_utc(2021, 1, 1), _utc(2022, 6, 1)),
    }
    rows = [
        {"time_series_id": ts, "time": t, "power": 1.0}
        for ts, (first, last) in coverage.items()
        for t in (first, last)
    ]
    df = pl.DataFrame(rows).cast(
        {
            "time_series_id": pl.Int32,
            "time": pl.Datetime("us", "UTC"),
            "power": pl.Float32,
        }
    )
    df.write_delta(power_delta_path)


@pytest.fixture
def cv_paths(tmp_path, monkeypatch) -> dict[str, str]:
    """Redirect Settings' data paths to a temp dir and supply dummy required secrets.

    Settings reads these from the environment, which takes precedence over any ``.env`` file, so
    the test never touches real data or credentials.
    """
    nged_path = tmp_path / "NGED"
    eligible_path = tmp_path / "eligible_time_series"
    nged_path.mkdir()
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(eligible_path))

    _write_synthetic_power(str(nged_path / "power_time_series.delta"))
    return {"eligible": str(eligible_path)}


def _read_eligible(eligible_path: str, fold_id: str) -> list[int]:
    return sorted(
        pl.read_delta(eligible_path)
        .filter(pl.col("fold_id") == fold_id)["time_series_id"]
        .to_list()
    )


def test_eligible_time_series_materialises_per_fold_population(cv_paths) -> None:
    assert materialize([eligible_time_series], partition_key="2022").success
    assert _read_eligible(cv_paths["eligible"], "2022") == [1, 2]


def test_eligible_time_series_differs_between_folds(cv_paths) -> None:
    """The eligible population is fold-specific: a later val_end excludes shorter series."""
    assert materialize([eligible_time_series], partition_key="2022").success
    assert materialize([eligible_time_series], partition_key="2023").success

    # Both fold partitions coexist (writing 2023 must not clobber 2022).
    assert _read_eligible(cv_paths["eligible"], "2022") == [1, 2]
    assert _read_eligible(cv_paths["eligible"], "2023") == [1]


def test_eligible_time_series_is_idempotent(cv_paths) -> None:
    """Re-materialising a fold overwrites its partition rather than appending duplicates."""
    assert materialize([eligible_time_series], partition_key="2022").success
    assert materialize([eligible_time_series], partition_key="2022").success

    fold_rows = pl.read_delta(cv_paths["eligible"]).filter(pl.col("fold_id") == "2022")
    assert len(fold_rows) == 2  # not 4
    assert sorted(fold_rows["time_series_id"].to_list()) == [1, 2]
