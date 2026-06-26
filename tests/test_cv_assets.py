"""Integration tests for the CV Dagster assets.

These materialise ``eligible_time_series`` in-process against synthetic power data written to a
temporary Delta table, exercising the real asset wiring (Dagster + Delta) for the single canonical
fold in ``conf/cv/default.yaml``. The pure, fold-specific eligibility logic is unit-tested in
``packages/ml_core/tests/test_cv_helpers.py``.
"""

from datetime import datetime, timezone

import polars as pl
import pytest
from contracts.ml_schemas import EligibleTimeSeries
from dagster import materialize
from deltalake import write_deltalake

from nged_substation_forecast.defs.cv_assets import eligible_time_series

# The single canonical fold (conf/cv/default.yaml): train 2024-04-01..2025-06-30,
# validate 2025-07-01..2026-06-30, min_training_months=6.
FOLD_ID = "mid_2025_to_mid_2026"


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _write_synthetic_power(power_delta_path: str) -> None:
    """Write a tiny synthetic power Delta with hand-picked coverage per time series.

    Only the per-series min/max observation times matter for eligibility, so two rows per series
    (first + last) suffice. Coverage is chosen so eligibility differs across series for the
    canonical fold (val_start 2025-07-01, val_end 2026-06-30, min_training_months=6 ⇒ first obs
    must be ≤ 2025-01-01 and last obs ≥ 2026-06-30):

    - ts1: 2024-06-01 .. 2026-07-01 -> eligible (enough history, reaches val_end).
    - ts2: 2024-06-01 .. 2026-03-01 -> not eligible (stops before val_end).
    - ts3: 2025-03-01 .. 2026-07-01 -> not eligible (< 6 months history before val_start).
    """
    coverage = {
        1: (_utc(2024, 6, 1), _utc(2026, 7, 1)),
        2: (_utc(2024, 6, 1), _utc(2026, 3, 1)),
        3: (_utc(2025, 3, 1), _utc(2026, 7, 1)),
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
    assert materialize([eligible_time_series], partition_key=FOLD_ID).success
    assert _read_eligible(cv_paths["eligible"], FOLD_ID) == [1]


def test_eligible_time_series_is_idempotent(cv_paths) -> None:
    """Re-materialising a fold overwrites its partition rather than appending duplicates."""
    assert materialize([eligible_time_series], partition_key=FOLD_ID).success
    assert materialize([eligible_time_series], partition_key=FOLD_ID).success

    fold_rows = pl.read_delta(cv_paths["eligible"]).filter(pl.col("fold_id") == FOLD_ID)
    assert len(fold_rows) == 1  # not 2
    assert fold_rows["time_series_id"].to_list() == [1]


def test_eligible_time_series_overwrite_is_partition_scoped(cv_paths) -> None:
    """The partition overwrite touches only its own fold, leaving sibling folds intact.

    Pre-seed the eligible table with a row for an unrelated fold (e.g. a prior leaderboard epoch),
    then materialise the canonical fold and assert the sibling partition survives.
    """
    seeded = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series(["prior_epoch"], dtype=pl.String),
                "time_series_id": pl.Series([99], dtype=pl.Int32),
            }
        )
    )
    write_deltalake(
        table_or_uri=cv_paths["eligible"],
        data=seeded.to_arrow(),
        mode="overwrite",
        partition_by=["fold_id"],
    )

    assert materialize([eligible_time_series], partition_key=FOLD_ID).success

    assert _read_eligible(cv_paths["eligible"], "prior_epoch") == [99]
    assert _read_eligible(cv_paths["eligible"], FOLD_ID) == [1]
