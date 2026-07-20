"""Integration tests for the CV Dagster assets.

These materialise ``eligible_time_series`` in-process against synthetic power data written to a
temporary Delta table, exercising the real asset wiring (Dagster + Delta) for the leaderboard fold
and the non-leaderboard ``smoke_test`` fold in ``conf/cv/default.yaml``. The pure, fold-specific
eligibility logic is unit-tested in ``packages/ml_core/tests/test_cv_helpers.py``.
"""

from datetime import datetime, timezone

import polars as pl
import pytest
from contracts.ml_schemas import EligibleTimeSeries
from dagster import materialize
from deltalake import write_deltalake

from nged_substation_forecast.defs.cv_assets import effective_capacity, eligible_time_series

# The leaderboard fold (conf/cv/default.yaml): train 2024-04-01..2025-06-30,
# validate 2025-07-01..2026-06-30, min_training_months=6.
FOLD_ID = "mid_2025_to_mid_2026"

# The non-leaderboard dev fold (conf/cv/default.yaml): validate 2025-02-01..2025-02-28 with the
# per-fold override min_training_months=1.
SMOKE_FOLD_ID = "smoke_test"


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _write_synthetic_power(power_delta_path: str) -> None:
    """Write a tiny synthetic power Delta with hand-picked coverage per time series.

    Only the per-series min/max observation times matter for eligibility, so two rows per series
    (first + last) suffice. Coverage is chosen so eligibility differs across series for the
    leaderboard fold (val_start 2025-07-01, val_end 2026-06-30, min_training_months=6 ⇒ first obs
    must be ≤ 2025-01-01 and last obs ≥ 2026-06-30):

    - ts1: 2024-06-01 .. 2026-07-01 -> eligible (enough history, reaches val_end).
    - ts2: 2024-06-01 .. 2026-03-01 -> not eligible (stops before val_end).
    - ts3: 2025-03-01 .. 2026-07-01 -> not eligible (< 6 months history before val_start).
    - ts4: 2024-12-01 .. 2025-03-01 -> not eligible for the leaderboard fold (stops before val_end),
      but eligible for the smoke fold *only because* its per-fold min_training_months=1 override is
      honoured: smoke val_start 2025-02-01 needs first obs ≤ 2025-01-01 (met by 2024-12-01), whereas
      the default 6 months would require ≤ 2024-08-01 (not met). So ts4 is the override's witness.
    """
    coverage = {
        1: (_utc(2024, 6, 1), _utc(2026, 7, 1)),
        2: (_utc(2024, 6, 1), _utc(2026, 3, 1)),
        3: (_utc(2025, 3, 1), _utc(2026, 7, 1)),
        4: (_utc(2024, 12, 1), _utc(2025, 3, 1)),
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
    effective_capacity_path = tmp_path / "effective_capacity"
    nged_path.mkdir()
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(eligible_path))
    monkeypatch.setenv("EFFECTIVE_CAPACITY_DATA_PATH", str(effective_capacity_path))

    _write_synthetic_power(str(nged_path / "power_time_series.delta"))
    return {"eligible": str(eligible_path), "effective_capacity": str(effective_capacity_path)}


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


def test_eligible_time_series_honours_per_fold_min_training_months_override(cv_paths) -> None:
    """The smoke fold's min_training_months=1 override widens eligibility to include ts4.

    ts4 (first obs 2024-12-01) would be excluded under the config-level default of 6 months but is
    eligible here because the fold's own override of 1 month is applied (see ts4 in the fixture).
    """
    assert materialize([eligible_time_series], partition_key=SMOKE_FOLD_ID).success
    assert _read_eligible(cv_paths["eligible"], SMOKE_FOLD_ID) == [1, 2, 4]


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


def test_effective_capacity_materialises_one_row_per_series(cv_paths) -> None:
    """The asset writes one full-history P99 row per time series with a plausible capacity.

    The synthetic power fixture has ``power == 1.0`` for every observation, so P99(|power|) == 1.0
    for all four series, and ``time`` is each series' latest observation.
    """
    assert materialize([effective_capacity]).success

    capacity = pl.read_delta(cv_paths["effective_capacity"]).sort("time_series_id")
    assert capacity["time_series_id"].to_list() == [1, 2, 3, 4]
    assert capacity["effective_capacity_mw"].to_list() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    # `time` is the latest observed timestep per series (see _write_synthetic_power coverage).
    assert capacity.filter(pl.col("time_series_id") == 1)["time"][0] == _utc(2026, 7, 1)


def test_effective_capacity_is_idempotent(cv_paths) -> None:
    """Re-materialising overwrites the whole table rather than appending duplicate rows."""
    assert materialize([effective_capacity]).success
    assert materialize([effective_capacity]).success

    assert pl.read_delta(cv_paths["effective_capacity"]).height == 4
