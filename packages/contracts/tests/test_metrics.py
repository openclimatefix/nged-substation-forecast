from datetime import UTC, datetime

import patito as pt
import pytest
from contracts.ml_schemas import Metrics


def _full_key_rows(
    window_ends: list[datetime], metric_values: list[float] | None = None
) -> pt.DataFrame[Metrics]:
    """`Metrics` rows sharing every key column except `window_end`, with the full key present.

    `metric_values` defaults to the same value for every row; pass distinct values to build rows
    that share their full primary key but differ in a non-key column.
    """
    n = len(window_ends)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    return (
        Metrics.DataFrame(
            {
                "time_series_id": [123] * n,
                "power_fcst_model_name": ["xgboost"] * n,
                "experiment_name": ["baseline"] * n,
                "fold_id": ["live"] * n,
                "evaluation_scope": ["production_monitoring"] * n,
                "horizon_slice": ["all"] * n,
                "metric_name": ["mae"] * n,
                "metric_param": ["all"] * n,
                "metric_value": metric_values if metric_values is not None else [1.23] * n,
                "window_start": [window_start] * n,
                "window_end": window_ends,
            }
        )
        .cast()
        .validate()
    )


def test_metrics_rejects_duplicate_primary_key():
    """Two rows sharing the full primary key mean the same window was scored twice."""
    window_end = datetime(2026, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="Duplicate entries found for primary key"):
        _full_key_rows(window_ends=[window_end, window_end])


def test_metrics_rejects_duplicate_primary_key_with_different_metric_value():
    """The check compares only `PRIMARY_KEY` columns, not the whole row.

    Two rows sharing the full key but disagreeing on `metric_value` are still a duplicate key —
    the same window scored twice with a different result, not two legitimately distinct rows.
    """
    window_end = datetime(2026, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="Duplicate entries found for primary key"):
        _full_key_rows(window_ends=[window_end, window_end], metric_values=[1.23, 4.56])


def test_metrics_accepts_rows_differing_only_in_window_end():
    """`window_end` is part of the key, so two windows for the same series are not duplicates."""
    _full_key_rows(window_ends=[datetime(2026, 1, 2, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)])


def test_metrics_validate_skips_uniqueness_check_when_key_columns_missing():
    """`compute_metrics()` validates before `experiment_name`/`evaluation_scope`/`window_start`/
    `window_end` exist on the frame, on rows that would collide on every column that *is* present.

    Those four columns are `allow_missing`, so `validate()` must not try to select them — it
    would raise `ColumnNotFoundError` rather than validate, breaking `compute_metrics()` and
    `_score_forecast_group` on their normal calling pattern.
    """
    rows = Metrics.DataFrame(
        {
            "time_series_id": [123, 123],
            "power_fcst_model_name": ["xgboost", "xgboost"],
            "fold_id": ["live", "live"],
            "horizon_slice": ["all", "all"],
            "metric_name": ["mae", "mae"],
            "metric_param": ["all", "all"],
            "metric_value": [1.23, 4.56],
        }
    ).cast()

    rows.validate(allow_superfluous_columns=True)
