"""Tests for ``write_forecast_metrics`` — the ``forecast_metrics`` storage format end-to-end.

Writes real (tiny) ``Metrics`` frames into a temp Delta table and asserts the on-disk Enum→String
cast and the per-``(experiment_name, fold_id)``-partition overwrite semantics.
"""

from pathlib import Path

import patito as pt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from contracts.ml_schemas import Metrics
from delta_store.forecast_metrics import write_forecast_metrics


def _make_metrics(
    experiment_name: str, fold_id: str, time_series_ids: list[int]
) -> pt.DataFrame[Metrics]:
    n = len(time_series_ids)
    return (
        Metrics.DataFrame(
            {
                "time_series_id": time_series_ids,
                "power_fcst_model_name": ["xgboost"] * n,
                "fold_id": [fold_id] * n,
                "horizon_slice": ["all"] * n,
                "metric_name": ["mae"] * n,
                "metric_param": ["all"] * n,
                "metric_value": [1.23] * n,
                "experiment_name": [experiment_name] * n,
            }
        )
        .cast()
        .validate()
    )


def test_enum_columns_cast_to_string_on_disk(tmp_path: Path) -> None:
    table = tmp_path / "forecast_metrics"
    write_forecast_metrics(
        _make_metrics("exp_a", "fold_a", [1]), table, experiment_name="exp_a", fold_id="fold_a"
    )

    parquet_files = list(table.rglob("*.parquet"))
    assert parquet_files
    schema = pq.ParquetFile(parquet_files[0]).schema_arrow
    for col in ("metric_name", "horizon_slice", "metric_param"):
        field_type = schema.field(col).type
        assert pa.types.is_string(field_type) or pa.types.is_large_string(field_type), (
            f"{col} written as {field_type}"
        )


def test_overwrite_is_partition_scoped_by_experiment_and_fold(tmp_path: Path) -> None:
    table = tmp_path / "forecast_metrics"
    write_forecast_metrics(
        _make_metrics("exp_a", "fold_a", [1, 2]), table, experiment_name="exp_a", fold_id="fold_a"
    )
    write_forecast_metrics(
        _make_metrics("exp_a", "fold_b", [3]), table, experiment_name="exp_a", fold_id="fold_b"
    )
    # Same fold_id as the first write, but a different experiment_name — the predicate must key on
    # both columns, or this overwrite would also wipe "exp_a"'s "fold_a" rows.
    write_forecast_metrics(
        _make_metrics("exp_b", "fold_a", [7]), table, experiment_name="exp_b", fold_id="fold_a"
    )
    write_forecast_metrics(
        _make_metrics("exp_a", "fold_a", [4, 5, 6]),
        table,
        experiment_name="exp_a",
        fold_id="fold_a",
    )

    partition_dirs = sorted(
        str(p.relative_to(table)) for p in table.glob("experiment_name=*/fold_id=*")
    )
    assert partition_dirs == [
        "experiment_name=exp_a/fold_id=fold_a",
        "experiment_name=exp_a/fold_id=fold_b",
        "experiment_name=exp_b/fold_id=fold_a",
    ]

    stored = pl.read_delta(str(table))
    a_ids = stored.filter((pl.col("experiment_name") == "exp_a") & (pl.col("fold_id") == "fold_a"))[
        "time_series_id"
    ]
    assert sorted(a_ids) == [4, 5, 6]
    assert stored.filter(pl.col("fold_id") == "fold_b")["time_series_id"].to_list() == [3]
    assert stored.filter(pl.col("experiment_name") == "exp_b")["time_series_id"].to_list() == [7]
