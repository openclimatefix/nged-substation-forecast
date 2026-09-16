"""Storage policy for the ``forecast_metrics`` Delta table.

No writer-properties, sort-order or precision tuning yet — see this package's ``__init__``
docstring for which tables carry that tuning. This module exists so the write itself goes through
``delta_store`` like every other table, and so the Enum→String cast below — on-disk-format
knowledge, not evaluation logic — lives next to the write it protects rather than in a Dagster
module.
"""

from pathlib import Path

import patito as pt
import polars as pl
from contracts.ml_schemas import Metrics
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import write_deltalake


def write_forecast_metrics(
    metrics: pt.DataFrame[Metrics],
    table_uri: str | Path,
    experiment_name: str,
    fold_id: str,
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write ``Metrics`` rows to the ``forecast_metrics`` Delta table.

    Casts the ``Enum`` columns (e.g. ``metric_name``, ``horizon_slice``) to ``String`` before
    writing — delta-rs stores Arrow dictionary arrays as plain String in Parquet, so the on-disk
    schema is always String. Re-sending Enum data on an overwrite would cause a schema-mismatch
    error. Performs an idempotent overwrite of the ``(experiment_name, fold_id)`` partition
    so re-materialising the asset replaces rows rather than duplicating them.

    Args:
        metrics: Fully populated ``Metrics`` rows, with all provenance columns set by
            ``enrich_metrics_rows()``.
        table_uri: Path or URI of the ``forecast_metrics`` Delta table.
        experiment_name: Experiment name; used in the Delta overwrite predicate to scope the
            replacement to this ``(experiment_name, fold_id)`` partition.
        fold_id: Fold identifier; used alongside ``experiment_name`` in the predicate.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
    """
    enum_cols = [c for c, dtype in metrics.schema.items() if isinstance(dtype, pl.Enum)]
    delta_data = metrics.with_columns(pl.col(c).cast(pl.String) for c in enum_cols).to_arrow()
    write_deltalake(
        table_or_uri=table_uri,
        data=delta_data,
        mode="overwrite",
        predicate=f"experiment_name = '{experiment_name}' AND fold_id = '{fold_id}'",
        partition_by=["experiment_name", "fold_id"],
        storage_options=typeddict_to_dict(storage_options),
    )
