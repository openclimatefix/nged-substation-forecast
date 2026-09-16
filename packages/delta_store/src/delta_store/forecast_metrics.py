"""Storage policy for the ``forecast_metrics`` Delta table.

See this package's ``__init__`` docstring for which tables carry writer-properties tuning. The
Enum→String cast below is on-disk-format knowledge, so it lives here rather than in evaluation
logic.
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
    *,
    experiment_name: str,
    fold_id: str,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write ``Metrics`` rows to the ``forecast_metrics`` Delta table.

    Casts the ``Enum`` columns (e.g. ``metric_name``, ``horizon_slice``) to ``String`` before
    writing. Without the cast, delta-rs writes the column as a dictionary-typed parquet column but
    records it as ``Utf8`` in the Delta log — the write itself succeeds, but a later
    ``pl.read_delta`` then raises ``SchemaError: data type mismatch for column ...: incoming:
    Enum([...]) != target: String`` (verified against deltalake 1.6.3 / polars 1.44.2). Casting
    before the write keeps the file's physical type and the log's logical type in step. Performs
    an idempotent overwrite of the ``(experiment_name, fold_id)`` partition so re-materialising the
    asset replaces rows rather than duplicating them.

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
