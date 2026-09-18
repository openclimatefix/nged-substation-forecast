"""Storage policy for the ``eligible_time_series`` Delta table.

See this package's ``__init__`` docstring for which tables carry writer-properties tuning.
"""

from pathlib import Path

import patito as pt
from contracts.ml_schemas import EligibleTimeSeries
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import write_deltalake


def write_eligible_time_series(
    eligible: pt.DataFrame[EligibleTimeSeries],
    table_uri: str | Path,
    *,
    fold_id: str,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write one fold's ``EligibleTimeSeries`` rows to the ``eligible_time_series`` Delta table.

    The table is partitioned by ``fold_id``; the write **replaces** that partition, so
    re-materialising a fold leaves one copy of its eligible population. ``fold_id`` is an explicit
    parameter rather than read off ``eligible``. A fold with zero eligible series must still clear
    its own partition, and an empty frame has no row to read a fold id from.

    Args:
        eligible: Validated eligible-series rows for the fold named by ``fold_id``. May be empty.
        table_uri: Path or URI of the ``eligible_time_series`` Delta table.
        fold_id: The fold this write replaces, used in the overwrite predicate.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
    """
    write_deltalake(
        table_or_uri=table_uri,
        data=eligible.to_arrow(),
        mode="overwrite",
        predicate=f"fold_id = '{fold_id}'",
        partition_by=["fold_id"],
        storage_options=typeddict_to_dict(storage_options),
    )
