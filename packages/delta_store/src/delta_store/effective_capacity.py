"""Storage policy for the ``effective_capacity`` Delta table.

See this package's ``__init__`` docstring for which tables carry writer-properties tuning.
"""

from pathlib import Path

import patito as pt
from contracts.power_schemas import EffectiveCapacity
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import write_deltalake


def write_effective_capacity(
    capacity: pt.DataFrame[EffectiveCapacity],
    table_uri: str | Path,
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Overwrite the ``effective_capacity`` Delta table with ``capacity``.

    The whole (small — one row per series) table is replaced on every write; there is no
    partitioning or predicate to scope the overwrite.

    Args:
        capacity: Validated capacity rows, one per ``time_series_id``.
        table_uri: Path or URI of the ``effective_capacity`` Delta table.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
    """
    write_deltalake(
        table_or_uri=table_uri,
        data=capacity.to_arrow(),
        mode="overwrite",
        storage_options=typeddict_to_dict(storage_options),
    )
