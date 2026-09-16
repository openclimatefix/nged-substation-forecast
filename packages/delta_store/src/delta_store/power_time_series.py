"""Storage policy for the ``power_time_series`` Delta table.

No writer-properties, sort-order or precision tuning yet — see this package's ``__init__``
docstring for which tables carry that tuning. This module exists so the write itself goes through
``delta_store`` like every other table, rather than being open-coded in a Dagster asset.
"""

from pathlib import Path

import patito as pt
from contracts.power_schemas import PowerTimeSeries
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import write_deltalake


def write_power_time_series(
    power_ts: pt.DataFrame[PowerTimeSeries],
    table_uri: str | Path,
    *,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Append ``PowerTimeSeries`` rows to the ``power_time_series`` Delta table.

    The table is partitioned by ``time_series_id``; the first write creates it. Appends only —
    the caller is responsible for de-duplicating rows already on disk before calling this (see
    ``power_time_series_and_metadata``'s use of ``select_new_rows``).

    Args:
        power_ts: Validated, de-duplicated new rows to append.
        table_uri: Path or URI of the ``power_time_series`` Delta table.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
    """
    write_deltalake(
        table_or_uri=table_uri,
        data=power_ts.to_arrow(),
        mode="append",
        partition_by=["time_series_id"],
        storage_options=typeddict_to_dict(storage_options),
    )
