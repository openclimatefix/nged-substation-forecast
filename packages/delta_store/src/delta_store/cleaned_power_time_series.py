"""Storage policy for the ``cleaned_power_time_series`` Delta table.

See this package's ``__init__`` docstring for which tables carry writer-properties tuning.
"""

from pathlib import Path

import patito as pt
from contracts.power_schemas import CleanedPowerTimeSeries
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import CommitProperties, DeltaTable, write_deltalake
from nged_data.cleaning import CleaningProvenance


class VacuumError(Exception):
    """The cleaned table was written, but deleting its superseded files failed."""


def write_cleaned_power_time_series(
    cleaned: pt.DataFrame[CleanedPowerTimeSeries],
    table_uri: str | Path,
    *,
    provenance: CleaningProvenance,
    storage_options: ObjectStoreOptions | None = None,
    retention_hours: int = 2,
) -> None:
    """Overwrite the ``cleaned_power_time_series`` Delta table with ``cleaned``, then vacuum it.

    The overwrite is one atomic Delta commit, partitioned by ``time_series_id``. Its commit
    records ``provenance``, so the next run can tell whether the table is up to date. Delta cannot
    keep row order within a partition, so the files on disk are not sorted by ``time``.

    The vacuum deletes the files the overwrite superseded; without it every overwrite leaves the
    previous copy on disk forever. A scan resolves its file list when it starts and only files
    tombstoned more than ``retention_hours`` ago are deleted, so ``retention_hours`` bounds how
    long the longest reader scan may take. The vacuum adds two commits (``VACUUM START`` and
    ``VACUUM END``) after the write, which `nged_data.cleaning.read_cleaning_provenance` skips.
    Both ``dry_run=False`` and ``enforce_retention_duration=False`` matter: ``dry_run`` defaults
    to ``True``, which lists files and deletes none, and delta-rs refuses a retention under 168
    hours without the second flag.

    Args:
        cleaned: Validated, sorted rows: every raw row, with its ``drop_reason``.
        table_uri: Path or URI of the ``cleaned_power_time_series`` Delta table.
        provenance: What the rows were built from, recorded in the write's commit.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
        retention_hours: How old a superseded file must be before the vacuum deletes it.

    Raises:
        VacuumError: The write succeeded and the vacuum failed. The new table is already in
            place, so a caller can treat this as a degradation rather than a failed write.
    """
    options = typeddict_to_dict(storage_options)
    write_deltalake(
        table_or_uri=table_uri,
        data=cleaned.to_arrow(),
        mode="overwrite",
        partition_by=["time_series_id"],
        storage_options=options,
        commit_properties=CommitProperties(custom_metadata=provenance.to_commit_metadata()),
    )
    try:
        DeltaTable(str(table_uri), storage_options=options).vacuum(
            retention_hours=retention_hours, dry_run=False, enforce_retention_duration=False
        )
    except Exception as err:
        raise VacuumError(f"Vacuuming {table_uri} failed after the write succeeded.") from err
