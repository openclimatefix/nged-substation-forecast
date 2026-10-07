"""Storage policy for the ``cleaned_power_time_series`` Delta table.

See this package's ``__init__`` docstring for which tables carry writer-properties tuning.
"""

import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final

import patito as pt
from contracts.power_schemas import CleanedPowerTimeSeries
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import CommitProperties, DeltaTable, write_deltalake

log = logging.getLogger(__name__)

_PROVENANCE_KEY_PREFIX: Final[str] = "cleaning_"
"""Prefix of every `CleaningProvenance` key in a Delta commit's `custom_metadata`."""

_HISTORY_WINDOW: Final[int] = 10
"""How many recent Delta commits `read_cleaning_provenance` walks back through to find the newest
`WRITE`. Every vacuum adds two commits after the write, so a small window is enough."""


class VacuumError(Exception):
    """The cleaned table was written, but deleting its superseded files failed."""


@dataclass(frozen=True)
class CleaningProvenance:
    """What a cleaned table was built from, recorded in the write's Delta commit.

    Each field becomes one `custom_metadata` key, named `cleaning_<field>`.

    Attributes:
        raw_table_id: The raw table's Delta table id. It guards against a deleted and rebuilt raw
            table, whose versions restart at 0.
        raw_version: The raw table's Delta version that the cleaning read.
        code_hash: The SHA-256 of the cleaning rules' source files when the cleaning ran.
        metadata_hash: A hash of the `TimeSeriesMetadata` table that the cleaning read: the
            SHA-256 of Polars' per-row `hash_rows` over that table sorted by `time_series_id`. The
            hash is stable between runs on one Polars version and changes when any value in the
            table changes. A change of Polars version can change every hash, which only causes one
            extra rebuild.
        git_sha: The git SHA of the code that did the cleaning, recorded for provenance only:
            `clean_nged_power_data` skips a rebuild when the other four fields match.
    """

    raw_table_id: str
    raw_version: int
    code_hash: str
    metadata_hash: str
    git_sha: str

    def to_commit_metadata(self) -> dict[str, str]:
        """Return the `custom_metadata` to record on the Delta write commit."""
        return {
            f"{_PROVENANCE_KEY_PREFIX}{name}": str(value) for name, value in asdict(self).items()
        }


def read_cleaning_provenance(
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> CleaningProvenance | None:
    """Read the `CleaningProvenance` from the newest `WRITE` commit of the cleaned table.

    A vacuum adds commits after the write, so this walks back through the table's recent history
    to the newest `WRITE`. Never raises: a missing table, an unreadable table, no `WRITE` commit
    in the window, or a missing key all read as "no provenance", which always means rebuild.

    Args:
        table_uri: Path or URI of the `cleaned_power_time_series` Delta table.
        storage_options: delta-rs object-store options for a remote `table_uri`.

    Returns:
        The provenance, or `None` when there is none.
    """
    try:
        options = typeddict_to_dict(storage_options) or {}
        if not DeltaTable.is_deltatable(str(table_uri), storage_options=options):
            return None
        history = DeltaTable(str(table_uri), storage_options=options).history(limit=_HISTORY_WINDOW)
        write_commit = next(
            (commit for commit in history if commit.get("operation") == "WRITE"), None
        )
        keys = {
            field.name: f"{_PROVENANCE_KEY_PREFIX}{field.name}"
            for field in fields(CleaningProvenance)
        }
        if write_commit is None or not all(key in write_commit for key in keys.values()):
            return None
        values: dict[str, Any] = {name: write_commit[key] for name, key in keys.items()}
        # Delta stores every custom_metadata value as a string.
        values["raw_version"] = int(values["raw_version"])
        return CleaningProvenance(**values)
    except Exception:
        log.warning(f"Could not read cleaning provenance from {table_uri}.", exc_info=True)
        return None


def write_cleaned_power_time_series(
    cleaned: pt.DataFrame[CleanedPowerTimeSeries],
    table_uri: str | Path,
    *,
    provenance: CleaningProvenance,
    storage_options: ObjectStoreOptions | None = None,
    retention_hours: int = 2,
) -> None:
    """Overwrite the ``cleaned_power_time_series`` Delta table with ``cleaned``, then vacuum it.

    The overwrite is one atomic Delta commit, and the table is partitioned by ``time_series_id``.
    That commit records ``provenance``, so the next run can tell whether the table is up to date.
    Delta cannot keep row order within a partition, so the files on disk are not sorted by
    ``time``.

    The vacuum deletes the files the overwrite superseded; without the vacuum every overwrite leaves
    the previous copy on disk forever. A scan resolves its file list when the scan starts, and the
    vacuum deletes only files tombstoned more than ``retention_hours`` ago. ``retention_hours``
    therefore bounds how long the longest reader scan may take. The vacuum adds two commits
    (``VACUUM START`` and ``VACUUM END``) after the write, which `read_cleaning_provenance` skips.
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
