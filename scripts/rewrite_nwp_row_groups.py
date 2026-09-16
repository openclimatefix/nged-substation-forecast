"""Rewrite every ``nwp`` Delta partition so its row groups are member-aligned.

`delta_store.nwp.write_nwp` lands one ensemble member per Parquet row group, which is what lets a
single-member read skip the rest of a partition. Partitions written before that layout existed hold
row groups spanning many members, and because an NWP partition is written once and never revisited,
they stay that way until something rewrites them. This script is that something: run it once per
table, then delete it.

Each partition is read back through the contract, re-validated, and written through ``write_nwp``,
which replaces that ``(nwp_model_id, init_time)`` partition and nothing else. A partition whose row
groups are already member-aligned is skipped, so an interrupted run resumes where it stopped.

**The rewrite roughly doubles the table on disk** until the superseded files are reclaimed, so
migrating a 123 GB table needs about 250 GB free. Reclaim with ``vacuum(full=True)``; plain
``vacuum()`` reports deleting the files and does not, for the reason recorded on
<https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/>.

Usage::

    uv run python scripts/rewrite_nwp_row_groups.py --table-uri data/NWP --dry-run
    uv run python scripts/rewrite_nwp_row_groups.py --table-uri data/NWP
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Final
from urllib.parse import unquote

import polars as pl
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from contracts.settings import get_settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from deltalake import DeltaTable

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def _unaligned_partitions(table_uri: str, storage_options: ObjectStoreOptions) -> list[datetime]:
    """Partitions holding a row group that spans more than one ``ensemble_member``.

    Reads each active file's Parquet footer, which is the only sound way to tell: the Delta log's
    own per-file statistics give a file's *overall* member range, which spans the whole ensemble
    whether the row groups inside are aligned or not. A file-count heuristic is not sound either —
    an incomplete ECMWF run is small enough to land in one file under either layout, and an
    incomplete run is the case that prunes worst.

    A file whose footer cannot be read is reported as unaligned, so the migration rewrites it
    rather than skipping it. Rewriting an already-aligned partition is wasted work; skipping an
    unaligned one leaves the defect in place.

    Args:
        table_uri: Path or URI of the ``nwp`` Delta table.
        storage_options: delta-rs object-store options; empty for a local path.

    Returns:
        The ``init_time`` of each partition needing a rewrite, in ascending order.
    """
    delta_table = DeltaTable(table_uri, storage_options=typeddict_to_dict(storage_options))
    actions = pl.DataFrame(delta_table.get_add_actions(flatten=True))
    filesystem, root = pa_fs.FileSystem.from_uri(table_uri)

    unaligned: set[datetime] = set()
    for row in actions.iter_rows(named=True):
        init_time = row["partition.init_time"]
        if init_time in unaligned:
            continue
        try:
            parquet = pq.ParquetFile(
                f"{root.rstrip('/')}/{unquote(row['path'])}", filesystem=filesystem
            )
            member_column = parquet.schema_arrow.get_field_index("ensemble_member")
            spans_one_member = all(
                (statistics := parquet.metadata.row_group(group).column(member_column).statistics)
                is not None
                and statistics.min == statistics.max
                for group in range(parquet.metadata.num_row_groups)
            )
        except OSError as error:
            _LOGGER.warning("could not read %s, rewriting its partition: %s", row["path"], error)
            spans_one_member = False
        if not spans_one_member:
            unaligned.add(init_time)
    return sorted(unaligned)


def _rewrite_partition(
    table_uri: str, init_time: datetime, storage_options: ObjectStoreOptions
) -> None:
    """Read one partition back through the contract and write it out member-aligned.

    Args:
        table_uri: Path or URI of the ``nwp`` Delta table.
        init_time: The single ``init_time`` value identifying the partition.
        storage_options: delta-rs object-store options; empty for a local path.
    """
    # Filtered on init_time alone although the partition key is (nwp_model_id, init_time):
    # NwpModelId has one member today, so an init_time identifies a partition. A second NWP model
    # would need this predicate widened, and would be a reason to revisit this script rather than
    # delete it.
    rows = (
        Nwp.scan_delta(table_uri, storage_options=storage_options)
        .filter(pl.col("init_time") == init_time)
        .collect()
    )
    write_nwp(
        nwp=Nwp.DataFrame(rows).cast().validate(),
        table_uri=table_uri,
        storage_options=storage_options,
    )


def main() -> None:
    """Rewrite each partition of the ``nwp`` table whose row groups are not member-aligned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-uri", default="", help="nwp Delta table; defaults to settings.nwp_data_path."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which partitions would be rewritten, and write nothing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = get_settings()
    table_uri = args.table_uri or settings.nwp_data_path
    storage_options = settings.storage_options

    pending = _unaligned_partitions(table_uri, storage_options)
    _LOGGER.info("%s has %d partitions to rewrite", table_uri, len(pending))

    for index, init_time in enumerate(pending, start=1):
        if args.dry_run:
            _LOGGER.info(
                "[%d/%d] %s would be rewritten", index, len(pending), init_time.isoformat()
            )
            continue
        started = time.perf_counter()
        _rewrite_partition(
            table_uri=table_uri, init_time=init_time, storage_options=storage_options
        )
        _LOGGER.info(
            "[%d/%d] %s rewritten in %.1fs",
            index,
            len(pending),
            init_time.isoformat(),
            time.perf_counter() - started,
        )

    if not args.dry_run and pending:
        _LOGGER.info(
            "Superseded files are still on disk and roughly double the table until reclaimed."
            " Use vacuum(full=True); plain vacuum() silently deletes nothing here — see"
            " https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/"
        )


if __name__ == "__main__":
    main()
