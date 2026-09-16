"""Rewrite every ``nwp`` Delta partition so its row groups are member-aligned.

`delta_store.nwp.write_nwp` lands one ensemble member per Parquet row group, which is what lets a
single-member read skip the rest of a partition. Partitions written before that layout existed hold
row groups spanning many members, and because an NWP partition is written once and never revisited,
they stay that way until something rewrites them. This script is that something: run it once per
table, then delete it.

Each partition is read back through the contract, re-validated, and written through ``write_nwp``,
which replaces that ``(nwp_model_id, init_time)`` partition and nothing else. A partition whose row
groups are already member-aligned is skipped, so an interrupted run resumes where it stopped and a
finished table costs one metadata pass.

Usage::

    uv run python scripts/rewrite_nwp_row_groups.py --table-uri data/NWP --dry-run
    uv run python scripts/rewrite_nwp_row_groups.py --table-uri data/NWP
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Final

import polars as pl
from contracts.settings import get_settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from deltalake import DeltaTable

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def _partition_file_counts(table_uri: str, storage_options: ObjectStoreOptions) -> pl.DataFrame:
    """Each ``init_time`` partition and how many Parquet files it holds, from the Delta log.

    Used to decide which partitions still need rewriting. The current `write_nwp` lands one file
    per partition, where the layout this script migrates away from always split a full ECMWF ENS
    run across two or more — so a single-file partition has already been rewritten. Reading the
    log rather than the data keeps this to one metadata call: listing partitions by scanning the
    table instead would read every row of a table that runs to tens of GB.

    Args:
        table_uri: Path or URI of the ``nwp`` Delta table.
        storage_options: delta-rs object-store options; empty for a local path.

    Returns:
        One row per partition, with columns ``init_time`` and ``n_files``, ordered by
        ``init_time``.
    """
    delta_table = DeltaTable(table_uri, storage_options=typeddict_to_dict(storage_options))
    actions = pl.DataFrame(delta_table.get_add_actions(flatten=True))
    return (
        actions.group_by("partition.init_time")
        .len(name="n_files")
        .rename({"partition.init_time": "init_time"})
        .sort("init_time")
    )


def _rewrite_partition(
    table_uri: str, init_time: datetime, storage_options: ObjectStoreOptions
) -> None:
    """Read one partition back through the contract and write it out member-aligned.

    Args:
        table_uri: Path or URI of the ``nwp`` Delta table.
        init_time: The single ``init_time`` value identifying the partition.
        storage_options: delta-rs object-store options; empty for a local path.
    """
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
    """Rewrite each partition of the ``nwp`` table that is not already member-aligned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-uri", default="", help="nwp Delta table; defaults to settings.nwp_data_path."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which partitions would be rewritten, and write nothing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite every partition, including ones that already hold a single file.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = get_settings()
    table_uri = args.table_uri or settings.nwp_data_path
    storage_options = settings.storage_options

    partitions = _partition_file_counts(table_uri, storage_options)
    _LOGGER.info("%s holds %d partitions", table_uri, partitions.height)

    total = partitions.height
    rewritten = 0
    for index, row in enumerate(partitions.iter_rows(named=True), start=1):
        init_time, n_files = row["init_time"], row["n_files"]
        if n_files == 1 and not args.force:
            _LOGGER.info("[%d/%d] %s already rewritten", index, total, init_time.isoformat())
            continue
        rewritten += 1
        if args.dry_run:
            _LOGGER.info("[%d/%d] %s would be rewritten", index, total, init_time.isoformat())
            continue
        started = time.perf_counter()
        _rewrite_partition(table_uri, init_time, storage_options)
        _LOGGER.info(
            "[%d/%d] %s rewritten in %.1fs",
            index,
            total,
            init_time.isoformat(),
            time.perf_counter() - started,
        )

    _LOGGER.info("%d partitions %s", rewritten, "to rewrite" if args.dry_run else "rewritten")


if __name__ == "__main__":
    main()
