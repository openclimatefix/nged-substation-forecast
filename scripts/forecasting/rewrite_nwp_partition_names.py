"""Rewrite every ``nwp`` Delta partition whose directory name uses the old ``deltalake`` format.

``deltalake`` 1.6.x named a datetime partition directory
``init_time=2026-11-23%2000%3A00%3A00.000000``. ``deltalake`` 1.7.0 names it
``init_time=2026-11-23T00%3A00%3A00.000000Z``. Polars infers the datetime
format from the first partition it lists and cannot parse the other format. A table holding
partitions in both formats therefore fails every ``Nwp.scan_delta`` collect with
``could not parse '2026-11-23 00:00:00.000000' with pattern 'DatetimeYMDZ'``. This script
rewrites each old-format partition through ``write_nwp``, so that the partition lands in a
directory with the new name. **Run the script with ``deltalake`` 1.7.0 or later installed**, which
the lockfile guarantees.

Each partition is read through the pyarrow reader, because the Polars reader is the one that
fails on a mixed table. The pyarrow reader takes partition values from the Delta log and ignores
directory names. The rows are then cast and validated against the ``Nwp`` data contract and
written through ``write_nwp``, which replaces that ``(nwp_model_id, init_time)`` partition and
nothing else. A partition that fails validation, such as a stored run that lacks
``total_cloud_cover_atmosphere``, is logged and skipped, and the script exits with status 1 after
the other partitions are done. That partition stays in the old format, so the table stays
unscannable until the run is re-materialised with the ``ecmwf_ens`` asset, which gives the
partition the new name. A partition whose value already ends in ``Z`` is skipped, so an
interrupted run resumes where it stopped and a finished table is left alone.

**The rewrite needs free disk space roughly equal to the size of the table** until the superseded
files are reclaimed with ``vacuum(full=True)``. A 123 GB table needs about 250 GB in total. The
script commits one partition at a time, so ``vacuum(full=True)`` can run between interrupted
runs. Plain ``vacuum()`` reports deleting the files but does not delete them. The reason is
recorded on
<https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/>.
A table small enough to recreate, such as the AWS table, can be deleted and re-ingested instead.

Usage::

    uv run python scripts/forecasting/rewrite_nwp_partition_names.py --table-uri data/NWP --dry-run
    uv run python scripts/forecasting/rewrite_nwp_partition_names.py --table-uri data/NWP
"""

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Final

import polars as pl
from contracts.settings import get_settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from deltalake import DeltaTable

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_NEW_FORMAT_SUFFIX: Final[str] = "Z"
"""Every ``init_time`` partition value that ``deltalake`` 1.7 writes ends in this character."""


def old_format_partitions(table_uri: str, storage_options: ObjectStoreOptions) -> list[datetime]:
    """Return the ``init_time`` of each partition whose value is in the old ``deltalake`` format.

    Reads the Delta log only, not the data files.
    """
    table = DeltaTable(table_uri, storage_options=typeddict_to_dict(storage_options))
    raw_values = {partition["init_time"] for partition in table.partitions()}
    return sorted(
        datetime.fromisoformat(value).replace(tzinfo=UTC)
        for value in raw_values
        if not value.endswith(_NEW_FORMAT_SUFFIX)
    )


def rewrite_partition(
    table_uri: str, init_time: datetime, storage_options: ObjectStoreOptions
) -> None:
    """Read one partition back through the contract and write it into a new-format directory.

    Args:
        table_uri: Path or URI of the ``nwp`` Delta table.
        init_time: The single ``init_time`` value identifying the partition.
        storage_options: delta-rs object-store options; empty for a local path.
    """
    # Filtered on init_time alone although the partition key is (nwp_model_id, init_time):
    # NwpModelId has one enum member today, so an init_time identifies a partition. A second
    # NWP model would need this predicate widened.
    rows = (
        pl.scan_delta(
            table_uri, storage_options=typeddict_to_dict(storage_options), use_pyarrow=True
        )
        .filter(pl.col("init_time") == init_time)
        .collect()
    )
    # `Nwp.scan_delta` adds a column that is absent from the whole table as nulls. The pyarrow
    # reader does not, so the same fill is applied here before validation.
    stored_columns = set(rows.columns)
    rows = rows.with_columns(
        pl.lit(None, dtype=Nwp.dtypes[column]).alias(column)
        for column in sorted(Nwp.columns_absent_from_older_partitions - stored_columns)
    )
    write_nwp(
        nwp=Nwp.DataFrame(rows).cast().validate(),
        table_uri=table_uri,
        storage_options=storage_options,
    )


def main() -> None:
    """Rewrite each old-format partition of the ``nwp`` table."""
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

    pending = old_format_partitions(table_uri, storage_options)
    _LOGGER.info("%s has %d partitions to rewrite", table_uri, len(pending))

    failed: list[datetime] = []
    for index, init_time in enumerate(pending, start=1):
        if args.dry_run:
            _LOGGER.info(
                "[%d/%d] %s would be rewritten", index, len(pending), init_time.isoformat()
            )
            continue
        started = time.perf_counter()
        try:
            rewrite_partition(
                table_uri=table_uri, init_time=init_time, storage_options=storage_options
            )
        except Exception:
            _LOGGER.exception("[%d/%d] %s failed", index, len(pending), init_time.isoformat())
            failed.append(init_time)
            continue
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

    if failed:
        _LOGGER.error(
            "%d partitions could not be rewritten and are still in the old format: %s."
            " Re-materialise each run with the ecmwf_ens asset.",
            len(failed),
            ", ".join(init_time.isoformat() for init_time in failed),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
