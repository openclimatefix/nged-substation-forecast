"""Rewrite every ``nwp`` Delta partition so its row groups are member-aligned.

`delta_store.nwp.write_nwp` lands one ensemble member per Parquet row group, which is what lets a
single-member read skip the rest of a partition. Partitions written before that layout existed
hold row groups spanning many members. A numerical-weather-prediction (NWP) partition is written
once and never revisited, so those partitions stay that way until a migration rewrites them. This
script is that migration.

**Run this script against the local table.** The member-aligned layout speeds up a single-member
read that spans many stored runs. The read spanning the most runs by far is the control-member
read the cross-validation assets do at training time, against the local table. The control member
is ``ensemble_member == 0``, the one unperturbed member of the ensemble. Nothing running on AWS
reads the back catalogue: the live forecast pins ``init_time`` to the one freshest run, which
every write lays out correctly anyway.

The ``view_forecasts`` dashboard does read about 17 stored runs from S3 with a single-member
filter. But an ``h3_index`` filter already cuts that query to one H3 cell in 1,671, and the query
returns in about 0.2 s. H3 is the hexagonal grid the gridded NWP is aggregated onto, and
``h3_index`` names one cell of that grid. So the S3 table is not worth rewriting for the
dashboard. Run this script against the S3 table if training ever moves to AWS. The script
rewrites only the partitions it measures as unaligned, so running the script later costs exactly
what running it now would.

Each partition is read back through its data contract, re-validated, and written through
``write_nwp``, which replaces that ``(nwp_model_id, init_time)`` partition and nothing else. The
contract is the Patito schema ``contracts.weather_schemas.Nwp``, which fixes every column of the
``nwp`` table and its type. A partition whose row groups are already member-aligned is skipped,
so an interrupted run resumes where it stopped.

**The rewrite roughly doubles the table on disk** until the superseded files are reclaimed, so
migrating a 123 GB table needs about 250 GB free. Reclaim with ``vacuum(full=True)``. Plain
``vacuum()`` reports deleting the files but does not delete them. The reason is recorded on
<https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/>.

Usage::

    uv run python scripts/forecasting/rewrite_nwp_row_groups.py --table-uri data/NWP --dry-run
    uv run python scripts/forecasting/rewrite_nwp_row_groups.py --table-uri data/NWP
"""

import argparse
import logging
import time
from datetime import datetime
from typing import Final
from urllib.parse import unquote

import polars as pl
import pyarrow as pa
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

    Reads each active file's Parquet footer, which is the only sound way to tell. The Delta log's
    own per-file statistics give a file's *overall* member range. That range spans the whole
    ensemble whether the row groups inside are aligned or not. A file-count heuristic is not
    sound either. An incomplete ECMWF run is small enough to land in one file under either
    layout, and an incomplete run is the case where a single-member read skips the least data.

    A partition counts as aligned when it holds at least as many row groups as the ensemble
    members its statistics span. Both counts are taken across the whole partition rather than
    within one file. A partition split across files has no member range of its own until every
    file's range is folded together. **Demanding that every row group hold exactly one member
    would not converge.** Where the row count does not divide evenly by the member count, the
    aligned layout still straddles one boundary by design. A ragged partition would therefore be
    rewritten on every pass, each pass writing another full copy of its data.

    A file whose footer cannot be read is reported as unaligned, so the migration rewrites it
    rather than skipping it. Rewriting an already-aligned partition is wasted work; skipping an
    unaligned partition leaves the defect in place.

    Args:
        table_uri: Path or URI of the ``nwp`` Delta table.
        storage_options: delta-rs object-store options; empty for a local path.

    Returns:
        The ``init_time`` of each partition needing a rewrite, in ascending order.
    """
    delta_table = DeltaTable(table_uri, storage_options=typeddict_to_dict(storage_options))
    actions = pl.DataFrame(delta_table.get_add_actions(flatten=True))
    filesystem, root = pa_fs.FileSystem.from_uri(table_uri)

    row_groups: dict[datetime, int] = {}
    lowest_member: dict[datetime, int] = {}
    highest_member: dict[datetime, int] = {}
    unreadable: set[datetime] = set()
    for row in actions.iter_rows(named=True):
        init_time = row["partition.init_time"]
        try:
            parquet = pq.ParquetFile(
                f"{root.rstrip('/')}/{unquote(row['path'])}", filesystem=filesystem
            )
        except (OSError, pa.ArrowInvalid) as error:
            # A truncated or empty footer is what an interrupted write leaves behind, and
            # pyarrow raises ArrowInvalid (a ValueError) rather than an OSError for it.
            _LOGGER.warning("could not read %s, rewriting its partition: %s", row["path"], error)
            unreadable.add(init_time)
            continue
        member_column = parquet.schema_arrow.get_field_index("ensemble_member")
        statistics = [
            parquet.metadata.row_group(group).column(member_column).statistics
            for group in range(parquet.metadata.num_row_groups)
        ]
        if any(statistic is None for statistic in statistics):
            unreadable.add(init_time)
            continue
        row_groups[init_time] = row_groups.get(init_time, 0) + len(statistics)
        lowest = min(statistic.min for statistic in statistics)
        highest = max(statistic.max for statistic in statistics)
        lowest_member[init_time] = min(lowest_member.get(init_time, lowest), lowest)
        highest_member[init_time] = max(highest_member.get(init_time, highest), highest)

    unaligned = unreadable | {
        init_time
        for init_time, count in row_groups.items()
        if count < highest_member[init_time] - lowest_member[init_time] + 1
        and init_time not in unreadable
    }
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
    # NwpModelId has one variant today, so an init_time identifies a partition. A second NWP model
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
