"""Storage policy for the ``nwp`` Delta table.

Owns everything about how ``Nwp`` rows are laid out on disk: the parquet writer properties, the
compression-friendly row order, and the significand-precision reduction of the continuous weather
variables. Callers write through `write_nwp` so it is impossible to land rows in the table
without this format applied.

Stores plain ``Float32`` + ``delta_store.precision.round_to_significand_bits`` — the technique
used by ``delta_store.power_forecasts``, but with *different* writer properties. Measured on real
NWP data (9 partitions spread across two years of history): ``BYTE_STREAM_SPLIT`` made every
continuous column *larger*, not smaller — the opposite of the ``power_forecasts`` result. Working
hypothesis: significand rounding collapses NWP values into a small set of repeats (many H3 cells
/ ensemble members round to the same value), which Parquet's *default* dictionary+RLE encoding
captures directly; ``BYTE_STREAM_SPLIT`` scatters that repetition across four separate byte
planes and loses more than it gains. ``power_forecasts``'s target values have no such repetition
(near-continuous ML output), so ``BYTE_STREAM_SPLIT`` wins there instead — the two tables need
different writer properties. See
<https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#storage-formats-measured-not-assumed>
for the measured GB/yr numbers, and
<https://openclimatefix.github.io/nged-substation-forecast/api/dynamical_data/> for the
member-early-sort read-speed benchmark.
"""

from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from contracts.weather_schemas import Nwp
from deltalake import WriterProperties, write_deltalake

from delta_store.precision import round_to_significand_bits

NWP_SIGNIFICAND_BITS: Final[int] = 13
"""Significand bits kept for every continuous NWP variable (1 implicit + 12 explicit fraction
bits) — the same budget as ``delta_store.power_forecasts.POWER_FCST_SIGNIFICAND_BITS``. Caps the
relative error at 2⁻¹³ ≈ 1.2×10⁻⁴. Measured max absolute error on real data: ≤0.004 °C for
temperature, ≤8 Pa (0.08 hPa) for mean-sea-level pressure — both well inside tolerance
(temperature ≤0.25 K, MSL pressure ≤1 hPa)."""

NWP_SORT_COLS: Final[tuple[str, ...]] = ("init_time", "ensemble_member", "valid_time", "h3_index")
"""Within-file row order for ``nwp`` writes — **member before valid_time** (the opposite
priority from ``power_forecasts``, which sorts member-adjacent for a different reason: there
it's about compressing near-duplicate ensemble values; here it's about row-group pruning).
Sorting ``ensemble_member`` early puts each member's rows in one contiguous block, which
`_member_aligned_row_group_size` then turns into one row group per member. A single-member predicate
then matches a single row group's min/max range and skips the rest of the partition instead of
decoding those rows. That pruning holds for **any** member, not only the control member: the sort
alone is not enough, because row groups that straddle member boundaries advertise the whole span
between their extremes.

Measured on the stored table, a single-member read decodes 1.96% of a partition's rows — one row
group in 51 — for every member: every partition we sampled held 51 row groups, each spanning a
single member, and the 51 together covered members 0 to 50. Reading 29 daily partitions, nine H3
cells, and the control member alone runs in 30 ms and 400 MB of peak resident memory, against 170 ms
and 2,200 MB for the same read of a table sorted ``valid_time``-first. The member-early sort holds
3.7% more stored bytes than the ``valid_time``-first sort. Measured on a real partition with half
its rows removed at random members, so that the member-to-row-group alignment degrades rather than
holding exactly, the worst member still reads 5.88% of the partition's rows, against that 1.96%
floor. The method and the full figures live
beside the storage measurements in
<https://openclimatefix.github.io/nged-substation-forecast/api/dynamical_data/>.

Two conditions have to hold for the predicate to reach the Parquet scan at all. The predicate must
survive ``Nwp.scan_delta``'s cast, which requires that cast to be a no-op (see the
``Nwp.ensemble_member`` field). And the row groups have to stay member-aligned, which is what
`_member_aligned_row_group_size` and `NWP_TARGET_FILE_SIZE_BYTES` exist to guarantee."""

NWP_TARGET_FILE_SIZE_BYTES: Final[int] = 2_000_000_000
"""Target size for each Parquet file delta-rs writes, sized to keep one partition in one file.

A daily partition of the European Centre for Medium-Range Weather Forecasts' ensemble
forecast (ECMWF ENS) is
~158 MB, so the target leaves more than a tenfold headroom.

**The file-size target is an optimisation, not a correctness requirement.** A partition that
outgrows the target still writes correctly and still prunes well, because the single Arrow chunk
and the member-aligned row-group size below do the real work. Measured on a real partition, a
single-member read touches 1.96% of rows when the partition lands in one file and 3.92% when
delta-rs splits it in two."""

NWP_ROW_GROUP_SIZE_LIMITS: Final[tuple[int, int]] = (1_024, 1_048_576)
"""Floor and ceiling clamped around the member-aligned row-group size.

The floor stops a frame with very few rows per member producing row groups too small to compress
or too numerous to track; the floor has not bound on any real ECMWF ENS run measured so far, where
one member occupies ~142,000 rows. The ceiling is Parquet's conventional maximum, which also bounds
the streaming engine's peak memory per morsel."""


def _member_aligned_row_group_size(nwp: pt.DataFrame[Nwp]) -> int:
    """Rows one ensemble member occupies, clamped to `NWP_ROW_GROUP_SIZE_LIMITS`.

    Setting Parquet's row-group size to the number of rows one member occupies, on a frame already
    sorted member-early by `NWP_SORT_COLS`, lands each ensemble member in a row group of its own. A
    single-member predicate then matches one row group's ``ensemble_member`` min/max range exactly,
    so the scan decodes 1/51 of the partition's rows instead of the wider span of rows a straddling
    row group would advertise.

    Derived from the frame rather than hard-coded so the alignment survives a change to the H3
    grid or the forecast horizon, both of which change how many rows one member occupies. Where
    the division is inexact the alignment degrades gently: a row group straddles two members
    instead of one member, rather than reverting to the full span. The measured degradation under
    an uneven member split is documented on `NWP_SORT_COLS`, beside the 1.96% floor that
    degradation qualifies.

    Args:
        nwp: The frame about to be written, carrying every ensemble member for one run.

    Returns:
        The row-group size to give `WriterProperties`, within `NWP_ROW_GROUP_SIZE_LIMITS`.
    """
    floor, ceiling = NWP_ROW_GROUP_SIZE_LIMITS
    rows_per_member = nwp.height // nwp.get_column("ensemble_member").n_unique()
    return min(max(rows_per_member, floor), ceiling)


def _writer_properties(*, max_row_group_size: int) -> WriterProperties:
    """Parquet writer properties for the ``nwp`` table, at the given row-group size.

    Deliberately **no** per-column encoding overrides (no ``BYTE_STREAM_SPLIT``,
    ``DELTA_BINARY_PACKED``, or disabled dictionary encoding) — see this module's docstring for why
    that choice, which won for ``power_forecasts``, measures worse here. Built per write rather
    than held as a module constant because the row-group size is derived from the frame.

    Args:
        max_row_group_size: Rows per Parquet row group, from `_member_aligned_row_group_size`.

    Returns:
        Writer properties to hand to ``write_deltalake``.
    """
    return WriterProperties(
        compression="ZSTD", compression_level=3, max_row_group_size=max_row_group_size
    )


def write_nwp(
    nwp: pt.DataFrame[Nwp],
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write one NWP run into the ``nwp`` Delta table in its storage format.

    Rounds every continuous weather variable to ``NWP_SIGNIFICAND_BITS`` significand bits, sorts
    rows by ``NWP_SORT_COLS``, and writes one row group per ensemble member (see
    `_member_aligned_row_group_size`) into a single file per partition. The table is
    partitioned by ``(nwp_model_id, init_time)``, matching ``Nwp.scan_delta``'s
    partition-pruning assumptions; the first write creates the table.

    The write **replaces** the ``(nwp_model_id, init_time)`` partition named by the frame's first
    row, so re-materialising an ``ecmwf_ens`` partition leaves one copy of the run.
    delta-rs checks every row against that predicate and rejects the whole write, table untouched,
    if any row falls outside it (confirmed empirically against ``deltalake`` 1.6.3, locally and on
    S3, on a partition column despite its percent-encoded Hive directory name). Two materialisations
    of the *same* partition at once contend and the loser raises ``CommitFailedError``; disjoint
    partitions do not.

    ``schema_mode="overwrite"`` is passed on every write, not just to migrate an old table. This
    safety claim holds only for a **widening** contract change, which is the only kind this
    function has been used for so far: confirmed empirically (``deltalake`` 1.6.3) that widening
    updates only the table's *logical* schema (the ``_delta_log`` metadata) to match the incoming
    frame's dtypes, and leaves every other partition's physical Parquet bytes untouched — a later
    read at the new logical dtype is correct and lossless even for a partition still physically
    stored at an older, narrower dtype.

    **A validated ``pt.DataFrame[Nwp]`` input is what makes the flag safe to leave on.** The
    ``nwp`` argument carries the full column set at the *current* contract's dtypes, so the only
    way this can ever change the table's schema is a deliberate widening of an ``Nwp`` dtype — it
    cannot silently drop a column.

    A **narrowing** contract change is a different, worse failure mode, also confirmed
    empirically: the write that narrows a column succeeds silently — ``schema_mode="overwrite"``
    accepts it at write time — and the table is left with a logical schema that its own,
    previously-written partitions can no longer safely satisfy. Nothing fails until a *later* read
    of the whole table (by anyone, not necessarily the writer that broke it), which then raises
    ``SchemaError: incoming dtype cannot safely cast to target dtype`` — a confusing error far
    removed from its cause, especially if further partitions land on the bad schema before anyone
    reads across all of them.

    **Leaving ``schema_mode="overwrite"`` on permanently is an accepted, disclosed risk rather
    than a defect.** Nothing in this contract's own dtype-widening history has ever narrowed a
    column, and there is no guard against a future change that does. Without
    ``schema_mode="overwrite"``, delta-rs instead auto-widens the incoming (narrower) data up to
    the table's existing (wider) schema at write time, and the table stays readable throughout.

    Args:
        nwp: Validated, non-empty NWP rows for a single ``(nwp_model_id, init_time)`` partition.
        table_uri: Path or URI of the ``nwp`` Delta table.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
    """
    continuous_vars = sorted(Nwp.continuous_var_names())
    rounded = nwp.with_columns(
        **{
            var: round_to_significand_bits(pl.col(var), keep_bits=NWP_SIGNIFICAND_BITS)
            for var in continuous_vars
        }
    ).sort(*NWP_SORT_COLS)

    # Combine the frame into one Arrow chunk rather than the 32 chunks a measured partition arrived
    # in. delta-rs consumes a multi-chunk table out of order when partitioning the write. That
    # out-of-order read scatters the member-sorted rows across row groups and widens every row
    # group's ensemble_member min/max range. Measured on a real partition, a single-member read went
    # from 1.96% of rows to 33% purely from the chunking. Combining allocates one extra copy of the
    # frame.
    prepared = rounded.to_arrow().combine_chunks()

    write_deltalake(
        table_or_uri=table_uri,
        data=prepared,
        mode="overwrite",
        schema_mode="overwrite",
        predicate=(
            f"nwp_model_id = '{nwp.item(0, 'nwp_model_id')}' "
            f"AND init_time = '{nwp.item(0, 'init_time').isoformat()}'"
        ),
        partition_by=["nwp_model_id", "init_time"],
        writer_properties=_writer_properties(
            max_row_group_size=_member_aligned_row_group_size(nwp)
        ),
        target_file_size=NWP_TARGET_FILE_SIZE_BYTES,
        storage_options=typeddict_to_dict(storage_options),
    )
