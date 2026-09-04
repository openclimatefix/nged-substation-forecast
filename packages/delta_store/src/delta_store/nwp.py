"""Storage policy for the ``nwp`` Delta table.

Owns everything about how ``Nwp`` rows are laid out on disk: the parquet writer properties, the
compression-friendly row order, and the significand-precision reduction of the continuous
weather variables. Callers write through :func:`write_nwp` so it is impossible to land rows in
the table without this format applied.

Stores plain ``Float32`` + ``delta_store.precision.round_to_significand_bits`` — the technique
used by ``delta_store.power_forecasts``, but with *different* writer properties. Measured on
real NWP data (9 partitions spread across two years of history): ``BYTE_STREAM_SPLIT`` made every
continuous column *larger*, not smaller — the opposite of the ``power_forecasts`` result.
Working hypothesis: significand rounding collapses NWP values into a small set of repeats (many
H3 cells / ensemble members round to the same value), which Parquet's *default* dictionary+RLE
encoding captures directly; ``BYTE_STREAM_SPLIT`` scatters that repetition across four separate
byte planes and loses more than it gains. ``power_forecasts``'s target values have no such
repetition (near-continuous ML output), so ``BYTE_STREAM_SPLIT`` wins there instead — the two
tables need different writer properties. See
<https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#storage-formats-measured-not-assumed>
for the measured GB/yr numbers, and
<https://openclimatefix.github.io/nged-substation-forecast/api/dynamical_data/> for the
member-early-sort read-speed benchmark (a single-member, 29-day, 9-cell collect: ~5x faster and
~5x less peak memory for a ~2% storage cost).
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
Sorting ``ensemble_member`` early means each ~1M-row Parquet row group spans only a handful of
member values instead of all ~51, so a single-member predicate (the control-member read every
training run does) can skip most row groups via min/max stats instead of decoding the whole
partition — provided that predicate reaches the Parquet scan unchanged, which requires
``Nwp.scan_delta``'s cast to be a no-op (see the ``Nwp.ensemble_member`` field). The speed and
storage cost of this ordering versus a ``valid_time``-first sort need re-measuring against real
production data; the old figures predate a period where that cast was not a no-op and are not
restored here."""

NWP_WRITER_PROPERTIES: Final[WriterProperties] = WriterProperties(
    compression="ZSTD", compression_level=3
)
"""Deliberately **no** per-column encoding overrides (no ``BYTE_STREAM_SPLIT``,
``DELTA_BINARY_PACKED``, or disabled dictionary encoding) — see this module's docstring for why
that choice, which won for ``power_forecasts``, measures worse here."""


def write_nwp(
    nwp: pt.DataFrame[Nwp],
    table_uri: str | Path,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write one NWP run into the ``nwp`` Delta table in its storage format.

    Rounds every continuous weather variable to ``NWP_SIGNIFICAND_BITS`` significand bits, sorts
    rows by ``NWP_SORT_COLS``, and writes with ``NWP_WRITER_PROPERTIES``. The table is
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
    stored at an older, narrower dtype. Since ``nwp``'s input is always an already-validated
    ``pt.DataFrame[Nwp]``, carrying the full column set at the *current* contract's dtypes, the
    only way this can ever change the table's schema is a deliberate future ``Nwp`` dtype change
    like this one — it cannot silently drop a column.

    A **narrowing** contract change is a different, worse failure mode, also confirmed
    empirically: the write that narrows a column succeeds silently — ``schema_mode="overwrite"``
    accepts it at write time — and the table is left with a logical schema that its own,
    previously-written partitions can no longer safely satisfy. Nothing fails until a *later* read
    of the whole table (by anyone, not necessarily the writer that broke it), which then raises
    ``SchemaError: incoming dtype cannot safely cast to target dtype`` — a confusing error far
    removed from its cause, especially if further partitions land on the bad schema before anyone
    reads across all of them. Without ``schema_mode="overwrite"``, delta-rs instead auto-widens
    the incoming (narrower) data up to the table's existing (wider) schema at write time, and the
    table stays readable throughout. This is an accepted, disclosed risk of leaving the flag on
    permanently, not a defect: nothing in this contract's own dtype-widening history has ever
    narrowed a column, and there is no guard against a future change that does.

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

    prepared = rounded.to_arrow()

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
        writer_properties=NWP_WRITER_PROPERTIES,
        storage_options=typeddict_to_dict(storage_options),
    )
