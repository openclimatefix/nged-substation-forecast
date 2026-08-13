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
<https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/> for the measured
GB/yr and read-latency numbers.
"""

from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts._uri import ObjectStoreOptions
from contracts.typing_utils import typeddict_to_dict
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
partition. Measured on a real 29-day/9-cell/control-member read: ~5x faster and ~5x less peak
memory than the previous ``valid_time``-first sort, for a ~2% storage cost."""

NWP_WRITER_PROPERTIES: Final[WriterProperties] = WriterProperties(
    compression="ZSTD", compression_level=3
)
"""Deliberately **no** per-column encoding overrides (no ``BYTE_STREAM_SPLIT``,
``DELTA_BINARY_PACKED``, or disabled dictionary encoding) — see this module's docstring for why
that choice, which won for ``power_forecasts``, measures worse here."""


def _partition_predicate(nwp: pt.DataFrame[Nwp]) -> str:
    """Build the SQL predicate naming the one Delta partition ``nwp`` belongs to.

    Read from the frame's first row rather than taken as an argument, so the predicate cannot
    disagree with the data written alongside it. Rows outside that partition are not silently
    dropped: delta-rs validates every row against the predicate and rejects the whole write,
    leaving the table untouched.
    """
    return (
        f"nwp_model_id = '{nwp.item(0, 'nwp_model_id')}' "
        f"AND init_time = '{nwp.item(0, 'init_time').isoformat()}'"
    )


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

    The write **replaces** the ``(nwp_model_id, init_time)`` partition it is given rather than
    appending to it, so re-materialising an ``ecmwf_ens`` partition — to pick up a run
    Dynamical.org has republished, or after one was killed between the Delta commit and Dagster
    recording success — costs a rewrite rather than a second copy of the run beside the first.
    ``Nwp.validate`` sees only the frame in hand, so an appended duplicate would land silently and
    fan out every later ``Nwp.scan_delta`` read.

    delta-rs' ``replaceWhere`` predicate matches a ``Timestamp`` partition column correctly, which
    is worth stating because the Hive directory holds a percent-encoded string: confirmed
    empirically that an ``isoformat()`` literal matches only the named partition, on a local path
    and on S3 alike, and works on a table that does not exist yet as well as on a partition that
    does not. Two materialisations of the *same* partition at once contend, and the loser raises
    ``CommitFailedError`` — one lost run rather than a table needing repair. Disjoint partitions do
    not contend, so the daily schedule and a backfill of other dates run happily together.

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

    # Strip the Patito model before the dict-cast: `nwp_model_id` is declared `Enum` for
    # in-memory type safety, but delta-rs can't store `Enum`/`Categorical` (see the "Delta Lake
    # dictionary-encoded columns" gotcha in the `polars-patito-gotchas` skill). A dict-cast on a
    # *model-bearing* frame would silently swallow the mapping and revert other columns to the
    # model's declared dtypes instead — strip first so this is a plain-Polars cast.
    prepared = pl.DataFrame._from_pydf(rounded._df).cast({"nwp_model_id": pl.String}).to_arrow()

    write_deltalake(
        table_or_uri=table_uri,
        data=prepared,
        mode="overwrite",
        predicate=_partition_predicate(nwp),
        partition_by=["nwp_model_id", "init_time"],
        writer_properties=NWP_WRITER_PROPERTIES,
        storage_options=typeddict_to_dict(storage_options),
    )
