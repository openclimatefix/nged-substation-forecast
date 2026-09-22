"""Storage policy for the internal ``power_forecasts`` Delta table.

Owns everything about how ``PowerForecast`` rows are laid out on disk: the parquet writer
properties (codec + per-column encodings), the compression-friendly row order, and the
``power_fcst`` precision reduction. Callers write through `write_power_forecasts` so it is
impossible to land rows in the table without this format applied.

Measured impact: the full 403.6M-row development table shrank from 6.33 GB to 0.73 GB when
rewritten into this format. The 6.33 GB figure is the delta-rs defaults: SNAPPY, dictionary
encoding, unsorted, full precision. See the ``POWER_FORECASTS_WRITER_PROPERTIES`` docstring for
the per-lever breakdown.
"""

from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import PowerForecast
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import ObjectStoreOptions
from deltalake import ColumnProperties, WriterProperties, write_deltalake

from delta_store.precision import round_to_significand_bits

POWER_FCST_SIGNIFICAND_BITS: Final[int] = 13
"""Significand bits kept when storing ``power_fcst`` (1 implicit + 12 explicit fraction bits).

Rounding to nearest at 13 significand bits caps the relative error at 2⁻¹³ ≈ 1.2×10⁻⁴, orders of
magnitude below forecast error. The rounding also zeroes the 11 low fraction bits. Those 11 bits are
otherwise pure entropy that defeats every compression codec, because nearly every full-precision
``power_fcst`` value is distinct. Mirrors the significand-rounding scheme ``delta_store.nwp``
applies to NWP data, but with different writer properties. See that module's docstring for why the
choice doesn't transfer between tables. See ``POWER_FORECASTS_WRITER_PROPERTIES`` for the measured
size impact.
"""

POWER_FORECASTS_SORT_COLS: Final[tuple[str, ...]] = PowerForecast.PRIMARY_KEY
"""Within-file row order for ``power_forecasts`` writes.

``power_fcst`` becomes locally smooth, and the timestamp columns become stepped sequences, when the
~51 ensemble members of one (series, init time, valid time) target sit on adjacent rows. Locally
smooth values and stepped sequences are exactly what the BYTE_STREAM_SPLIT and DELTA_BINARY_PACKED
encodings in ``POWER_FORECASTS_WRITER_PROPERTIES`` need to compress well. Leading with
``time_series_id`` also lets parquet row-group statistics prune scans that filter on one series.

Defined as ``PowerForecast.PRIMARY_KEY`` rather than repeating its columns, because the set that
identifies a row uniquely is exactly the set whose adjacency makes a row compress. The *order* is
this module's own choice, and ``test_sort_cols_lead_with_time_series_id`` pins that order. A
reordered primary key is still a correct key but a worse layout, so this tuple is not free to follow
a reordered primary key.
"""

_TIMESTAMP_COLUMN_PROPERTIES: Final[ColumnProperties] = ColumnProperties(
    encoding="DELTA_BINARY_PACKED", dictionary_enabled=False
)
"""Delta-encode timestamps: sorted microsecond timestamps have tiny, near-constant deltas."""

POWER_FORECASTS_WRITER_PROPERTIES: Final[WriterProperties] = WriterProperties(
    compression="ZSTD",
    compression_level=3,
    column_properties={
        "valid_time": _TIMESTAMP_COLUMN_PROPERTIES,
        "power_fcst_init_time": _TIMESTAMP_COLUMN_PROPERTIES,
        "nwp_init_time": _TIMESTAMP_COLUMN_PROPERTIES,
        "power_fcst": ColumnProperties(encoding="BYTE_STREAM_SPLIT", dictionary_enabled=False),
    },
)
"""Parquet writer settings for the ``power_forecasts`` Delta table.

``power_fcst`` is ~76% of the bytes. The delta-rs defaults (SNAPPY + dictionary encoding everywhere)
leave that column essentially uncompressed. Measured on the table's largest single file (18.3M rows,
105 MB): ZSTD-3 alone → 80% of the original size; adding these column encodings plus the
``POWER_FORECASTS_SORT_COLS`` row order → 50%; adding the ``POWER_FCST_SIGNIFICAND_BITS``
precision reduction → 29%. That file is the table's least compressible slice. Across the full table
the format achieved 11.5%.
"""


def write_power_forecasts(
    forecasts: pt.DataFrame[PowerForecast],
    table_uri: str | Path,
    *,
    replace_partition: tuple[str, str] | None = None,
    replace_predicate_extra: str | None = None,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write ``PowerForecast`` rows to the ``power_forecasts`` Delta table in its storage format.

    Applies the three storage levers before writing: rounds ``power_fcst`` to
    ``POWER_FCST_SIGNIFICAND_BITS`` significand bits, sorts rows by
    ``POWER_FORECASTS_SORT_COLS``, and writes with ``POWER_FORECASTS_WRITER_PROPERTIES``.

    The table is partitioned by ``(experiment_name, fold_id)``. A multi-chunk materialisation passes
    ``replace_partition`` on its first chunk, and ``None`` (append) on the rest. Passing
    ``replace_partition`` on the first chunk overwrites that partition, so prior rows are always
    replaced, even when the first chunk is empty.

    **Every row of ``forecasts`` must satisfy the predicate the overwrite path builds.** delta-rs
    checks each row against that predicate. If any row falls outside the predicate, delta-rs rejects
    the whole write — ``DeltaError``, nothing committed — rather than writing the rows that do
    match. So a frame spanning two ``(experiment_name, fold_id)`` pairs fails under a single
    ``replace_partition``, and a frame spanning two ``power_fcst_init_time`` values fails under a
    narrowing ``replace_predicate_extra``. Confirmed empirically against ``deltalake`` 1.6.3: a
    frame carrying one row outside the predicate raised ``Invalid data found: 1 rows failed
    validation check``. The table was left at both its previous row count and its previous Delta
    version. ``write_nwp`` documents the same delta-rs behaviour and is no safer. ``write_nwp``'s
    predicate is built from the frame's *first row* (``nwp.item(0, ...)``), so a frame spanning two
    ``(nwp_model_id, init_time)`` partitions is rejected there too. Both functions leave it to the
    caller to pass a frame the predicate covers, and neither checks before writing.

    The append path takes no predicate, so the append path performs no row-by-row check. Two
    identical appends leave two copies, confirmed empirically, because nothing deduplicates on
    ``PowerForecast.PRIMARY_KEY`` at write time. ``PowerForecast``'s uniqueness constraint is
    checked by ``validate()``, on the way in, not by the table.

    **What stops a retry double-counting is the caller's chunking discipline, not anything in this
    function.** The cross-validation asset ``cv_power_forecasts`` restarts its chunk loop from the
    first chunk on every materialisation, and that first chunk always passes ``replace_partition``.
    The first chunk therefore clears whatever an interrupted run left in the ``(experiment_name,
    fold_id)`` partition, before the later chunks append. A caller that appended without an
    overwriting first chunk would silently double the partition's rows instead — see [principle 10,
    every write is atomic and
    idempotent](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#10-every-write-is-atomic-and-idempotent-and-every-failure-is-confined-to-one-partition).

    Args:
        forecasts: Validated forecast rows. ``experiment_name`` and ``fold_id`` are ``String``,
            which is their ``PowerForecast`` dtype. ``String`` is exactly what delta-rs needs for
            Hive-style partition directories, so no cast is required.
        table_uri: Path or URI of the ``power_forecasts`` Delta table.
        replace_partition: ``(experiment_name, fold_id)`` to overwrite, or ``None`` to append.
            Passed explicitly rather than derived from ``forecasts`` so an *empty* first chunk
            still clears the partition.
        replace_predicate_extra: An additional ``AND``-ed SQL predicate clause narrowing the
            overwrite below the ``(experiment_name, fold_id)`` partition. For example,
            ``"power_fcst_init_time = '2026-07-04T06:00:00+00:00'"`` lets ``live_forecasts`` replace
            one 6-hourly slot's rows without wiping the rest of the ``"live"`` fold's
            partition. The production service forecasts every 6 hours, and stores every one of
            those runs under the reserved ``fold_id`` of ``"live"`` rather than under a
            cross-validation fold. delta-rs' ``replaceWhere`` supports predicates on non-partition
            columns (confirmed empirically: a `datetime.isoformat()` literal round-trips correctly
            against a ``Timestamp`` column). Only meaningful alongside ``replace_partition``.
        storage_options: delta-rs object-store options (credentials/endpoint) for a remote
            ``table_uri``; ``None``/empty for a local path.
    """
    prepared = (
        forecasts.with_columns(
            power_fcst=round_to_significand_bits(
                pl.col("power_fcst"), keep_bits=POWER_FCST_SIGNIFICAND_BITS
            )
        )
        .sort(*POWER_FORECASTS_SORT_COLS)
        .to_arrow()
    )
    if replace_partition is not None:
        experiment_name, fold_id = replace_partition
        predicate = f"experiment_name = '{experiment_name}' AND fold_id = '{fold_id}'"
        if replace_predicate_extra is not None:
            predicate = f"{predicate} AND {replace_predicate_extra}"
        write_deltalake(
            table_or_uri=table_uri,
            data=prepared,
            mode="overwrite",
            predicate=predicate,
            partition_by=["experiment_name", "fold_id"],
            writer_properties=POWER_FORECASTS_WRITER_PROPERTIES,
            storage_options=typeddict_to_dict(storage_options),
        )
    else:
        write_deltalake(
            table_or_uri=table_uri,
            data=prepared,
            mode="append",
            partition_by=["experiment_name", "fold_id"],
            writer_properties=POWER_FORECASTS_WRITER_PROPERTIES,
            storage_options=typeddict_to_dict(storage_options),
        )
