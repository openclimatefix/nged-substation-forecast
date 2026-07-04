"""Storage policy for the internal ``power_forecasts`` Delta table.

Owns everything about how ``PowerForecast`` rows are laid out on disk: the parquet writer
properties (codec + per-column encodings), the compression-friendly row order, and the
``power_fcst`` precision reduction. Callers write through :func:`write_power_forecasts` so it is
impossible to land rows in the table without this format applied.

Measured impact: rewriting the full 403.6M-row development table into this format shrank it from
6.33 GB (delta-rs defaults: SNAPPY, dictionary encoding, unsorted, full precision) to 0.73 GB.
See the ``POWER_FORECASTS_WRITER_PROPERTIES`` docstring for the per-lever breakdown.
"""

from pathlib import Path
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import PowerForecast
from deltalake import ColumnProperties, WriterProperties, write_deltalake

from delta_store.precision import round_to_significand_bits

POWER_FCST_SIGNIFICAND_BITS: Final[int] = 13
"""Significand bits kept when storing ``power_fcst`` (1 implicit + 12 explicit fraction bits).

Rounding to nearest at 13 significand bits caps the relative error at 2⁻¹³ ≈ 1.2×10⁻⁴ — orders
of magnitude below forecast error — and zeroes the 11 low fraction bits, which are otherwise
pure entropy that defeats every compression codec (nearly every full-precision ``power_fcst``
value is distinct). Mirrors the significand-rounding scheme ``delta_store.nwp`` applies to NWP
data — but with different writer properties; see that module's docstring for why the choice
doesn't transfer between tables. See ``POWER_FORECASTS_WRITER_PROPERTIES`` for the measured
size impact.
"""

POWER_FORECASTS_SORT_COLS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
)
"""Within-file row order for ``power_forecasts`` writes.

Placing the ~51 ensemble members of one (series, init time, valid time) target on adjacent rows
makes ``power_fcst`` locally smooth and the timestamp columns stepped sequences — exactly what
the BYTE_STREAM_SPLIT and DELTA_BINARY_PACKED encodings in ``POWER_FORECASTS_WRITER_PROPERTIES``
need to compress well. Leading with ``time_series_id`` also lets parquet row-group statistics
prune scans that filter on one series.
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

The delta-rs defaults (SNAPPY + dictionary encoding everywhere) leave ``power_fcst`` — ~76% of
the bytes — essentially uncompressed. Measured on the table's largest single file (18.3M rows,
105 MB): ZSTD-3 alone → 80% of the original size; adding these column encodings plus the
``POWER_FORECASTS_SORT_COLS`` row order → 50%; adding the ``POWER_FCST_SIGNIFICAND_BITS``
precision reduction → 29%. (That file is the table's least compressible slice — across the full
table the format achieved 11.5%.)
"""


def write_power_forecasts(
    forecasts: pt.DataFrame[PowerForecast],
    table_uri: str | Path,
    *,
    replace_partition: tuple[str, str] | None = None,
) -> None:
    """Write ``PowerForecast`` rows to the ``power_forecasts`` Delta table in its storage format.

    Applies the three storage levers before writing: rounds ``power_fcst`` to
    ``POWER_FCST_SIGNIFICAND_BITS`` significand bits, sorts rows by
    ``POWER_FORECASTS_SORT_COLS``, and writes with ``POWER_FORECASTS_WRITER_PROPERTIES``.

    The table is partitioned by ``(experiment_name, fold_id)``. A multi-chunk materialisation
    passes ``replace_partition`` on its first chunk — overwriting that partition so prior rows
    are always replaced, even when the first chunk is empty — and ``None`` (append) on the rest.

    Args:
        forecasts: Validated forecast rows. ``experiment_name`` / ``fold_id`` are ``String``
            (their ``PowerForecast`` dtype), which is exactly what delta-rs needs for Hive-style
            partition directories — no cast required.
        table_uri: Path or URI of the ``power_forecasts`` Delta table.
        replace_partition: ``(experiment_name, fold_id)`` to overwrite, or ``None`` to append.
            Passed explicitly rather than derived from ``forecasts`` so an *empty* first chunk
            still clears the partition.
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
        write_deltalake(
            table_or_uri=table_uri,
            data=prepared,
            mode="overwrite",
            predicate=f"experiment_name = '{experiment_name}' AND fold_id = '{fold_id}'",
            partition_by=["experiment_name", "fold_id"],
            writer_properties=POWER_FORECASTS_WRITER_PROPERTIES,
        )
    else:
        write_deltalake(
            table_or_uri=table_uri,
            data=prepared,
            mode="append",
            partition_by=["experiment_name", "fold_id"],
            writer_properties=POWER_FORECASTS_WRITER_PROPERTIES,
        )
