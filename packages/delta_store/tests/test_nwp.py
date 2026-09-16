"""Tests for ``write_nwp`` — the ``nwp`` storage format end-to-end.

Writes real (tiny) ``Nwp`` frames into a temp Delta table and asserts the on-disk format:
ZSTD with parquet's *default* encodings (measured better than ``BYTE_STREAM_SPLIT`` /
``DELTA_BINARY_PACKED`` for this table — see ``delta_store.nwp``), member-early sort within
each file, every continuous variable rounded to ``NWP_SIGNIFICAND_BITS``, successive runs landing
as separate ``(nwp_model_id, init_time)`` Hive partitions, and a re-written run replacing its own
partition and no other.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from contracts.weather_schemas import Nwp
from delta_store.nwp import (
    NWP_ROW_GROUP_SIZE_LIMITS,
    NWP_SIGNIFICAND_BITS,
    NWP_SORT_COLS,
    write_nwp,
)
from deltalake import write_deltalake

_T0 = datetime(2025, 6, 1, tzinfo=UTC)

_CONTINUOUS_BASE_VALUES = {
    "temperature_2m": 15.7031,
    "dew_point_temperature_2m": 9.1234,
    "wind_speed_10m": 5.6789,
    "wind_direction_10m": 123.456,
    "wind_speed_100m": 8.9101,
    "wind_direction_100m": 234.567,
    "pressure_surface": 101_234.5,
    "pressure_reduced_to_mean_sea_level": 101_567.8,
    "geopotential_height_500hpa": 5_432.1,
    "downward_long_wave_radiation_flux_surface": 312.34,
    "downward_short_wave_radiation_flux_surface": 456.78,
    "precipitation_surface": 0.00123,
}
"""Per-variable bases well inside each field's ``ge``/``le`` bounds, with full-entropy
mantissas so the significand-rounding assertions have something to measure."""


def _make_nwp(
    n: int = 6, *, init_time: datetime = _T0, n_members: int | None = None
) -> pt.DataFrame[Nwp]:
    """Build a valid ``Nwp`` frame with deliberately unsorted key columns.

    With ``n_members`` set, the ``n`` rows are spread over that many members, interleaved so each
    member's rows are scattered through the frame before the writer sorts them.
    """
    members = list(range(n - 1, -1, -1)) if n_members is None else [i % n_members for i in range(n)]
    rows = {
        # Reverse-ordered members and cycling valid_times so the writer's sort has work to do.
        "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * n,
        "init_time": [init_time] * n,
        "valid_time": [init_time + timedelta(hours=(i % 3) + 1) for i in range(n)],
        "ensemble_member": members,
        "h3_index": [100 + i for i in range(n)],
        "categorical_precipitation_type_surface": [1] * n,
        **{
            # Cycle the scaling so a large frame stays inside each variable's contract range;
            # for the small frames the other tests build this is identical to scaling by i.
            var: [base * (1 + 0.003 * (i % 100)) for i in range(n)]
            for var, base in _CONTINUOUS_BASE_VALUES.items()
        },
    }
    return Nwp.DataFrame(rows).cast().validate()


def test_on_disk_format(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(), table)

    parquet_files = list(table.rglob("*.parquet"))
    assert parquet_files
    for parquet_file in parquet_files:
        row_group = pq.ParquetFile(parquet_file).metadata.row_group(0)
        columns = {
            row_group.column(i).path_in_schema: row_group.column(i)
            for i in range(row_group.num_columns)
        }
        for name, column in columns.items():
            assert column.compression == "ZSTD", f"{name} written as {column.compression}"
            # Unlike power_forecasts, this table measures *worse* with the special encodings —
            # write_nwp must stick to parquet's defaults (see the delta_store.nwp docstring).
            assert "BYTE_STREAM_SPLIT" not in column.encodings, name
            assert "DELTA_BINARY_PACKED" not in column.encodings, name

        rows = pl.read_parquet(parquet_file)
        # nwp_model_id and init_time are Hive partition values, not parquet columns.
        key = rows.select(k=pl.struct([c for c in NWP_SORT_COLS if c in rows.columns]))
        assert key["k"].is_sorted()


def test_every_member_lands_in_its_own_row_group(tmp_path: Path) -> None:
    """Each row group covers one ensemble member, so any member's predicate skips the rest.

    The bug this guards against survived a check on the control member alone: an unaligned write
    still prunes for member 0, because 0 is the minimum and only a row group containing it can
    have a minimum of 0. The members in the middle of the range are the case that breaks.
    """
    n_members = 4
    # Twice the clamp floor, so the derived size and the floor differ and the test sees the
    # derivation rather than the clamp.
    rows_per_member = 2 * NWP_ROW_GROUP_SIZE_LIMITS[0]
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(n_members * rows_per_member, n_members=n_members), table)

    spans = []
    for parquet_file in table.rglob("*.parquet"):
        parquet = pq.ParquetFile(parquet_file)
        member_column = parquet.schema_arrow.get_field_index("ensemble_member")
        for group in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(group).column(member_column).statistics
            assert statistics is not None, "row-group statistics are required to prune"
            spans.append((statistics.min, statistics.max))

    assert len(spans) == n_members, f"expected one row group per member, got {spans}"
    assert all(low == high for low, high in spans), f"a row group spans several members: {spans}"
    assert {low for low, _ in spans} == set(range(n_members))


def test_row_groups_stay_member_aligned_when_the_frame_arrives_in_many_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-chunk Arrow table still lands one ensemble member per row group.

    delta-rs consumes a multi-chunk table out of order while partitioning a write, which is what
    scatters members across row groups. Polars only leaves a frame multi-chunk after the sort when
    it has many threads to sort with, and ``conftest.py`` pins ``POLARS_MAX_THREADS`` to 4, so the
    chunking is reproduced here by slicing the sorted table exactly as a higher thread count would.
    """
    chunks = 8
    unchunked = pl.DataFrame.to_arrow

    def _sliced(frame: pl.DataFrame) -> pa.Table:
        table = unchunked(frame)
        size = table.num_rows // chunks
        return pa.concat_tables(
            [
                table.slice(
                    index * size,
                    size if index < chunks - 1 else table.num_rows - index * size,
                )
                for index in range(chunks)
            ]
        )

    monkeypatch.setattr(pl.DataFrame, "to_arrow", _sliced)

    n_members = 4
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(n_members * NWP_ROW_GROUP_SIZE_LIMITS[0], n_members=n_members), table)

    spans = []
    for parquet_file in table.rglob("*.parquet"):
        parquet = pq.ParquetFile(parquet_file)
        member_column = parquet.schema_arrow.get_field_index("ensemble_member")
        for group in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(group).column(member_column).statistics
            spans.append((statistics.min, statistics.max))

    assert all(low == high for low, high in spans), f"a row group spans several members: {spans}"


def test_the_row_group_limits_bracket_a_real_ensemble_run() -> None:
    """The clamp never binds on a real ECMWF ENS run, which is the case it must not alter.

    Pinned here because the alignment tests size their fixtures *from* these limits, so they grow
    and shrink with the constant and can never see it change. A floor raised above a real run's
    rows-per-member, or a ceiling lowered below it, would silently undo the alignment the rest of
    this module exists to check, with every test still green. This is the same guard, and for the
    same reason, as `test_the_nwp_significand_bits_is_thirteen`.
    """
    rows_per_member_in_a_real_run = 1_671 * 85  # H3 cells over GB x forecast steps
    floor, ceiling = NWP_ROW_GROUP_SIZE_LIMITS
    assert floor < rows_per_member_in_a_real_run < ceiling


def test_continuous_vars_rounded_to_significand_bits(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    n = 6
    write_nwp(_make_nwp(n), table)

    stored_df = pl.read_delta(str(table))
    discarded = np.uint32((1 << (23 - (NWP_SIGNIFICAND_BITS - 1))) - 1)
    for var, base in _CONTINUOUS_BASE_VALUES.items():
        stored = np.sort(stored_df[var].to_numpy())
        expected = np.sort(np.array([base * (1 + 0.003 * i) for i in range(n)], dtype=np.float32))
        rel_err = np.abs(stored - expected) / np.abs(expected)
        assert (rel_err <= 2.0**-NWP_SIGNIFICAND_BITS).all(), var
        # np.dtype(np.uint32), not a bare np.uint32 — see the `ty-workarounds` skill.
        assert (stored.view(np.dtype(np.uint32)) & discarded == 0).all(), var


def test_the_nwp_significand_bits_is_thirteen() -> None:
    """Pins the literal, not a value derived from the constant.

    ``test_continuous_vars_rounded_to_significand_bits`` above computes its tolerance *from*
    ``NWP_SIGNIFICAND_BITS``, so it stays green no matter what the constant is set to. The
    ``power_forecasts`` sibling significand constant carries a matching relative-error claim in
    ``PowerForecast.power_fcst``'s description that a changed constant would leave silently wrong.
    """
    assert NWP_SIGNIFICAND_BITS == 13


def test_successive_runs_create_separate_partitions(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    n = 4
    t1 = _T0 + timedelta(days=1)
    write_nwp(_make_nwp(n), table)
    write_nwp(_make_nwp(n, init_time=t1), table)

    partition_dirs = sorted(
        p.name for p in (table / "nwp_model_id=ECMWF_ENS_0_25_degree").iterdir() if p.is_dir()
    )
    assert len(partition_dirs) == 2
    assert all(name.startswith("init_time=") for name in partition_dirs)

    # Both partitions' rows are intact, and the table round-trips through the read contract.
    scan = Nwp.scan_delta(table)
    collected = scan.collect()
    assert collected.height == 2 * n
    assert collected.filter(pl.col("init_time") == t1).height == n
    assert collected.schema["temperature_2m"] == pl.Float32
    assert collected.schema["nwp_model_id"] == pl.String


def test_rewriting_a_run_replaces_only_its_own_partition(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    n = 4
    init_times = [_T0 + timedelta(days=day) for day in range(3)]
    for init_time in init_times:
        write_nwp(_make_nwp(n, init_time=init_time), table)
    before = Nwp.scan_delta(table).collect().sort(*NWP_SORT_COLS)

    # The replacement run is a different length, so the row count alone says which copy survived.
    replaced = init_times[1]
    write_nwp(_make_nwp(2, init_time=replaced), table)
    after = Nwp.scan_delta(table).collect().sort(*NWP_SORT_COLS)

    assert after.height == 2 * n + 2, "the re-written run landed alongside the first copy"
    assert after.filter(pl.col("init_time") != replaced).equals(
        before.filter(pl.col("init_time") != replaced)
    ), "the replace predicate reached beyond its own (nwp_model_id, init_time) partition"


def test_scan_pushes_filters_into_the_parquet_scan(tmp_path: Path) -> None:
    """A filter on ``ensemble_member``, ``h3_index`` or ``nwp_model_id`` through the production
    read path (``Nwp.scan_delta(...).filter(...)``) must push all the way into the Parquet scan —
    it should appear inside a ``SELECTION`` line in ``.explain()``, not as a ``FILTER`` node sitting
    above a ``WITH_COLUMNS`` cast. The latter shape means every row is decoded before the filter
    ever runs, defeating row-group pruning.

    Fails on `main` today: reproduced directly — today's `explain()` shows no `SELECTION` line for
    any of these three columns, because ``Nwp``'s then-declared dtypes (``UInt8``/``UInt64``/
    ``Enum``) don't match what's physically on disk, so ``Nwp.scan_delta``'s cast is not a no-op
    and sits between the scan and the filter.
    """
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(), table)

    for column, value in (
        ("ensemble_member", 0),
        ("h3_index", 100),
        ("nwp_model_id", "ECMWF_ENS_0_25_degree"),
    ):
        plan = Nwp.scan_delta(table).filter(pl.col(column) == value).explain()
        assert "SELECTION" in plan, f"{column}: predicate pushdown lost — plan:\n{plan}"


def test_categorical_ptype_missing_sentinel_round_trips(tmp_path: Path) -> None:
    """``categorical_precipitation_type_surface``'s own field description names ``255``
    ("Missing") as a legitimate value, but the pre-fix ``UInt8``-declared column couldn't survive
    it: ``write_deltalake`` raised ``Cast error: Can't cast value 255 to type Int8`` for any value
    ``>= 128``. Fails on `main` today for the identical reason."""
    table = tmp_path / "nwp"
    n = 3
    nwp = _make_nwp(n).with_columns(
        categorical_precipitation_type_surface=pl.lit(255, dtype=pl.Int16)
    )

    write_nwp(nwp, table)

    collected = Nwp.scan_delta(table).collect()
    assert collected["categorical_precipitation_type_surface"].to_list() == [255] * n


def test_scan_reads_correctly_across_old_narrow_and_new_wide_physical_partitions(
    tmp_path: Path,
) -> None:
    """No data rewrite is needed for the ``categorical_precipitation_type_surface`` widening
    (``UInt8``/physical ``int8`` -> ``Int16``/physical ``int16``): ``write_nwp``'s permanent
    ``schema_mode="overwrite"`` updates only the table's *logical* schema, and delta-rs promotes
    each Parquet file's physical type to the table's current logical type on read — it does not
    require every file to already agree.

    Simulates a partition written before this dtype change (physically ``int8``, built directly
    rather than through ``write_nwp`` since that function only ever writes the current, ``int16``
    physical layout now) alongside one written after (physically ``int16``, via the real
    ``write_nwp``), and checks the whole table reads back correctly with no error. New coverage,
    not a `main`-failing regression in the strict sense — it guards the "no rewrite is needed"
    claim from silently rotting.
    """
    table = tmp_path / "nwp"
    n = 3
    old_init_time = _T0
    new_init_time = _T0 + timedelta(days=1)

    old_arrow = _make_nwp(n, init_time=old_init_time).to_arrow()
    narrowed_schema = old_arrow.schema.set(
        old_arrow.schema.get_field_index("categorical_precipitation_type_surface"),
        pa.field("categorical_precipitation_type_surface", pa.int8()),
    )
    write_deltalake(
        table_or_uri=table,
        data=old_arrow.cast(narrowed_schema),
        mode="overwrite",
        partition_by=["nwp_model_id", "init_time"],
    )

    write_nwp(_make_nwp(n, init_time=new_init_time), table)

    collected = Nwp.scan_delta(table).collect()
    assert collected.height == 2 * n
    assert collected["categorical_precipitation_type_surface"].to_list() == [1] * (2 * n)
