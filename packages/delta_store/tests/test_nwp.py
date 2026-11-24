"""Tests for ``write_nwp`` — the ``nwp`` storage format end-to-end.

Writes real (tiny) ``Nwp`` frames into a temp Delta table and asserts the on-disk format: ZSTD
with parquet's *default* encodings (measured better than ``BYTE_STREAM_SPLIT`` /
``DELTA_BINARY_PACKED`` for this table — see ``delta_store.nwp``), member-early sort within each
file, every continuous variable rounded to ``NWP_SIGNIFICAND_BITS``, successive runs landing as
separate ``(nwp_model_id, init_time)`` Hive partitions, and a re-written run replacing its own
partition and no other.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from contracts.weather_schemas import Nwp, variables_known_empty_at
from delta_store.nwp import (
    NWP_ROW_GROUP_SIZE_LIMITS,
    NWP_SIGNIFICAND_BITS,
    NWP_SORT_COLS,
    write_nwp,
)
from deltalake import DeltaTable, write_deltalake
from polars.testing import assert_frame_equal

# After every window in `NWP_VARIABLE_EMPTY_WINDOWS`, so that populated
# `categorical_precipitation_type_surface` and `total_cloud_cover_atmosphere` columns are valid.
_T0 = datetime(2026, 1, 1, tzinfo=UTC)

_CONTINUOUS_BASE_VALUES = {
    "temperature_2m": 15.7031,
    "dew_point_temperature_2m": 9.1234,
    "wind_u_10m": -3.4567,
    "wind_v_10m": 5.6789,
    "wind_u_100m": 1.2345,
    "wind_v_100m": 8.9101,
    "pressure_surface": 101_234.5,
    "pressure_reduced_to_mean_sea_level": 101_567.8,
    "geopotential_height_500hpa": 5_432.1,
    "downward_long_wave_radiation_flux_surface": 312.34,
    "downward_short_wave_radiation_flux_surface": 456.78,
    "precipitation_surface": 0.00123,
    "total_cloud_cover_atmosphere": 45.678,
}
"""Per-variable bases well inside each field's ``ge``/``le`` bounds, with full-entropy
mantissas so the significand-rounding assertions have something to measure."""


def _make_nwp(
    n: int = 6, *, init_time: datetime = _T0, n_members: int | None = None
) -> pt.DataFrame[Nwp]:
    """Build a valid ``Nwp`` frame with deliberately unsorted key columns.

    With ``n_members`` set, the ``n`` rows are spread over that many members, interleaved so each
    member's rows are scattered through the frame before the writer sorts them. A variable in
    ``NWP_VARIABLE_EMPTY_WINDOWS`` is null when ``init_time`` is inside its window, as the contract
    requires.
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
    if "total_cloud_cover_atmosphere" in variables_known_empty_at(init_time):
        rows["total_cloud_cover_atmosphere"] = [None] * n
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
    scatters members across row groups. Polars only leaves a frame multi-chunk after the sort
    when it has many threads to sort with, and ``conftest.py`` pins ``POLARS_MAX_THREADS`` to 4,
    so the chunking is reproduced here by slicing the sorted table exactly as a higher thread
    count would.
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
        assert (stored.view(np.uint32) & discarded == 0).all(), var


def test_the_nwp_significand_bits_is_thirteen() -> None:
    """Pins the literal, not a value derived from the constant.

    ``test_continuous_vars_rounded_to_significand_bits`` above computes its tolerance *from*
    ``NWP_SIGNIFICAND_BITS``, so it stays green no matter what the constant is set to. The
    ``power_forecasts`` sibling significand constant carries a matching relative-error claim in
    ``PowerForecast.power_fcst``'s description that a changed constant would leave silently
    wrong.
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
    """A filter on ``ensemble_member``, ``h3_index`` or ``nwp_model_id`` through the production read
    path (``Nwp.scan_delta(...).filter(...)``) must push all the way into the Parquet scan — it
    should appear inside a ``SELECTION`` line in ``.explain()``, not as a ``FILTER`` node sitting
    above a ``WITH_COLUMNS`` cast. The latter shape means every row is decoded before the filter
    ever runs, defeating row-group pruning.

    Fails on `main` today: reproduced directly — today's `explain()` shows no `SELECTION` line
    for any of these three columns, because ``Nwp``'s then-declared dtypes (``UInt8``/``UInt64``/
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
    ("Missing") as a legitimate value, but the pre-fix ``UInt8``-declared column couldn't
    survive it: ``write_deltalake`` raised ``Cast error: Can't cast value 255 to type Int8``
    for any value ``>= 128``. Fails on `main` today for the identical reason."""
    table = tmp_path / "nwp"
    n = 3
    nwp = _make_nwp(n).with_columns(
        categorical_precipitation_type_surface=pl.lit(255, dtype=pl.Int16)
    )

    write_nwp(nwp, table)

    collected = Nwp.scan_delta(table).collect()
    assert collected["categorical_precipitation_type_surface"].to_list() == [255] * n


@pytest.mark.parametrize("schema_change", ["polar", "missing", "extra"])
def test_writer_rejects_changed_column_set_without_modifying_table(
    tmp_path: Path, schema_change: str
) -> None:
    table = tmp_path / "existing"
    old = _make_nwp()
    if schema_change == "polar":
        old = old.with_columns(
            **{
                f"wind_{quantity}_{height}": pl.lit(value, dtype=pl.Float32)
                for height in ("10m", "100m")
                for quantity, value in (("speed", 5.0), ("direction", 180.0))
            }
        ).drop([f"wind_{axis}_{height}" for height in ("10m", "100m") for axis in ("u", "v")])
    elif schema_change == "missing":
        old = old.drop("wind_v_100m")
    else:
        old = old.with_columns(unexpected_column=pl.lit(1))
    write_deltalake(
        table_or_uri=table, data=old.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )
    before = pl.read_delta(str(table)).sort("h3_index")
    with pytest.raises(ValueError, match="incompatible column set") as error:
        write_nwp(nwp=_make_nwp(init_time=_T0 + timedelta(days=1)), table_uri=table)
    assert str(table) in str(error.value)
    assert DeltaTable(str(table)).version() == 0
    assert_frame_equal(before, pl.read_delta(str(table)).sort("h3_index"), check_exact=True)


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


def _file_digests(table: Path) -> dict[str, str]:
    """SHA-256 of every Parquet file the table's current version lists, keyed by relative path."""
    return {
        str(path.relative_to(table)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(table.rglob("*.parquet"))
    }


def _write_table_without_total_cloud(table: Path, *, init_time: datetime) -> None:
    """Write a partition in the schema the table had before `total_cloud_cover_atmosphere`.

    Built with `write_deltalake` directly, because `write_nwp` only writes the current schema.
    """
    old = _make_nwp(init_time=init_time).drop("total_cloud_cover_atmosphere")
    write_deltalake(
        table_or_uri=table, data=old.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )


def test_a_table_without_total_cloud_accepts_a_partition_that_has_it(tmp_path: Path) -> None:
    """The back-fill adds `total_cloud_cover_atmosphere` to a table that never had it.

    The existing partition's Parquet file must survive byte for byte, still be listed by the new
    table version, and read the new column as null.
    """
    table = tmp_path / "nwp"
    old_init_time = _T0
    new_init_time = _T0 - timedelta(days=700)
    _write_table_without_total_cloud(table, init_time=old_init_time)
    old_digests = _file_digests(table)
    old_listed_files = set(DeltaTable(str(table)).file_uris())

    write_nwp(_make_nwp(init_time=new_init_time), table)

    new_digests = _file_digests(table)
    assert {path: new_digests.get(path) for path in old_digests} == old_digests
    assert old_listed_files <= set(DeltaTable(str(table)).file_uris())
    assert DeltaTable(str(table)).version() == 1
    collected = Nwp.scan_delta(table).collect()
    for init_time, has_cloud in ((old_init_time, False), (new_init_time, True)):
        partition = collected.filter(pl.col("init_time") == init_time)
        assert partition.height == 6
        assert (partition["total_cloud_cover_atmosphere"].null_count() == 0) is has_cloud
        assert (partition["total_cloud_cover_atmosphere"].null_count() == 6) is not has_cloud
    Nwp.validate(
        collected.filter(pl.col("init_time") == new_init_time)
    )  # the new partition is whole


def test_total_cloud_is_rounded_to_significand_bits(tmp_path: Path) -> None:
    table = tmp_path / "nwp"
    nwp = _make_nwp()

    write_nwp(nwp, table)

    stored = pl.read_delta(str(table)).sort("h3_index")
    expected = nwp.sort("h3_index")["total_cloud_cover_atmosphere"].to_numpy()
    kept = stored["total_cloud_cover_atmosphere"].to_numpy()
    assert not np.array_equal(kept, expected)  # rounding changed the values...
    np.testing.assert_allclose(
        kept, expected, rtol=2.0**-NWP_SIGNIFICAND_BITS
    )  # ...but only a little


def test_scan_reads_total_cloud_as_null_from_a_table_that_lacks_the_column(
    tmp_path: Path,
) -> None:
    table = tmp_path / "nwp"
    _write_table_without_total_cloud(table, init_time=_T0)

    collected = Nwp.scan_delta(table).collect()

    assert collected.height == 6
    assert collected["total_cloud_cover_atmosphere"].dtype == pl.Float32
    assert collected["total_cloud_cover_atmosphere"].null_count() == 6
    assert set(collected.columns) == set(Nwp.model_fields)


def test_scan_filters_still_prune_partitions_of_a_table_that_lacks_the_column(
    tmp_path: Path,
) -> None:
    table = tmp_path / "nwp"
    _write_table_without_total_cloud(table, init_time=_T0)

    plan = Nwp.scan_delta(table).filter(pl.col("init_time") == _T0).explain()

    assert "SELECTION" in plan or "FILTER" in plan


def test_rewriting_a_partition_of_a_mixed_table_replaces_only_that_partition(
    tmp_path: Path,
) -> None:
    table = tmp_path / "nwp"
    _write_table_without_total_cloud(table, init_time=_T0)
    write_nwp(_make_nwp(init_time=_T0 + timedelta(days=1)), table)
    old_digests = _file_digests(table)

    write_nwp(_make_nwp(init_time=_T0 + timedelta(days=1)), table)

    after = _file_digests(table)
    kept = {
        path
        for path in old_digests
        if path.startswith("nwp_model_id=ECMWF_ENS_0_25_degree/init_time=2026-01-01")
    }
    assert kept
    assert {path: after.get(path) for path in kept} == {path: old_digests[path] for path in kept}
    collected = Nwp.scan_delta(table).collect()
    assert collected.height == 12


def test_writer_rejects_a_frame_that_lacks_a_column_the_table_already_has(
    tmp_path: Path,
) -> None:
    """Only a *table* may lack `total_cloud_cover_atmosphere`; a frame must carry it."""
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(), table)

    incoming = _make_nwp(init_time=_T0 + timedelta(days=1)).drop("total_cloud_cover_atmosphere")
    with pytest.raises(ValueError, match="incompatible column set"):
        write_nwp(nwp=incoming, table_uri=table)
    assert DeltaTable(str(table)).version() == 0


def test_the_partition_directory_name_keeps_the_format_the_migration_script_expects(
    tmp_path: Path,
) -> None:
    """A datetime partition's value must end in ``Z``, as ``deltalake`` 1.7 writes it.

    ``scripts/forecasting/rewrite_nwp_partition_names.py`` tells a migrated partition from an
    unmigrated one by that suffix. Polars also infers the datetime format from the first partition
    directory it lists, so a table holding two formats cannot be scanned. If this test fails, a
    ``deltalake`` upgrade changed the format again: migrate the existing tables before accepting
    the upgrade.
    """
    table = tmp_path / "nwp"
    write_nwp(_make_nwp(), table)

    partition_dirs = [
        p.name for p in (table / "nwp_model_id=ECMWF_ENS_0_25_degree").iterdir() if p.is_dir()
    ]
    assert partition_dirs == ["init_time=2026-01-01T00%3A00%3A00.000000Z"]
    assert [p["init_time"] for p in DeltaTable(table).partitions()] == [
        "2026-01-01T00:00:00.000000Z"
    ]
