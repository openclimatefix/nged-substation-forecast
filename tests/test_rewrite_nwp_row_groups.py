"""Tests for `scripts/forecasting/rewrite_nwp_row_groups.py`.

The script is a one-shot migration that will be pointed at the whole ``nwp`` table with no
rehearsal, so the part worth testing is the part that decides what *not* to touch. Two ways of
getting that decision wrong are silent: reporting nothing leaves every partition unmigrated while
exiting cleanly, and reporting an already-aligned partition rewrites the table on every pass
without ever converging.
"""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Final

import patito as pt
from contracts.weather_schemas import Nwp
from delta_store.nwp import NWP_SORT_COLS, write_nwp
from deltalake import WriterProperties, write_deltalake

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_PATH: Final[Path] = REPO_ROOT / "scripts" / "forecasting" / "rewrite_nwp_row_groups.py"
"""The script under test, imported by path because `scripts/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `rewrite_nwp_row_groups.py` from its path in `scripts/`."""
    spec = importlib.util.spec_from_file_location("rewrite_nwp_row_groups", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rewrite_nwp_row_groups = _load_script()

_MEMBERS: Final[int] = 4
_ROWS_PER_MEMBER: Final[int] = 1_024


def _load_delta_store_tests() -> ModuleType:
    """Import `packages/delta_store/tests/test_nwp.py` for its `Nwp` frame builder.

    Borrowed rather than duplicated: the builder already knows each continuous variable's
    contract range, and a second copy of that table here would drift from the schema.
    """
    path = REPO_ROOT / "packages" / "delta_store" / "tests" / "test_nwp.py"
    spec = importlib.util.spec_from_file_location("delta_store_test_nwp", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_make_nwp = _load_delta_store_tests()._make_nwp


def _run(init_time: datetime, *, rows: int) -> pt.DataFrame[Nwp]:
    """Build a valid ``Nwp`` run whose members are interleaved before the writer sorts them."""
    return _make_nwp(rows, init_time=init_time, n_members=_MEMBERS)


def _write_unaligned(run: pt.DataFrame[Nwp], table: Path) -> None:
    """Write a run the way the pre-migration layout did: row groups spanning many members."""
    write_deltalake(
        table_or_uri=table,
        data=run.sort(*NWP_SORT_COLS).to_arrow(),
        mode="append",
        partition_by=["nwp_model_id", "init_time"],
        writer_properties=WriterProperties(compression="ZSTD", compression_level=3),
    )


def test_only_the_unaligned_partition_is_reported(tmp_path: Path) -> None:
    aligned_at = datetime(2025, 6, 1, tzinfo=UTC)
    unaligned_at = datetime(2025, 6, 2, tzinfo=UTC)
    table = tmp_path / "nwp"
    write_nwp(nwp=_run(aligned_at, rows=_MEMBERS * _ROWS_PER_MEMBER), table_uri=table)
    _write_unaligned(_run(unaligned_at, rows=_MEMBERS * _ROWS_PER_MEMBER), table)

    assert rewrite_nwp_row_groups._unaligned_partitions(str(table), {}) == [unaligned_at]


def test_a_partition_whose_rows_do_not_divide_by_its_members_converges(tmp_path: Path) -> None:
    """A ragged run is reported once, then never again.

    `delta_store.nwp` aligns a ragged run to within one straddling row group by design, which is
    as good as that run gets. Demanding an exact member-per-row-group match would rewrite it on
    every pass, each pass writing another full copy of the data.
    """
    ragged_at = datetime(2025, 6, 3, tzinfo=UTC)
    table = tmp_path / "nwp"
    _write_unaligned(_run(ragged_at, rows=_MEMBERS * _ROWS_PER_MEMBER + 3), table)
    assert rewrite_nwp_row_groups._unaligned_partitions(str(table), {}) == [ragged_at]

    rewrite_nwp_row_groups._rewrite_partition(
        table_uri=str(table), init_time=ragged_at, storage_options={}
    )
    assert rewrite_nwp_row_groups._unaligned_partitions(str(table), {}) == []


def test_an_unreadable_footer_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """Truncation raises `pyarrow.ArrowInvalid`, not `OSError`, and must still fail safe."""
    init_time = datetime(2025, 6, 4, tzinfo=UTC)
    table = tmp_path / "nwp"
    write_nwp(nwp=_run(init_time, rows=_MEMBERS * _ROWS_PER_MEMBER), table_uri=table)
    assert rewrite_nwp_row_groups._unaligned_partitions(str(table), {}) == []

    parquet_file = next(table.rglob("*.parquet"))
    parquet_file.write_bytes(parquet_file.read_bytes()[:2048])

    assert rewrite_nwp_row_groups._unaligned_partitions(str(table), {}) == [init_time]
