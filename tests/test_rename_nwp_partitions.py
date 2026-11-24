"""Tests for `scripts/forecasting/rename_nwp_partitions.py`.

The script writes a Delta commit itself instead of going through ``deltalake``. The tests check
what a bug in that commit would break silently: a scan that still fails, a row lost or
duplicated, a later ``write_nwp`` that no longer replaces the right partition, and a failure that
leaves the table unreadable.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Final

import polars as pl
import pytest
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from deltalake import DeltaTable

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
_MODEL_DIR: Final[str] = "nwp_model_id=ECMWF_ENS_0_25_degree"


def _load(name: str, path: Path) -> ModuleType:
    """Import a module by path, because `scripts/` and the package tests are not importable."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rename_nwp_partitions = _load(
    "rename_nwp_partitions", REPO_ROOT / "scripts" / "forecasting" / "rename_nwp_partitions.py"
)
_make_nwp = _load(
    "delta_store_test_nwp", REPO_ROOT / "packages" / "delta_store" / "tests" / "test_nwp.py"
)._make_nwp

_RUNS: Final[list[datetime]] = [
    datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(3)
]


def _segments(run: datetime) -> tuple[str, str]:
    """Return a run's partition directory name in the new and the old ``deltalake`` format."""
    new = f"init_time={run:%Y-%m-%d}T00%3A00%3A00.000000Z"
    old = f"init_time={run:%Y-%m-%d}%2000%3A00%3A00.000000"
    return new, old


def _make_old_format(table: Path, run: datetime) -> None:
    """Make one run's partition look as if ``deltalake`` 1.6.x had written it.

    The helper renames the directory and rewrites the same strings in the Delta log. The log holds
    the path percent-encoded twice and the partition value as plain text.
    """
    new, old = _segments(run)
    (table / _MODEL_DIR / new).rename(table / _MODEL_DIR / old)
    for log_file in (table / "_delta_log").glob("*.json"):
        log_file.write_text(
            log_file.read_text()
            .replace(new.replace("%", "%25"), old.replace("%", "%25"))
            .replace(f"{run:%Y-%m-%d}T00:00:00.000000Z", f"{run:%Y-%m-%d} 00:00:00.000000")
        )


_KEY: Final[list[str]] = ["init_time", "ensemble_member", "valid_time", "h3_index"]


@pytest.fixture
def mixed_table(tmp_path: Path) -> Path:
    """A table with two old-format partitions and one new-format partition."""
    table = tmp_path / "nwp"
    for run in _RUNS:
        write_nwp(nwp=_make_nwp(4, init_time=run), table_uri=table)
    _make_old_format(table, _RUNS[0])
    _make_old_format(table, _RUNS[1])
    return table


def test_a_mixed_table_cannot_be_scanned_until_the_script_has_run(mixed_table: Path) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="could not parse"):
        Nwp.scan_delta(mixed_table).collect()
    before = pl.scan_delta(mixed_table, use_pyarrow=True).collect().sort(*_KEY)

    assert rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False) == 2

    after = Nwp.scan_delta(mixed_table).collect().sort(*_KEY)
    assert after.height == 12
    assert after.select(before.columns).equals(before.cast(after.select(before.columns).schema))
    assert sorted(p["init_time"] for p in DeltaTable(mixed_table).partitions()) == [
        f"{run:%Y-%m-%d}T00:00:00.000000Z" for run in _RUNS
    ]


def test_a_dry_run_changes_nothing(mixed_table: Path) -> None:
    log_before = sorted(p.name for p in (mixed_table / "_delta_log").iterdir())

    assert rename_nwp_partitions.rename_partitions(mixed_table, dry_run=True) == 2

    assert sorted(p.name for p in (mixed_table / "_delta_log").iterdir()) == log_before


def test_a_second_run_finds_nothing_to_rename(mixed_table: Path) -> None:
    rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False)
    version = DeltaTable(mixed_table).version()

    assert rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False) == 0

    assert DeltaTable(mixed_table).version() == version


def test_write_nwp_still_replaces_only_its_own_partition_after_the_rename(
    mixed_table: Path,
) -> None:
    rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False)

    write_nwp(nwp=_make_nwp(2, init_time=_RUNS[0]), table_uri=mixed_table)

    scanned = Nwp.scan_delta(mixed_table).collect()
    assert scanned.filter(pl.col("init_time") == _RUNS[0]).height == 2
    assert scanned.height == 2 + 4 + 4


def test_a_failed_commit_leaves_the_table_readable_and_a_rerun_finishes(
    mixed_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail at the flush of the commit file, after every file has been linked."""
    log_before = sorted(p.name for p in (mixed_table / "_delta_log").iterdir())

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(rename_nwp_partitions.os, "fsync", fail)
        with pytest.raises(OSError, match="disk full"):
            rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False)

    assert sorted(p.name for p in (mixed_table / "_delta_log").iterdir()) == log_before
    assert pl.scan_delta(mixed_table, use_pyarrow=True).collect().height == 12

    assert rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False) == 2
    assert Nwp.scan_delta(mixed_table).collect().height == 12


def test_a_table_with_writer_features_is_refused(
    mixed_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = SimpleNamespace(min_writer_version=7, writer_features=("deletionVectors",))
    monkeypatch.setattr(DeltaTable, "protocol", lambda _self: protocol)

    with pytest.raises(RuntimeError, match="writer version"):
        rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False)


def test_restore_undoes_the_migration(mixed_table: Path) -> None:
    version_before = DeltaTable(mixed_table).version()
    rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False)

    DeltaTable(mixed_table).restore(version_before)

    assert sorted(p["init_time"] for p in DeltaTable(mixed_table).partitions()) == [
        f"{_RUNS[0]:%Y-%m-%d} 00:00:00.000000",
        f"{_RUNS[1]:%Y-%m-%d} 00:00:00.000000",
        f"{_RUNS[2]:%Y-%m-%dT00:00:00.000000Z}",
    ]
    assert pl.scan_delta(mixed_table, use_pyarrow=True).collect().height == 12
