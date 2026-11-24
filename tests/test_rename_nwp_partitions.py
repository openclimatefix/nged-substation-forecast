"""Tests for `scripts/forecasting/rename_nwp_partitions.py`.

The script edits the Delta log of the whole ``nwp`` table by hand, so the tests check what a bug
there would break silently: a scan that still fails, a row lost or duplicated, a later
``write_nwp`` that no longer replaces the right partition, and a half-finished rename.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
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

    Renames the directory and rewrites the same strings in the Delta log. The log holds the path
    percent-encoded twice and the partition value as plain text.
    """
    new, old = _segments(run)
    (table / _MODEL_DIR / new).rename(table / _MODEL_DIR / old)
    for log_file in (table / "_delta_log").glob("*.json"):
        log_file.write_text(
            log_file.read_text()
            .replace(new.replace("%", "%25"), old.replace("%", "%25"))
            .replace(f"{run:%Y-%m-%d}T00:00:00.000000Z", f"{run:%Y-%m-%d} 00:00:00.000000")
        )


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
    for run in _RUNS:
        new, old = _segments(run)
        assert (mixed_table / _MODEL_DIR / new).is_dir()
        assert not (mixed_table / _MODEL_DIR / old).exists()


_KEY: Final[list[str]] = ["init_time", "ensemble_member", "valid_time", "h3_index"]


def test_a_dry_run_changes_nothing(mixed_table: Path) -> None:
    log_before = sorted(p.name for p in (mixed_table / "_delta_log").iterdir())

    assert rename_nwp_partitions.rename_partitions(mixed_table, dry_run=True) == 2

    assert sorted(p.name for p in (mixed_table / "_delta_log").iterdir()) == log_before
    assert (mixed_table / _MODEL_DIR / _segments(_RUNS[0])[1]).is_dir()


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


def test_a_failed_commit_renames_the_directories_back(
    mixed_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail at the moment of the commit, after every directory has been renamed."""

    def fail(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("another writer took the version")

    monkeypatch.setattr(rename_nwp_partitions.json, "dumps", fail)

    with pytest.raises(RuntimeError, match="another writer"):
        rename_nwp_partitions.rename_partitions(mixed_table, dry_run=False)

    for run in _RUNS[:2]:
        new, old = _segments(run)
        assert (mixed_table / _MODEL_DIR / old).is_dir()
        assert not (mixed_table / _MODEL_DIR / new).exists()
