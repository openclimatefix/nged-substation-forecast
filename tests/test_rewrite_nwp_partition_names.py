"""Tests for `scripts/forecasting/rewrite_nwp_partition_names.py`.

The script is a one-shot migration that is pointed at a whole ``nwp`` table. The test builds a
table that mixes the old and the new ``deltalake`` partition directory names, which is the state
that makes `Nwp.scan_delta` raise, and checks that the script repairs it and then leaves it alone.
"""

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Final

import polars as pl
import pytest
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
_MODEL_DIR: Final[str] = "nwp_model_id=ECMWF_ENS_0_25_degree"
_NEW_NAME: Final[str] = "init_time=2026-01-02T00%3A00%3A00.000000Z"
_OLD_NAME: Final[str] = "init_time=2026-01-02%2000%3A00%3A00.000000"


def _load(name: str, path: Path) -> ModuleType:
    """Import a module by path, because `scripts/` and the package tests are not importable."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rewrite_nwp_partition_names = _load(
    "rewrite_nwp_partition_names",
    REPO_ROOT / "scripts" / "forecasting" / "rewrite_nwp_partition_names.py",
)
_make_nwp = _load(
    "delta_store_test_nwp", REPO_ROOT / "packages" / "delta_store" / "tests" / "test_nwp.py"
)._make_nwp

_FIRST_RUN: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
_SECOND_RUN: Final[datetime] = _FIRST_RUN + timedelta(days=1)


def _rename_second_partition_to_the_old_format(table: Path) -> None:
    """Make the second run's partition look as if ``deltalake`` 1.6.x had written it.

    Renames the directory and rewrites the same two strings in the Delta log. The log holds the
    path percent-encoded twice and the partition value as plain text.
    """
    (table / _MODEL_DIR / _NEW_NAME).rename(table / _MODEL_DIR / _OLD_NAME)
    for log_file in (table / "_delta_log").glob("*.json"):
        text = log_file.read_text()
        text = text.replace(_NEW_NAME.replace("%", "%25"), _OLD_NAME.replace("%", "%25")).replace(
            "2026-01-02T00:00:00.000000Z", "2026-01-02 00:00:00.000000"
        )
        for line in text.splitlines():
            json.loads(line)  # fail loudly if the edit broke the log
        log_file.write_text(text)


@pytest.fixture
def mixed_table(tmp_path: Path) -> Path:
    table = tmp_path / "nwp"
    write_nwp(nwp=_make_nwp(4, init_time=_FIRST_RUN), table_uri=table)
    write_nwp(nwp=_make_nwp(4, init_time=_SECOND_RUN), table_uri=table)
    _rename_second_partition_to_the_old_format(table)
    return table


def test_a_mixed_table_cannot_be_scanned_until_the_script_has_run(mixed_table: Path) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="could not parse"):
        Nwp.scan_delta(mixed_table).collect()

    pending = rewrite_nwp_partition_names.old_format_partitions(str(mixed_table), {})
    assert pending == [_SECOND_RUN]
    rewrite_nwp_partition_names.rewrite_partition(
        table_uri=str(mixed_table), init_time=_SECOND_RUN, storage_options={}
    )

    collected = Nwp.scan_delta(mixed_table).collect()
    assert collected.height == 8
    assert collected.filter(pl.col("init_time") == _SECOND_RUN).height == 4


def test_a_migrated_table_has_nothing_left_to_rewrite(mixed_table: Path) -> None:
    rewrite_nwp_partition_names.rewrite_partition(
        table_uri=str(mixed_table), init_time=_SECOND_RUN, storage_options={}
    )

    assert rewrite_nwp_partition_names.old_format_partitions(str(mixed_table), {}) == []
