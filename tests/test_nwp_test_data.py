"""Tests for ``tests/_nwp_test_data.py`` — the shared synthetic-``Nwp`` fixture writer.

``write_test_nwp`` hand-copies part of ``delta_store.nwp.write_nwp``'s storage layout rather than
calling it directly — its docstring explains why: ``write_nwp``'s significand rounding would
perturb some of this fixture's exact values. This module is the guard against that hand-copy
silently drifting from the real writer, and it pins the contract-validation gate that stops a
hand-rolled, contract-invalid fixture reaching a test undetected.
"""

from datetime import UTC, datetime
from pathlib import Path

import patito as pt
import pytest
from _nwp_test_data import nwp_records, write_test_nwp
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from deltalake import DeltaTable
from patito.exceptions import DataFrameValidationError

_INIT_TIME = datetime(2025, 6, 1, tzinfo=UTC)


def _as_real_nwp(records: list[dict]) -> pt.DataFrame[Nwp]:
    """Build the same rows as a validated ``Nwp`` frame, independently of ``write_test_nwp``.

    Uses Patito's own no-arg ``.cast()`` (casts every column to the model's declared dtype)
    rather than ``write_test_nwp``'s hand-written cast dict, so this doesn't just check
    ``write_test_nwp`` against its own casting logic.
    """
    columnar = {key: [record[key] for record in records] for key in records[0]}
    return Nwp.DataFrame(columnar).cast().validate()


def test_partition_layout_matches_write_nwp(tmp_path: Path) -> None:
    """Pin ``write_test_nwp``'s hard-copied ``partition_by`` to ``write_nwp``'s real layout.

    If ``write_nwp``'s partitioning ever changes, this fails and names exactly what to update in
    ``write_test_nwp`` — see that function's docstring for why it can't just call ``write_nwp``
    for the whole fixture instead.
    """
    records = nwp_records(cell=1, day=_INIT_TIME, members=(0,))

    fixture_table = tmp_path / "fixture"
    write_test_nwp(str(fixture_table), records)

    real_table = tmp_path / "real"
    write_nwp(nwp=_as_real_nwp(records), table_uri=real_table)

    assert (
        DeltaTable(str(fixture_table)).metadata().partition_columns
        == DeltaTable(str(real_table)).metadata().partition_columns
    )


def test_rejects_contract_invalid_records(tmp_path: Path) -> None:
    """``write_test_nwp`` calls ``Nwp.validate`` before writing, so a bad fixture raises here.

    Guards against a repeat of the earlier fixture that wrote contract-invalid NWP that no test
    noticed: an out-of-range ``wind_direction_10m`` (``Nwp`` declares ``ge=0, le=360``) must raise,
    not write.
    """
    records = nwp_records(cell=1, day=_INIT_TIME, members=(0,))
    records[0]["wind_direction_10m"] = 400.0

    with pytest.raises(DataFrameValidationError):
        write_test_nwp(str(tmp_path / "invalid"), records)
