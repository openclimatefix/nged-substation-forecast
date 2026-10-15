"""Self-checks for experiment 00037's log parser and detectors (an experiment has no pytest suite).

uv run python ml-experiments/00037-switching-first-look/e00037_checks.py
"""

# ruff: noqa: DTZ001
import ast
from datetime import UTC, datetime
from pathlib import Path

import e00037_detect as detect
import e00037_logs as logs
import numpy as np
import polars as pl
from e00037_grid import GRID_LENGTH, GRID_START

HERE = Path(__file__).parent


def check_actions() -> None:
    """Classification of the action strings and switch keys."""
    assert logs.classify_action(action="Earth Close") == "earth"
    assert logs.classify_action(action="Close") == "close"
    assert logs.classify_action(action="Confirm Closed") == "close"
    assert logs.classify_action(action="Open to break load") == "open"
    assert logs.classify_action(action="Confirm Open (dead)") == "open"
    assert logs.classify_action(action="Issue permit") == "other"
    assert logs.switch_key(location="2) 33kV 12AB1234, Alford") == "12AB1234"
    assert logs.switch_key(location="b) Some text, more") == "Some text"
    assert logs.switch_key(location=None) is None


def _actions(*rows: tuple[str, str, int]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"sheet": "1-2", "switch": s, "kind": k, "t": datetime(2024, 1, 1, h), "job": "j"}
            for k, s, h in rows
        ]
    ).sort("sheet", "switch", "t")


def check_state_machine() -> None:
    """Duplicate close, open before close, unpaired close, and two switches on one sheet."""
    frame = logs.reconstruct_intervals(
        confirmed=_actions(("close", "A", 1), ("close", "A", 2), ("open", "A", 5), ("open", "A", 6))
    )
    assert frame.height == 1
    assert frame["hours"][0] == 4.0
    assert logs.reconstruct_intervals(confirmed=_actions(("open", "A", 1))).height == 0
    unpaired = logs.reconstruct_intervals(confirmed=_actions(("close", "A", 1)))
    assert unpaired.height == 1
    assert unpaired["end"][0] is None
    two = logs.reconstruct_intervals(
        confirmed=_actions(("close", "A", 1), ("open", "A", 3), ("close", "B", 2), ("open", "B", 9))
    )
    assert sorted(two["hours"].to_list()) == [2.0, 7.0]


def check_time() -> None:
    """Local-to-UTC on a GMT date, a BST date, and the October fall-back hour."""
    assert logs.to_utc(local=datetime(2024, 1, 15, 12)) == datetime(2024, 1, 15, 12, tzinfo=UTC)
    assert logs.to_utc(local=datetime(2024, 7, 15, 12)) == datetime(2024, 7, 15, 11, tzinfo=UTC)
    assert logs.to_utc(local=datetime(2024, 10, 27, 1, 30)) == datetime(
        2024, 10, 27, tzinfo=UTC
    ).replace(hour=0, minute=30)
    assert logs.to_utc(local=datetime(2024, 3, 31, 1, 30)) is None
    close = datetime(2022, 1, 1, 0, 10, tzinfo=UTC)
    opened = datetime(2022, 1, 1, 1, 10, tzinfo=UTC)
    assert logs.interval_stamps(close_utc=close, open_utc=opened) == (1, 4)


def _flat() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(0)
    return {s: rng.normal(0, 0.1, GRID_LENGTH) for s in (8, 10, 12, 14, 24, 25)}


def check_level_shift() -> None:
    """A synthetic step is recovered in size and stamp."""
    values = np.zeros(GRID_LENGTH)
    values[5000:] = 2.0
    shift = detect.level_shift(values=values, window=8)
    assert abs(shift[5000] - 2.0) < 1e-9
    assert int(np.nanargmax(shift)) == 5000


def check_balance() -> None:
    """A drop split across two neighbours is accepted; a common-mode rise is rejected."""
    series = _flat()
    series[10][5000:] -= 9.0
    series[8][5000:] += 5.0
    series[14][5000:] += 4.0
    inputs = detect.prepare(residuals=series, window=8)
    table = detect.alarms(inputs=inputs, detector="D1", threshold=4.0)
    donor = table.filter(pl.col("series") == 10, (pl.col("stamp") - 5000).abs() <= 1)
    assert donor.height == 1
    assert sorted(int(p.split(":")[0]) for p in donor["partners"][0].split(";")) == [8, 14]
    common = _flat()
    for s in (10, 8, 14):
        common[s][5000:] += 5.0
    rejected = detect.alarms(
        inputs=detect.prepare(residuals=common, window=8), detector="D1", threshold=4.0
    )
    assert rejected.filter((pl.col("stamp") - 5000).abs() <= 1).height == 0


def check_no_log_access() -> None:
    """The detector module and what it imports never read a switching log."""
    forbidden = ("e00037_logs", "e00037_oracle", "logs_raw", "intervals.parquet", "oracle_")
    for name in ("e00037_detect.py", "e00037_common.py", "e00037_grid.py"):
        source = (HERE / name).read_text()
        imported = {
            n.module if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Import | ast.ImportFrom)
            for a in getattr(n, "names", [])
        }
        assert not any(f in str(imported) for f in forbidden), (name, imported)
        assert not any(f in source for f in forbidden[2:]), name


def main() -> None:
    """Run every check."""
    assert GRID_START.tzinfo is not None
    for check in (
        check_actions,
        check_state_machine,
        check_time,
        check_level_shift,
        check_balance,
        check_no_log_access,
    ):
        check()
        print("ok", check.__name__)


if __name__ == "__main__":
    main()
