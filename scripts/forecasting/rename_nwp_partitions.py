"""Rename every ``nwp`` Delta partition that uses the old ``deltalake`` partition-value format.

``deltalake`` 1.6.x stored the ``init_time`` partition value of a run as
``2026-11-23 00:00:00.000000``, and named the partition directory
``init_time=2026-11-23%2000%3A00%3A00.000000``. ``deltalake`` 1.7.0 stores
``2026-11-23T00:00:00.000000Z`` and names the directory
``init_time=2026-11-23T00%3A00%3A00.000000Z``. Polars cannot parse the two value formats in one
table, so a table holding both fails every ``Nwp.scan_delta`` collect with
``could not parse '2026-11-23 00:00:00.000000' with pattern 'DatetimeYMDZ'``. This script gives
every old-format partition the new format without copying any data.

**The script renames each old-format partition directory and then records the rename in one new
Delta commit.** The commit removes each data file at its old path and adds the same file, with the
same size and statistics, at its new path with the new partition value. No Parquet file is read or
rewritten, so the script takes seconds, needs no free disk space, and runs on a table of any size.
The script works on a local table only, because it renames directories.

**The script undoes itself if the commit fails**, by renaming the directories back. To undo a
finished run by hand, delete the new commit file that the script reports, and rename each
directory back.

The script reads the Delta log directly, replaying the latest checkpoint and the commits after
it. It then checks the result against what ``deltalake`` itself reads, and refuses to commit if
the two disagree on which files are active. After the commit, the script checks that every active
file exists and that every partition value is in the new format.

**Run the script with ``deltalake`` 1.7.0 or later installed, while nothing else writes to the
table.** A write that lands between the script's read of the log and its commit makes the commit
fail, and the script then renames the directories back.

Usage::

    uv run python scripts/forecasting/rename_nwp_partitions.py --table-uri data/NWP --dry-run
    uv run python scripts/forecasting/rename_nwp_partitions.py --table-uri data/NWP
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote, unquote

import polars as pl
import pyarrow.parquet as pq
from contracts.settings import get_settings
from deltalake import DeltaTable

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_NEW_FORMAT_SUFFIX: Final[str] = "Z"
"""Every ``init_time`` partition value that ``deltalake`` 1.7 writes ends in this character."""

_Action = dict[str, Any]
"""One ``add`` action of the Delta log, as the JSON object under its ``add`` key."""


def new_format_value(old_value: str) -> str:
    """Return an old-format partition value in the format that ``deltalake`` 1.7 writes."""
    return datetime.fromisoformat(old_value).replace(tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _directory_segment(value: str) -> str:
    """Return the ``init_time=...`` directory name that ``deltalake`` gives a partition value."""
    return "init_time=" + quote(value, safe="")


def _log_segment(value: str) -> str:
    """Return ``_directory_segment`` as the Delta log spells it: percent-encoded a second time."""
    return quote(_directory_segment(value), safe="=")


def _json_value(value: Any) -> Any:
    """Turn a value that pyarrow read from a checkpoint into the JSON the commit files hold."""
    if isinstance(value, list) and all(
        isinstance(item, tuple) and len(item) == 2 for item in value
    ):
        return {key: _json_value(item) for key, item in value}  # an Arrow map
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def active_add_actions(table_path: Path) -> dict[str, _Action]:
    """Replay the Delta log and return each active file's ``add`` action, keyed by its path.

    Starts from the latest checkpoint named in ``_last_checkpoint``, if there is one, and applies
    every commit file after it. A multi-part checkpoint is read part by part.
    """
    log_dir = table_path / "_delta_log"
    active: dict[str, _Action] = {}
    checkpoint_version = -1
    last_checkpoint = log_dir / "_last_checkpoint"
    if last_checkpoint.exists():
        info = json.loads(last_checkpoint.read_text())
        checkpoint_version = int(info["version"])
        parts = sorted(log_dir.glob(f"{checkpoint_version:020d}.checkpoint*.parquet"))
        for part in parts:
            for row in pq.read_table(part, columns=["add"]).column("add").to_pylist():
                if row is not None:
                    action = {key: _json_value(value) for key, value in row.items()}
                    active[action["path"]] = {
                        key: value for key, value in action.items() if value is not None
                    }
    for commit_file in sorted(log_dir.glob("[0-9]" * 20 + ".json")):
        if int(commit_file.stem) <= checkpoint_version:
            continue
        for line in commit_file.read_text().splitlines():
            if not line.strip():
                continue
            action = json.loads(line)
            if "add" in action:
                active[action["add"]["path"]] = action["add"]
            elif "remove" in action:
                active.pop(action["remove"]["path"], None)
    return active


def _old_format_actions(actions: dict[str, _Action]) -> list[_Action]:
    return [
        action
        for action in actions.values()
        if not action["partitionValues"]["init_time"].endswith(_NEW_FORMAT_SUFFIX)
    ]


def _renamed(action: _Action) -> _Action:
    """Return the ``add`` action for the same file at its new path and partition value."""
    old_value = action["partitionValues"]["init_time"]
    new_value = new_format_value(old_value)
    old_segment, new_segment = _log_segment(old_value), _log_segment(new_value)
    if old_segment not in action["path"]:
        raise ValueError(f"{action['path']} does not contain the segment {old_segment}")
    return {
        **action,
        "path": action["path"].replace(old_segment, new_segment),
        "partitionValues": {**action["partitionValues"], "init_time": new_value},
        "dataChange": True,
    }


def _disk_path(table_path: Path, log_path: str) -> Path:
    """Return the file's location on disk: the log's path, with its second encoding undone."""
    return table_path / unquote(log_path)


def _active_paths(table: DeltaTable) -> set[str]:
    """Return the log path of every file that ``deltalake`` reads as active."""
    return set(pl.DataFrame(table.get_add_actions(flatten=True))["path"].to_list())


def _flat_actions(table: DeltaTable) -> list[tuple[Any, ...]]:
    """Return what ``deltalake`` reads for each active file, other than its path and partition."""
    flat = pl.DataFrame(table.get_add_actions(flatten=True))
    kept = [name for name in flat.columns if name != "path" and not name.startswith("partition.")]
    return sorted(tuple(str(value) for value in row) for row in flat.select(kept).rows())


def rename_partitions(table_path: Path, *, dry_run: bool) -> int:
    """Rename every old-format partition and record the renames in one new commit.

    Returns:
        The number of partitions renamed, or that would be renamed when ``dry_run`` is true.
    """
    table = DeltaTable(table_path)
    actions = active_add_actions(table_path)
    if set(actions) != _active_paths(table):
        raise RuntimeError(
            "The files that the replayed Delta log calls active differ from the files that"
            " deltalake reads. Nothing was changed."
        )
    old = _old_format_actions(actions)
    partitions = sorted({a["partitionValues"]["init_time"] for a in old})
    _LOGGER.info("%s has %d partitions to rename (%d files)", table_path, len(partitions), len(old))
    if dry_run or not old:
        return len(partitions)

    before = _flat_actions(table)
    version = table.version()
    moved: list[tuple[Path, Path]] = []
    new_actions: list[_Action] = []
    try:
        for action in old:
            new_action = _renamed(action)
            source, target = (
                _disk_path(table_path, action["path"]),
                _disk_path(table_path, new_action["path"]),
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            moved.append((source, target))
            new_actions.append(new_action)
        now = int(time.time() * 1000)
        lines = [
            {
                "commitInfo": {
                    "timestamp": now,
                    "operation": "RENAME PARTITIONS",
                    "operationParameters": {"reason": "deltalake 1.7 partition value format"},
                    "engineInfo": "rename_nwp_partitions.py",
                }
            },
            *(
                {
                    "remove": {
                        "path": a["path"],
                        "deletionTimestamp": now,
                        "dataChange": True,
                        "extendedFileMetadata": True,
                        "partitionValues": a["partitionValues"],
                        "size": a["size"],
                        **({"stats": a["stats"]} if "stats" in a else {}),
                    }
                }
                for a in old
            ),
            *({"add": a} for a in new_actions),
        ]
        commit_file = table_path / "_delta_log" / f"{version + 1:020d}.json"
        with commit_file.open("x") as handle:  # "x" fails if another writer took the version
            handle.write("\n".join(json.dumps(line, separators=(",", ":")) for line in lines))
            handle.write("\n")
    except BaseException:
        for source, target in reversed(moved):
            target.rename(source)
            if not any(target.parent.iterdir()):
                target.parent.rmdir()  # the new directory, now empty
        raise

    after_table = DeltaTable(table_path)
    still_old = [p for p in after_table.partitions() if not p["init_time"].endswith("Z")]
    missing = [p for p in _active_paths(after_table) if not _disk_path(table_path, p).exists()]
    unchanged = _flat_actions(after_table) == before
    if after_table.version() != version + 1 or still_old or missing or not unchanged:
        raise RuntimeError(
            f"The commit {commit_file} was written but its check failed: version"
            f" {after_table.version()}, {len(still_old)} old-format partitions, {len(missing)}"
            " missing files, or changed statistics. Delete the commit file and rename the"
            " directories back to undo it."
        )
    for source, _ in moved:
        if not any(source.parent.iterdir()):
            source.parent.rmdir()  # the old directory, now empty
    _LOGGER.info("Wrote %s, and renamed %d directories", commit_file, len(moved))
    return len(partitions)


def main() -> None:
    """Rename the old-format partitions of the ``nwp`` table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-uri", default="", help="nwp Delta table; defaults to settings.nwp_data_path."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be renamed, and change nothing."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    table_path = Path(args.table_uri or get_settings().nwp_data_path)
    rename_partitions(table_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
