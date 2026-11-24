"""Give every ``nwp`` Delta partition in the old ``deltalake`` partition-value format the new one.

``deltalake`` 1.6.x stored the ``init_time`` partition value of a run as
``2026-11-23 00:00:00.000000``, and named the partition directory
``init_time=2026-11-23%2000%3A00%3A00.000000``. ``deltalake`` 1.7.0 stores
``2026-11-23T00:00:00.000000Z`` and names the directory
``init_time=2026-11-23T00%3A00%3A00.000000Z``. Polars cannot parse the two value formats in one
table, so a table holding both fails every ``Nwp.scan_delta`` collect with
``could not parse '2026-11-23 00:00:00.000000' with pattern 'DatetimeYMDZ'``. This script gives
every old-format partition the new format without copying any data.

**The script hard-links each data file of an old-format partition into the directory that
``deltalake`` 1.7 would have used, writes one new Delta commit, and then deletes the old links.**
The commit removes each data file at its old path and adds the same file, with the same size and
statistics, at its new path with the new partition value. A hard link is a second name for the
same file on disk, so no Parquet file is read, rewritten, or copied. The script takes seconds and
needs no free disk space. It works on a local table only, because it creates hard links.

**A crash at any point before the commit leaves the table readable.** The only trace is the new
links, which a rerun reuses, or which ``vacuum`` removes. The commit file is created atomically,
so the log never holds a partly written commit. If the script fails after the commit, the table
is valid, the old links remain, and the script reports what failed. To undo a finished run, restore
the table to the version before the commit with ``DeltaTable.restore`` while no later commit
exists. Do not delete the commit file once another writer has committed after it.

The script reads the Delta log directly, replaying the latest checkpoint and the commits after
it. It refuses to run if the table's protocol has writer features, and it refuses to commit if
its replay of the log and ``deltalake`` disagree on which files are active. After the commit, the
script checks that every active file exists and that every partition value is in the new format.

**Run the script with ``deltalake`` 1.7.0 or later installed, while nothing else writes to the
table.** A write that lands between the script's read of the log and its commit makes the commit
fail, and a rerun then succeeds.

Usage::

    uv run python scripts/forecasting/rename_nwp_partitions.py --table-uri data/NWP --dry-run
    uv run python scripts/forecasting/rename_nwp_partitions.py --table-uri data/NWP
"""

import argparse
import json
import logging
import os
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


def _write_commit(commit_file: Path, lines: list[dict[str, Any]]) -> None:
    """Create ``commit_file`` atomically, failing if that Delta version is already taken.

    Writes the whole commit to a temporary file in the same folder, flushes it to disk, and then
    hard-links the temporary file to the commit's name. The link either appears complete or does
    not appear, and raises ``FileExistsError`` if another writer took the version. A crash or a
    full disk therefore cannot leave a partly written commit in the log.
    """
    temporary = commit_file.with_name(commit_file.name + ".tmp")
    try:
        with temporary.open("w") as handle:
            handle.write("\n".join(json.dumps(line, separators=(",", ":")) for line in lines))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, commit_file)
    finally:
        temporary.unlink(missing_ok=True)


def rename_partitions(table_path: Path, *, dry_run: bool) -> int:
    """Give every old-format partition the new format, recording it in one new commit.

    Hard-links each data file to its new path, writes one commit that removes the old paths and
    adds the new ones, and then deletes the old links. The hard links make every step before the
    commit harmless: a crash leaves the table intact, plus links that a rerun reuses.

    Args:
        table_path: Local path of the ``nwp`` Delta table.
        dry_run: When true, report what would be changed and change nothing.

    Returns:
        The number of partitions changed, or that would be changed when ``dry_run`` is true.
    """
    table = DeltaTable(table_path)
    protocol = table.protocol()
    if protocol.min_writer_version > 2 or protocol.writer_features:
        raise RuntimeError(
            f"The table needs writer version {protocol.min_writer_version} with features"
            f" {protocol.writer_features}, which the hand-written commit does not support."
        )
    actions = active_add_actions(table_path)
    if set(actions) != _active_paths(table):
        raise RuntimeError(
            "The files that the replayed Delta log calls active differ from the files that"
            " deltalake reads. Nothing was changed."
        )
    old = _old_format_actions(actions)
    partitions = sorted({a["partitionValues"]["init_time"] for a in old})
    _LOGGER.info("%s has %d partitions to change (%d files)", table_path, len(partitions), len(old))
    if dry_run or not old:
        return len(partitions)

    before = _flat_actions(table)
    version = table.version()
    commit_file = table_path / "_delta_log" / f"{version + 1:020d}.json"
    new_actions = [_renamed(action) for action in old]
    pairs = [
        (_disk_path(table_path, action["path"]), _disk_path(table_path, new_action["path"]))
        for action, new_action in zip(old, new_actions, strict=True)
    ]
    created: list[Path] = []
    committed = False
    try:
        for source, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.samefile(source):
                continue  # linked by an earlier run that stopped before its commit
            os.link(source, target)
            created.append(target)
        now = int(time.time() * 1000)
        lines: list[dict[str, Any]] = [
            {
                "commitInfo": {
                    "timestamp": now,
                    "operation": "RELOCATE PARTITIONS",
                    "operationParameters": {"reason": "deltalake 1.7 partition value format"},
                    "engineInfo": "rename_nwp_partitions.py",
                }
            },
            *(
                {
                    "remove": {
                        "path": a["path"],
                        "deletionTimestamp": now,
                        "dataChange": False,
                        "extendedFileMetadata": True,
                        "partitionValues": a["partitionValues"],
                        "size": a["size"],
                        **({"stats": a["stats"]} if "stats" in a else {}),
                    }
                }
                for a in old
            ),
            *({"add": {**a, "dataChange": False}} for a in new_actions),
        ]
        _write_commit(commit_file, lines)
        committed = True
    except BaseException:
        if not committed:
            for target in reversed(created):
                target.unlink(missing_ok=True)
                if target.parent.exists() and not any(target.parent.iterdir()):
                    target.parent.rmdir()
        raise

    after_table = DeltaTable(table_path)
    still_old = [p for p in after_table.partitions() if not p["init_time"].endswith("Z")]
    missing = [p for p in _active_paths(after_table) if not _disk_path(table_path, p).exists()]
    unchanged = _flat_actions(after_table) == before
    if after_table.version() < version + 1 or still_old or missing or not unchanged:
        raise RuntimeError(
            f"The commit {commit_file} was written, but its check failed: version"
            f" {after_table.version()}, {len(still_old)} old-format partitions, {len(missing)}"
            " missing files, or changed statistics. The old links were left in place. Investigate"
            " before deleting anything."
        )
    for source, _ in pairs:
        source.unlink(missing_ok=True)
        if source.parent.exists() and not any(source.parent.iterdir()):
            source.parent.rmdir()  # the old directory, now empty
    _LOGGER.info("Wrote %s, and moved %d files to new-format directories", commit_file, len(pairs))
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
