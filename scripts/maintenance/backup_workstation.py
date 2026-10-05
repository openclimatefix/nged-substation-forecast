"""Back up the workstation's data, MLflow store, and production model to a second disk.

Each run writes one dated snapshot directory under ``--destination``, such as
``2026-10-05T183000Z/``, holding a full copy of every source. The snapshot copies:

- the three storage roots in ``Settings`` — ``data_path_internal``, ``data_path_delivery``, and
  ``local_artifacts_path`` — which on the workstation are all ``data/``, and so hold the NWP and
  NGED power tables, the forecasts, the metrics, the study outputs under ``data/studies/``, and the
  production model;
- the MLflow artifacts directory ``mlruns/`` at the repository root, which holds the trained models;
- ``literature/`` at the repository root, the git-ignored library of papers and reference documents;
- the MLflow database named by ``mlflow_tracking_uri``, copied through SQLite's own backup routine.
  A plain file copy of an SQLite database that MLflow is writing to can capture a half-finished
  transaction; the backup routine always produces a consistent file.

**Unchanged files cost no disk space, because each snapshot hard-links them to the previous
snapshot.** ``rsync --link-dest`` creates a hard link instead of a copy for every file whose size,
modification time, and permissions match the previous snapshot. Delta tables never modify a file
in place, so after the first snapshot each new one stores only the files written since the last
run. Every snapshot is still a complete, independent copy: deleting an old snapshot never damages
a newer one.

**Keeping old snapshots is what protects the backup from a mistaken deletion on the workstation.**
A file deleted from the workstation is missing from the next snapshot but stays in every earlier
one. Each snapshot is also an exact copy of its sources at one moment, so restoring a directory
from a snapshot with ``rsync --delete`` reproduces that directory as it was, with no newer Delta
commit left behind to override the restored version.

A snapshot is written under a ``.partial`` name and renamed only once every copy has succeeded, so
an interrupted run never looks complete and is never used as the base for the next snapshot's hard
links. Delete a leftover ``.partial`` directory by hand.

**The script refuses to run from a git worktree.** ``mlruns/``, ``literature/``, ``.env``, and a
relative MLflow database path are all found relative to the checkout the script runs from, and in
a worktree those are missing or different.

**The script also refuses to run if the destination is on the same disk as any source.** If the
backup disk is not mounted, ``/mnt/wd_18tb`` is an empty directory on the system disk, and a backup
written there would fill the system disk while protecting nothing.

``rsync -a`` copies a symlink as a symlink and never follows it, so a link inside a source that
points at the backup disk cannot make the backup copy itself.

Run it while no Dagster run is in progress. A Delta table copied while a Dagster run is writing
to it, or vacuuming it, can be captured with a transaction log that names a data file the copy
missed. The script finishes by listing every file that changed while it ran, and a long list
means the snapshot is worth repeating. How to restore from a snapshot is on
<https://openclimatefix.github.io/nged-substation-forecast/live_service/backup/>.

Usage::

    uv run python scripts/maintenance/backup_workstation.py
    uv run python scripts/maintenance/backup_workstation.py --destination /mnt/other/backups
"""

import argparse
import json
import logging
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from contracts.settings import PROJECT_ROOT, get_settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_DESTINATION: Final[Path] = Path("/mnt/wd_18tb/nged-substation-forecast-backups")
"""Where snapshots go unless ``--destination`` says otherwise: the workstation's 18 TB disk."""

SNAPSHOT_NAME_FORMAT: Final[str] = "%Y-%m-%dT%H%M%SZ"
"""``strftime`` format of a snapshot directory's name, in UTC, so names sort by time."""

PARTIAL_SUFFIX: Final[str] = ".partial"
"""Suffix of a snapshot that is still being written, or whose run was interrupted."""

MLFLOW_DB_NAME: Final[str] = "mlflow.db"
"""File name of the MLflow database inside a snapshot."""

MANIFEST_NAME: Final[str] = "sources.json"
"""File inside a snapshot mapping each copied directory's name to the path it was copied from."""

_RSYNC_EXIT_FILES_VANISHED: Final[int] = 24
"""``rsync`` exit code for "some source files vanished before they could be transferred"."""


@dataclass(frozen=True)
class BackupSource:
    """One directory to copy into each snapshot.

    Attributes:
        name: The directory's name inside a snapshot.
        path: The absolute, symlink-resolved directory on the workstation.
    """

    name: str
    path: Path


def collect_sources(roots: Mapping[str, str]) -> list[BackupSource]:
    """Turn named root directories into the de-duplicated list of directories to copy.

    A root nested inside another root, or equal to one, is dropped, because copying the outer
    root already copies it. On the workstation all three ``Settings`` roots are ``data/``, so they
    collapse to one source.

    Args:
        roots: Each root's name inside a snapshot, mapped to its path. Paths are resolved through
            symlinks before comparing, so ``data/`` and the directory it links to count as equal.

    Returns:
        One ``BackupSource`` per existing, non-nested root, in the order ``roots`` lists them.
        A root that does not exist is skipped with a warning.

    Raises:
        ValueError: If a root is a remote URI such as ``s3://bucket/data``, which ``rsync`` cannot
            read.
    """
    resolved: dict[str, Path] = {}
    for name, root in roots.items():
        if "://" in root:
            raise ValueError(f"{name} is the remote URI {root!r}; this script copies local paths.")
        path = Path(root).resolve()
        if not path.is_dir():
            _LOGGER.warning("Skipping %s: %s is not a directory.", name, path)
            continue
        resolved[name] = path

    sources: list[BackupSource] = []
    for name, path in resolved.items():
        if any(source.path == path or path.is_relative_to(source.path) for source in sources):
            continue
        # Drop any already-kept source this one contains, so the outer directory wins whatever
        # order the roots arrive in.
        sources = [source for source in sources if not source.path.is_relative_to(path)]
        sources.append(BackupSource(name=name, path=path))
    return sources


def mlflow_db_path(tracking_uri: str, project_root: Path) -> Path:
    """Return the local SQLite file an MLflow ``sqlite:///`` tracking URI names.

    MLflow resolves a relative SQLite path against the directory MLflow was started in, which in
    this repository is always the repository root.

    Args:
        tracking_uri: The ``mlflow_tracking_uri`` setting, such as ``sqlite:///mlflow.db``.
        project_root: The repository root, which a relative path is resolved against.

    Returns:
        The database file's absolute path.

    Raises:
        ValueError: If the URI is not a ``sqlite:///`` URI, such as a remote tracking server.
    """
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        raise ValueError(f"mlflow_tracking_uri {tracking_uri!r} is not a local SQLite database.")
    path = Path(tracking_uri.removeprefix(prefix))
    return path if path.is_absolute() else project_root / path


def check_main_checkout(project_root: Path) -> None:
    """Raise unless ``project_root`` is the repository's main checkout rather than a worktree.

    Args:
        project_root: The checkout the script is running from.

    Raises:
        RuntimeError: If ``project_root`` is a git worktree.
    """
    common_dir = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    main_checkout = Path(common_dir).parent
    if main_checkout.resolve() != project_root.resolve():
        raise RuntimeError(
            f"{project_root} is a git worktree. Run this script from {main_checkout} instead."
        )


def check_separate_device(sources: Sequence[BackupSource], destination: Path) -> None:
    """Raise unless the destination is on a different filesystem from every source.

    Args:
        sources: The directories about to be copied.
        destination: The snapshot directory's parent. It need not exist yet; its nearest existing
            ancestor is checked instead.

    Raises:
        RuntimeError: If the destination shares a filesystem with a source. On the workstation
            that almost always means the backup disk is not mounted.
    """
    existing = destination
    while not existing.exists():
        existing = existing.parent
    destination_device = existing.stat().st_dev
    for source in sources:
        if source.path.stat().st_dev == destination_device:
            raise RuntimeError(
                f"{destination} is on the same disk as {source.path}. Is the backup disk mounted?"
            )


def warn_about_unbacked_artifacts(db_path: Path, sources: Sequence[BackupSource]) -> None:
    """Log a warning for each MLflow experiment whose artifacts no source contains.

    MLflow stores each experiment's artifact location as an absolute path. An experiment created
    from a git worktree points inside that worktree, so its trained models are not in the
    repository's ``mlruns/`` and are lost when the worktree is removed.

    Args:
        db_path: The MLflow SQLite database.
        sources: The directories about to be copied.
    """
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT name, artifact_location FROM experiments").fetchall()
    for name, location in rows:
        path = Path(location.removeprefix("file://"))
        if not any(path.is_relative_to(source.path) for source in sources):
            _LOGGER.warning(
                "MLflow experiment %r stores its artifacts at %s, which this backup does not copy.",
                name,
                location,
            )


def find_previous_snapshot(destination: Path) -> Path | None:
    """Return the newest complete snapshot under ``destination``, or ``None`` if there is none.

    Args:
        destination: The directory holding the snapshots.

    Returns:
        The snapshot directory with the latest timestamp name. A ``.partial`` directory, or any
        directory whose name is not a timestamp, is ignored.
    """
    if not destination.is_dir():
        return None
    snapshots: list[Path] = []
    for child in destination.iterdir():
        try:
            datetime.strptime(child.name, SNAPSHOT_NAME_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            continue
        if child.is_dir():
            snapshots.append(child)
    return max(snapshots, default=None)


def _rsync(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run ``rsync`` with ``args``, tolerating only the files-vanished exit code.

    Args:
        args: Every argument after ``rsync`` itself.

    Returns:
        The finished process, with its standard output captured as text.

    Raises:
        subprocess.CalledProcessError: If ``rsync`` exits with any other non-zero code.
    """
    process = subprocess.run(["rsync", *args], capture_output=True, text=True, check=False)
    if process.returncode == _RSYNC_EXIT_FILES_VANISHED:
        _LOGGER.warning("Some files were deleted while rsync was copying them:\n%s", process.stderr)
    elif process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=process.args,
            output=process.stdout,
            stderr=process.stderr,
        )
    return process


def _copy_source(source: BackupSource, snapshot: Path, previous: Path | None) -> None:
    """Copy one source into ``snapshot``, hard-linking files unchanged since ``previous``.

    Args:
        source: The directory to copy.
        snapshot: The snapshot being written.
        previous: The newest complete snapshot, or ``None`` on the first run.
    """
    args = ["-aH"]
    if previous is not None and (previous / source.name).is_dir():
        args.append(f"--link-dest={previous / source.name}")
    _LOGGER.info("Copying %s to %s", source.path, snapshot / source.name)
    # A trailing slash on the source copies the directory's contents, not the directory itself.
    _rsync([*args, f"{source.path}/", f"{snapshot / source.name}/"])


def _back_up_sqlite(db_path: Path, destination: Path) -> None:
    """Copy an SQLite database with SQLite's backup routine, then check the copy's integrity.

    Args:
        db_path: The live database.
        destination: Where the copy goes. Must not exist yet.

    Raises:
        RuntimeError: If SQLite's integrity check of the copy reports a fault.
    """
    with (
        closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(destination)) as copy,
    ):
        source.backup(copy)
        result = copy.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"Integrity check of {destination} failed: {result}")


def run_backup(
    sources: Sequence[BackupSource], db_path: Path, destination: Path, now: datetime
) -> Path:
    """Write one complete snapshot of every source and the MLflow database.

    Args:
        sources: The directories to copy.
        db_path: The MLflow SQLite database.
        destination: The directory holding the snapshots. Created if missing.
        now: The time the snapshot is named after.

    Returns:
        The finished snapshot directory.

    Raises:
        FileExistsError: If a snapshot, finished or partial, already has this run's name.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for leftover in destination.glob(f"*{PARTIAL_SUFFIX}"):
        _LOGGER.warning("%s is left from an interrupted run; delete it by hand.", leftover)

    previous = find_previous_snapshot(destination)
    name = now.astimezone(UTC).strftime(SNAPSHOT_NAME_FORMAT)
    final = destination / name
    partial = destination / f"{name}{PARTIAL_SUFFIX}"
    if final.exists() or partial.exists():
        raise FileExistsError(f"A snapshot named {name} already exists in {destination}.")
    partial.mkdir()

    for source in sources:
        _copy_source(source=source, snapshot=partial, previous=previous)
    _back_up_sqlite(db_path=db_path, destination=partial / MLFLOW_DB_NAME)
    manifest = {source.name: str(source.path) for source in sources}
    manifest[MLFLOW_DB_NAME] = str(db_path)
    (partial / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    partial.rename(final)
    return final


def files_changed_since(sources: Sequence[BackupSource], snapshot: Path) -> list[str]:
    """List every source file that differs from its copy in ``snapshot``.

    Run straight after a backup, the list holds only the files that changed while the backup ran.
    An empty list means the snapshot matches the workstation exactly, compared on size and
    modification time.

    Args:
        sources: The directories that were copied.
        snapshot: The finished snapshot.

    Returns:
        One ``rsync --itemize-changes`` line per differing file, prefixed with its source's name.
    """
    changed: list[str] = []
    for source in sources:
        process = _rsync(
            ["-aHn", "--itemize-changes", f"{source.path}/", f"{snapshot / source.name}/"]
        )
        changed += [f"{source.name}: {line}" for line in process.stdout.splitlines()]
    return changed


def main() -> None:
    """Parse arguments, back up, and report what changed while the backup ran."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Directory holding the snapshots (default: {DEFAULT_DESTINATION}).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    check_main_checkout(PROJECT_ROOT)
    settings = get_settings()
    sources = collect_sources(
        {
            "data_internal": settings.data_path_internal,
            "data_delivery": settings.data_path_delivery,
            "local_artifacts": settings.local_artifacts_path,
            "mlruns": str(PROJECT_ROOT / "mlruns"),
            "literature": str(PROJECT_ROOT / "literature"),
        }
    )
    db_path = mlflow_db_path(tracking_uri=settings.mlflow_tracking_uri, project_root=PROJECT_ROOT)
    check_separate_device(sources=sources, destination=args.destination)
    warn_about_unbacked_artifacts(db_path=db_path, sources=sources)

    snapshot = run_backup(
        sources=sources, db_path=db_path, destination=args.destination, now=datetime.now(tz=UTC)
    )
    changed = files_changed_since(sources=sources, snapshot=snapshot)
    if changed:
        _LOGGER.warning(
            "%d files changed while the backup ran:\n%s", len(changed), "\n".join(changed)
        )
    _LOGGER.info("Backup complete: %s", snapshot)


if __name__ == "__main__":
    main()
