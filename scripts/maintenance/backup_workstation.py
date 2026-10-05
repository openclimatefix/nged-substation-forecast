"""Back up the workstation's data, MLflow store, Dagster history, and secrets to a second disk.

Each run writes one dated snapshot directory under ``--destination``, such as
``2026-10-05T183000Z/``, holding a full copy of every source. The snapshot copies:

- the three storage roots in ``Settings`` — ``data_path_internal``, ``data_path_delivery``, and
  ``local_artifacts_path`` — which on the workstation are all ``data/``, and so hold the NWP and
  NGED power tables, the forecasts, the metrics, the study outputs under ``data/studies/``, and the
  production model;
- the MLflow artifacts directory ``mlruns/`` at the repository root, which holds the trained models;
- ``literature/`` at the repository root, the git-ignored library of papers and reference documents;
- Dagster's run history: ``dagster_history/`` at the repository root, which the ``base_dir`` in
  ``$DAGSTER_HOME/dagster.yaml`` names, and ``$DAGSTER_HOME`` itself;
- the MLflow database named by ``mlflow_tracking_uri``;
- the credential files in ``SECRET_FILES``, which the docs cannot recreate, under ``secrets/``.

A missing ``Settings`` root or MLflow database stops the run, because a snapshot without them
would look complete while missing the data that matters most. The other directories are skipped
with a warning when absent.

**Every SQLite database is copied through SQLite's own backup routine, never as a plain file.**
Dagster keeps its run history in SQLite databases in write-ahead-log mode, where recent commits sit
in a ``-wal`` file beside the database. A file copy can capture the database and its log at
different moments, and so capture a torn database. The backup routine reads through the log and
always produces a consistent file.

**Unchanged files cost no disk space, because each snapshot hard-links them to the previous
snapshot.** ``rsync --link-dest`` creates a hard link instead of a copy for every file whose size,
modification time, and permissions match the newest earlier snapshot holding that source. Delta
tables never modify a file in place, so after the first snapshot each new one stores only the
files written since the last run. SQLite databases are the exception: each is copied afresh every
run, which costs about 450 MB per run for Dagster's history. Every snapshot is still a complete,
independent copy: deleting an old snapshot never damages a newer one.

**Keeping old snapshots is what protects the backup from a mistaken deletion on the workstation.**
A file deleted from the workstation is missing from the next snapshot but stays in every earlier
one. Each snapshot is also an exact copy of its sources at one moment, so restoring a directory
from a snapshot with ``rsync --delete`` reproduces that directory as it was, with no newer Delta
commit left behind to override the restored version.

A snapshot is written under a ``.partial`` name and renamed only once every copy has succeeded, so
an interrupted run never looks complete and is never used as the base for the next snapshot's hard
links. Delete a leftover ``.partial`` directory by hand.

**The script refuses to run from a git worktree.** ``mlruns/``, ``literature/``,
``dagster_history/``, the secret files, and a relative MLflow database path are all found relative
to the checkout the script runs from, and in a worktree those are missing or different.

**The script also refuses to run if the destination is on the same disk as any source.** If the
backup disk is not mounted, ``/mnt/wd_18tb`` is an empty directory on the system disk, and a backup
written there would fill the system disk while protecting nothing.

``rsync -a`` copies a symlink as a symlink and never follows it, so a link inside a source that
points at the backup disk cannot make the backup copy itself.

``scripts/maintenance/systemd/`` holds the systemd user units that run the script once a day.
Run it while no Dagster run is in progress. A Delta table copied while a Dagster run is writing
to it, or vacuuming it, can be captured with a transaction log that names a data file the copy
missed. The script finishes by listing every entry that changed while it ran, and a long list
means the snapshot is worth repeating. How to restore from a snapshot is on
<https://openclimatefix.github.io/nged-substation-forecast/live_service/backup/>.

Usage::

    uv run python scripts/maintenance/backup_workstation.py
    uv run python scripts/maintenance/backup_workstation.py --destination /mnt/other/backups
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from contracts.settings import PROJECT_ROOT, get_settings

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_DESTINATION: Final[Path] = Path("/mnt/wd_18tb/nged-substation-forecast-backups")
"""Where snapshots go unless ``--destination`` says otherwise: the workstation's 18 TB disk."""

SECRET_FILES: Final[tuple[str, ...]] = (
    ".env",
    ".google_account_for_weathernext3.json",
    "packages/dashboard/.env.s3",
)
"""Credential files, relative to the repository root, copied into each snapshot's ``secrets/``."""

SNAPSHOT_NAME_FORMAT: Final[str] = "%Y-%m-%dT%H%M%SZ"
"""``strftime`` format of a snapshot directory's name, in UTC, so names sort by time."""

PARTIAL_SUFFIX: Final[str] = ".partial"
"""Suffix of a snapshot that is still being written, or whose run was interrupted."""

MLFLOW_DB_NAME: Final[str] = "mlflow.db"
"""File name of the MLflow database inside a snapshot."""

SECRETS_DIR_NAME: Final[str] = "secrets"
"""Directory inside a snapshot holding the files in ``SECRET_FILES``."""

MANIFEST_NAME: Final[str] = "sources.json"
"""File inside a snapshot mapping each copied directory's name to the path it was copied from."""

_SQLITE_PATTERNS: Final[tuple[str, ...]] = ("*.db", "*.db-wal", "*.db-shm", "*.db-journal")
"""Files ``rsync`` skips in a source holding SQLite databases; the backup routine copies them."""

_RSYNC_EXIT_FILES_VANISHED: Final[int] = 24
"""``rsync`` exit code for "some source files vanished before they could be transferred"."""


@dataclass(frozen=True)
class BackupSource:
    """One directory to copy into each snapshot.

    Attributes:
        name: The directory's name inside a snapshot.
        path: The directory on the workstation. ``collect_sources`` resolves it through symlinks.
        required: Whether a missing directory stops the run. When False, a missing directory is
            skipped with a warning.
        holds_sqlite: Whether the directory holds SQLite databases, which are then copied through
            SQLite's backup routine instead of by ``rsync``.
    """

    name: str
    path: Path
    required: bool = True
    holds_sqlite: bool = False


def local_path(name: str, uri: str) -> Path:
    """Return ``uri`` as a local path, refusing a remote URI that ``rsync`` cannot read.

    Args:
        name: The setting ``uri`` came from, named in the error.
        uri: A ``Settings`` storage root.

    Returns:
        ``uri`` as a ``Path``.

    Raises:
        ValueError: If ``uri`` is a remote URI such as ``s3://bucket/data``.
    """
    if "://" in uri:
        raise ValueError(f"{name} is the remote URI {uri!r}; this script copies local paths.")
    return Path(uri)


def collect_sources(candidates: Sequence[BackupSource]) -> list[BackupSource]:
    """Resolve the candidate directories and drop any that another candidate already contains.

    A directory nested inside another candidate, or equal to one, is dropped, because copying the
    outer directory already copies it. On the workstation all three ``Settings`` roots are
    ``data/``, so they collapse to one source.

    Args:
        candidates: The directories to copy. Paths are resolved through symlinks before comparing,
            so ``data/`` and the directory it links to count as equal.

    Returns:
        One ``BackupSource`` per existing, non-nested candidate, with its path resolved, in the
        order ``candidates`` lists them.

    Raises:
        FileNotFoundError: If a required candidate is not a directory.
    """
    existing: list[BackupSource] = []
    for candidate in candidates:
        path = candidate.path.resolve()
        if path.is_dir():
            existing.append(replace(candidate, path=path))
        elif candidate.required:
            raise FileNotFoundError(f"{candidate.name}: {path} is not a directory.")
        else:
            _LOGGER.warning("Skipping %s: %s is not a directory.", candidate.name, path)

    sources: list[BackupSource] = []
    for candidate in existing:
        if any(candidate.path.is_relative_to(source.path) for source in sources):
            continue
        # Drop any already-kept source this one contains, so the outer directory wins whatever
        # order the candidates arrive in.
        sources = [source for source in sources if not source.path.is_relative_to(candidate.path)]
        sources.append(candidate)
    return sources


def build_candidates(
    data_path_internal: str,
    data_path_delivery: str,
    local_artifacts_path: str,
    project_root: Path,
    dagster_home: str | None,
) -> list[BackupSource]:
    """List every directory the backup copies, before ``collect_sources`` drops duplicates.

    Args:
        data_path_internal: ``Settings.data_path_internal``.
        data_path_delivery: ``Settings.data_path_delivery``.
        local_artifacts_path: ``Settings.local_artifacts_path``.
        project_root: The repository root, which holds ``mlruns/``, ``literature/``, and
            ``dagster_history/``.
        dagster_home: The ``DAGSTER_HOME`` environment variable, or ``None`` if it is unset.

    Returns:
        The three required ``Settings`` roots, then the optional directories. The two Dagster
        directories are marked as holding SQLite databases.
    """
    candidates = [
        BackupSource(
            name="data_internal",
            path=local_path(name="data_path_internal", uri=data_path_internal),
        ),
        BackupSource(
            name="data_delivery",
            path=local_path(name="data_path_delivery", uri=data_path_delivery),
        ),
        BackupSource(
            name="local_artifacts",
            path=local_path(name="local_artifacts_path", uri=local_artifacts_path),
        ),
        BackupSource(name="mlruns", path=project_root / "mlruns", required=False),
        BackupSource(name="literature", path=project_root / "literature", required=False),
        BackupSource(
            name="dagster_history",
            path=project_root / "dagster_history",
            required=False,
            holds_sqlite=True,
        ),
    ]
    if dagster_home:
        candidates.append(
            BackupSource(
                name="dagster_home", path=Path(dagster_home), required=False, holds_sqlite=True
            )
        )
    else:
        _LOGGER.warning("Skipping dagster_home: DAGSTER_HOME is not set.")
    return candidates


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


def check_separate_device(paths: Sequence[Path], destination: Path) -> None:
    """Raise unless the destination is on a different filesystem from every path.

    Args:
        paths: The directories about to be copied, and the directory holding the MLflow
            database. The repository root is always among them, so the check holds even when
            every optional directory is absent.
        destination: The snapshot directory's parent. It need not exist yet; its nearest existing
            ancestor is checked instead.

    Raises:
        RuntimeError: If the destination shares a filesystem with a path. On the workstation that
            almost always means the backup disk is not mounted.
    """
    existing = destination
    while not existing.exists():
        existing = existing.parent
    destination_device = existing.stat().st_dev
    for path in paths:
        if path.stat().st_dev == destination_device:
            raise RuntimeError(
                f"{destination} is on the same disk as {path}. Is the backup disk mounted?"
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
        path = Path(location.removeprefix("file://")).resolve()
        if not any(path.is_relative_to(source.path) for source in sources):
            _LOGGER.warning(
                "MLflow experiment %r stores its artifacts at %s, which this backup does not copy.",
                name,
                location,
            )


def list_snapshots(destination: Path) -> list[Path]:
    """Return every complete snapshot under ``destination``, newest first.

    Args:
        destination: The directory holding the snapshots.

    Returns:
        The snapshot directories, sorted newest first. A ``.partial`` directory, or any directory
        whose name is not a timestamp, is left out. Empty if ``destination`` does not exist.
    """
    if not destination.is_dir():
        return []
    snapshots: list[Path] = []
    for child in destination.iterdir():
        try:
            datetime.strptime(child.name, SNAPSHOT_NAME_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            continue
        if child.is_dir():
            snapshots.append(child)
    return sorted(snapshots, reverse=True)


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


def _excludes(source: BackupSource) -> list[str]:
    """Return the ``rsync`` arguments that skip a source's SQLite files, if it holds any.

    The ``--include`` comes first because ``rsync`` applies the first rule that matches: it keeps
    a directory whose name happens to end in ``.db``, which the excludes would otherwise drop.
    """
    if not source.holds_sqlite:
        return []
    return ["--include=*/", *(f"--exclude={pattern}" for pattern in _SQLITE_PATTERNS)]


def _back_up_sqlite(db_path: Path, destination: Path) -> None:
    """Copy an SQLite database with SQLite's backup routine, then check the copy's integrity.

    Args:
        db_path: The live database.
        destination: Where the copy goes. Must not exist yet: SQLite writes into an existing file
            in place, which would also change every snapshot hard-linked to it.

    Raises:
        sqlite3.DatabaseError: If ``db_path`` is not a database, or SQLite's integrity check of
            the copy reports a fault.
    """
    with (
        closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(destination)) as copy,
    ):
        source.backup(copy)
        result = copy.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise sqlite3.DatabaseError(f"Integrity check of {destination} failed: {result}")


def _copy_sqlite_files(source: BackupSource, target: Path) -> None:
    """Copy every SQLite database in ``source`` into ``target`` through the backup routine.

    A file named like a database that SQLite cannot read is copied as a plain file with a
    warning, so one damaged Dagster run record cannot stop the whole backup. A symlink named like
    a database is copied as a symlink, as ``rsync -a`` copies every other symlink.

    Args:
        source: A source whose ``holds_sqlite`` is True.
        target: The source's directory inside the snapshot.
    """
    for db_path in sorted(source.path.rglob("*.db")):
        copy = target / db_path.relative_to(source.path)
        if db_path.is_symlink():
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.symlink_to(db_path.readlink())
            continue
        if not db_path.is_file():
            continue
        copy.parent.mkdir(parents=True, exist_ok=True)
        try:
            _back_up_sqlite(db_path=db_path, destination=copy)
        except sqlite3.DatabaseError as error:
            _LOGGER.warning("Copying %s as a plain file: %s", db_path, error)
            shutil.copy2(db_path, copy)


def _copy_source(source: BackupSource, snapshot: Path, link_dest: Path | None) -> None:
    """Copy one source into ``snapshot``, hard-linking files unchanged since ``link_dest``.

    Args:
        source: The directory to copy.
        snapshot: The snapshot being written.
        link_dest: The same source's directory in the newest earlier snapshot holding it, or
            ``None`` if no earlier snapshot holds it.
    """
    args = ["-aH", *_excludes(source=source)]
    if link_dest is not None:
        args.append(f"--link-dest={link_dest}")
    _LOGGER.info("Copying %s to %s", source.path, snapshot / source.name)
    # A trailing slash on the source copies the directory's contents, not the directory itself.
    _rsync(args=[*args, f"{source.path}/", f"{snapshot / source.name}/"])
    if source.holds_sqlite:
        _copy_sqlite_files(source=source, target=snapshot / source.name)


def _copy_secrets(project_root: Path, secret_files: Sequence[str], target: Path) -> list[str]:
    """Copy each existing secret file into ``target``, keeping its path relative to the root.

    Args:
        project_root: The repository root the paths in ``secret_files`` are relative to.
        secret_files: The credential files to copy.
        target: The snapshot's secrets directory, readable by its owner only.

    Returns:
        The paths in ``secret_files`` that existed and were copied.
    """
    target.mkdir(mode=0o700)
    copied: list[str] = []
    for relative in secret_files:
        path = project_root / relative
        if not path.is_file():
            _LOGGER.warning("Skipping secret file %s: it does not exist.", path)
            continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / relative)
        copied.append(relative)
    return copied


def run_backup(
    sources: Sequence[BackupSource],
    db_path: Path,
    project_root: Path,
    secret_files: Sequence[str],
    destination: Path,
    now: datetime,
) -> Path:
    """Write one complete snapshot of every source, the MLflow database, and the secret files.

    Args:
        sources: The directories to copy.
        db_path: The MLflow SQLite database.
        project_root: The repository root the paths in ``secret_files`` are relative to.
        secret_files: The credential files to copy.
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

    snapshots = list_snapshots(destination)
    name = now.astimezone(UTC).strftime(SNAPSHOT_NAME_FORMAT)
    final = destination / name
    partial = destination / f"{name}{PARTIAL_SUFFIX}"
    if final.exists():
        raise FileExistsError(f"A snapshot named {name} already exists in {destination}.")
    partial.mkdir()  # Raises FileExistsError if this run's partial snapshot already exists.

    for source in sources:
        link_dest = next(
            (snapshot / source.name for snapshot in snapshots if (snapshot / source.name).is_dir()),
            None,
        )
        _copy_source(source=source, snapshot=partial, link_dest=link_dest)
    _back_up_sqlite(db_path=db_path, destination=partial / MLFLOW_DB_NAME)
    copied = _copy_secrets(
        project_root=project_root, secret_files=secret_files, target=partial / SECRETS_DIR_NAME
    )

    manifest = {source.name: str(source.path) for source in sources}
    manifest[MLFLOW_DB_NAME] = str(db_path)
    for relative in copied:
        manifest[f"{SECRETS_DIR_NAME}/{relative}"] = str(project_root / relative)
    (partial / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    partial.rename(final)
    return final


def entries_changed_since(sources: Sequence[BackupSource], snapshot: Path) -> list[str]:
    """List every file or directory in the sources that differs from its copy in ``snapshot``.

    Run straight after a backup, the list holds only the entries created, changed, or deleted
    while the backup ran. An empty list means the snapshot matches the workstation exactly,
    compared on each file's size and modification time. SQLite databases are left out, because
    their copies never match the live file's modification time.

    Args:
        sources: The directories that were copied.
        snapshot: The finished snapshot.

    Returns:
        One ``rsync --itemize-changes`` line per differing entry, prefixed with its source's name.
    """
    changed: list[str] = []
    for source in sources:
        process = _rsync(
            args=[
                "-an",
                # A directory's modification time changes whenever a file inside it does, so
                # comparing it only repeats what the file lines already say.
                "--omit-dir-times",
                "--delete",
                "--itemize-changes",
                *_excludes(source=source),
                f"{source.path}/",
                f"{snapshot / source.name}/",
            ]
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
    candidates = build_candidates(
        data_path_internal=settings.data_path_internal,
        data_path_delivery=settings.data_path_delivery,
        local_artifacts_path=settings.local_artifacts_path,
        project_root=PROJECT_ROOT,
        dagster_home=os.environ.get("DAGSTER_HOME"),
    )
    sources = collect_sources(candidates=candidates)

    db_path = mlflow_db_path(tracking_uri=settings.mlflow_tracking_uri, project_root=PROJECT_ROOT)
    if not db_path.is_file():
        raise FileNotFoundError(f"The MLflow database {db_path} does not exist.")
    check_separate_device(
        paths=[*(source.path for source in sources), db_path.parent, PROJECT_ROOT],
        destination=args.destination,
    )
    warn_about_unbacked_artifacts(db_path=db_path, sources=sources)

    snapshot = run_backup(
        sources=sources,
        db_path=db_path,
        project_root=PROJECT_ROOT,
        secret_files=SECRET_FILES,
        destination=args.destination,
        now=datetime.now(tz=UTC),
    )
    changed = entries_changed_since(sources=sources, snapshot=snapshot)
    if changed:
        _LOGGER.warning(
            "%d entries changed while the backup ran:\n%s", len(changed), "\n".join(changed)
        )
    _LOGGER.info("Backup complete: %s", snapshot)


if __name__ == "__main__":
    main()
