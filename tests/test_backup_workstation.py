"""Tests for `scripts/maintenance/backup_workstation.py`.

The script runs unattended once a day and nobody reads its output closely, so the tests target
the failures that would leave a backup looking fine while it protects nothing: copying onto the
system disk, building a snapshot on an interrupted one, swallowing an `rsync` error, silently
storing every file twice, copying a SQLite database as a torn file, and missing a source.
"""

import importlib.util
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_PATH: Final[Path] = REPO_ROOT / "scripts" / "maintenance" / "backup_workstation.py"
"""The script under test, imported by path because `scripts/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `backup_workstation.py` from its path in `scripts/`."""
    spec = importlib.util.spec_from_file_location("backup_workstation", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backup_workstation = _load_script()
BackupSource = backup_workstation.BackupSource

_FIRST_RUN: Final[datetime] = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)
_SECOND_RUN: Final[datetime] = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)
_THIRD_RUN: Final[datetime] = datetime(2026, 10, 19, 18, 0, tzinfo=UTC)


def _make_db(path: Path) -> Path:
    """Create a small SQLite database with an MLflow-shaped ``experiments`` table."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE experiments (name TEXT, artifact_location TEXT)")
        connection.execute("INSERT INTO experiments VALUES ('baseline', '/elsewhere/mlruns/1')")
        connection.commit()
    return path


def _make_sources(tmp_path: Path) -> list[BackupSource]:
    """Create two source directories holding one file each."""
    data = tmp_path / "data"
    (data / "power_forecasts").mkdir(parents=True)
    (data / "power_forecasts" / "part-0.parquet").write_text("forecasts")
    literature = tmp_path / "literature"
    literature.mkdir()
    (literature / "paper.pdf").write_text("paper")
    return backup_workstation.collect_sources(
        [BackupSource(name="data", path=data), BackupSource(name="literature", path=literature)]
    )


def _backup(
    tmp_path: Path, sources: list[BackupSource], now: datetime, secret_files: tuple[str, ...] = ()
) -> Path:
    """Back up ``sources`` and ``tmp_path / "mlflow.db"`` into ``tmp_path / "backups"``."""
    db_path = tmp_path / "mlflow.db"
    if not db_path.exists():
        _make_db(db_path)
    return backup_workstation.run_backup(
        sources=sources,
        db_path=db_path,
        project_root=tmp_path,
        secret_files=secret_files,
        destination=tmp_path / "backups",
        now=now,
    )


def test_run_backup_copies_every_source_and_the_database(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)

    snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    assert snapshot.name == "2026-10-05T180000Z"
    assert (snapshot / "data" / "power_forecasts" / "part-0.parquet").read_text() == "forecasts"
    assert (snapshot / "literature" / "paper.pdf").read_text() == "paper"
    with closing(sqlite3.connect(snapshot / "mlflow.db")) as connection:
        assert connection.execute("SELECT name FROM experiments").fetchall() == [("baseline",)]
    assert backup_workstation.entries_changed_since(sources=sources, snapshot=snapshot) == []


def test_plain_source_copies_a_database_file_with_rsync(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    (tmp_path / "data" / "cache.db").write_text("study cache")

    snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    assert (snapshot / "data" / "cache.db").read_text() == "study cache"


def test_run_backup_copies_secret_files_into_an_owner_only_directory(tmp_path: Path) -> None:
    (tmp_path / "packages" / "dashboard").mkdir(parents=True)
    (tmp_path / "packages" / "dashboard" / ".env.s3").write_text("KEY=1")
    sources = _make_sources(tmp_path)

    snapshot = _backup(
        tmp_path, sources, _FIRST_RUN, secret_files=("packages/dashboard/.env.s3", ".env")
    )

    secrets = snapshot / "secrets"
    assert (secrets / "packages" / "dashboard" / ".env.s3").read_text() == "KEY=1"
    assert not (secrets / ".env").exists()
    assert secrets.stat().st_mode & 0o777 == 0o700


def test_snapshot_hard_links_unchanged_files_to_the_newest_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    forecasts = tmp_path / "data" / "power_forecasts" / "part-0.parquet"

    _backup(tmp_path, sources, _FIRST_RUN)
    forecasts.write_text("forecasts, rewritten")
    second = _backup(tmp_path, sources, _SECOND_RUN)
    third = _backup(tmp_path, sources, _THIRD_RUN)

    relative = Path("data") / "power_forecasts" / "part-0.parquet"
    assert (second / relative).stat().st_ino == (third / relative).stat().st_ino


def test_snapshot_links_against_the_newest_snapshot_holding_each_source(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    first = _backup(tmp_path, sources, _FIRST_RUN)
    literature_only = [source for source in sources if source.name == "literature"]
    _backup(tmp_path, literature_only, _SECOND_RUN)

    third = _backup(tmp_path, sources, _THIRD_RUN)

    relative = Path("data") / "power_forecasts" / "part-0.parquet"
    assert (first / relative).stat().st_ino == (third / relative).stat().st_ino


def test_run_backup_keeps_hard_links_inside_a_source(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    (tmp_path / "literature" / "copy.pdf").hardlink_to(tmp_path / "literature" / "paper.pdf")

    snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    paper = (snapshot / "literature" / "paper.pdf").stat()
    assert paper.st_ino == (snapshot / "literature" / "copy.pdf").stat().st_ino


def test_deleted_file_survives_in_the_earlier_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)

    first = _backup(tmp_path, sources, _FIRST_RUN)
    (tmp_path / "literature" / "paper.pdf").unlink()
    second = _backup(tmp_path, sources, _SECOND_RUN)

    assert (first / "literature" / "paper.pdf").exists()
    assert not (second / "literature" / "paper.pdf").exists()


def test_rsync_failure_leaves_only_a_partial_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    unreadable = tmp_path / "literature" / "locked.pdf"
    unreadable.write_text("locked")
    unreadable.chmod(0o000)

    try:
        with pytest.raises(subprocess.CalledProcessError):
            _backup(tmp_path, sources, _FIRST_RUN)
    finally:
        unreadable.chmod(0o600)

    assert backup_workstation.list_snapshots(tmp_path / "backups") == []
    assert (tmp_path / "backups" / "2026-10-05T180000Z.partial").is_dir()


def test_sqlite_source_copies_wal_commits_through_the_backup_routine(tmp_path: Path) -> None:
    history = tmp_path / "dagster_history"
    history.mkdir()
    (history / "broken.db").write_text("not a database")
    sources = backup_workstation.collect_sources(
        [BackupSource(name="dagster_history", path=history, holds_sqlite=True)]
    )

    # Keep the live connection open with checkpointing off, so the committed row sits only in
    # runs.db-wal, as it does while Dagster is running.
    with closing(sqlite3.connect(history / "runs.db")) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("PRAGMA wal_autocheckpoint=0")
        live.execute("CREATE TABLE runs (id INTEGER)")
        live.execute("INSERT INTO runs VALUES (1)")
        live.commit()
        snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    copy = snapshot / "dagster_history"
    with closing(sqlite3.connect(copy / "runs.db")) as connection:
        assert connection.execute("SELECT id FROM runs").fetchall() == [(1,)]
    assert not (copy / "runs.db-wal").exists()
    assert (copy / "broken.db").read_text() == "not a database"
    assert backup_workstation.entries_changed_since(sources=sources, snapshot=snapshot) == []


def test_list_snapshots_ignores_partial_and_unrelated_directories(tmp_path: Path) -> None:
    for name in ("2026-10-05T180000Z", "2026-10-19T180000Z", "2026-10-12T180000Z"):
        (tmp_path / name).mkdir()
    (tmp_path / "2026-10-26T180000Z.partial").mkdir()
    (tmp_path / "notes").mkdir()

    assert [path.name for path in backup_workstation.list_snapshots(tmp_path)] == [
        "2026-10-19T180000Z",
        "2026-10-12T180000Z",
        "2026-10-05T180000Z",
    ]


def test_list_snapshots_returns_empty_for_missing_destination(tmp_path: Path) -> None:
    assert backup_workstation.list_snapshots(tmp_path / "absent") == []


def test_run_backup_refuses_to_overwrite_a_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    _backup(tmp_path, sources, _FIRST_RUN)

    with pytest.raises(FileExistsError):
        _backup(tmp_path, sources, _FIRST_RUN)


def test_run_backup_refuses_to_overwrite_a_partial_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    (tmp_path / "backups" / "2026-10-05T180000Z.partial").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        _backup(tmp_path, sources, _FIRST_RUN)


def test_entries_changed_since_reports_created_and_deleted_files(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    (tmp_path / "literature" / "new.pdf").write_text("new")
    (tmp_path / "data" / "power_forecasts" / "part-0.parquet").unlink()

    changed = backup_workstation.entries_changed_since(sources=sources, snapshot=snapshot)
    assert any(line.startswith("literature: ") and line.endswith("new.pdf") for line in changed)
    assert any(line.startswith("data: *deleting") for line in changed)


def test_entries_changed_since_ignores_a_directory_timestamp(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    # A timestamp well away from the backup's, because rsync compares times to the second.
    os.utime(tmp_path / "literature", times=(0, 0))

    assert backup_workstation.entries_changed_since(sources=sources, snapshot=snapshot) == []


def test_collect_sources_drops_nested_and_duplicate_roots(tmp_path: Path) -> None:
    (tmp_path / "data" / "production_model").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "data")

    sources = backup_workstation.collect_sources(
        [
            BackupSource(name="production_model", path=tmp_path / "data" / "production_model"),
            BackupSource(name="data", path=tmp_path / "data"),
            BackupSource(name="link", path=tmp_path / "link"),
        ]
    )

    assert sources == [BackupSource(name="data", path=tmp_path / "data")]


def test_collect_sources_skips_a_missing_optional_root(tmp_path: Path) -> None:
    sources = backup_workstation.collect_sources(
        [BackupSource(name="mlruns", path=tmp_path / "mlruns", required=False)]
    )

    assert sources == []


def test_collect_sources_refuses_a_missing_required_root(tmp_path: Path) -> None:
    (tmp_path / "data").symlink_to(tmp_path / "unmounted")

    with pytest.raises(FileNotFoundError, match="data_internal"):
        backup_workstation.collect_sources(
            [BackupSource(name="data_internal", path=tmp_path / "data")]
        )


def test_local_path_rejects_a_remote_root() -> None:
    with pytest.raises(ValueError, match="remote URI"):
        backup_workstation.local_path(name="data_path_internal", uri="s3://bucket/data")


def test_mlflow_db_path_resolves_a_relative_path_against_the_project_root(tmp_path: Path) -> None:
    db_path = backup_workstation.mlflow_db_path(
        tracking_uri="sqlite:///mlflow.db", project_root=tmp_path
    )

    assert db_path == tmp_path / "mlflow.db"


def test_mlflow_db_path_keeps_an_absolute_path(tmp_path: Path) -> None:
    db_path = backup_workstation.mlflow_db_path(
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db", project_root=Path("/elsewhere")
    )

    assert db_path == tmp_path / "mlflow.db"


def test_mlflow_db_path_rejects_a_tracking_server() -> None:
    with pytest.raises(ValueError, match="not a local SQLite"):
        backup_workstation.mlflow_db_path(
            tracking_uri="http://localhost:5000", project_root=Path("/repo")
        )


def test_check_separate_device_refuses_the_same_disk(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Is the backup disk mounted"):
        backup_workstation.check_separate_device(
            paths=[tmp_path], destination=tmp_path / "unmounted" / "backups"
        )


def test_warn_about_unbacked_artifacts_names_only_the_experiment_outside_the_sources(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            "INSERT INTO experiments VALUES ('kept', ?)", (f"file://{tmp_path}/data/mlruns/2",)
        )
        connection.commit()

    backup_workstation.warn_about_unbacked_artifacts(db_path=db_path, sources=sources)

    assert "'baseline'" in caplog.text
    assert "'kept'" not in caplog.text


def test_check_main_checkout_refuses_a_worktree(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=main, check=True)
    commit = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty"]
    subprocess.run([*commit, "-m", "init"], cwd=main, check=True)
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "-q", str(worktree)], cwd=main, check=True)

    backup_workstation.check_main_checkout(main)
    with pytest.raises(RuntimeError, match="is a git worktree"):
        backup_workstation.check_main_checkout(worktree)


def test_build_candidates_marks_only_the_dagster_directories_as_sqlite(tmp_path: Path) -> None:
    candidates = backup_workstation.build_candidates(
        data_path_internal="/data",
        data_path_delivery="/data",
        local_artifacts_path="/data",
        project_root=tmp_path,
        dagster_home="/home/user/dagster_home",
    )

    by_name = {candidate.name: candidate for candidate in candidates}
    assert {name for name, c in by_name.items() if c.holds_sqlite} == {
        "dagster_history",
        "dagster_home",
    }
    assert {name for name, c in by_name.items() if c.required} == {
        "data_internal",
        "data_delivery",
        "local_artifacts",
    }
    assert by_name["dagster_history"].path == tmp_path / "dagster_history"
    assert by_name["dagster_home"].path == Path("/home/user/dagster_home")


def test_build_candidates_skips_dagster_home_when_unset(tmp_path: Path) -> None:
    candidates = backup_workstation.build_candidates(
        data_path_internal="/data",
        data_path_delivery="/data",
        local_artifacts_path="/data",
        project_root=tmp_path,
        dagster_home=None,
    )

    assert "dagster_home" not in {candidate.name for candidate in candidates}


def test_check_separate_device_checks_every_path_not_only_the_first(tmp_path: Path) -> None:
    # /proc is always its own filesystem, so only the second path shares the destination's disk.
    with pytest.raises(RuntimeError, match="Is the backup disk mounted"):
        backup_workstation.check_separate_device(
            paths=[Path("/proc"), tmp_path], destination=tmp_path / "backups"
        )


def test_sqlite_source_keeps_a_directory_and_a_symlink_named_like_a_database(
    tmp_path: Path,
) -> None:
    history = tmp_path / "dagster_history"
    (history / "logs.db").mkdir(parents=True)
    (history / "logs.db" / "run.log").write_text("log")
    (history / "latest.db").symlink_to("logs.db")
    sources = backup_workstation.collect_sources(
        candidates=[BackupSource(name="dagster_history", path=history, holds_sqlite=True)]
    )

    snapshot = _backup(tmp_path, sources, _FIRST_RUN)

    copy = snapshot / "dagster_history"
    assert (copy / "logs.db" / "run.log").read_text() == "log"
    assert (copy / "latest.db").readlink() == Path("logs.db")
