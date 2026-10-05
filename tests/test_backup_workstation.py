"""Tests for `scripts/maintenance/backup_workstation.py`.

The script is run by hand once a week and nobody reads its output closely, so the tests target the
failures that would leave a backup looking fine while it protects nothing: copying onto the system
disk, building a snapshot on an interrupted one, silently storing every file twice, and missing a
source.
"""

import importlib.util
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

_FIRST_RUN: Final[datetime] = datetime(2026, 10, 5, 18, 0, tzinfo=UTC)
_SECOND_RUN: Final[datetime] = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)


def _make_db(path: Path) -> Path:
    """Create a small SQLite database with an MLflow-shaped ``experiments`` table."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE experiments (name TEXT, artifact_location TEXT)")
        connection.execute("INSERT INTO experiments VALUES ('baseline', '/elsewhere/mlruns/1')")
        connection.commit()
    return path


def _make_sources(tmp_path: Path) -> list:
    """Create two source directories holding one file each."""
    data = tmp_path / "data"
    (data / "power_forecasts").mkdir(parents=True)
    (data / "power_forecasts" / "part-0.parquet").write_text("forecasts")
    literature = tmp_path / "literature"
    literature.mkdir()
    (literature / "paper.pdf").write_text("paper")
    return backup_workstation.collect_sources({"data": str(data), "literature": str(literature)})


def test_run_backup_copies_every_source_and_the_database(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")

    snapshot = backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=tmp_path / "backups", now=_FIRST_RUN
    )

    assert snapshot.name == "2026-10-05T180000Z"
    assert (snapshot / "data" / "power_forecasts" / "part-0.parquet").read_text() == "forecasts"
    assert (snapshot / "literature" / "paper.pdf").read_text() == "paper"
    with closing(sqlite3.connect(snapshot / "mlflow.db")) as connection:
        assert connection.execute("SELECT name FROM experiments").fetchall() == [("baseline",)]
    assert backup_workstation.files_changed_since(sources=sources, snapshot=snapshot) == []


def test_second_snapshot_hard_links_unchanged_files(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")
    destination = tmp_path / "backups"

    first = backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=destination, now=_FIRST_RUN
    )
    second = backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=destination, now=_SECOND_RUN
    )

    relative = Path("data") / "power_forecasts" / "part-0.parquet"
    assert (first / relative).stat().st_ino == (second / relative).stat().st_ino


def test_deleted_file_survives_in_the_earlier_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")
    destination = tmp_path / "backups"

    first = backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=destination, now=_FIRST_RUN
    )
    (tmp_path / "literature" / "paper.pdf").unlink()
    second = backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=destination, now=_SECOND_RUN
    )

    assert (first / "literature" / "paper.pdf").exists()
    assert not (second / "literature" / "paper.pdf").exists()


def test_find_previous_snapshot_ignores_partial_and_unrelated_directories(tmp_path: Path) -> None:
    (tmp_path / "2026-10-05T180000Z").mkdir()
    (tmp_path / "2026-10-12T180000Z.partial").mkdir()
    (tmp_path / "notes").mkdir()

    assert backup_workstation.find_previous_snapshot(tmp_path) == tmp_path / "2026-10-05T180000Z"


def test_find_previous_snapshot_returns_none_for_missing_destination(tmp_path: Path) -> None:
    assert backup_workstation.find_previous_snapshot(tmp_path / "absent") is None


def test_run_backup_refuses_to_overwrite_a_snapshot(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")
    destination = tmp_path / "backups"
    backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=destination, now=_FIRST_RUN
    )

    with pytest.raises(FileExistsError):
        backup_workstation.run_backup(
            sources=sources, db_path=db_path, destination=destination, now=_FIRST_RUN
        )


def test_files_changed_since_reports_a_file_written_after_the_backup(tmp_path: Path) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")
    snapshot = backup_workstation.run_backup(
        sources=sources, db_path=db_path, destination=tmp_path / "backups", now=_FIRST_RUN
    )

    (tmp_path / "literature" / "new.pdf").write_text("new")

    changed = backup_workstation.files_changed_since(sources=sources, snapshot=snapshot)
    assert len(changed) == 1
    assert changed[0].startswith("literature: ")
    assert changed[0].endswith("new.pdf")


def test_collect_sources_drops_nested_and_duplicate_roots(tmp_path: Path) -> None:
    (tmp_path / "data" / "production_model").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "data")

    sources = backup_workstation.collect_sources(
        {
            "production_model": str(tmp_path / "data" / "production_model"),
            "data": str(tmp_path / "data"),
            "link": str(tmp_path / "link"),
        }
    )

    assert sources == [backup_workstation.BackupSource(name="data", path=tmp_path / "data")]


def test_collect_sources_skips_a_missing_root(tmp_path: Path) -> None:
    sources = backup_workstation.collect_sources({"mlruns": str(tmp_path / "mlruns")})

    assert sources == []


def test_collect_sources_rejects_a_remote_root() -> None:
    with pytest.raises(ValueError, match="remote URI"):
        backup_workstation.collect_sources({"data": "s3://bucket/data"})


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
    sources = _make_sources(tmp_path)

    with pytest.raises(RuntimeError, match="Is the backup disk mounted"):
        backup_workstation.check_separate_device(
            sources=sources, destination=tmp_path / "unmounted" / "backups"
        )


def test_warn_about_unbacked_artifacts_names_the_experiment(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sources = _make_sources(tmp_path)
    db_path = _make_db(tmp_path / "mlflow.db")

    backup_workstation.warn_about_unbacked_artifacts(db_path=db_path, sources=sources)

    assert "'baseline'" in caplog.text


def test_check_main_checkout_refuses_a_worktree(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=main, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=main,
        check=True,
    )
    worktree = tmp_path / "worktree"
    subprocess.run(["git", "worktree", "add", "-q", str(worktree)], cwd=main, check=True)

    backup_workstation.check_main_checkout(main)
    with pytest.raises(RuntimeError, match="is a git worktree"):
        backup_workstation.check_main_checkout(worktree)
