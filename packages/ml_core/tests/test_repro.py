"""Unit tests for the reproducibility provenance helpers (`ml_core._repro`).

These exercise real ``git`` subprocesses in throwaway repos and real Delta tables in ``tmp_path``
— no MLflow and no network — so they run in the default (non-integration) suite.
"""

import subprocess
from pathlib import Path

import polars as pl
from deltalake import write_deltalake
from ml_core._repro import (
    ABSENT,
    UNKNOWN,
    get_delta_versions,
    get_git_info,
    provenance_tags,
)


def _init_repo(path: Path) -> None:
    """Create a git repo at ``path`` with one committed file and a usable identity."""
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "tracked.txt").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)


def test_git_info_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    info = get_git_info(cwd=tmp_path)
    assert len(info["git_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in info["git_sha"])
    assert info["git_dirty"] == "false"


def test_git_info_dirty_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n")  # modify a tracked file
    assert get_git_info(cwd=tmp_path)["git_dirty"] == "true"


def test_git_info_outside_repo_is_unknown_never_raises(tmp_path: Path) -> None:
    # tmp_path has no .git — the git commands fail and both values degrade to the sentinel.
    info = get_git_info(cwd=tmp_path)
    assert info == {"git_sha": UNKNOWN, "git_dirty": UNKNOWN}


def test_git_info_never_raises_on_bad_cwd(tmp_path: Path) -> None:
    # A nonexistent working directory makes subprocess raise FileNotFoundError *for the cwd* — a
    # different path from the out-of-repo case — which the broad guard must still swallow.
    info = get_git_info(cwd=tmp_path / "does_not_exist")
    assert info == {"git_sha": UNKNOWN, "git_dirty": UNKNOWN}


def test_delta_version_zero_then_increments_on_append(tmp_path: Path) -> None:
    table_path = str(tmp_path / "tbl")
    frame = pl.DataFrame({"a": [1, 2, 3]}).to_arrow()
    write_deltalake(table_path, frame)
    assert get_delta_versions({"tbl": table_path}) == {"delta_version__tbl": "0"}

    write_deltalake(table_path, frame, mode="append")
    assert get_delta_versions({"tbl": table_path}) == {"delta_version__tbl": "1"}


def test_delta_version_missing_table_is_absent(tmp_path: Path) -> None:
    versions = get_delta_versions({"nope": str(tmp_path / "does_not_exist")})
    assert versions == {"delta_version__nope": ABSENT}


def test_provenance_tags_are_stage_prefixed(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    table_path = str(tmp_path / "tbl")
    write_deltalake(table_path, pl.DataFrame({"a": [1]}).to_arrow())

    # get_git_info here uses the real repo of the test run (default cwd); we only assert the keys
    # and the Delta-version value, which are deterministic.
    tags = provenance_tags("train", {"power": table_path})
    assert set(tags) == {
        "train_git_sha",
        "train_git_dirty",
        "train_delta_version__power",
    }
    assert tags["train_delta_version__power"] == "0"


def test_provenance_tags_without_delta_paths_is_git_only() -> None:
    tags = provenance_tags("register")
    assert set(tags) == {"register_git_sha", "register_git_dirty"}
