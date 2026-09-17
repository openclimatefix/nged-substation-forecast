"""`check_prose_only.py` proves a sweep of docstrings and comments changed nothing else.

The guard's one unforgivable outcome is a silent pass, so most of the tests below check that an
input it cannot read produces a failing verdict rather than an `OK` line. Its first silent pass
came from `git show <rev>:<path>` resolving `<path>` against the repository root: run from any
subdirectory every lookup failed, every file was reported `new file`, and the guard exited zero
having compared nothing.
"""

from pathlib import Path

import pytest
from _prose_review_scripts import commit, load, make_repo, run

check_prose_only = load("check_prose_only")

DOCSTRING_MODULE = '''"""Spatial helpers."""


def area() -> int:
    """Return the area."""
    return 1
'''


def _repo_with_module(tmp_path: Path, *, where: str = "pkg/mod.py") -> tuple[Path, Path, str]:
    """A repository holding one committed Python module, and the revision it was committed at."""
    root = tmp_path / "repo"
    path = root / where
    path.parent.mkdir(parents=True)
    make_repo(root)
    path.write_text(DOCSTRING_MODULE, encoding="utf-8")
    return root, path, commit(root)


def test_a_path_relative_to_a_subdirectory_is_read_at_the_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _root, path, rev = _repo_with_module(tmp_path)
    path.write_text(DOCSTRING_MODULE.replace("Spatial helpers", "Spatial helper functions"))
    monkeypatch.chdir(path.parent)
    assert check_prose_only.verdict_for(rev=rev, path=Path("mod.py")) == "prose-only"


def test_a_behaviour_change_is_caught_from_a_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _root, path, rev = _repo_with_module(tmp_path)
    path.write_text(DOCSTRING_MODULE.replace("return 1", "return 2"))
    monkeypatch.chdir(path.parent)
    assert check_prose_only.verdict_for(rev=rev, path=Path("mod.py")) == "BEHAVIOUR CHANGED"


def test_a_file_the_branch_created_is_still_reported_as_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _root, path, rev = _repo_with_module(tmp_path)
    fresh = path.parent / "fresh.py"
    fresh.write_text(DOCSTRING_MODULE, encoding="utf-8")
    monkeypatch.chdir(path.parent)
    assert check_prose_only.verdict_for(rev=rev, path=Path("fresh.py")) == "new file"


def test_a_file_deleted_from_the_working_tree_fails_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, path, rev = _repo_with_module(tmp_path)
    path.unlink()
    monkeypatch.chdir(root)
    verdict = check_prose_only.verdict_for(rev=rev, path=Path("pkg/mod.py"))
    assert verdict in check_prose_only.FAILING_VERDICTS


def test_an_unparseable_file_at_the_revision_blames_the_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "repo"
    path = root / "mod.py"
    root.mkdir()
    make_repo(root)
    path.write_text("def broken(:\n", encoding="utf-8")
    rev = commit(root)
    path.write_text(DOCSTRING_MODULE, encoding="utf-8")
    monkeypatch.chdir(root)
    assert check_prose_only.verdict_for(rev=rev, path=Path("mod.py")) == "unparseable at rev"


def test_an_unparseable_working_tree_file_blames_the_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, path, rev = _repo_with_module(tmp_path, where="mod.py")
    path.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.chdir(root)
    assert check_prose_only.verdict_for(rev=rev, path=Path("mod.py")) == "unparseable"


def test_checking_no_python_file_at_all_fails_rather_than_printing_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, _path, rev = _repo_with_module(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_prose_only.py", rev, "pkg/mod.pyi"])
    with pytest.raises(SystemExit) as raised:
        check_prose_only.main()
    assert raised.value.code
    assert "OK - every Python file changed prose only" not in capsys.readouterr().out


def test_a_revision_that_does_not_exist_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, _path, _rev = _repo_with_module(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_prose_only.py", "no-such-rev", "pkg/mod.py"])
    with pytest.raises(SystemExit) as raised:
        check_prose_only.main()
    assert raised.value.code


def test_a_prose_only_sweep_of_a_whole_repository_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, path, rev = _repo_with_module(tmp_path)
    path.write_text(
        DOCSTRING_MODULE.replace("Return the area.", "Return the area in square metres.")
    )
    run(root, "add", "-A")
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_prose_only.py", rev, "pkg/mod.py"])
    check_prose_only.main()
    assert "OK - every Python file changed prose only" in capsys.readouterr().out
