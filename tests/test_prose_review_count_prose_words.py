"""`count_prose_words.py` gives a sweep's before-and-after figures, which have to be comparable.

A before figure that is not the same kind of measurement as the after figure is worse than no
figure at all, because a pull-request body then reports a sweep that rewrote far more or far less
than it really did. The tests below pin the inputs that used to produce a wrong number or a
traceback: a directory named at `--rev`, a file that does not parse, a non-UTF-8 file, and a path
whose suffix is neither `.py` nor `.md`.
"""

from pathlib import Path

import pytest
from _prose_review_scripts import commit, load, make_repo

count_prose_words = load("count_prose_words")

MODULE = '''"""Two prose words."""

x = 1  # three more prose words
'''


def test_a_directory_at_a_revision_is_not_counted_as_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    make_repo(root)
    (root / "pkg" / "mod.py").write_text(MODULE, encoding="utf-8")
    rev = commit(root)
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", "--rev", rev, "pkg"])
    with pytest.raises(SystemExit) as raised:
        count_prose_words.main()
    assert raised.value.code
    printed = capsys.readouterr().out
    assert "not a file" in printed
    assert "\n      0     0  TOTAL" in printed


def test_a_python_file_that_does_not_parse_is_reported_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "broken.py"
    path.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", str(path)])
    with pytest.raises(SystemExit) as raised:
        count_prose_words.main()
    assert raised.value.code
    assert "unparseable" in capsys.readouterr().out


def test_a_non_utf8_file_is_reported_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "latin.md"
    path.write_bytes(b"caf\xe9 au lait\n")
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", str(path)])
    with pytest.raises(SystemExit) as raised:
        count_prose_words.main()
    assert raised.value.code
    assert "unreadable" in capsys.readouterr().out


def test_a_path_that_is_neither_python_nor_markdown_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "data.json"
    path.write_text('{"one": "two three four"}\n', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", str(path)])
    with pytest.raises(SystemExit) as raised:
        count_prose_words.main()
    assert raised.value.code
    assert "not prose" in capsys.readouterr().out


def test_a_valueless_rev_flag_is_rejected_rather_than_raising_an_index_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", "docs/index.md", "--rev"])
    with pytest.raises(SystemExit) as raised:
        count_prose_words.main()
    assert raised.value.code


def test_help_is_not_read_as_a_file_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", "--help"])
    with pytest.raises(SystemExit) as raised:
        count_prose_words.main()
    assert raised.value.code == 0


def test_a_missing_path_at_a_revision_still_counts_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path / "repo"
    root.mkdir()
    make_repo(root)
    (root / "mod.py").write_text(MODULE, encoding="utf-8")
    rev = commit(root)
    (root / "fresh.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", "--rev", rev, "fresh.py"])
    count_prose_words.main()
    assert "(absent)" in capsys.readouterr().out


def test_a_path_relative_to_a_subdirectory_is_read_at_the_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    make_repo(root)
    (root / "pkg" / "mod.py").write_text(MODULE, encoding="utf-8")
    rev = commit(root)
    monkeypatch.chdir(root / "pkg")
    monkeypatch.setattr("sys.argv", ["count_prose_words.py", "--rev", rev, "mod.py"])
    count_prose_words.main()
    assert "(absent)" not in capsys.readouterr().out


def test_python_prose_counts_docstrings_and_comments_and_no_code(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text(MODULE, encoding="utf-8")
    prose = count_prose_words.prose_of(path=path, source=MODULE)
    assert len(prose.split()) == 7
