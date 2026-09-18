"""`check_comment_wrap.py` finds the half-empty line a splice leaves inside a comment block.

The defect is invisible to every other guard in this repo: `ruff` does not reformat comment text,
and `reflow_python_prose.py` declines a block whose lines already fit. Three commits on one
branch each shipped one. The tests below pin down both halves of the design — which lines count
as stranded, and the rule that the gate is the count rising rather than the count being non-zero,
because prose written before the guard existed holds short lines that are nobody's defect.
"""

from pathlib import Path

import pytest
from _prose_review_scripts import commit, load, make_repo

check_comment_wrap = load("check_comment_wrap")

STRANDED = """\
x = 1

# The highest surface wind speed outside a tornado recorded by the World Meteorological
# Organisation (WMO) in its
# World Weather and Climate Extremes Archive says 113.2 metres per second, which this bound clears.
y = 2
"""
"""A block whose middle line a splice left at 27 characters."""

TIDY = """\
x = 1

# The highest surface wind speed outside a tornado recorded by the World Meteorological
# Organisation (WMO) in its World Weather and Climate Extremes Archive says 113.2 metres per
# second, which this bound clears.
y = 2
"""
"""The same comment, wrapped so only the last line falls short."""


def _short(text: str) -> list[str]:
    """The stranded lines the guard reports in `text`."""
    return check_comment_wrap._short_lines(text)


def test_a_stranded_middle_line_is_reported():
    assert _short(STRANDED) == ["# Organisation (WMO) in its"]


def test_a_tidily_wrapped_block_reports_nothing():
    assert _short(TIDY) == []


def test_the_last_line_of_a_block_may_be_short():
    text = "# A full line of comment prose that runs past the threshold without trouble.\n# End.\n"
    assert _short(text) == []


def test_a_one_line_comment_is_not_a_block():
    assert _short("# Short.\nx = 1\n") == []


def test_a_bare_hash_ends_the_block_so_the_line_before_it_may_be_short():
    text = (
        "# Short.\n#\n"
        "# A second paragraph inside the same run of comment lines, long enough to pass.\n"
    )
    assert _short(text) == []


def test_indentation_counts_towards_the_width():
    text = (
        "        # per-row mask,\n"
        "        # costing about five times the peak memory on a predict-sized frame.\n"
    )
    assert _short(text) == ["# per-row mask,"]


@pytest.mark.parametrize(
    "exempt",
    [
        "# noqa: E501",
        "# fmt: off",
        "# ruff: noqa",
        "# type: ignore[arg-type]",
        "# ------------------------------",
        "# > a quoted line",
    ],
)
def test_one_exempt_line_skips_the_whole_block(exempt: str):
    text = (
        f"# Short.\n{exempt}\n"
        "# A closing line long enough that it would not be reported on its own.\n"
    )
    assert _short(text) == []


def test_a_hand_aligned_block_is_skipped():
    text = (
        "# name     meaning\n# alpha    the first\n"
        "# beta     the second, described at enough length to matter.\n"
    )
    assert _short(text) == []


def test_a_short_line_before_a_url_is_left_alone():
    text = "# The derive-from-root convention and the per-table overrides:\n# https://example.invalid/a/very/long/path/that/cannot/be/wrapped/anywhere\n"
    assert _short(text) == []


def test_a_short_line_before_an_indented_sample_is_left_alone():
    text = "# Run the network-gated test by hand:\n#     uv run pytest --run-network -m network\n"
    assert _short(text) == []


def _repo(tmp_path: Path, before: str) -> tuple[Path, Path, str]:
    """A repository holding one committed module, and the revision it was committed at."""
    root = tmp_path / "repo"
    path = root / "mod.py"
    root.mkdir()
    make_repo(root)
    path.write_text(before, encoding="utf-8")
    return root, path, commit(root)


def test_a_sweep_that_strands_a_line_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    _root, path, _rev = _repo(tmp_path, TIDY)
    path.write_text(STRANDED, encoding="utf-8")
    assert check_comment_wrap._report(before=TIDY, after=STRANDED, path=str(path)) is True
    printed = capsys.readouterr().out
    assert "went from 0 to 1" in printed
    assert "# Organisation (WMO) in its" in printed


def test_a_sweep_that_repairs_a_stranded_line_passes():
    assert check_comment_wrap._report(before=STRANDED, after=TIDY, path="mod.py") is False


def test_a_short_line_the_file_already_had_is_not_a_finding():
    assert check_comment_wrap._report(before=STRANDED, after=STRANDED, path="mod.py") is False


def test_the_command_exits_non_zero_when_a_line_was_stranded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, path, rev = _repo(tmp_path, TIDY)
    path.write_text(STRANDED, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", rev])
    with pytest.raises(SystemExit) as raised:
        check_comment_wrap.main()
    assert raised.value.code != 0


def test_the_command_passes_on_a_tidy_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, path, rev = _repo(tmp_path, STRANDED)
    path.write_text(TIDY, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", rev])
    check_comment_wrap.main()
    assert "PASS - 1 file" in capsys.readouterr().out


def test_a_file_the_branch_created_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, _path, rev = _repo(tmp_path, TIDY)
    (root / "fresh.py").write_text(STRANDED, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", rev, "fresh.py"])
    check_comment_wrap.main()
    assert "PASS - 0 files" in capsys.readouterr().out
