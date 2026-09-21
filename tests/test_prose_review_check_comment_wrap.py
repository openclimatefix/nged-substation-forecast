"""`check_comment_wrap.py` finds the half-empty line a splice leaves inside a comment block.

The defect is invisible to every other guard in this repo: `ruff` does not reformat comment text,
and `reflow_python_prose.py` declines a block whose lines already fit. Three commits on one
branch each shipped a stranded line. The tests below pin down both halves of the design — which
lines count as stranded, and the rule that the gate is the count rising rather than being non-zero,
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
    """A line that straddles the threshold: 53 characters of text, 61 with its indentation.

    Measuring the stripped line would report it; measuring the indented line, as `SHORT` says,
    correctly stays silent. A fixture under the threshold both ways cannot tell the two apart.
    """
    text = (
        "        # a line of comment prose just under fifty-five wide,\n"
        "        # costing about five times the peak memory on a predict-sized frame.\n"
    )
    assert _short(text) == []


def test_an_indented_line_short_both_ways_is_still_reported():
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
        "# ty: ignore[unresolved-attribute]",
        "# -*- coding: utf-8 -*-",
        "#!/usr/bin/env python3",
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
    """26 characters, so the exemption is what keeps it silent rather than the threshold."""
    text = (
        "# The per-table overrides:\n"
        "# https://example.invalid/a/very/long/path/that/cannot/be/wrapped/anywhere\n"
    )
    assert _short(text) == []


def test_a_short_line_before_a_mid_sentence_url_is_still_reported():
    """A line that merely mentions a URL is prose a wrapper would repack, so it hides no strand."""
    text = (
        "# A long enough first line of comment prose to sit above the threshold easily.\n"
        "# a splice\n"
        "# See https://example.com/x for the rest of this long explanation right here.\n"
    )
    assert _short(text) == ["# a splice"]


def test_a_hash_inside_a_string_literal_is_not_a_comment():
    """A regex scan reads this docstring's example as a comment block; `tokenize` does not."""
    text = (
        "def f():\n"
        '    """Doc.\n'
        "\n"
        "        # The highest surface wind speed outside a tornado recorded by the World\n"
        "        # Meteorological Organisation (WMO) in its\n"
        "        # World Weather and Climate Extremes Archive:\n"
        '    """\n'
    )
    assert _short(text) == []


def test_a_file_that_does_not_tokenise_yields_no_blocks():
    assert (
        _short("def f(:\n# short\n# a second line long enough to run past the threshold.\n") == []
    )


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


def test_an_unresolvable_ref_fails_rather_than_passing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Every per-file lookup reads a git failure as "new file", so a bad ref would pass silently."""
    root, _path, _rev = _repo(tmp_path, TIDY)
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", "no-such-ref-at-all", "mod.py"])
    with pytest.raises(SystemExit) as raised:
        check_comment_wrap.main()
    assert raised.value.code != 0


def test_a_path_that_does_not_exist_fails_rather_than_passing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A mistyped path would otherwise be read as a file the branch created, and pass."""
    root, _path, rev = _repo(tmp_path, TIDY)
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", rev, "modd.py"])
    with pytest.raises(SystemExit) as raised:
        check_comment_wrap.main()
    assert raised.value.code != 0


def test_a_path_is_resolved_against_the_working_directory_not_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Run from a subdirectory, `<ref>:<path>` resolves against the root and compares nothing.

    The nested file is deliberately named something the repository root does not also hold. A
    shared name would let the root-relative lookup succeed on the wrong file, and the test would
    then pass whether or not the path is resolved correctly.
    """
    root, _path, _rev = _repo(tmp_path, TIDY)
    nested = root / "pkg"
    nested.mkdir()
    (nested / "only_here.py").write_text(TIDY, encoding="utf-8")
    rev = commit(root)
    (nested / "only_here.py").write_text(STRANDED, encoding="utf-8")
    monkeypatch.chdir(nested)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", rev, "only_here.py"])
    with pytest.raises(SystemExit) as raised:
        check_comment_wrap.main()
    assert raised.value.code != 0
    assert "went from 0 to 1" in capsys.readouterr().out


def test_the_failure_message_prints_the_width_that_was_measured(
    capsys: pytest.CaptureFixture[str],
):
    """`SHORT` measures the indented line, so reporting the stripped width contradicts the gate."""
    tidy = (
        "        # A full line of comment prose that runs past the threshold without trouble.\n"
        "        # and a closing line.\n"
    )
    stranded = (
        "        # A full line of comment prose that runs past the threshold without trouble.\n"
        "        # a splice\n"
        "        # and a closing line that runs past the threshold without any trouble at all.\n"
    )
    assert check_comment_wrap._report(before=tidy, after=stranded, path="mod.py") is True
    assert "18 chars: # a splice" in capsys.readouterr().out


def test_a_file_the_branch_created_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    root, _path, rev = _repo(tmp_path, TIDY)
    (root / "fresh.py").write_text(STRANDED, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["check_comment_wrap.py", rev, "fresh.py"])
    check_comment_wrap.main()
    assert "PASS - 0 files" in capsys.readouterr().out
