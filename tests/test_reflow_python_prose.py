"""Tests for `scripts/lint/reflow_python_prose.py`.

The comment half of this script shipped broken and stayed broken silently: `_is_prose_comment`
was handed each line with the line's own `#` still attached, and `NOT_PROSE` matches a bare `#`,
so every comment line matched the "this is not prose" pattern on its own marker and
`_reflow_comment_block` bailed for every block in the repository.
`test_overlong_prose_comment_block_is_reflowed` is the regression: it fails on the version that
shipped and passes on the fixed one.

Every other comment test pins a block the script must leave exactly as written — some refused by
`_is_prose_comment`, some never collected as a block in the first place — because the fix widens
what the script is willing to rewrite. Each of those fixtures carries a line past `WIDTH`, so
what refuses the block is the block's own content rather than the width gate
`_reflow_comment_block` applies before anything else.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "scripts" / "lint" / "reflow_python_prose.py"
"""The script under test, imported by path because `scripts/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `reflow_python_prose.py` from its path in `scripts/lint/`.

    Its own directory goes on `sys.path` first, because the script does a bare `import
    markdown_wrap` that only resolves when run directly, where Python puts the script's directory
    there itself.
    """
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("reflow_python_prose", SCRIPT_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))
    return module


reflow_python_prose = _load_script()

WIDTH: Final[int] = reflow_python_prose.WIDTH
"""The column the script wraps at, so a fixture can be built just over the limit."""

OVERLONG_PROSE_LINE: Final[str] = (
    "# This ordinary prose comment is long enough on its own that the block around it runs past the"
    " hundred character limit this repository wraps at.\n"
)
"""One overlong prose line, so a refusal fixture is not refused by the width gate instead."""


def test_overlong_prose_comment_block_is_reflowed() -> None:
    """A two-line prose comment running past the limit gets repacked.

    The regression for the leading-`#` bug: the shipped version returned this source unchanged.
    """
    source = (
        "# This explanatory comment is deliberately long enough that its first physical line runs"
        " well past the hundred character limit the repo wraps at.\n"
        "# A second line follows it so the run counts as a block.\n"
        "x = 1\n"
    )

    result = reflow_python_prose.reflow_python_prose(source)

    assert result != source, "the comment block was not reflowed at all"
    comment_lines = [line for line in result.splitlines() if line.startswith("#")]
    assert all(len(line) <= WIDTH for line in comment_lines), comment_lines
    assert "x = 1" in result


def test_comment_block_inside_the_limit_is_left_alone() -> None:
    """Two short comment lines stay on two lines, because neither of them overflows.

    The words would fit on one line, and a greedy repack would put them there. The script
    deliberately does not: its job is wrapping prose that runs past `WIDTH`, and where nothing
    runs past `WIDTH` the line break belongs to whoever wrote the comment.
    """
    source = "# A short comment line.\n# And a second short line under it.\nx = 1\n"

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_reflowing_is_idempotent() -> None:
    """Running the script over its own output changes nothing the second time."""
    source = (
        "# This explanatory comment is deliberately long enough that its first physical line runs"
        " well past the hundred character limit the repo wraps at.\n"
        "# A second line follows it so the run counts as a block.\n"
        "x = 1\n"
    )

    once = reflow_python_prose.reflow_python_prose(source)

    assert reflow_python_prose.reflow_python_prose(once) == once


def test_embedded_noqa_directive_is_not_reflowed() -> None:
    """A `# noqa` sitting after padding, not at the line's start, keeps its whole block untouched.

    This is `storage.py`'s aligned-key comment. Reflowing the block would move the directive off
    the line it suppresses, so the suppression would stop applying and the line would start
    failing `E501` instead.
    """
    source = (
        "# The sample key below is the shape the ingest writes, and this sentence is long enough"
        " that it runs past the limit.\n"
        "# key/2024-01-01.json  ->  date  # noqa: E501\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_hand_aligned_diagram_is_not_reflowed() -> None:
    """An interior run of two or more spaces is alignment, and collapsing it kills the diagram."""
    source = (
        "# The sample key below is the shape the ingest writes, and this sentence is long enough"
        " that it runs past the limit.\n"
        "# key/2024-01-01.json  ->  date\n"
        "#                 ^^^^     ext\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_directive_comment_at_the_start_of_a_line_is_not_reflowed() -> None:
    """A `# noqa` anchored at the start of its own text keeps its block untouched."""
    source = (
        "# noqa: E501\n"
        "# This second line is deliberately long enough that the block as a whole runs past the"
        " hundred character limit.\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_url_only_comment_is_not_reflowed() -> None:
    """A comment holding nothing but a URL keeps its block whole, because a split URL is dead."""
    source = (
        # The `\n` is a separate piece so that `check_docs_links.py`, which reads this file as
        # text, does not read the escape as part of the URL and report the page as missing.
        "# https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/"
        "\n"
        "# This second line is deliberately long enough that the block as a whole runs past the"
        " hundred character limit.\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_a_lone_comment_line_is_not_a_block() -> None:
    """One overlong comment on its own is left alone, because a block needs two or more lines."""
    source = (
        "# A single overlong comment line that runs well past the hundred character limit but has"
        " no sibling under it.\n"
        "x = 1\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_trailing_inline_comment_is_not_reflowed() -> None:
    """A comment sharing its line with code is never part of a block, so it survives untouched."""
    source = (
        "x = 1  # This trailing comment is deliberately long enough to run past the hundred"
        " character limit the repo wraps at.\n"
        "y = 2  # And so is this one, on the physical line directly under it, to make a run.\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_blockquote_comment_is_not_reflowed() -> None:
    """A `>` line inside a comment keeps its block whole, rather than aborting the run.

    Wrapping a blockquote adds a `>` marker to each line it wraps onto, and `_flatten` strips
    only the `#` marker, so the added markers read as new words and the round-trip assertion in
    `reflow_python_prose` fires — which would end a batch run part-way, after the files already
    processed had been written.
    """
    source = OVERLONG_PROSE_LINE + (
        "# > In production, never raise because an input is absent or stale: degrade, widen the"
        " uncertainty bands, and record the degradation on the row.\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_doctest_comment_is_not_reflowed() -> None:
    """A `>>>` example inside a comment survives, because its line breaks are part of it."""
    source = OVERLONG_PROSE_LINE + "# >>> reflow_python_prose(source)\n"

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_formatter_directive_is_not_merged_into_the_prose_beside_it() -> None:
    """`# fmt: off` keeps its own line, because text merged onto it stops ruff reading it.

    A merged directive is the quietest failure this script can cause: the formatter goes on to
    reformat the block the directive was switched off to protect, and reports nothing.
    """
    source = "# fmt: off\n" + OVERLONG_PROSE_LINE

    assert reflow_python_prose.reflow_python_prose(source) == source


@pytest.mark.parametrize(
    "directive",
    [
        "# ruff: noqa: E501",
        "# pylint: disable=invalid-name",
        "# mypy: ignore-errors",
        "# pyright: strict",
        "# isort: skip_file",
        "# flake8: noqa",
        "# coverage: ignore",
        "# nosec",
        "# pragma: no cover",
    ],
)
def test_a_tool_directive_keeps_its_block_untouched(directive: str) -> None:
    """Each tool's own directive line stays on its own line, so that tool still reads it."""
    source = f"{directive}\n{OVERLONG_PROSE_LINE}"

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_prose_opening_with_a_directive_word_is_still_reflowed() -> None:
    """A sentence opening with a word a directive also opens with is prose, and is wrapped."""
    source = (
        "# Typescript is not a language this repository uses, and this sentence is long enough to"
        " push the block past the hundred character limit.\n"
        "# A second prose line follows it.\n"
    )

    result = reflow_python_prose.reflow_python_prose(source)

    assert result != source
    assert all(len(line) <= WIDTH for line in result.splitlines())


def test_banner_rule_is_not_merged_into_its_title() -> None:
    """A rule of dashes above a section title keeps its own line, so the banner survives."""
    source = (
        "# ---------------------------------------------------------------------------\n"
        "# In-process integration tests (file-based MLflow)\n" + OVERLONG_PROSE_LINE
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_double_hash_comment_block_is_not_reflowed() -> None:
    """A `##` block is left alone, because its second `#` is a marker rather than a word.

    Pins that `_is_prose_comment` drops exactly one leading `#` before testing `NOT_PROSE`:
    dropping two would read a `##` line as prose and repack the block into `# # ...`.
    """
    source = (
        "## This commented-out line is long enough that the block runs past the hundred character"
        " limit the repository wraps at.\n"
        "## A second line follows it.\n"
    )

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_embedded_hash_without_padding_is_not_reflowed() -> None:
    """An embedded `#` refuses the block even where no hand alignment accompanies it.

    `NOT_PROSE` tests for an embedded `#` and for an interior run of spaces separately, so a
    fixture carrying both would be refused whichever of the two the code actually applied.
    """
    source = OVERLONG_PROSE_LINE + "# key/2024-01-01.json -> date # noqa: E501\n"

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_commented_out_code_is_not_reflowed() -> None:
    """A commented-out function keeps its line breaks, which are what make it readable."""
    source = OVERLONG_PROSE_LINE + "# def f(x: int) -> int:\n#     return x + 1\n"

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_tab_inside_a_comment_block_is_not_reflowed() -> None:
    """A tab is alignment or indentation this script cannot reproduce, so its block is skipped."""
    source = OVERLONG_PROSE_LINE + "#\tTabbed alignment column\n"

    assert reflow_python_prose.reflow_python_prose(source) == source


def test_docstring_reflow_still_works() -> None:
    """The docstring half of the script is unaffected by the comment fix."""
    source = (
        "def f() -> None:\n"
        '    """A summary line.\n'
        "\n"
        "    This body paragraph is deliberately long enough that it runs well past the hundred"
        " character limit the repository wraps its prose at.\n"
        '    """\n'
    )

    result = reflow_python_prose.reflow_python_prose(source)

    assert result != source
    assert all(len(line) <= WIDTH for line in result.splitlines())
