"""Tests for `scripts/lint/check_docs_links.py`'s sibling, `scripts/lint/reflow_python_prose.py`.

The comment half of this script shipped broken and stayed broken silently. `_is_prose_comment` was
handed each line with the line's own `#` still attached, and `NOT_PROSE` matches a bare `#`, so
every comment line matched the "this is not prose" pattern on its own marker and
`_reflow_comment_block` bailed for every block in the repository. Nothing failed, because the
script's only safety net compares the words before and against the words after, and a block that
was never touched trivially passes that comparison. `test_overlong_prose_comment_block_is_reflowed`
is the regression: it fails on the version that shipped and passes on the fixed one.

The tests either side of it pin the lines that must *not* be reflowed, because the fix widens what
the script is willing to rewrite and a too-eager version would destroy the hand-aligned comments
`NOT_PROSE` was written to protect.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

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
    spec = importlib.util.spec_from_file_location("reflow_python_prose", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reflow_python_prose = _load_script()

WIDTH: Final[int] = reflow_python_prose.WIDTH
"""The column the script wraps at, so a fixture can be built just over the limit."""


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
        "# https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/\n"
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
