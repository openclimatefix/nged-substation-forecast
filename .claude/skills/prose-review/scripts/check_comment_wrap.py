"""Check that a sweep of Python comments left no half-empty line in the middle of a block.

An apply script splices a new sentence into one line of a comment block and then re-wraps that
block. Where the re-wrap does not run — `reflow_python_prose.py` declines a block whose lines all
fit inside the width — the spliced line keeps whatever break the splice gave it, and a line of a
dozen characters sits in the middle of an otherwise full block::

    # The highest surface wind speed outside a tornado recorded by the World Meteorological
    # Organisation (WMO) in its
    # World Weather and Climate Extremes Archive:

Three separate commits on one branch each left a line like that, and neither `ruff`, `ruff format`
nor `reflow_python_prose.py` reports one: ruff does not reformat a comment's text, and the reflow
declines the block before it looks. Only a reader notices, which is why this check exists.

Usage::

    python3 check_comment_wrap.py <git-ref> [path ...]
    python3 check_comment_wrap.py origin/main packages/contracts/src/contracts/ml_schemas.py
    python3 check_comment_wrap.py origin/main          # every .py file changed since that ref

Exits non-zero when a file carries more short mid-block lines than it did at the ref. The gate is
the change rather than the total, because prose written before this check existed holds short lines
that are nobody's defect — a worked example whose lines are deliberately parallel, a comment ending
a paragraph early. Those are stable across a sweep, so they cancel, and a line the sweep itself
stranded does not.

The check covers `#` comments only. A docstring's prose has list items, `Args:` entries and indented
examples whose short lines are all legitimate and all move when the prose around them is rewritten,
so the same counting there reports mostly noise. A stranded line inside a docstring has to be caught
by reading.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

SHORT: Final[int] = 55
"""A mid-block comment line under this width is short enough to have been left by a splice.

Every stranded line found so far measured 35 characters or fewer, and the shortest legitimate one
in this repo measures 41, so the threshold sits above the defects with room to spare. Raising it
costs nothing on an unchanged file, because the gate compares counts rather than reading the total.
"""

COMMENT: Final[re.Pattern[str]] = re.compile(r"^\s*#(\s|$)")
"""A line whose first non-whitespace character opens a comment."""

BLANK: Final[re.Pattern[str]] = re.compile(r"^\s*#\s*$")
"""A bare `#`, which separates two paragraphs inside one block and so ends a run."""

DIRECTIVE: Final[re.Pattern[str]] = re.compile(
    r"^#\s*(noqa|type:|fmt:|ruff:|pylint:|mypy:|pyright:|isort:|flake8:|coverage:|nosec|pragma:"
    r"|[-=*_~]{3,}\s*$|>)"
)
"""A comment whose line breaks are not the wrapper's to choose: a tool directive, a banner, a quote.

One of these anywhere in a block exempts the whole block, because a directive or a banner rule is
what tells the reader where the block's own divisions fall.
"""


def _at_ref(ref: str, path: str) -> str | None:
    """The file's text as of `ref`, or None when it did not exist there."""
    shown = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return shown.stdout if shown.returncode == 0 else None


def _changed_since(ref: str) -> list[str]:
    """Every Python file that differs from `ref`."""
    listed = subprocess.run(
        ["git", "diff", "--name-only", ref, "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in listed.stdout.splitlines() if Path(line).exists()]


def _blocks(lines: list[str]) -> list[list[str]]:
    """Every run of two or more consecutive comment lines, split at each bare `#`."""
    blocks: list[list[str]] = []
    run: list[str] = []
    for line in lines:
        if COMMENT.match(line) and not BLANK.match(line):
            run.append(line)
            continue
        if len(run) > 1:
            blocks.append(run)
        run = []
    if len(run) > 1:
        blocks.append(run)
    return blocks


def _body(line: str) -> str:
    """What the comment says, with the leading indentation and the `#` removed."""
    return line.strip()[1:]


def _short_lines(text: str) -> list[str]:
    """Every comment line under `SHORT` characters that is not the last line of its block.

    A block is skipped whole in three cases, each one a reason its author chose the line breaks:
    a directive or banner line anywhere in it, two consecutive spaces somewhere in its text (the
    mark of a hand-aligned table), and — per line — a following line holding a URL or starting with
    its own indentation, because a wrapper cannot pull either of those up.
    """
    found: list[str] = []
    for block in _blocks(text.splitlines()):
        if any(DIRECTIVE.match(line.strip()) for line in block):
            continue
        if any("  " in _body(line).strip() for line in block):
            continue
        for line, following in itertools.pairwise(block):
            if "://" in following or _body(following).startswith("    "):
                continue
            if len(line.rstrip()) < SHORT:
                found.append(line.strip())
    return found


def _report(*, before: str, after: str, path: str) -> bool:
    """Print every short line the sweep added. Returns True when the count rose."""
    was, now = _short_lines(before), _short_lines(after)
    if len(now) <= len(was):
        return False
    known = set(was)
    print(f"FAIL  {path}: short mid-block comment lines went from {len(was)} to {len(now)}")
    for line in now:
        if line not in known:
            print(f"      {len(line)} chars: {line}")
    return True


def main() -> None:
    """Compare every named Python file against a git ref and report the lines a splice stranded."""
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref, paths = sys.argv[1], sys.argv[2:]
    failed = False
    checked = 0
    for path in paths or _changed_since(ref):
        before = _at_ref(ref, path)
        if before is None:
            continue
        checked += 1
        after = Path(path).read_text(encoding="utf-8")
        failed |= _report(before=before, after=after, path=path)

    if failed:
        sys.exit("\nFAIL - re-wrap each block listed above, or join the line by hand")
    subject = "1 file" if checked == 1 else f"{checked} files"
    print(f"PASS - {subject} stranded no comment line that {ref} had wrapped")


if __name__ == "__main__":
    main()
