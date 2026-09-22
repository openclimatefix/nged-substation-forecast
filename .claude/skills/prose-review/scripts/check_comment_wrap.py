"""Check that a sweep of Python comments left no half-empty line in the middle of a block.

An apply script splices a new sentence into one line of a comment block and then re-wraps that
block. Where the re-wrap does not run — `reflow_python_prose.py` declines a block whose lines all
fit inside the width — the spliced line keeps whatever break the splice gave it, and a line of a
dozen characters sits in the middle of an otherwise full block::

    # The highest surface wind speed outside a tornado recorded by the World Meteorological
    # Organisation (WMO) in its
    # World Weather and Climate Extremes Archive:

Three separate commits on one branch each left a line like that. Neither `ruff`, `ruff format`
nor `reflow_python_prose.py` reports a stranded line. Ruff does not reformat a comment's text at
all. The reflow decides whether to run on a block by asking only whether its lines fit the width,
so the reflow never reaches the question of where the breaks fall. Only a reader notices.

Usage::

    python3 check_comment_wrap.py <git-ref> [path ...]
    python3 check_comment_wrap.py origin/main packages/contracts/src/contracts/ml_schemas.py
    python3 check_comment_wrap.py origin/main          # every .py file changed since that ref

Exits non-zero when a file carries more short mid-block lines than it did at the ref. The gate is
the change rather than the total, because prose written before this check existed holds short lines
that are nobody's defect — a worked example whose lines are deliberately parallel, a comment ending
a paragraph early. A short line that predates the sweep appears in both counts and cancels out.
A line the sweep itself stranded appears only in the later count, so it raises the total and the
check fails.

The check covers `#` comments only, and reads them with `tokenize`, so a `#` inside a string
literal is not mistaken for one — this module's own worked example above would otherwise be read
as a comment block. A docstring's prose has list items, `Args:` entries, and indented
examples whose short lines are all legitimate and all move when the prose around them is rewritten,
so the same counting there reports mostly noise. A stranded line inside a docstring has to be caught
by reading.
"""

from __future__ import annotations

import io
import itertools
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Final

SHORT: Final[int] = 55
"""A mid-block comment line under this width is short enough to have been left by a splice.

Every stranded line found so far measured 28 characters or fewer, and this repo holds legitimate
short lines from 8 characters up, so width alone does not separate the two. What separates them is
that a legitimate short line predates the sweep, so it appears in both counts and cancels. Raising
the threshold changes nothing on an unchanged file, for that same reason.
"""

COMMENT: Final[re.Pattern[str]] = re.compile(r"^\s*#(\s|$)")
"""A line whose first non-whitespace character opens a comment."""

BLANK: Final[re.Pattern[str]] = re.compile(r"^\s*#\s*$")
"""A bare `#`, which separates two paragraphs inside one block and so ends a run."""

DIRECTIVE: Final[re.Pattern[str]] = re.compile(
    r"^#!|^#\s*(noqa|type:|ty:|-\*-|fmt:|ruff:|pylint:|mypy:|pyright:|isort:|flake8:|coverage:"
    r"|nosec|pragma:|[-=*_~]{3,}\s*$|>)",
    re.IGNORECASE,
)
"""A comment whose line breaks are not the wrapper's to choose: a tool directive, a banner, a quote.

A directive or a banner rule anywhere in a block exempts the whole block, because the directive or
the rule is what tells the reader where the block's own divisions fall. `ty:` and `-*-` are here
for a second reason: the failure message tells the operator to join a reported line into the line
below it, and joining a `# ty: ignore` into the line below switches the suppression off silently.
"""

URL_ONLY: Final[re.Pattern[str]] = re.compile(r"^#\s*\S+://\S+\s*$")
"""A comment line holding nothing but a URL, which no wrapper can pull up into the line above.

The test is the whole line rather than `"://" in line`, because a line that merely mentions a URL
mid-sentence is ordinary prose a wrapper would repack, and exempting it hides a real strand.
"""


def _verify_revision(ref: str) -> None:
    """Exit unless `ref` names a commit in the repository the current directory sits in.

    Every per-file lookup below reads a git failure as "the file is new", which is a pass. A
    revision nobody can resolve would therefore turn the whole run into a pass, so it is settled
    once, here, before any file is looked at.

    Args:
        ref: The revision named on the command line.
    """
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.exit(f"FAIL - {ref!r} does not name a commit in this repository")


def _at_ref(ref: str, path: str) -> str | None:
    """The file's text as of `ref`, or None when it did not exist there.

    `<ref>:./<path>` is git's cwd-relative form. Without the `./`, git resolves the path against
    the repository root, so run from any subdirectory every lookup fails, every file is reported
    as new, and this guard exits zero having compared nothing.
    """
    shown = subprocess.run(
        ["git", "show", f"{ref}:./{path}"],
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


def _blocks(source: str) -> list[list[str]]:
    """Every run of two or more consecutive whole-line `#` comments, split at each bare `#`.

    The runs come from `tokenize` rather than from a regex over the raw lines, because a `#`
    inside a string literal is not a comment. A regex scan reads a shell example, a markdown
    heading, or an illustrative comment block inside a docstring as a comment block, and this
    module's own docstring carries exactly such an example.

    A run also ends where the indentation changes, so a comment block inside a nested body is
    never joined to the one above it. A file that does not tokenise yields no blocks.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError, SyntaxError:
        return []

    blocks: list[list[str]] = []
    run: list[str] = []
    previous: tuple[int, int] | None = None
    for token in tokens:
        if token.type != tokenize.COMMENT or token.line[: token.start[1]].strip() != "":
            continue
        lineno, indent = token.start
        contiguous = previous == (lineno - 1, indent)
        if not contiguous or BLANK.match(token.line.strip()):
            if len(run) > 1:
                blocks.append(run)
            run = []
        if not BLANK.match(token.line.strip()):
            run.append(token.line.rstrip("\n"))
        previous = (lineno, indent)
    if len(run) > 1:
        blocks.append(run)
    return blocks


def _body(line: str) -> str:
    """What the comment says, with the leading indentation and the `#` removed."""
    return line.strip()[1:]


def _short_pairs(text: str) -> list[tuple[int, str]]:
    """Each stranded line as `(measured width, the line's text)`.

    The width is measured with the indentation included, matching `SHORT`, while the text is
    reported without it. Carrying both is what lets the failure message print the width that
    triggered the finding rather than the width of the text it prints.

    Two of the exemptions skip a whole block, each naming a reason its author chose the line
    breaks: a directive or banner line anywhere in the block, and two consecutive spaces somewhere
    in its text, which is the mark of a hand-aligned table. The third exemption skips a single
    line rather than its block: a line is left alone when the line after it holds nothing but a
    URL, or starts with its own indentation, because a wrapper cannot pull a URL or an indented
    line up.
    """
    found: list[tuple[int, str]] = []
    for block in _blocks(text):
        if any(DIRECTIVE.match(line.strip()) for line in block):
            continue
        if any("  " in _body(line).strip() for line in block):
            continue
        for line, following in itertools.pairwise(block):
            if URL_ONLY.match(following.strip()) or _body(following).startswith("    "):
                continue
            if len(line.rstrip()) < SHORT:
                found.append((len(line.rstrip()), line.strip()))
    return found


def _short_lines(text: str) -> list[str]:
    """Every comment line under `SHORT` characters (55) that is not the last line of its block."""
    return [line for _, line in _short_pairs(text)]


def _report(*, before: str, after: str, path: str) -> bool:
    """Print every short line the sweep added. Returns True when the count rose."""
    was, now = _short_pairs(before), _short_pairs(after)
    if len(now) <= len(was):
        return False
    known = {line for _, line in was}
    print(f"FAIL  {path}: short mid-block comment lines went from {len(was)} to {len(now)}")
    for width, line in now:
        if line not in known:
            print(f"      {width} chars: {line}")
    return True


def main() -> None:
    """Compare every named Python file against a git ref and report the lines a splice stranded.

    Two checks run before any comparison, because this guard reads both an unresolvable ref and a
    path that is not there as "nothing to compare", and would otherwise exit zero having compared
    nothing. A path that exists on disk but not at the ref is a file the branch created, which is
    a genuine skip rather than a mistake.
    """
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref, paths = sys.argv[1], sys.argv[2:]
    _verify_revision(ref)
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        sys.exit(f"FAIL - no such file: {', '.join(missing)}")

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
