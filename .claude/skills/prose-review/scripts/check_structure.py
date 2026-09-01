"""Check that a sentence-level sweep changed the words and nothing else.

A sweep that splits sentences and renames pronouns must leave a page's structure exactly as it
found it: the same links, the same bold and code spans, the same list items, the same headings.
Every defect this catches was written by an apply script and then passed `pymarkdown scan`,
`mkdocs build --strict`, `check_render_loss.py` and `check_information_loss.py` — a citation that
lost the `](url)` half of its link and stopped being a link, a re-wrap that merged two numbered
list items into one, and one that swallowed the closing `---` of a skill file's YAML frontmatter.
Counting is what found all three.

Usage::

    python3 check_structure.py <git-ref> [path ...]
    python3 check_structure.py HEAD docs/roadmap/capacity-estimation.md
    python3 check_structure.py 34a8164a          # every markdown file changed since that ref

Exits non-zero when a count *falls*. A sweep can legitimately add a link or a code span, because
naming the noun a pronoun stood for often means writing `file` or `prose-review` where the
sentence said "it". No sweep can legitimately lose one. Increases are therefore reported and not
gated, and so are over-long lines, which a page carrying unwrapped paragraphs would list every
time.

Use this after a sentence sweep, not after a restructure: a restructure is *meant* to move
headings and list items, and `check_information_loss.py` is the check that fits it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

COUNTS: Final[dict[str, str]] = {
    "markdown links": r"\]\(",
    "link openers": r"(?<!\!)\[",
    "bold markers": r"\*\*",
    "code-span markers": r"`",
    "list items": r"(?m)^\s*(?:[-*+]|\d+[.)])\s+",
    "headings": r"(?m)^#{1,6} ",
    "blank lines": r"(?m)^$",
    "table rows": r"(?m)^\s*\|",
}
"""What must be identical before and after. Each name is what the reader sees when it moves."""

UNBALANCED: Final[dict[str, str]] = {
    "bold markers": r"\*\*",
    "code-span markers": r"`",
}
"""Markers that come in pairs, checked per paragraph so an unclosed span cannot hide in a total."""

LONG_LINE: Final[int] = 105
"""A wrapped line above this width means a re-flow ran at the wrong width, or did not run."""

FRONTMATTER: Final[re.Pattern[str]] = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
"""A skill file's YAML block, whose indented lines count as list items unless it is stripped."""


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
    """Every markdown file that differs from `ref`, tracked or not."""
    listed = subprocess.run(
        ["git", "diff", "--name-only", ref, "--", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in listed.stdout.splitlines() if Path(line).exists()]


def _body(text: str) -> str:
    """The prose, with any YAML frontmatter removed so its indented lines are not counted."""
    return FRONTMATTER.sub("", text, count=1)


def _unbalanced_paragraphs(text: str, pattern: str) -> int:
    """How many blank-line-separated paragraphs carry an odd number of `pattern`."""
    return sum(1 for para in text.split("\n\n") if len(re.findall(pattern, para)) % 2)


def _report_counts(*, raw_before: str, raw_after: str, path: str) -> bool:
    """Print every structural count that moved. Returns True when one fell."""
    before, after = raw_before, raw_after
    failed = False
    before, after = _body(before), _body(after)
    for name, pattern in COUNTS.items():
        was, now = len(re.findall(pattern, before)), len(re.findall(pattern, after))
        if was > now:
            print(f"FAIL  {path}: {name} fell from {was} to {now}")
            failed = True
        elif was < now:
            print(f"NOTE  {path}: {name} rose from {was} to {now}")
    for name, pattern in UNBALANCED.items():
        was = _unbalanced_paragraphs(before, pattern)
        now = _unbalanced_paragraphs(after, pattern)
        if now > was:
            print(f"FAIL  {path}: paragraphs with unpaired {name} went from {was} to {now}")
            failed = True
    if raw_before.endswith("\n") and not raw_after.endswith("\n"):
        print(f"FAIL  {path}: the trailing newline was stripped")
        failed = True
    if raw_before.startswith("---\n") and not FRONTMATTER.match(raw_after):
        print(f"FAIL  {path}: the YAML frontmatter block no longer closes")
        failed = True
    return failed


def _report_long_lines(*, before: str, after: str, path: str) -> None:
    """List lines over `LONG_LINE` characters that the earlier version did not have."""
    known = {line for line in before.splitlines() if len(line) > LONG_LINE}
    new = [line for line in after.splitlines() if len(line) > LONG_LINE and line not in known]
    for line in new:
        print(f"NOTE  {path}: line of {len(line)} characters - {line[:60]}...")


def main() -> None:
    """Compare every named file against a git ref and report what moved structurally."""
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ref, paths = sys.argv[1], sys.argv[2:]
    failed = False
    checked = 0
    for path in paths or _changed_since(ref):
        before = _at_ref(ref, path)
        if before is None:
            continue
        after = Path(path).read_text(encoding="utf-8")
        checked += 1
        failed |= _report_counts(raw_before=before, raw_after=after, path=path)
        _report_long_lines(before=before, after=after, path=path)

    subject = "1 file keeps" if checked == 1 else f"{checked} files keep"
    if failed:
        sys.exit("\nFAIL - the sweep lost structure, not only words")
    print(f"PASS - {subject} every link, span, list item and heading held at {ref}")


if __name__ == "__main__":
    main()
