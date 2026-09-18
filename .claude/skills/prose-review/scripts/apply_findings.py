"""Apply a sentence sweep's findings to hard-wrapped markdown, refusing the edits that corrupt it.

A sweep is one reading of a text against CLAUDE.md's prose rules, run by a sub-agent — a separate
Claude instance with its own context. The sweep reports each fault as a finding: the sentence it
wants changed, and the sentence it wants instead. Turning a finding into an edit is harder than it
looks, and every guard below exists because its absence
silently damaged a page that then passed `pymarkdown scan`, `mkdocs build --strict` and
`check_information_loss.py`:

- The agent quotes the sentence with the markdown stripped, so `[Gijon et al. (2025)](url) write`
  comes back as `Gijon et al. (2025) write` and a wrap-tolerant literal search misses it. Matching
  runs against a projection of the file that drops the markup but keeps an offset map back to it.
- The replacement is written without markup too, so splicing it in verbatim would delete every
  link and bold marker the sentence carried. Only the runs that actually changed take the
  replacement's characters; the rest keep the raw file's.
- The offset map has to record where each character's markup *ends*, not only where the character
  itself sits. A map of bare character positions resumes the raw text before a closing backtick, so
  a serial comma inserted after `n_h3_cells` is written as `` `n_h3_cells,` `` — the comma inside
  the code span, the backtick count unchanged, and every count of the markers either side of the
  edit unmoved.
- A split whose full stop lands on a closing marker used to delete that marker. The markup counts
  either side of the splice must agree, or the edit is refused rather than written.
- A bolded lead — the sentence CLAUDE.md asks each paragraph to open with, written in `**` — is the
  one span whose full stop belongs *inside* its markers, and the splice pulls it there. Every span
  that does not open its block takes its punctuation outside instead. `prose_splice._lead_marker`
  carries the counts both halves of that rule rest on.
- A quote can match inside a fenced code block, where the words are a command rather than prose.
  Splicing there rewrites the command, and every check downstream passes. A finding whose span
  reaches into a fence is refused. So is a finding landing in a file's YAML frontmatter, where the
  text is configuration rather than prose.
- Re-wrapping the whole file buries the change, so only the unit the splice landed in is
  re-wrapped, at `markdown_wrap.WIDTH`.
- A replacement that spans a different number of lines from the text it replaced invalidates any
  line index taken before the splice. Unit boundaries are recomputed on the spliced text, which is
  what stops two list items merging into one.
- The file's last block carries the trailing newline as an empty final line, which re-flowing the
  unit would swallow. The empty lines are held aside and put back, so an edit landing in the last
  paragraph of a page leaves the file ending the way it started.

Usage::

    python3 apply_findings.py findings.json
    python3 apply_findings.py findings.json --apply
    python3 apply_findings.py findings.json --apply --merge-base 34a8164a

`findings.json` holds a list of objects, each with `file`, `quote` and `replacement`. Nothing is
written without `--apply`; the dry run reports what each finding would do. With `--merge-base`, a
finding whose sentence already exists at that ref is prose the current branch did not write, so it
is skipped and listed rather than applied.

Exits non-zero when any finding could not be applied, so a batch that half-lands is visible.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prose_splice import (
    MARKER,
    block_at,
    fenced_regions,
    markup_intact,
    plain,
    project,
    splice,
)

from scripts.lint.markdown_wrap import WIDTH

UNWRAPPABLE: Final[re.Pattern[str]] = re.compile(r"^(\s*\||\s*#|\s*>|\s{4,}\S|```)")
"""A table row, heading, quote, indented block or fence — none of which may be re-flowed."""

FRONTMATTER: Final[re.Pattern[str]] = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
"""A skill file's YAML block. Its indented lines read as list markers, so it is never re-flowed."""


class Finding(TypedDict):
    """One reported edit: the sentence as the agent quoted it, and what it should say instead."""

    file: str
    quote: str
    replacement: str


StatusType = Literal[
    "applied", "pre-existing", "no match", "ambiguous", "markup refused", "code block"
]


def trim_common_tail(*, quote: str, replacement: str) -> tuple[str, str]:
    """Drop the words the quote and the replacement share at the end.

    A sub-agent routinely stops its quote before a trailing clause the file actually carries, and
    every change it proposes sits in the head. Trimming the shared tail lets the truncated quote
    match.
    """
    quote_words = re.sub(r"\s+", " ", quote).strip().split()
    replacement_words = re.sub(r"\s+", " ", replacement).strip().split()
    shared = 0
    while (
        shared < min(len(quote_words), len(replacement_words)) - 1
        and quote_words[-1 - shared] == replacement_words[-1 - shared]
    ):
        shared += 1
    if shared:
        quote_words, replacement_words = quote_words[:-shared], replacement_words[:-shared]
    return " ".join(quote_words), " ".join(replacement_words)


def locate(*, raw: str, quote: str) -> tuple[tuple[int, int] | None, int]:
    """Find `quote` in the projection of `raw`.

    Args:
        raw: The file text to search, before projection.
        quote: The text to find, matched after both it and `raw` have their runs of whitespace
            collapsed, so a quote that was re-wrapped still matches.

    Returns:
        `((start, end), 1)` in projection coordinates when the quote occurs exactly once, and
        `(None, count)` otherwise, so the caller can tell a missing quote from an ambiguous one.
    """
    projected, _ = project(raw)
    needle = re.sub(r"\s+", " ", quote).strip()
    hits: list[int] = []
    start = projected.find(needle)
    while start != -1:
        hits.append(start)
        start = projected.find(needle, start + 1)
    if len(hits) != 1:
        return None, len(hits)
    return (hits[0], hits[0] + len(needle)), 1


def units(block: str) -> list[tuple[int, int]]:
    """The `(first_line, last_line + 1)` bounds of each wrapping unit in `block`.

    A block is a run of lines with no blank line in it. A bullet list written without blank lines
    between its items is therefore one block, so re-wrapping whole blocks would reflow every
    sibling of the item that changed. A unit is finer: a run of lines starting at a list marker,
    or the whole block where it carries no markers.
    """
    lines = block.split("\n")
    starts = [0, *(index for index in range(1, len(lines)) if MARKER.match(lines[index]))]
    return list(zip(starts, [*starts[1:], len(lines)], strict=True))


def _unit_parts(unit_lines: list[str]) -> tuple[str, str, str]:
    """The unit's first-line prefix, its continuation indent, and its text as one line."""
    marker_match = MARKER.match(unit_lines[0])
    marker = marker_match.group(1) if marker_match else ""
    leading = re.match(r"^\s*", unit_lines[0])
    indent = " " * len(marker) if marker else (leading.group(0) if leading else "")
    body = re.sub(r"\s+", " ", " ".join(unit_lines)).strip()
    if marker:
        body = body[len(marker.strip()) + 1 :]
    return marker or indent, indent, body


def rewrap(unit_lines: list[str], width: int) -> list[str]:
    """Re-flow one unit at `width`, keeping its list marker and its indent."""
    first, indent, body = _unit_parts(unit_lines)
    return textwrap.wrap(
        body,
        width=width,
        initial_indent=first,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _unit_at(text: str, offset: int) -> tuple[int, str, int, int, int]:
    """The block start, block text, unit bounds and line index of the unit holding `offset`."""
    block_start, block = block_at(text, offset)
    line = block.count("\n", 0, offset - block_start)
    low, high = next((lo, hi) for lo, hi in units(block) if lo <= line < hi)
    return block_start, block, low, high, line


def _trailing_blanks(unit_lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a unit into the lines carrying text and the empty lines after them.

    Blocks are separated by blank lines, so the only unit that can end in one is the last of a
    file that ends with a newline: splitting that block on newlines leaves an empty final line.
    Re-flowing the empty line away strips the file's trailing newline. It also leaves the wrapper
    measuring an empty line as part of the paragraph, so the width it computes for the unit is
    the width of a paragraph that is not there.
    """
    end = len(unit_lines)
    while end and not unit_lines[end - 1].strip():
        end -= 1
    return unit_lines[:end], unit_lines[end:]


def _reflow(*, raw: str, spliced: str, offset: int) -> str:
    """Re-wrap only the unit the splice landed in, at `WIDTH`.

    The unit is bounded on `raw` and recomputed on `spliced`, because a replacement spanning a
    different number of lines from the text it replaced moves every line index taken beforehand.
    """
    _, original_block, low, high, _ = _unit_at(raw, offset)
    original_unit, _ = _trailing_blanks(original_block.split("\n")[low:high])
    if len(original_unit) < 2 or any(UNWRAPPABLE.match(line) for line in original_unit):
        return spliced

    block_start, block, low, high, line = _unit_at(spliced, offset)
    lines = block.split("\n")
    head, tail = lines[low:line], lines[line:high]
    tail, blanks = _trailing_blanks(tail)
    reflowed = lines[:low] + head + rewrap(tail, WIDTH) + blanks + lines[high:]
    return spliced[:block_start] + "\n".join(reflowed) + spliced[block_start + len(block) :]


@functools.cache
def _text_at_ref(ref: str, path: str) -> str:
    """The projection of `path` as of `ref`, or an empty string when the file did not exist."""
    shown = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return plain(shown.stdout) if shown.returncode == 0 else ""


def apply_one(*, raw: str, finding: Finding, merge_base: str | None) -> tuple[str, StatusType]:
    """Apply one finding to `raw`, returning the new text and what happened.

    `raw` comes back unchanged for every status other than `applied`, so a refused edit costs
    nothing and the batch carries on.
    """
    quote, replacement = trim_common_tail(
        quote=finding["quote"], replacement=finding["replacement"]
    )
    if merge_base and plain(quote) in _text_at_ref(merge_base, finding["file"]):
        return raw, "pre-existing"

    span, hits = locate(raw=raw, quote=quote)
    if span is None:
        return raw, "ambiguous" if hits > 1 else "no match"

    start, end = span
    spans = project(raw)[1]
    offset, last = spans[start].text, spans[end - 1].right
    frontmatter = FRONTMATTER.match(raw)
    if frontmatter and offset < frontmatter.end():
        return raw, "markup refused"
    if any(low < last and offset < high for low, high in fenced_regions(raw)):
        return raw, "code block"

    spliced = splice(raw=raw, start=start, end=end, replacement=replacement)
    if not markup_intact(before=block_at(raw, offset)[1], after=block_at(spliced, offset)[1]):
        return raw, "markup refused"
    return _reflow(raw=raw, spliced=spliced, offset=offset), "applied"


def main() -> None:
    """Apply every finding in the file named on the command line, reporting each one's status."""
    arguments = sys.argv[1:]
    if not arguments:
        sys.exit(__doc__)
    write = "--apply" in arguments
    merge_base = None
    if "--merge-base" in arguments:
        merge_base = arguments[arguments.index("--merge-base") + 1]
    findings: list[Finding] = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))

    tally: dict[StatusType, int] = {}
    for finding in findings:
        path = Path(finding["file"])
        raw = path.read_text(encoding="utf-8")
        updated, status = apply_one(raw=raw, finding=finding, merge_base=merge_base)
        tally[status] = tally.get(status, 0) + 1
        if status != "applied":
            print(f"{status.upper():>14}  {finding['file']}  {finding['quote'][:70]}")
        if write and status == "applied":
            path.write_text(updated, encoding="utf-8")

    print()
    for status, count in sorted(tally.items()):
        print(f"{count:>5}  {status}")
    if not write:
        print("\ndry run - nothing written; pass --apply to write")
    if tally.keys() - {"applied", "pre-existing"}:
        sys.exit("\nFAIL - some findings could not be applied; fix or hand-edit them")


if __name__ == "__main__":
    main()
