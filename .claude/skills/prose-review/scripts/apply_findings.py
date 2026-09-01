"""Apply a sentence sweep's findings to hard-wrapped markdown, refusing the edits that corrupt it.

A sub-agent reports a finding as the sentence it wants changed and the sentence it wants instead.
Turning that into an edit is harder than it looks, and every guard below exists because its absence
silently damaged a page that then passed `pymarkdown scan`, `mkdocs build --strict` and
`check_information_loss.py`:

- The agent quotes the sentence with the markdown stripped, so `[Gijon et al. (2025)](url) write`
  comes back as `Gijon et al. (2025) write` and a wrap-tolerant literal search misses it. Matching
  runs against a projection of the file that drops the markup but keeps an offset map back to it.
- The replacement is written without markup too, so splicing it in verbatim would delete every
  link and bold marker the sentence carried. Only the runs that actually changed take the
  replacement's characters; the rest keep the raw file's.
- A split whose full stop lands on a closing `**` or `](url)` deletes that marker. The markup
  counts either side of the splice must agree, or the edit is refused rather than written.
- Re-wrapping the whole file buries the change, and re-wrapping at the wrong width reflows every
  line it touches. The width is solved from the lines being replaced, per unit.
- A replacement that spans a different number of lines from the text it replaced invalidates any
  line index taken before the splice. Unit boundaries are recomputed on the spliced text, which is
  what stops two list items merging into one.

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

import difflib
import functools
import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, TypedDict

LINK: Final[re.Pattern[str]] = re.compile(r"\[([^\]\[]*)\]\((?:[^()]|\([^()]*\))*\)")
"""A markdown link, whose label survives the projection and whose target does not."""

MARKER: Final[re.Pattern[str]] = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")
"""The bullet or number that starts a list item, and therefore starts a new wrapping unit."""

UNWRAPPABLE: Final[re.Pattern[str]] = re.compile(r"^(\s*\||\s*#|\s*>|\s{4,}\S|```)")
"""A table row, heading, quote, indented block or fence — none of which may be re-flowed."""

MARKUP: Final[tuple[str, ...]] = ("**", "`", "[", "](")
"""The markers whose count must be identical before and after a splice."""

WIDTH_RANGE: Final[tuple[int, int]] = (88, 104)
"""The wrap widths to try when solving a unit's width. This repo's pages sit between 94 and 100."""

FALLBACK_WIDTH: Final[int] = 99
"""Used only for a file whose every unit is a single line, so no width can be solved from it."""

FRONTMATTER: Final[re.Pattern[str]] = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
"""A skill file's YAML block. Its indented lines read as list markers, so it is never re-flowed."""


class Finding(TypedDict):
    """One reported edit: the sentence as the agent quoted it, and what it should say instead."""

    file: str
    quote: str
    replacement: str


StatusType = Literal["applied", "pre-existing", "no match", "ambiguous", "markup refused"]

PushType = Callable[[str, int], None]
"""Appends one character to the projection, recording the raw offset it came from."""


@functools.lru_cache(maxsize=64)
def project(raw: str) -> tuple[str, tuple[int, ...]]:
    """Strip the markup from `raw`, returning the plain text and where each character came from.

    Args:
        raw: The file's text, hard wrapping and markdown included.

    Returns:
        `(plain, offsets)` where `plain[i]` is the character at `raw[offsets[i]]`, every run of
        whitespace has become one space, and the markup a sub-agent's quote will not carry — link
        targets, emphasis markers, code fences — has been dropped.
    """
    plain: list[str] = []
    offsets: list[int] = []

    def push(char: str, at: int) -> None:
        if char.isspace():
            if plain and plain[-1] != " ":
                plain.append(" ")
                offsets.append(at)
        else:
            plain.append(char)
            offsets.append(at)

    index, length = 0, len(raw)
    while index < length:
        link = LINK.match(raw, index)
        if link:
            # The label is hard-wrapped too, so it needs the same whitespace collapsing.
            for position, char in enumerate(link.group(1)):
                push(char, link.start(1) + position)
            index = link.end()
            continue
        if raw[index] in "*`":
            index += 1
            continue
        if raw[index] == "_":
            index = _push_underscores(raw=raw, index=index, push=push)
            continue
        push(raw[index], index)
        index += 1
    return "".join(plain), tuple(offsets)


def _push_underscores(*, raw: str, index: int, push: PushType) -> int:
    """Drop a run of underscores that marks emphasis; keep one that sits inside an identifier.

    `mid_2025_to_mid_2026` and `mae__all` are names, and dropping their underscores makes the
    projection unmatchable against a quote that spells them correctly.
    """
    end = index
    while end < len(raw) and raw[end] == "_":
        end += 1
    inside_word = index > 0 and raw[index - 1].isalnum() and end < len(raw) and raw[end].isalnum()
    if not inside_word:
        return end
    for position in range(index, end):
        push(raw[position], position)
    return end


def plain(text: str) -> str:
    """The projection of a standalone string, for comparing a quote that carries markdown."""
    return project(text)[0].strip()


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


def splice(*, raw: str, start: int, end: int, replacement: str) -> str:
    """Rewrite the projection span `[start, end)` of `raw` as `replacement`.

    Unchanged runs keep whatever markup the raw text carries there, so the links and bold markers
    the agent's replacement omits survive. Only the runs that genuinely differ take the
    replacement's own characters.
    """
    projected, offsets = project(raw)
    old = projected[start:end]
    new = re.sub(r"\s+", " ", replacement).strip()
    out: list[str] = []
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, old_lo, old_hi, new_lo, new_hi in matcher.get_opcodes():
        if tag == "equal":
            out.append(raw[offsets[start + old_lo] : offsets[start + old_hi - 1] + 1])
        elif tag in ("replace", "insert"):
            out.append(new[new_lo:new_hi])
    return raw[: offsets[start]] + "".join(out) + raw[offsets[end - 1] + 1 :]


def markup_intact(*, before: str, after: str) -> bool:
    """True when the splice left every link, bold span and code span in `before` whole.

    A split whose full stop lands inside a link label or between two bold markers breaks the markup
    silently: the page still lints, still builds, and the link simply stops being a link.
    """
    if after.count("**") % 2 or after.count("`") % 2:
        return False
    return all(before.count(marker) == after.count(marker) for marker in MARKUP)


def units(block: str) -> list[tuple[int, int]]:
    """The `(first_line, last_line + 1)` bounds of each wrapping unit in `block`.

    A bullet list written without blank lines between its items is one block, so re-wrapping whole
    blocks would reflow every sibling of the item that changed. A unit is finer: a run of lines
    starting at a list marker, or the whole block where it carries no markers.
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


def solve_width(unit_lines: list[str]) -> int | None:
    """The wrap width that reproduces `unit_lines` exactly, or None when no width does.

    Solving beats assuming: this repo's pages are wrapped anywhere between 94 and 100 characters,
    and applying the wrong width re-flows every line of the unit instead of the lines that changed.
    """
    if len(unit_lines) < 2 or any(UNWRAPPABLE.match(line) for line in unit_lines[1:]):
        return None
    low, high = WIDTH_RANGE
    for width in range(low, high + 1):
        if rewrap(unit_lines, width) == unit_lines:
            return width
    return None


@functools.cache
def modal_width(path: str) -> int:
    """The width most of this file's units solve to, for a unit that solves to none."""
    solved: list[int] = []
    for block in Path(path).read_text(encoding="utf-8").split("\n\n"):
        lines = block.split("\n")
        solved += [width for lo, hi in units(block) if (width := solve_width(lines[lo:hi]))]
    return max(set(solved), key=solved.count) if solved else FALLBACK_WIDTH


def _block_at(text: str, offset: int) -> tuple[int, str]:
    """The start offset and text of the blank-line-separated block containing `offset`."""
    position = 0
    for block in text.split("\n\n"):
        if position <= offset <= position + len(block):
            return position, block
        position += len(block) + 2
    raise AssertionError(f"offset {offset} falls outside every block")


def _unit_at(text: str, offset: int) -> tuple[int, str, int, int, int]:
    """The block start, block text, unit bounds and line index of the unit holding `offset`."""
    block_start, block = _block_at(text, offset)
    line = block.count("\n", 0, offset - block_start)
    low, high = next((lo, hi) for lo, hi in units(block) if lo <= line < hi)
    return block_start, block, low, high, line


def _reflow(*, raw: str, spliced: str, offset: int, path: str) -> str:
    """Re-wrap only the unit the splice landed in, at the width that unit was already wrapped at.

    The unit is solved on `raw` and recomputed on `spliced`, because a replacement spanning a
    different number of lines from the text it replaced moves every line index taken beforehand.
    """
    _, original_block, low, high, _ = _unit_at(raw, offset)
    original_unit = original_block.split("\n")[low:high]
    if len(original_unit) == 1 or any(UNWRAPPABLE.match(line) for line in original_unit):
        return spliced
    width = solve_width(original_unit) or modal_width(path)

    block_start, block, low, high, line = _unit_at(spliced, offset)
    lines = block.split("\n")
    head, tail = lines[low:line], lines[line:high]
    reflowed = lines[:low] + head + rewrap(tail, width) + lines[high:]
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
    offset = project(raw)[1][start]
    frontmatter = FRONTMATTER.match(raw)
    if frontmatter and offset < frontmatter.end():
        return raw, "markup refused"

    spliced = splice(raw=raw, start=start, end=end, replacement=replacement)
    if not markup_intact(before=_block_at(raw, offset)[1], after=_block_at(spliced, offset)[1]):
        return raw, "markup refused"
    return _reflow(raw=raw, spliced=spliced, offset=offset, path=finding["file"]), "applied"


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
