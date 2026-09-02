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
- The offset map has to record where each character's markup *ends*, not only where the character
  itself sits. A map of bare character positions resumes the raw text before a closing backtick, so
  a serial comma inserted after `n_h3_cells` is written as `` `n_h3_cells,` `` — the comma inside
  the code span, the backtick count unchanged, and every markup check below satisfied.
- A split whose full stop lands on a closing marker used to delete that marker. The markup counts
  either side of the splice must agree, or the edit is refused rather than written.
- A bolded lead is the one span whose full stop belongs *inside* its markers, and the splice pulls
  it there. The docs carry 371 leads written `**Lead.**` against 6 written `**Lead**.`, while every
  span that does not open its block takes its punctuation outside.
- A quote can match inside a fenced code block, where the words are a command rather than prose.
  Splicing there rewrites the command, and every check downstream passes. A finding whose span
  reaches into a fence is refused, as one landing in YAML frontmatter already is.
- Re-wrapping the whole file buries the change, and re-wrapping at the wrong width reflows every
  line it touches. The width is solved from the lines being replaced, per unit.
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

import difflib
import functools
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

LINK: Final[re.Pattern[str]] = re.compile(r"\[(?P<body>[^\]\[]*)\]\((?:[^()]|\([^()]*\))*\)")
"""A markdown link, whose label survives the projection and whose target does not."""

CODE_SPAN: Final[re.Pattern[str]] = re.compile(
    r"(?P<fence>`+)(?P<body>(?:[^\n]|\n(?![ \t]*\n))+?)(?P=fence)(?!`)"
)
"""A code span, whose content survives the projection and whose backtick fences do not.

The body cannot span a blank line, so an unmatched backtick in one paragraph cannot swallow the
next: the fences of a fenced block never pair on one line either, and fall through to the
character-by-character path that simply drops them.
"""

MARKER: Final[re.Pattern[str]] = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")
"""The bullet or number that starts a list item, and therefore starts a new wrapping unit."""

UNWRAPPABLE: Final[re.Pattern[str]] = re.compile(r"^(\s*\||\s*#|\s*>|\s{4,}\S|```)")
"""A table row, heading, quote, indented block or fence — none of which may be re-flowed."""

MARKUP: Final[tuple[str, ...]] = ("**", "`", "[", "](")
"""The markers whose count must be identical before and after a splice."""

SENTENCE_STOPS: Final[str] = ".!?"
"""The punctuation that ends a sentence, and so ends a bolded lead inside the lead's own markers."""

WIDTH_RANGE: Final[tuple[int, int]] = (88, 104)
"""The wrap widths to try when solving a unit's width. This repo's pages sit between 94 and 100."""

FALLBACK_WIDTH: Final[int] = 99
"""Used only for a file whose every unit is a single line, so no width can be solved from it."""

FRONTMATTER: Final[re.Pattern[str]] = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
"""A skill file's YAML block. Its indented lines read as list markers, so it is never re-flowed."""

FENCE: Final[re.Pattern[str]] = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[^`\n]*$")
"""The line opening or closing a fenced code block, matched against one line at a time.

The indent is unbounded rather than CommonMark's three characters, because a fence inside a list
item is indented to that item's content column: every fenced block on the code-style page is, and
two of the six on the getting-started page are. A line that deep which is not a fence is inside an
indented code block anyway, where an edit is no more welcome.

Nothing may follow the marker except an info string carrying no backtick, which is CommonMark's own
rule and is what tells a fence from an inline ```code span``` that a hard wrap has pushed to the
start of a line. Read as a fence, such a span opens a region no later line closes, and every
finding in the rest of the file is then refused.
"""

QUOTE_MARKERS: Final[re.Pattern[str]] = re.compile(r"(?:[ \t]*>)+[ \t]*")
"""The blockquote markers a line may carry before its content, however deeply nested."""

BOLD: Final[str] = "**"
"""The emphasis marker a bolded lead is written with, and the only one a stop is moved into."""


class Finding(TypedDict):
    """One reported edit: the sentence as the agent quoted it, and what it should say instead."""

    file: str
    quote: str
    replacement: str


StatusType = Literal[
    "applied", "pre-existing", "no match", "ambiguous", "markup refused", "code block"
]


class CharSpan(NamedTuple):
    """Where one projected character sits in the raw text, together with the markup glued to it.

    `text` is the character itself. `left` and `right` widen that to include the markup that must
    stay outside anything the splice writes: the opening backtick, `[` or `**` before the
    character, and the closing backtick, `](url)` or `**` after it. A comma inserted at that
    boundary therefore lands after the closing marker, which is where this repo's prose puts it —
    215 commas sit after a closing `**` across the docs and none inside one.
    """

    left: int
    text: int
    right: int


class _Projection:
    """The markup-stripped text built so far, and where in the raw text each character came from.

    Markup is dropped from the text but not forgotten: `open_at` and `close_at` widen the bounds of
    the character beside it, which is what keeps a spliced comma outside the code span, link or
    bold span it follows.
    """

    def __init__(self) -> None:
        self.plain: list[str] = []
        self.lefts: list[int] = []
        self.texts: list[int] = []
        self.rights: list[int] = []
        self.pending: int | None = None
        self.emphasis: list[int] = []

    @property
    def spans(self) -> tuple[CharSpan, ...]:
        """One `CharSpan` per projected character, in projection order."""
        return tuple(
            CharSpan(left=left, text=text, right=right)
            for left, text, right in zip(self.lefts, self.texts, self.rights, strict=True)
        )

    def open_at(self, offset: int) -> None:
        """Record markup opening at `offset`, for the next character to carry in its `left`."""
        if self.pending is None:
            self.pending = offset

    def close_at(self, offset: int) -> None:
        """Extend the last character's `right` over markup that ends at `offset`."""
        if self.rights:
            self.rights[-1] = offset

    def push(self, char: str, at: int) -> None:
        """Append the character at `raw[at]`, collapsing a run of whitespace into one space."""
        if char.isspace():
            if not self.plain or self.plain[-1] == " ":
                # The rest of a collapsed whitespace run stays inside the space it became, so no
                # splice can land between two spaces the projection merged into one.
                if self.rights and self.rights[-1] == at:
                    self.close_at(at + 1)
                return
            char = " "
        self.plain.append(char)
        self.lefts.append(at if self.pending is None else self.pending)
        self.texts.append(at)
        self.rights.append(at + 1)
        self.pending = None

    def take_bracketed(self, match: re.Match[str]) -> None:
        """Push a code span's or a link label's body, gluing the brackets to its end characters.

        The body is hard-wrapped like any other prose, so it needs the same whitespace collapsing
        `push` applies to everything else.
        """
        self.open_at(match.start())
        pushed_before = len(self.plain)
        for position, char in enumerate(match.group("body")):
            self.push(char, match.start("body") + position)
        if len(self.plain) > pushed_before:
            self.close_at(match.end())

    def take_emphasis(self, *, raw: str, index: int) -> int:
        """Bind the run of asterisks at `index` to the character it closes, or the one it opens.

        A run closing an open span of the same width binds to the character before it, so a comma
        spliced at that boundary lands outside the markers. A run followed by text opens a span and
        binds to the character after it. A run doing neither — the `*` starting a bullet — is
        dropped and bound to nothing, which leaves any splice across it to be refused.
        """
        end = index
        while end < len(raw) and raw[end] == "*":
            end += 1
        width = end - index
        closes_an_open_span = bool(self.emphasis) and self.emphasis[-1] == width
        if closes_an_open_span and index > 0 and not raw[index - 1].isspace():
            self.emphasis.pop()
            self.close_at(end)
        elif end < len(raw) and not raw[end].isspace():
            self.emphasis.append(width)
            self.open_at(index)
        return end


@functools.lru_cache(maxsize=64)
def project(raw: str) -> tuple[str, tuple[CharSpan, ...]]:
    """Strip the markup from `raw`, returning the plain text and where each character came from.

    Args:
        raw: The file's text, hard wrapping and markdown included.

    Returns:
        `(plain, spans)` where `plain[i]` is the character at `raw[spans[i].text]`, every run of
        whitespace has become one space, and the markup a sub-agent's quote will not carry — link
        targets, emphasis markers, code fences — has been dropped. Each span's `left` and `right`
        bounds carry that markup, so a splice writes an inserted comma after the closing backtick
        rather than inside the code span.
    """
    projection = _Projection()
    index, length = 0, len(raw)
    while index < length:
        bracketed = LINK.match(raw, index) or CODE_SPAN.match(raw, index)
        if bracketed:
            projection.take_bracketed(bracketed)
            index = bracketed.end()
            continue
        if raw[index] == "*":
            index = projection.take_emphasis(raw=raw, index=index)
            continue
        if raw[index] == "`":
            index += 1
            continue
        if raw[index] == "_":
            index = _push_underscores(raw=raw, index=index, projection=projection)
            continue
        projection.push(raw[index], index)
        index += 1
    return "".join(projection.plain), projection.spans


def _push_underscores(*, raw: str, index: int, projection: _Projection) -> int:
    """Drop a run of underscores that marks emphasis; keep one that sits inside an identifier.

    `mid_2025_to_mid_2026` and `mae__all` are names, and dropping their underscores makes the
    projection unmatchable against a quote that spells them correctly. A dropped run is bound to
    nothing, so a splice across an underscore-emphasised span is refused rather than written — the
    docs use `**` for emphasis, so the case has not come up.
    """
    end = index
    while end < len(raw) and raw[end] == "_":
        end += 1
    inside_word = index > 0 and raw[index - 1].isalnum() and end < len(raw) and raw[end].isalnum()
    if not inside_word:
        return end
    for position in range(index, end):
        projection.push(raw[position], position)
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
    replacement's own characters, and a rewritten run keeps the brackets around it: an identifier
    renamed inside a code span comes back still spanned, and a comma inserted after one comes back
    after the closing backtick.

    A deleted run keeps the brackets around it too, so cutting the last word of a bolded lead
    leaves the lead bolded. The exception is a run that is exactly one span's whole content, where
    keeping the brackets would leave `[](url)` or an empty pair of backticks behind: there the
    brackets go with the words, and `markup_intact` refuses the edit when that unbalances the
    paragraph.
    """
    projected, spans = project(raw)
    old = projected[start:end]
    new = re.sub(r"\s+", " ", replacement).strip()
    out: list[str] = []
    written_to = spans[start].left
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, old_lo, old_hi, new_lo, new_hi in matcher.get_opcodes():
        if tag == "insert":
            _append_replacement(out=out, text=new[new_lo:new_hi], raw=raw, written_to=written_to)
            continue
        first, last = spans[start + old_lo], spans[start + old_hi - 1]
        if tag == "equal":
            out.append(raw[first.left : last.right])
            written_to = last.right
        elif tag in ("replace", "delete"):
            opening, closing = raw[first.left : first.text], raw[last.text + 1 : last.right]
            if tag == "delete" and opening and closing:
                opening, closing = "", ""
            if opening:
                # An empty chunk would hide the closing marker the stop may have to move inside.
                out.append(opening)
                written_to = first.text
            _append_replacement(out=out, text=new[new_lo:new_hi], raw=raw, written_to=written_to)
            if closing:
                out.append(closing)
            written_to = last.right
    return raw[: spans[start].left] + "".join(out) + raw[spans[end - 1].right :]


def _append_replacement(*, out: list[str], text: str, raw: str, written_to: int) -> None:
    """Append replacement text, pulling a sentence-ending stop inside a bolded lead's markers.

    `written_to` is the raw offset the last raw chunk in `out` ran to, which is what says whether
    the text is being written straight after a closing `**`.
    """
    ends_a_sentence = bool(text) and text[0] in SENTENCE_STOPS
    marker = _lead_marker(raw=raw, at=written_to) if ends_a_sentence else ""
    if marker and out and out[-1].endswith(marker):
        out[-1] = out[-1][: -len(marker)] + text[0] + marker
        text = text[1:]
    if text:
        out.append(text)


def _lead_marker(*, raw: str, at: int) -> str:
    """The emphasis run ending at `at` when it closes a span opening its own block, else `""`.

    A bolded lead's full stop belongs inside its markers and every other span's punctuation belongs
    outside. Counted over the 78 markdown files under `docs/`, in the repository root and in
    `.claude/skills/`: a lead opening a paragraph carries the stop inside its `**` 451 times
    against 6 that do not, a lead on a list item 404 times against 1, and a lead in a blockquote 38
    times against 1. A bold span in the middle of a sentence goes the other way — 144 commas and 70
    full stops sit after its closing `**`, against no comma and 2 full stops inside one. A lead is
    therefore recognised through a blockquote's `>` and a list item's bullet alike.

    Only `**` moves a stop. Single-asterisk emphasis was never counted, so the script leaves the
    stop where the reviewer's replacement put it.
    """
    if not raw.endswith(BOLD, 0, at):
        return ""
    block_start, block = _block_at(raw, at)
    opener = raw.rfind(BOLD, block_start, at - len(BOLD))
    if opener == -1:
        return ""
    # A blockquote's markers are not text, so drop the leading run of them before asking what
    # precedes the lead: what is left is either nothing or the one list marker `MARKER` describes.
    # Only the leading run — a `>` later in the prefix is an arrow or a comparison, and `-> ` read
    # as a bullet would pull the stop inside a bold span that opens nothing.
    before_opener = block[: opener - block_start]
    quoted = QUOTE_MARKERS.match(before_opener)
    before_opener = before_opener[quoted.end() :] if quoted else before_opener
    return BOLD if not before_opener.strip() or MARKER.fullmatch(before_opener) else ""


@functools.lru_cache(maxsize=64)
def fenced_regions(raw: str) -> tuple[tuple[int, int], ...]:
    """The raw `[start, end)` bounds of every fenced code block in `raw`.

    A reviewer quotes prose, but the quote can still match a comment inside a shell snippet: "create
    the virtualenv and install all workspace packages" is a comment inside a fenced block on the
    getting-started page, and a serial comma spliced into it rewrites the command. A block left
    unclosed runs to the end of the file, which is how the renderer reads it too.
    """
    regions: list[tuple[int, int]] = []
    opener, opened_at, offset = "", 0, 0
    for line in raw.splitlines(keepends=True):
        match = FENCE.match(line)
        marker = match.group("fence") if match else ""
        if not opener:
            opener, opened_at = marker, offset
        elif marker and marker[0] == opener[0] and len(marker) >= len(opener):
            regions.append((opened_at, offset + len(line)))
            opener = ""
        offset += len(line)
    if opener:
        regions.append((opened_at, len(raw)))
    return tuple(regions)


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


def _trailing_blanks(unit_lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a unit into the lines carrying text and the empty lines after them.

    Blocks are separated by blank lines, so the only unit that can end in one is the last of a file
    that ends with a newline: splitting that block on newlines leaves an empty final line.
    Re-flowing the empty line away strips the file's trailing newline, and stops the unit's own
    width from solving.
    """
    end = len(unit_lines)
    while end and not unit_lines[end - 1].strip():
        end -= 1
    return unit_lines[:end], unit_lines[end:]


def _reflow(*, raw: str, spliced: str, offset: int, path: str) -> str:
    """Re-wrap only the unit the splice landed in, at the width that unit was already wrapped at.

    The unit is solved on `raw` and recomputed on `spliced`, because a replacement spanning a
    different number of lines from the text it replaced moves every line index taken beforehand.
    """
    _, original_block, low, high, _ = _unit_at(raw, offset)
    original_unit, _ = _trailing_blanks(original_block.split("\n")[low:high])
    if len(original_unit) < 2 or any(UNWRAPPABLE.match(line) for line in original_unit):
        return spliced
    width = solve_width(original_unit) or modal_width(path)

    block_start, block, low, high, line = _unit_at(spliced, offset)
    lines = block.split("\n")
    head, tail = lines[low:line], lines[line:high]
    tail, blanks = _trailing_blanks(tail)
    reflowed = lines[:low] + head + rewrap(tail, width) + blanks + lines[high:]
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
