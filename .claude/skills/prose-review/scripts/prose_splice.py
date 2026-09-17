"""Project markdown to plain text, splice a sweep's replacement back in, and refuse the rest.

`apply_findings.py` edits markdown files and `apply_findings_py.py` edits the docstrings and
comments of Python files, but a docstring *is* markdown — mkdocstrings renders it onto an API page
— so the ways a splice corrupts one are the ways it corrupts the other. This module holds
everything both appliers need for that, and each keeps only what differs: how it finds the prose to
search (markdown blocks against `ast` and `tokenize`), and how it re-wraps what it wrote.

Three guards live here, and each exists because a splice without it silently damaged a file that
then passed every linter in the repo:

- **The offset map records where each character's markup ends, not only where the character sits.**
  A map of bare character positions resumes the raw text before a closing backtick, so a serial
  comma inserted after `n_h3_cells` is written as `` `n_h3_cells,` `` — the comma inside the code
  span, the backtick count unchanged, and every count-based check satisfied. `CharSpan` widens each
  projected character to the markup glued to it, and `splice` writes between those bounds.
- **A splice crossing a markup boundary deletes the closing marker.** Replacing `` `write_nwp`
  helper `` with `the writer` leaves one unbalanced backtick, and the rest of the paragraph renders
  as code. `markup_intact` counts the markers either side and refuses an edit that changes a count.
- **A quote can match inside a code sample, where the words are a command rather than prose.**
  Splicing there rewrites a command a reader is meant to copy and run. `fenced_regions` gives the
  bounds of every fenced block so the caller can refuse a match that reaches one.

A comment block opens every continuation line with a `#`, which a quote never carries. `project`
takes a `continuation` pattern for that: whitespace containing a newline swallows whatever the
pattern matches after it, so a sentence wrapped across three comment lines projects to one line and
the offset map still points at real characters.
"""

import difflib
import functools
import re
from typing import Final, Literal, NamedTuple

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

MARKUP: Final[tuple[str, ...]] = ("**", "`", "[", "](")
"""The markers whose count must be identical before and after a splice."""

SENTENCE_STOPS: Final[str] = ".!?"
"""The punctuation that ends a sentence, and so ends a bolded lead inside the lead's own markers."""

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

OpcodeType = Literal["replace", "delete", "insert", "equal"]
"""What `difflib` says to do with one run of the matched text."""

BOLD: Final[str] = "**"
"""The emphasis marker a bolded lead is written with, and the only one a stop is moved into."""


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

    Markup is dropped from the text but not forgotten: `open_at` and `close_at` widen the bounds
    of the character beside it, which is what keeps a spliced comma outside the code span, link
    or bold span it follows.
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

    def take_emphasis(self, *, raw: str, index: int, end: int) -> int:
        """Bind the run of asterisks at `index` to the character it closes, or the one it opens.

        A run closing an open span of the same width binds to the character before it, so a comma
        spliced at that boundary lands outside the markers. A run followed by text opens a span
        and binds to the character after it. A run doing neither — the `*` starting a bullet — is
        dropped and bound to nothing, which leaves any splice across it to be refused.
        """
        stop = index
        while stop < end and raw[stop] == "*":
            stop += 1
        width = stop - index
        closes_an_open_span = bool(self.emphasis) and self.emphasis[-1] == width
        if closes_an_open_span and index > 0 and not raw[index - 1].isspace():
            self.emphasis.pop()
            self.close_at(stop)
        elif stop < end and not raw[stop].isspace():
            self.emphasis.append(width)
            self.open_at(index)
        return stop


@functools.lru_cache(maxsize=256)
def project(
    raw: str,
    *,
    start: int = 0,
    end: int | None = None,
    continuation: re.Pattern[str] | None = None,
) -> tuple[str, tuple[CharSpan, ...]]:
    """Strip the markup from `raw[start:end]`, returning the plain text and where it came from.

    Args:
        raw: The file's text, hard wrapping and markdown included.
        start: Where to begin, for a caller projecting one docstring or comment block rather than
            a whole file.
        end: Where to stop, or None for the end of `raw`.
        continuation: A pattern for whatever opens a continuation line and is not prose — a
            comment block's `#` and the space after it. Matched only after a run of whitespace
            carrying a newline, and swallowed into that whitespace when it matches.

    Returns:
        `(plain, spans)` where `plain[i]` is the character at `raw[spans[i].text]`, every run of
        whitespace has become one space, and the markup a sub-agent's quote will not carry — link
        targets, emphasis markers, code fences — has been dropped. Each span's `left` and `right`
        bounds carry that markup, so a splice writes an inserted comma after the closing backtick
        rather than inside the code span.
    """
    projection = _Projection()
    index = start
    length = len(raw) if end is None else end
    while index < length:
        if continuation is not None and raw[index].isspace():
            index = _take_continuation(
                raw=raw, index=index, end=length, continuation=continuation, projection=projection
            )
            continue
        bracketed = LINK.match(raw, index, length) or CODE_SPAN.match(raw, index, length)
        if bracketed:
            projection.take_bracketed(bracketed)
            index = bracketed.end()
            continue
        if raw[index] == "*":
            index = projection.take_emphasis(raw=raw, index=index, end=length)
            continue
        if raw[index] == "`":
            index += 1
            continue
        if raw[index] == "_":
            index = _push_underscores(raw=raw, index=index, end=length, projection=projection)
            continue
        projection.push(raw[index], index)
        index += 1
    return "".join(projection.plain), projection.spans


def _take_continuation(
    *, raw: str, index: int, end: int, continuation: re.Pattern[str], projection: _Projection
) -> int:
    """Collapse the whitespace run at `index`, swallowing the marker that opens the next line.

    The whole run — the newline, the next line's indent and its `#` marker — stays inside the one
    space it became, so no splice can land on a character the file needs where it is.
    """
    stop = index
    saw_newline = False
    while stop < end and raw[stop].isspace():
        saw_newline = saw_newline or raw[stop] == "\n"
        stop += 1
    if saw_newline:
        match = continuation.match(raw, stop, end)
        if match:
            stop = match.end()
    projection.push(" ", index)
    projection.close_at(stop)
    return stop


def _push_underscores(*, raw: str, index: int, end: int, projection: _Projection) -> int:
    """Drop a run of underscores that marks emphasis; keep one that sits inside an identifier.

    `mid_2025_to_mid_2026` and `mae__all` are names, and dropping their underscores makes the
    projection unmatchable against a quote that spells them correctly. A dropped run is bound to
    nothing, so a splice across an underscore-emphasised span is refused rather than written —
    the docs use `**` for emphasis, so the case has not come up.
    """
    stop = index
    while stop < end and raw[stop] == "_":
        stop += 1
    inside_word = index > 0 and raw[index - 1].isalnum() and stop < end and raw[stop].isalnum()
    if not inside_word:
        return stop
    for position in range(index, stop):
        projection.push(raw[position], position)
    return stop


def plain(text: str) -> str:
    """The projection of a standalone string, for comparing a quote that carries markdown.

    Args:
        text: A quote, a replacement, or a whole file read at a git revision.

    Returns:
        The text with its markdown markers dropped and its whitespace collapsed.
    """
    return project(text)[0].strip()


def splice(
    *,
    raw: str,
    start: int,
    end: int,
    replacement: str,
    projection: tuple[str, tuple[CharSpan, ...]] | None = None,
) -> str:
    """Rewrite the projection span `[start, end)` of `raw` as `replacement`.

    Unchanged runs keep whatever markup the raw text carries there, so the links and bold markers
    the agent's replacement omits survive. Only the runs that genuinely differ take the
    replacement's own characters, and a rewritten run keeps the brackets around it: an identifier
    renamed inside a code span comes back still spanned, and a comma inserted after one comes
    back after the closing backtick.

    A deleted run keeps the brackets around it too, so cutting the last word of a bolded lead
    leaves the lead bolded. The exception is a run that is exactly one span's whole content,
    where keeping the brackets would leave `[](url)` or an empty pair of backticks behind: there
    the brackets go with the words, and `markup_intact` refuses the edit when that unbalances the
    paragraph.

    Args:
        raw: The full text of the file.
        start: Where the matched quote begins in the projection.
        end: Where the matched quote ends in the projection.
        replacement: The sentence proposed in place of the matched quote.
        projection: The `(plain, spans)` pair the bounds were taken in, for a caller that
            projected one docstring or comment block rather than the whole of `raw`. None
            projects the whole of `raw`, which is what a markdown file needs.

    Returns:
        `raw` with the matched span rewritten.
    """
    projected, spans = project(raw) if projection is None else projection
    old = projected[start:end]
    new = re.sub(r"\s+", " ", replacement).strip()
    out: list[str] = []
    written_to = spans[start].left
    for tag, old_lo, old_hi, new_lo, new_hi in _opcodes(old=old, new=new):
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


@functools.lru_cache(maxsize=256)
def _opcodes(*, old: str, new: str) -> list[tuple[OpcodeType, int, int, int, int]]:
    """The `difflib` opcodes turning `old` into `new`, computed once per pair.

    Args:
        old: The matched text as the file reads it, projected.
        new: The replacement, with its whitespace collapsed.

    Returns:
        The opcode tuples, in order.
    """
    return difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()


def changed_bounds(*, old: str, new: str) -> tuple[int, int] | None:
    """The first and last offsets of `old` a splice would rewrite, or None where nothing changes.

    A caller refusing a splice that reaches somewhere it must not — across a paragraph break, into
    a code sample — has to test the run that really changes rather than the whole matched quote,
    because every unchanged run is copied from the raw file character for character.

    Args:
        old: The matched text as the file reads it, projected.
        new: The replacement, with its whitespace collapsed.

    Returns:
        `(first, last)` offsets into `old`, or None when the two texts are identical.
    """
    changed = [(low, high) for tag, low, high, _, _ in _opcodes(old=old, new=new) if tag != "equal"]
    if not changed:
        return None
    return changed[0][0], changed[-1][1]


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

    A bolded lead's full stop belongs inside its markers and every other span's punctuation
    belongs outside. Counted over the 78 markdown files under `docs/`, in the repository root and
    in `.claude/skills/`: a lead opening a paragraph carries the stop inside its `**` 451 times
    against 6 that do not, a lead on a list item 404 times against 1, and a lead in a blockquote
    38 times against 1. A bold span in the middle of a sentence goes the other way — 144 commas
    and 70 full stops sit after its closing `**`, against no comma and 2 full stops inside one. A
    lead is therefore recognised through a blockquote's `>` and a list item's bullet alike.

    Only `**` moves a stop. Single-asterisk emphasis was never counted, so the script leaves the
    stop where the reviewer's replacement put it.
    """
    if not raw.endswith(BOLD, 0, at):
        return ""
    block_start, block = block_at(raw, at)
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


def block_at(text: str, offset: int) -> tuple[int, str]:
    """The start offset and text of the blank-line-separated block containing `offset`.

    Args:
        text: The file's text.
        offset: A raw character offset inside it.

    Returns:
        Where the block starts, and the block itself.
    """
    position = 0
    for block in text.split("\n\n"):
        if position <= offset <= position + len(block):
            return position, block
        position += len(block) + 2
    raise AssertionError(f"offset {offset} falls outside every block")


@functools.lru_cache(maxsize=64)
def fenced_regions(raw: str) -> tuple[tuple[int, int], ...]:
    """The raw `[start, end)` bounds of every fenced code block in `raw`.

    A reviewer quotes prose, but the quote can still match a comment inside a shell snippet:
    "create the virtualenv and install all workspace packages" is a comment inside a fenced block
    on the getting-started page, and a serial comma spliced into it rewrites the command. A block
    left unclosed runs to the end of the file, which is how the renderer reads it too.

    Args:
        raw: The text to scan.

    Returns:
        One `(start, end)` pair per fenced block, fence lines included.
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

    A split whose full stop lands inside a link label or between two bold markers breaks the
    markup silently: the page still lints, still builds, and the link simply stops being a link.

    Args:
        before: The paragraph, docstring or comment block as it read before the splice.
        after: The same region as it reads after.

    Returns:
        True when every marker's count is unchanged and none is left unpaired.
    """
    if after.count("**") % 2 or after.count("`") % 2:
        return False
    return all(before.count(marker) == after.count(marker) for marker in MARKUP)
