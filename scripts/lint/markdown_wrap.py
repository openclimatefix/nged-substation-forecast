"""The one wrap width every docs-reflow script in this repo targets, and a whole-file reflow.

This repository's markdown prose is hard-wrapped at `WIDTH`, and `pymarkdown`'s `MD013` rule
holds the prose there. A script that rewraps one line of a page therefore does not have to solve
which width that page is already at before it can match the lines around the edit. Every reflow
script imports `WIDTH` from here rather than assuming a width or solving for each page's width.
"""

import re
from typing import Final

WIDTH: Final[int] = 100
"""Column at which prose in `docs/`, the READMEs, and the skill files is wrapped."""

FRONTMATTER: Final[re.Pattern[str]] = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
"""A skill file's YAML block, left untouched: its indented lines would otherwise read as a list."""

FENCE: Final[re.Pattern[str]] = re.compile(r"^([ \t]*)(`{3,}|~{3,})")
"""The line opening or closing a fenced code block."""

HEADING: Final[re.Pattern[str]] = re.compile(r"^\s*#")

TABLE_ROW: Final[re.Pattern[str]] = re.compile(r"^\s*\|")

LIST_MARKER: Final[re.Pattern[str]] = re.compile(r"^(\s*(?:[-*+]|\d+[.)])\s+)")
"""The bullet or number that opens a list item, and so starts a new wrapping unit."""

QUOTE_MARKER: Final[re.Pattern[str]] = re.compile(r"^(\s*(?:>\s?)+)")
"""The `>` markers a blockquote line opens with, however deeply nested."""

BLOCK_OPENER: Final[re.Pattern[str]] = re.compile(r"^(#|>|`{3,}|~{3,})|^[-*+]\Z|^\d+[.)]\Z")
"""A whitespace-split token that opens a new block wherever it starts a line, wrapping or not.

A wrapped continuation line landing on one of these tokens is read as a heading, a blockquote, a
list item, or a fence rather than as the rest of the sentence it belongs to — the same class of
bug `mkdocs-authoring` documents for `#`, generalised to every other token CommonMark treats the
same way. `#` and `>` open their block with no space required after the marker, so a token merely
*starting* with `#` or `>` is risky. The markers `-`, `*`, `+`, and an ordered marker each need
the following space that `_tokenise` has already split off, so only a token that *is* the bare
marker is risky.
"""


def _tokenise(text: str) -> list[str]:
    """Split `text` on whitespace, gluing a `BLOCK_OPENER` word to the word before it.

    A link's destination is never split from the label word before it, because nothing in
    `[label](destination)` puts whitespace between `]` and `(`. The label itself does wrap across
    lines, and renders correctly when it does: `packages/delta_store/README.md` carries a link
    whose label breaks after "Storage formats: measured, not". So the label needs no
    special-casing.

    A word that happens to start with `#`, `>`, a list marker, or a fence marker is read as
    opening that block wherever it lands at the start of a line, so it can never be the first
    word on a wrapped line. Gluing that word to its predecessor keeps the pair on one line
    together and lets `_wrap` treat the pair as one token, rather than discovering the clash
    after the line is already full and having to push the width past `WIDTH` to fix the clash.
    """
    words = text.split()
    glued: list[str] = []
    for word in words:
        if glued and BLOCK_OPENER.match(word):
            glued[-1] = f"{glued[-1]} {word}"
        else:
            glued.append(word)
    return glued


def _wrap(words: list[str], *, initial_indent: str, subsequent_indent: str) -> list[str]:
    """Greedily wrap `words` to `WIDTH`."""
    lines: list[str] = []
    current = initial_indent
    at_line_start = True
    for word in words:
        candidate = f"{current}{word}" if at_line_start else f"{current} {word}"
        if len(candidate) > WIDTH and not at_line_start:
            lines.append(current)
            current = f"{subsequent_indent}{word}"
        else:
            current = candidate
        at_line_start = False

    lines.append(current)
    return [line for line in lines if line.strip()]


def _reflow_unit(unit_lines: list[str]) -> list[str]:
    """Reflow one list item, blockquote, or plain paragraph to `WIDTH`, preserving its prefix.

    A blockquote marker is stripped and reapplied to every output line uniformly, before the list
    or plain-paragraph handling runs on what is left. Stripping the marker first is what lets a
    list item nested inside a blockquote (each source line prefixed `> - `) keep its own marker
    rather than being read as ordinary quoted prose and merged with its siblings into one
    paragraph.

    A paragraph with no marker of its own can still be a list item's body: Python-Markdown treats
    any indent at or past the item's content column as part of that item, including a second
    paragraph separated from the marker line by a blank line. An unmarked paragraph carries that
    indent on its own first line, so preserving whatever indent the first de-quoted line already
    has — rather than flattening every unmarked paragraph to column 0 — is what keeps the
    paragraph inside the list item it belongs to instead of closing the list and starting a new
    top-level paragraph.
    """
    quote_match = QUOTE_MARKER.match(unit_lines[0])
    quote_prefix = quote_match.group(1) if quote_match else ""
    lines = (
        [QUOTE_MARKER.sub("", line, count=1) for line in unit_lines] if quote_prefix else unit_lines
    )

    marker_match = LIST_MARKER.match(lines[0])
    if marker_match:
        marker = marker_match.group(1)
        prefix, indent = f"{quote_prefix}{marker}", f"{quote_prefix}{' ' * len(marker)}"
        stripped = [LIST_MARKER.sub("", lines[0]), *lines[1:]]
        body = " ".join(" ".join(stripped).split())
        return _wrap(_tokenise(body), initial_indent=prefix, subsequent_indent=indent)

    match = re.match(r"[ \t]*", lines[0])
    own_indent = match.group() if match else ""
    indent = f"{quote_prefix}{own_indent}"
    body = " ".join(" ".join(lines).split())
    return _wrap(_tokenise(body), initial_indent=indent, subsequent_indent=indent)


def _is_blank_quote_line(line: str) -> bool:
    """Whether `line` is a blockquote line carrying no content — e.g. a bare `>`.

    CommonMark reads this the way it reads a blank line at the top level: it separates two
    paragraphs while keeping both inside the same `<blockquote>`. Folding it into a wrapping unit
    like ordinary quote content would erase the paragraph break and merge the two into one.
    """
    quote_match = QUOTE_MARKER.match(line)
    return bool(quote_match) and not line[quote_match.end() :].strip()


def _is_unwrappable(line: str) -> bool:
    """Whether `line` opens a block whose line breaks carry meaning and must survive untouched.

    There is no case here for a `$$...$$` display-math block: MathJax reads a newline inside one
    as ordinary whitespace, so reflowing it is safe, but only because every token in a formula is
    whitespace-separated the same way a word is — the corpus carries no construct where that
    isn't true. A LaTeX block relying on a significant literal newline would need its own case
    here.
    """
    return bool(
        HEADING.match(line)
        or TABLE_ROW.match(line)
        or FENCE.match(line)
        or not line.strip()
        or _is_blank_quote_line(line)
    )


def _units(lines: list[str]) -> list[tuple[int, int]]:
    """The `(first_line, last_line + 1)` bounds of each wrapping unit in a flowable block.

    A unit is a run of lines starting at a list marker or a blockquote marker, or the whole block
    where it carries neither — so a list written without blank lines between its items reflows
    one item at a time rather than merging every sibling into one paragraph.
    """
    starts = [
        0,
        *(
            i
            for i in range(1, len(lines))
            if LIST_MARKER.match(QUOTE_MARKER.sub("", lines[i], count=1))
            or bool(QUOTE_MARKER.match(lines[i])) != bool(QUOTE_MARKER.match(lines[i - 1]))
        ),
    ]
    return list(zip(starts, [*starts[1:], len(lines)], strict=True))


def _flatten(text: str) -> str:
    """`text` with its per-line blockquote markers dropped and all remaining whitespace collapsed.

    A blockquote's `>` repeats once per line, so rewrapping to a different line count
    legitimately changes how many appear. Comparing the flattened text is what lets
    `reflow_text`'s own round-trip check tell a rewrap from a change to the words.
    """
    lines = [QUOTE_MARKER.sub("", line) for line in text.split("\n")]
    return "".join("\n".join(lines).split())


def reflow_text(source: str) -> str:
    """Rewrap every prose paragraph, list item, and blockquote in `source` to `WIDTH`.

    Headings, tables, fenced code blocks, and a leading YAML frontmatter block are left
    untouched. There is no exemption for a CommonMark *indented* code block (4+ spaces, no
    fence): the `code-style` skill requires every code sample in this repo to be fenced, and a
    4-space indent with no marker is otherwise indistinguishable from a nested list item's
    continuation, which does need rewrapping. A scan of all 96 markdown files in the repository
    found no unfenced indented code block outside a list, so an indented, unmarked line is
    treated as prose rather than risk silently skipping list continuations.

    `source` must end in a newline, as every file in this repo does.
    """
    frontmatter = ""
    body = source
    if match := FRONTMATTER.match(source):
        frontmatter = match.group()
        body = source[match.end() :]

    lines = body.split("\n")
    out: list[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence or _is_unwrappable(line):
            out.append(line)
            i += 1
            continue

        block_start = i
        while i < len(lines) and not FENCE.match(lines[i]) and not _is_unwrappable(lines[i]):
            i += 1
        block = lines[block_start:i]

        for lo, hi in _units(block):
            out.extend(_reflow_unit(block[lo:hi]))

    rewritten = frontmatter + "\n".join(out)
    assert _flatten(rewritten) == _flatten(source), "reflow changed the text, not just the wrapping"
    return rewritten
