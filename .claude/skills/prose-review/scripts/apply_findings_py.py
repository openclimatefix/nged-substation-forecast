"""Apply a sentence sweep's findings to the docstrings and comments of hard-wrapped Python files.

A docstring is markdown — mkdocstrings renders it onto an API page — so the ways a splice corrupts
one are the ways it corrupts a `.md` page. The projection, the splice and the two markup guards
therefore live in `prose_splice.py` and are shared with `apply_findings.py`, the markdown sibling.
What is left here is everything a Python file needs and a markdown file does not:

- **`ast` and `tokenize` find the prose, rather than markdown blocks.** A docstring contributes the
  span between its quotes and a run of whole-line `#` comments contributes one span, so a sentence
  the file wrapped across three comment lines is one searchable run. Nothing outside those spans is
  searched, so a quote whose words also appear in a runtime string or an identifier cannot reach
  them.
- **A comment block opens every continuation line with a `#`**, which a sub-agent's quote never
  carries. `prose_splice.project` takes the marker as its `continuation` pattern and swallows it
  into the whitespace it follows.
- **A span reaching across a blank line would weld two paragraphs together**, because the
  replacement is a single line. Only the run that really changes is tested, since every unchanged
  run is copied from the file character for character.
- **A quote can match inside a code sample.** A fenced block and a reStructuredText `::` literal
  block both carry commands a reader is meant to copy and run, and a splice there rewrites the
  command while every check downstream passes. A match reaching one is refused.
- **The splice leaves a line far over the 100-character limit**, so every edited file goes through
  `scripts/lint/reflow_python_prose.py`, the engine that wrapped it originally, and then through
  `wrap_overlong` for the lines that engine declines: a Google-style `Args:` body, which it reads
  as an indented code block, and a docstring carrying hand alignment anywhere in it.
- **A splice can close a string early or comment out a line of code**, so the re-wrap is gated on
  the result parsing, and `check_prose_only.py` compares the syntax trees over the whole branch
  afterwards.

Usage::

    python3 apply_findings_py.py findings.json
    python3 apply_findings_py.py findings.json --apply
    python3 apply_findings_py.py findings.json --apply --merge-base 7657db1c

`findings.json` holds a list of objects, each with `file`, `quote` and `replacement`; any other key
is ignored, so a sweep's own `rule` and `why` fields can stay in the file. Each `file` is resolved
against the current directory, and so is each `--merge-base` lookup. Nothing is written without
`--apply`. With `--merge-base`, a finding whose sentence already exists at that revision is prose
the branch did not write, so it is skipped and listed rather than applied. Findings naming a file
that is not Python are reported as `not python` and left for `apply_findings.py`.
"""

import argparse
import ast
import bisect
import functools
import io
import json
import re
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypedDict

# Both the repo's own re-wrapper and this script's markdown sibling are found from the script's
# own location, not from the directory the script is run in, so the paths in `findings.json` are
# the only thing the current directory decides. `scripts/lint` goes on the path as well as the
# repository root, because `reflow_python_prose` imports `markdown_wrap` by its bare name.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "lint"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prose_splice import (
    CharSpan,
    changed_bounds,
    fenced_regions,
    markup_intact,
    plain,
    project,
    splice,
)

from scripts.lint.markdown_wrap import LIST_MARKER, WIDTH, _is_unwrappable, _tokenise, _wrap
from scripts.lint.reflow_python_prose import (
    DIRECTIVE_COMMENT,
    INTERIOR_DOUBLE_SPACE,
    NOT_PROSE,
    URL_ONLY_COMMENT,
    reflow_python_prose,
)

COMMENT_CONTINUATION: Final[re.Pattern[str]] = re.compile(r"#[ \t]*")
"""The marker opening a comment block's continuation line, which the projection swallows."""

COMMENT_MARKER: Final[re.Pattern[str]] = re.compile(r"^([ \t]*#[ \t]?)")
"""A whole-line comment's indent and `#`, which every line of its block repeats.

The trailing space is optional because `#text` is legal Python. Reading the marker off the line
rather than assuming `# ` is what stops a re-wrap of such a line eating its first character.
"""

BLANK_LINE: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*#?[ \t]*\n")
"""A paragraph break inside a docstring or a comment block, which no splice may cross."""

REST_LITERAL: Final[re.Pattern[str]] = re.compile(r"::[ \t]*$")
"""The `::` ending the line that opens a reStructuredText literal block.

The block is whatever follows, indented deeper than the `::` line itself — this module's own
``Usage::`` block is one. `reflow_python_prose` declines a docstring carrying one, and `locate`
and `wrap_overlong` both decline the lines themselves, because they are a command a reader copies
and runs rather than prose.
"""

MissType = Literal["no match", "ambiguous"]
"""Why a quote could not be located in exactly one place."""

RefusalType = Literal["crosses blank line", "code block", "markup refused"]
"""Why a located quote could not be edited where it sits."""

StatusType = MissType | RefusalType | Literal["applied", "pre-existing", "not python", "no-op"]
"""What became of one finding."""

FINDING_KEYS: Final[tuple[str, ...]] = ("file", "quote", "replacement")
"""The three keys every finding must carry, each holding a string."""

APPLICABLE: Final[frozenset[str]] = frozenset({"applied", "pre-existing", "not python", "no-op"})
"""The statuses that are not a failure, and so do not make the run exit non-zero."""


class Finding(TypedDict):
    """One reported sentence and the sentence proposed in its place."""

    file: str
    quote: str
    replacement: str


class Unit(NamedTuple):
    """One run of prose in a Python file, and where it sits in the raw text."""

    start: int
    end: int
    is_comment: bool


class Match(NamedTuple):
    """Where a quote was found: the unit holding it, that unit's projection, and the bounds."""

    unit: Unit
    projected: str
    spans: tuple[CharSpan, ...]
    start: int
    stop: int


@functools.lru_cache(maxsize=16)
def _line_starts(source: str) -> tuple[int, ...]:
    """The raw offset each line of `source` begins at.

    Args:
        source: The full text of a Python file.

    Returns:
        One offset per line, in order.
    """
    offsets = [0]
    for line in source.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    return tuple(offsets[:-1])


def _line_of(source: str, offset: int) -> int:
    """The zero-based index of the line `offset` falls on.

    Args:
        source: The full text of a Python file.
        offset: A raw character offset into it.

    Returns:
        The line's index.
    """
    return bisect.bisect_right(_line_starts(source), offset) - 1


def _string_statements(source: str) -> list[ast.Expr]:
    """Every bare string-literal statement in `source` — the docstrings, whatever they document.

    Args:
        source: The full text of a Python file.

    Returns:
        The statements, in the order `ast.walk` yields them.
    """
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and node.end_lineno is not None
        and node.end_col_offset is not None
    ]


@functools.lru_cache(maxsize=16)
def prose_units(source: str) -> tuple[Unit, ...]:
    """Return the raw bounds of every docstring and comment block in `source`.

    A docstring contributes the span between its quotes. Consecutive whole-line `#` comments at the
    same indent contribute one span between them, so a sentence wrapped across three comment lines
    is one searchable run; a trailing comment on a line of code is its own span.

    Args:
        source: The full text of a Python file.

    Returns:
        The units, in the order they appear.
    """
    units: list[Unit] = []
    starts = _line_starts(source)

    for node in _string_statements(source):
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        literal_start = starts[node.lineno - 1] + node.col_offset
        literal_end = starts[node.end_lineno - 1] + node.end_col_offset
        literal = source[literal_start:literal_end]
        quote = '"""' if literal.count('"""') >= 2 else ("'''" if literal.count("'''") >= 2 else "")
        opening = len(literal) - len(literal.lstrip("rbfuRBFU"))
        opening += len(quote) if quote else 1
        closing = len(quote) if quote else 1
        units.append(Unit(literal_start + opening, literal_end - closing, is_comment=False))

    run: tuple[int, int] | None = None
    run_indent = -1
    run_line = -2
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line_start = starts[token.start[0] - 1]
        before = source[line_start : line_start + token.start[1]]
        start = line_start + token.start[1] + len(token.string) - len(token.string.lstrip("#"))
        end = line_start + token.end[1]
        whole_line = before.strip() == ""
        # A run continues only onto the very next line at the same indent. Without the line test,
        # two comment blocks separated by a docstring or by code merge into one span, and a quote
        # can then match the code between them.
        continues_run = (
            whole_line
            and run is not None
            and token.start[1] == run_indent
            and token.start[0] == run_line + 1
        )
        if continues_run and run is not None:
            run = (run[0], end)
            run_line = token.start[0]
            continue
        if run is not None:
            units.append(Unit(run[0], run[1], is_comment=True))
            run = None
        if whole_line:
            run, run_indent, run_line = (start, end), token.start[1], token.start[0]
        else:
            units.append(Unit(start, end, is_comment=True))
    if run is not None:
        units.append(Unit(run[0], run[1], is_comment=True))
    return tuple(units)


def project_unit(*, raw: str, unit: Unit) -> tuple[str, tuple[CharSpan, ...]]:
    """Project one prose unit to a single markup-stripped line, with its map back into `raw`.

    Args:
        raw: The full text of the file.
        unit: The docstring or comment block to project.

    Returns:
        The projected text, and one `CharSpan` per projected character.
    """
    return project(
        raw,
        start=unit.start,
        end=unit.end,
        continuation=COMMENT_CONTINUATION if unit.is_comment else None,
    )


def normalise(text: str) -> str:
    """Return `text` as one line, with a comment block's `#` markers removed.

    Args:
        text: A quote or a replacement as a sub-agent wrote it.

    Returns:
        The text with runs of whitespace collapsed and stray `#` markers dropped.
    """
    collapsed = " ".join(text.split())
    collapsed = re.sub(r"(?:^|(?<= ))#+ ?", "", collapsed)
    return collapsed.strip()


@functools.cache
def _existing_at(merge_base: str, path: str) -> str | None:
    """Return `path` at `merge_base`, projected to one line, or None where it did not exist.

    `<rev>:./<path>` is git's cwd-relative form. Without the `./`, git resolves the path against
    the repository root instead, so every lookup from a subdirectory fails and the gate this
    function implements silently stops gating anything.

    `normalise` runs before `plain` because a comment block's `#` markers are not markdown: without
    dropping them, a sentence the file wrapped across two comment lines carries a `#` in the middle
    and no quote ever matches it.

    Args:
        merge_base: The git revision that predates the branch's prose.
        path: The file to read, as the finding names it.

    Returns:
        The old text with its comment markers, markup and whitespace collapsed, or None.
    """
    completed = subprocess.run(
        ["git", "show", f"{merge_base}:./{path}"], capture_output=True, text=True, check=False
    )
    return plain(normalise(completed.stdout)) if completed.returncode == 0 else None


def _docstring_bodies(source: str) -> list[range]:
    """The zero-based line indices strictly inside each docstring in `source`.

    The line carrying the opening quotes and the line carrying the closing quotes are both left
    out, because each carries something that is not prose.

    Args:
        source: The full text of a Python file.

    Returns:
        One range of line indices per docstring.
    """
    bodies: list[range] = []
    for node in _string_statements(source):
        assert node.end_lineno is not None
        bodies.append(range(node.lineno, node.end_lineno - 1))
    return bodies


@functools.lru_cache(maxsize=16)
def code_sample_lines(source: str) -> frozenset[int]:
    """The zero-based index of every line of `source` that is a code sample rather than prose.

    Two shapes qualify. A fenced block's lines, fences included, are found the same way
    `apply_findings.py` finds them in a markdown file. A reStructuredText `::` literal block is
    whatever follows a line ending `::`, indented deeper than that line — `reflow_python_prose`
    declines a docstring carrying one, and `markdown_wrap` has no case for it at all.

    Args:
        source: The full text of a Python file.

    Returns:
        The line indices no splice and no re-wrap may touch.
    """
    lines = source.split("\n")
    sample: set[int] = set()
    for low, high in fenced_regions(source):
        sample.update(range(_line_of(source, low), _line_of(source, max(low, high - 1)) + 1))
    for body in _docstring_bodies(source):
        sample.update(_literal_block_lines(lines=lines, body=body))
    return frozenset(sample)


def _below(*, line: str, indent: int) -> bool:
    """Whether `line` belongs to a literal block opened by a line indented `indent` columns.

    Args:
        line: The line to test.
        indent: The indent of the line ending `::`.

    Returns:
        True for a blank line or one indented deeper than the opener.
    """
    return not line.strip() or len(line) - len(line.lstrip()) > indent


def _literal_block_lines(*, lines: list[str], body: range) -> set[int]:
    """The lines of one docstring that belong to a reStructuredText `::` literal block.

    Args:
        lines: Every line of the file.
        body: The line indices strictly inside the docstring.

    Returns:
        The indices of the block's own lines, the opening `::` line excluded.
    """
    marked: set[int] = set()
    index = body.start
    while index < body.stop:
        line = lines[index]
        index += 1
        if not REST_LITERAL.search(line):
            continue
        opener_indent = len(line) - len(line.lstrip())
        while index < body.stop and _below(line=lines[index], indent=opener_indent):
            marked.add(index)
            index += 1
    return marked


def locate(*, raw: str, quote: str) -> Match | MissType:
    """Find `quote` among the prose units of `raw`, or return why it could not be found.

    Args:
        raw: The full text of a Python file.
        quote: The sentence to find, projected to one line.

    Returns:
        Where the quote sits, or a status naming the reason no unique match was found.
    """
    needle = plain(quote)
    hits: list[Match] = []
    for unit in prose_units(raw):
        projected, spans = project_unit(raw=raw, unit=unit)
        position = projected.find(needle)
        while position != -1:
            hits.append(Match(unit, projected, spans, position, position + len(needle)))
            position = projected.find(needle, position + 1)
    if len(hits) > 1:
        return "ambiguous"
    return hits[0] if hits else "no match"


def _changed_raw(*, raw: str, found: Match, bounds: tuple[int, int]) -> str:
    """The raw text the splice would rewrite, from its first changed character to its last.

    Args:
        raw: The full text of the file.
        found: Where the quote was located.
        bounds: The changed run's bounds within the matched quote.

    Returns:
        The raw text of that run, which is empty-ish for a pure insertion.
    """
    low, high = bounds
    last = len(found.spans) - 1
    first_index = min(found.start + low, last)
    last_index = min(found.start + max(high, low + 1) - 1, last)
    return raw[found.spans[first_index].left : found.spans[last_index].right]


def _refused(*, raw: str, found: Match, replacement: str) -> RefusalType | Literal["no-op"] | None:
    """Why the located quote cannot be edited where it sits, or None where it can.

    Args:
        raw: The full text of the file.
        found: Where the quote was located.
        replacement: The sentence proposed in its place.

    Returns:
        The status to report, or None to go ahead and splice.
    """
    sample = code_sample_lines(raw)
    first = _line_of(raw, found.spans[found.start].text)
    last = _line_of(raw, found.spans[found.stop - 1].right - 1)
    if any(number in sample for number in range(first, last + 1)):
        return "code block"
    bounds = changed_bounds(old=found.projected[found.start : found.stop], new=replacement)
    if bounds is None:
        return "no-op"
    if BLANK_LINE.search(_changed_raw(raw=raw, found=found, bounds=bounds)):
        return "crosses blank line"
    return None


def apply_one(*, raw: str, finding: Finding, merge_base: str | None) -> tuple[str, StatusType]:
    """Apply one finding to `raw`, or return `raw` unchanged with the reason it was refused.

    Args:
        raw: The full text of the file.
        finding: The sentence to change and the sentence to write instead.
        merge_base: A revision whose prose the branch did not write, or None to skip that gate.

    Returns:
        The updated text, and what became of the finding.
    """
    quote = normalise(finding["quote"])
    replacement = normalise(finding["replacement"])
    if quote == replacement:
        return raw, "no-op"

    if merge_base is not None:
        old = _existing_at(merge_base, finding["file"])
        if old is not None and plain(quote) in old:
            return raw, "pre-existing"

    found = locate(raw=raw, quote=quote)
    if isinstance(found, str):
        return raw, found

    refusal = _refused(raw=raw, found=found, replacement=replacement)
    if refusal is not None:
        return raw, refusal

    spliced = splice(
        raw=raw,
        start=found.start,
        end=found.stop,
        replacement=replacement,
        projection=(found.projected, found.spans),
    )
    grew = len(spliced) - len(raw)
    before = raw[found.unit.start : found.unit.end]
    after = spliced[found.unit.start : found.unit.end + grew]
    if not markup_intact(before=before, after=after):
        return raw, "markup refused"
    return spliced, "applied"


@functools.lru_cache(maxsize=16)
def wrappable_lines(source: str) -> frozenset[int]:
    """Return the zero-based index of every line of `source` that is prose end to end.

    A line qualifies when re-wrapping it can move only prose: every line strictly inside a
    docstring, and every whole-line `#` comment. The line carrying a docstring's opening quotes and
    the line carrying its closing quotes are both excluded, because each carries something that is
    not prose.

    Testing an offset range instead gets both ends wrong, and both failures are silent. A
    docstring's closing-quote line *starts* inside the span, so a range test accepts it and the
    re-wrap then pulls the closing quotes up into the paragraph and the code below into the string.
    A comment block's first line starts *before* its span, so a range test rejects it and no
    comment block is ever wrapped.

    Args:
        source: The full text of a Python file.

    Returns:
        The zero-based indices of the lines a re-wrap may touch.
    """
    wrappable: set[int] = set(comment_lines(source))
    for body in _docstring_bodies(source):
        wrappable.update(body)
    return frozenset(wrappable)


@functools.lru_cache(maxsize=16)
def comment_lines(source: str) -> frozenset[int]:
    """Return the zero-based index of every whole-line `#` comment in `source`.

    A line that looks like a comment is not always one: `# Heading` inside a docstring is markdown,
    and the tests that decide whether a comment may be re-wrapped are not the tests that decide it
    for a docstring. `tokenize` is what tells the two apart.

    Args:
        source: The full text of a Python file.

    Returns:
        The comment lines' indices.
    """
    return frozenset(
        token.start[0] - 1
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and token.line[: token.start[1]].strip() == ""
    )


def _is_prose_comment(text: str) -> bool:
    """Whether one whole-line comment is prose, rather than a directive or hand-aligned text.

    `text` carries its own leading `#`, which `DIRECTIVE_COMMENT` and `URL_ONLY_COMMENT` are
    anchored on. `NOT_PROSE` is searched in what follows the marker instead, because it matches a
    bare `#` too and would otherwise match the marker on every comment line ever written.

    Args:
        text: The comment line with its indent stripped, `#` included.

    Returns:
        True when re-wrapping the line can move only prose.
    """
    if DIRECTIVE_COMMENT.match(text) or URL_ONLY_COMMENT.match(text):
        return False
    return not NOT_PROSE.search(text[1:])


def _prefix_of(*, line: str, is_comment: bool) -> str:
    """The indent, and the `#` marker where there is one, that every line of a block repeats.

    Args:
        line: The line opening the block.
        is_comment: True when `tokenize` says the line is a whole-line comment.

    Returns:
        The prefix `_wrap` re-adds to each output line.
    """
    marker = COMMENT_MARKER.match(line) if is_comment else None
    return marker.group(1) if marker else line[: len(line) - len(line.lstrip())]


def _is_wrappable_prose(*, line: str, prefix: str, is_comment: bool) -> bool:
    """Whether re-wrapping `line` can move only prose.

    Every line a paragraph grows onto has to pass this, not only the long line that started it: a
    `# noqa: E501` welded into the paragraph above it stops suppressing the line it was written
    for, a hand-aligned diagram collapses to single spaces and points at nothing, and a heading or
    a table row loses the line break that makes it one.

    Args:
        line: The line to test.
        prefix: The block's prefix, which `_is_unwrappable` must not see.
        is_comment: True when `tokenize` says the line is a whole-line comment.

    Returns:
        True when the line is prose end to end.
    """
    if is_comment:
        return _is_prose_comment(line.strip())
    body = line[len(prefix) :]
    # `_is_unwrappable` has to see the text without its indent. A docstring inside a function body
    # is indented four spaces or more, which markdown reads as a code block, so testing the raw
    # line marks every deeply-indented line unwrappable and nothing gets fixed.
    return not _is_unwrappable(body) and not INTERIOR_DOUBLE_SPACE.search(body)


def _rewrap(*, block: Sequence[str], prefix: str) -> list[str]:
    """Re-flow one paragraph to `WIDTH`, keeping its prefix and any list marker's hanging indent.

    Args:
        block: The lines of the paragraph, prefix included.
        prefix: The indent and comment marker every line repeats.

    Returns:
        The re-flowed lines.
    """
    body = [line[len(prefix) :] for line in block]
    marker = LIST_MARKER.match(body[0])
    hanging = " " * len(marker.group(1)) if marker else ""
    # A paragraph carrying a bare URL is re-wrapped like any other: `_wrap` puts a word longer than
    # the width on a line of its own, which is where this repo already writes its URLs, and `E501`
    # does not flag a line whose overflow is one unbreakable token.
    return _wrap(
        _tokenise(" ".join(body)),
        initial_indent=prefix,
        subsequent_indent=f"{prefix}{hanging}",
    )


def wrap_overlong(path: Path) -> int:
    """Re-wrap every prose line left over `markdown_wrap.WIDTH` in `path`, and return how many.

    `reflow_python_prose.py` hands each docstring to `markdown_wrap.reflow_text`, which reads a
    Google-style ``Args:`` or ``Returns:`` body as an indented code block and therefore never
    re-flows it, and it declines a docstring carrying hand alignment anywhere in it. A splice
    landing in one of those leaves a single long line that no other tool will wrap and that
    `ruff`'s `E501` then rejects. Wrapping runs forward from the long line to the end of its
    paragraph, so the lines above the splice keep the wrapping they had.

    Args:
        path: The Python file to wrap.

    Returns:
        How many paragraphs were re-wrapped.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")
    prose = wrappable_lines(source)
    comments = comment_lines(source)
    samples = code_sample_lines(source)

    def usable(number: int) -> bool:
        return number in prose and number not in samples

    wrapped = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        prefix = _prefix_of(line=line, is_comment=index in comments)
        if (
            len(line) <= WIDTH
            or not usable(index)
            or not _is_wrappable_prose(line=line, prefix=prefix, is_comment=index in comments)
        ):
            index += 1
            continue
        block = [line]
        while index + len(block) < len(lines):
            follower = index + len(block)
            nxt = lines[follower]
            if not nxt.strip() or not nxt.startswith(prefix) or not usable(follower):
                break
            if not _is_wrappable_prose(
                line=nxt, prefix=prefix, is_comment=follower in comments
            ) or LIST_MARKER.match(nxt[len(prefix) :]):
                break
            block.append(nxt)
        # Re-wrapping a paragraph that was already correct reproduces it exactly, so the count
        # below tracks the paragraphs that really changed rather than the ones merely looked at.
        rewrapped = _rewrap(block=block, prefix=prefix)
        if rewrapped != block:
            wrapped += 1
        lines[index : index + len(block)] = rewrapped
        index += len(rewrapped)

    rewritten = "\n".join(lines)
    # A re-wrap that welds the closing quotes into a paragraph leaves a file that still looks
    # plausible and no longer parses, so the write is gated on the result parsing.
    ast.parse(rewritten)
    if rewritten != source:
        path.write_text(rewritten, encoding="utf-8")
    return wrapped


def reflow(paths: Sequence[Path]) -> None:
    """Re-wrap the prose of every path given, using the repo's own re-wrapper and then this one.

    Args:
        paths: The files that were edited.
    """
    for path in paths:
        source = path.read_text(encoding="utf-8")
        rewritten = reflow_python_prose(source)
        if rewritten != source:
            path.write_text(rewritten, encoding="utf-8")
            print(f"{path}: reflowed")
        wrapped = wrap_overlong(path)
        if wrapped:
            print(f"{path}: wrapped {wrapped} over-long line(s) the reflow could not reach")


def invalid_findings(findings: object) -> list[str]:
    """Every reason the loaded JSON is not a batch this script can apply.

    A finding missing its `quote` used to reach `apply_one` and die there with a bare `KeyError`,
    taking every edit the batch had already made with it.

    Args:
        findings: Whatever `json.loads` returned.

    Returns:
        One sentence per problem, empty when the batch is well formed.
    """
    if not isinstance(findings, list):
        return ["the file must hold a list of findings"]
    problems: list[str] = []
    for number, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict):
            problems.append(f"finding {number} is not an object")
            continue
        missing = [key for key in FINDING_KEYS if not isinstance(finding.get(key), str)]
        if missing:
            problems.append(f"finding {number} is missing a string {' and '.join(missing)}")
    return problems


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Read the command line, settling `--help`, a valueless flag and an unknown flag alike.

    Args:
        argv: The arguments after the program name.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Apply a sentence sweep's findings to Python docstrings and comments."
    )
    parser.add_argument("findings", type=Path, help="a JSON list of findings to apply")
    parser.add_argument(
        "--apply", action="store_true", help="write the edits; the default is a dry run"
    )
    parser.add_argument(
        "--merge-base", help="skip a finding whose sentence already exists at this revision"
    )
    return parser.parse_args(list(argv))


def main() -> None:
    """Apply every finding in the file named on the command line, reporting each one's status."""
    arguments = _parse_arguments(sys.argv[1:])
    findings = json.loads(arguments.findings.read_text(encoding="utf-8"))
    problems = invalid_findings(findings)
    if problems:
        sys.exit("\n".join([f"FAIL - {arguments.findings} is not a batch of findings:", *problems]))

    tally: dict[StatusType, int] = {}
    edited: dict[Path, str] = {}
    for finding in findings:
        path = Path(finding["file"])
        if path.suffix != ".py":
            tally["not python"] = tally.get("not python", 0) + 1
            print(f"{'NOT PYTHON':>18}  {finding['file']}  {finding['quote'][:64]}")
            continue
        raw = edited.get(path, path.read_text(encoding="utf-8"))
        updated, status = apply_one(raw=raw, finding=finding, merge_base=arguments.merge_base)
        tally[status] = tally.get(status, 0) + 1
        if status == "applied":
            edited[path] = updated
        else:
            print(f"{status.upper():>18}  {finding['file']}  {finding['quote'][:64]}")

    if arguments.apply:
        for path, updated in edited.items():
            path.write_text(updated, encoding="utf-8")
        reflow(sorted(edited))

    print()
    for status, count in sorted(tally.items()):
        print(f"{count:>5}  {status}")
    if not arguments.apply:
        print("\ndry run - nothing written; pass --apply to write")
    elif edited:
        print(f"\nwrote {len(edited)} file(s); run check_prose_only.py next")
    if tally.keys() - APPLICABLE:
        sys.exit("\nFAIL - some findings could not be applied; fix or hand-edit them")


if __name__ == "__main__":
    main()
