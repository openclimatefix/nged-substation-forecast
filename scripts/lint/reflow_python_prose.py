"""Rewrap docstring and prose-comment text in Python source files to `markdown_wrap.WIDTH`.

`#690`/`#762` hard-wrapped every `.md` file's prose to `markdown_wrap.WIDTH`; this script applies
the same wrapping to the prose inside `.py` files, which the markdown reflow never touched. Two
kinds of text are in scope:

- **Docstrings** — a module, class, function or async-function docstring, found via `ast` the same
  way `lint_docstring_markdown.py` finds them. `mkdocstrings` renders these as markdown, so they
  reflow with `markdown_wrap.reflow_text`, the same engine that reflows a `.md` file.
- **Prose comment blocks** — a run of two or more consecutive whole-line `#` comments at the same
  indent, wrapped with the same greedy word-wrap `markdown_wrap._wrap` uses. A comment carrying a
  linter directive (`noqa`, `type:`, `ty:`), a shebang, or a single trailing inline comment is left
  alone, since none of those is prose.

Neither kind of text is touched by `ruff format`, which reformats code but leaves comments and
string literals as the author wrote them (see the `E5` entry in `pyproject.toml`'s `select` list) —
this script exists because nothing else in the toolchain does this reflow.

A docstring or comment block is reflowed at `WIDTH` minus its own indentation, since the indent is
real characters on the source line and counts against the 100-character budget the same as the
prose does. Reflowing operates on the dedented text so `markdown_wrap.reflow_text`'s own units
(paragraphs, list items, fenced code) still work the way they do for a `.md` file; the result is
re-indented and the original closing-quote style (same line as the last word, or its own line) is
preserved.

Usage::

    python3 scripts/lint/reflow_python_prose.py <file.py> [<file.py> ...]
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path
from typing import Final

import markdown_wrap
from markdown_wrap import WIDTH, reflow_text

DOCSTRING_OPEN: Final[re.Pattern[str]] = re.compile(r'^(?P<prefix>[a-zA-Z]*)(?P<quote>"""|\'\'\')')
"""The prefix (`r`, `f`, ... or none) and triple-quote a docstring literal opens with."""

DIRECTIVE_COMMENT: Final[re.Pattern[str]] = re.compile(
    r"^#!|^#\s*(noqa\b|type:|ty:|-\*-)", re.IGNORECASE
)
"""A comment that is a directive, not prose, and so must be left exactly as written."""

URL_ONLY_COMMENT: Final[re.Pattern[str]] = re.compile(r"^#\s*\S+://\S+\s*$")
"""A comment whose entire content is one URL — nothing to wrap, and wrapping would break it."""

INTERIOR_DOUBLE_SPACE: Final[re.Pattern[str]] = re.compile(r"\S {2,}\S")
"""A run of 2+ spaces between two non-space characters — hand alignment, not prose.

`test_features.py`'s `NWP run A: ... 06:00  (init 00:00 + 6h delay)` pads before a parenthetical to
line up with the sibling `NWP run B:` line below it; collapsing that padding to a single space the
way ordinary prose is collapsed destroys the alignment. Requiring a non-space character on both
sides excludes a line's own leading indent, which is real structure (a nested list item, a
blockquote) that `reflow_text` already understands and must still be free to reflow.
"""

NOT_PROSE: Final[re.Pattern[str]] = re.compile(r"#|  ")
"""A `#` or an interior run of 2+ spaces inside a comment's text, neither of which plain prose has.

A `#` here is a second, embedded directive — `storage.py`'s aligned-key comment carries a `# noqa:
E501` after two spaces of padding, not at the line's own start, so `DIRECTIVE_COMMENT` alone (which
only matches at the start of the text) does not catch it. An interior double space is the same
comment's arrow diagram, hand-aligned to point at three substrings of a sample key — the alignment
is the content, and collapsing it to a single space the way ordinary prose is collapsed would
destroy it.
"""

COMMENT_MARKER: Final[re.Pattern[str]] = re.compile(r"^[ \t]*#[ \t]?")
"""A whole-line `#` comment's marker, for `_flatten` to strip before comparing two texts' words.

A comment block's `#` repeats once per physical line, the same way a blockquote's `>` does in
`markdown_wrap._flatten` — rewrapping a block to a different line count legitimately changes how
many `#` markers appear, so comparing the raw text (marker and all) before and after reflowing would
flag every successful comment rewrap as a corruption.
"""


def _dedent_lines(lines: list[str]) -> tuple[list[str], int]:
    """Dedent every line in `lines` by their common leading-whitespace margin.

    `lines[0]` is assumed already unindented (the text right after an opening `\"\"\"`, or the
    first line of a comment block after its `#` marker is stripped) and is excluded from the
    margin calculation, mirroring `lint_docstring_markdown._dedent_docstring`.
    """
    margin = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            margin = min(margin, len(line) - len(stripped))
    if margin == sys.maxsize:
        margin = 0
    return [lines[0], *(line[margin:] for line in lines[1:])], margin


def _reflow_indented(body_lines: list[str], margin: int, *, extra_indent: int = 0) -> list[str]:
    """Reflow `body_lines` (already dedented by `margin`) at `WIDTH - margin - extra_indent`.

    Temporarily narrows `markdown_wrap.WIDTH` for the call, since `reflow_text` always wraps to
    the module-level `WIDTH` and has no width parameter of its own — the indent this function
    re-adds afterwards, plus any further prefix the caller re-adds itself (`extra_indent` — a
    comment's `#` marker, the space after it, and its own indent, which `_reflow_comment_block`
    adds after this returns), has to fit inside the same 100-character budget as the text.
    """
    original_width = markdown_wrap.WIDTH
    markdown_wrap.WIDTH = WIDTH - margin - extra_indent
    try:
        reflowed = reflow_text("\n".join(body_lines) + "\n")
    finally:
        markdown_wrap.WIDTH = original_width
    reflowed_lines = reflowed[:-1].split("\n")  # drop the newline `reflow_text` requires as input
    margin_str = " " * margin
    rest = (f"{margin_str}{line}" if line else line for line in reflowed_lines[1:])
    return [reflowed_lines[0], *rest]


def _reflow_docstring(content: str, quote_width: int) -> str:
    """Reflow one docstring's raw (unevaluated) text between its quotes, preserving its style.

    `content` is the exact source text between the opening and closing triple quotes — not the
    escaped-and-evaluated string value, since this is a text edit to the source, not to the
    string the source produces. Preserves whether the closing quotes sit on their own line or
    trail the last word, matching whichever style the docstring already used.

    `quote_width` (`len(prefix) + len(quote)`, 3 for a plain triple-double-quote) is spent as
    `extra_indent`, to hold the opening quote's own budget for whichever line the first paragraph
    wraps onto — not only line one, since a docstring with no blank line after its summary (every
    docstring in a `**/tests/**` file, which `pydocstyle`'s `D205` does not run over) reflows the
    summary into the same paragraph as the body.

    A docstring with no blank line anywhere in its body — the same `D205`-exempt case, since a
    D205-compliant docstring always has one after the summary — reserves `quote_width` *twice*
    instead: its whole body is one paragraph, short enough that `ruff format` may later collapse
    it onto a single physical line carrying both quotes, independently of how this function
    packed the text, and the closing quote's width has to be held for that line too or the
    collapsed line overflows `WIDTH`. Reserving it on every docstring, not only this one, would
    push a `D205`-compliant summary that already fits `WIDTH - margin - quote_width` over the
    narrower budget and wrap it onto two lines — which `pydocstyle` then reads as a missing blank
    line, because `D205` requires the summary to be the entire first physical line.

    Leaves the docstring untouched if its dedented body carries a line indented 4 or more columns
    (beyond the first line, which is never indented): `markdown_wrap.reflow_text` understands a
    CommonMark list item or blockquote, but a Google-convention `Args:`/`Returns:`/`Raises:`
    section — 44 files in this repo have one — is neither; its `name: description` entries carry
    no marker `reflow_text` recognises, so it reads the whole section as one paragraph and merges
    every parameter into a single run-on line. The same threshold also catches a reST `::`
    literal block (an indented shell command in a docstring elsewhere in this repo), which
    `reflow_text` has no fenced-code exemption for either. A plain paragraph that happens to
    carry 1-3 columns of leftover indentation from a previous hand-wrap is common and harmless to
    reflow normally, which is why the threshold is 4, not 1.

    Also leaves the docstring untouched if any line carries an `INTERIOR_DOUBLE_SPACE` — the same
    hand-alignment signal `_is_prose_comment` already exempts a comment block for.
    """
    if "\n" not in content:
        return content  # single-line docstring: already one paragraph, nothing to reflow

    lines = content.expandtabs().split("\n")
    dedented, margin = _dedent_lines([lines[0].lstrip(), *lines[1:]])

    if any(len(line) - len(line.lstrip()) >= 4 for line in dedented[1:] if line.strip()):
        return content
    if any(INTERIOR_DOUBLE_SPACE.search(line) for line in dedented):
        return content

    own_line_close = dedented[-1].strip() == ""
    closing_extra = dedented[-1] if own_line_close else None
    body = dedented[:-1] if own_line_close else dedented

    has_internal_blank = any(not line.strip() for line in body)
    extra_indent = quote_width if has_internal_blank else quote_width * 2
    new_lines = _reflow_indented(body, margin, extra_indent=extra_indent)
    if own_line_close:
        new_lines.append(f"{' ' * margin}{closing_extra}")
    return "\n".join(new_lines)


def _char_col(line: str, byte_col: int) -> int:
    """Convert a UTF-8 *byte* column offset into a character index into `line`.

    `ast` reports a byte offset for `col_offset`/`end_col_offset` on a line with non-ASCII text,
    so this is what `line[byte_col:]`-style slicing needs to land on the right character instead.
    `ast.col_offset`/`end_col_offset` are documented as UTF-8 byte offsets, not character
    offsets; this repo's prose deliberately keeps en-dashes and typographic quotes
    (`RUF001`-`RUF003` are disabled for exactly that reason), so a docstring spanning a line with
    one of those needs this conversion or its splice point lands mid-character, or short by
    however many extra bytes those characters cost.
    """
    return len(line.encode("utf-8")[:byte_col].decode("utf-8"))


def _reflow_docstrings_in(source: str, tree: ast.Module) -> str:
    """Reflow every module/class/function/async-function docstring in `source`.

    Collects each docstring's absolute character span from the original `source` before rewriting
    any of them, then splices in reverse order (highest offset first) so an earlier edit's
    offsets stay valid while a later one is applied — the same reason `implement-issue`-style
    multi-edit scripts always work back to front.
    """
    lines = source.splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    nodes: list[ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = [
        tree,
        *(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        ),
    ]
    edits: list[tuple[int, int, str]] = []
    for node in nodes:
        if ast.get_docstring(node, clean=False) is None:
            continue
        stmt = node.body[0]
        segment = ast.get_source_segment(source, stmt)
        if segment is None:
            continue
        match = DOCSTRING_OPEN.match(segment)
        if match is None:
            continue  # not a triple-quoted literal (e.g. a bare `'x'`) — leave it alone
        prefix, quote = match.group("prefix"), match.group("quote")
        content = segment[len(prefix) + len(quote) : -len(quote)]
        new_content = _reflow_docstring(content, len(prefix) + len(quote))
        if new_content == content:
            continue
        # A parsed `ast.Expr` statement always carries concrete end position info; the `| None` in
        # the stub is for a node built by hand rather than by `ast.parse`.
        assert stmt.end_lineno is not None
        assert stmt.end_col_offset is not None
        start_char_col = _char_col(lines[stmt.lineno - 1], stmt.col_offset)
        end_char_col = _char_col(lines[stmt.end_lineno - 1], stmt.end_col_offset)
        abs_start = line_starts[stmt.lineno - 1] + start_char_col
        abs_end = line_starts[stmt.end_lineno - 1] + end_char_col
        edits.append((abs_start, abs_end, f"{prefix}{quote}{new_content}{quote}"))

    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _is_prose_comment(text: str) -> bool:
    """Whether a standalone `#` comment line is prose, rather than a directive or non-prose line.

    `text` still carries its own leading `#`, which `DIRECTIVE_COMMENT`/`URL_ONLY_COMMENT` need
    anchored there. `NOT_PROSE` is checked against `text[1:]` instead, because it also matches a
    bare `#` — searching the full `text` would match that leading marker on every single-`#`
    comment line and always report `False`, which is exactly the bug this slice fixes.
    """
    if DIRECTIVE_COMMENT.match(text) or URL_ONLY_COMMENT.match(text):
        return False
    return not NOT_PROSE.search(text[1:])


def _comment_blocks(source: str) -> list[tuple[int, int, int]]:
    """The `(first_line, last_line, indent)` of every run of 2+ consecutive whole-line `#` comments.

    A whole-line comment is one where nothing but whitespace precedes the `#` on its physical line
    — a trailing inline comment (`x = 1  # units: MW`) is a single line by construction and so is
    never part of a 2+-line run, and is left untouched without needing a separate check for it.
    Line numbers are 1-indexed, matching `tokenize`.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return []

    standalone = [
        (tok.start[0], tok.start[1], tok.line)
        for tok in tokens
        if tok.type == tokenize.COMMENT and tok.line[: tok.start[1]].strip() == ""
    ]

    blocks: list[tuple[int, int, int]] = []
    run_start: int | None = None
    run_indent = 0
    prev_line = -2
    for lineno, indent, _line in [*standalone, (-1, 0, "")]:
        if run_start is not None and (lineno != prev_line + 1 or indent != run_indent):
            if prev_line - run_start >= 1:
                blocks.append((run_start, prev_line, run_indent))
            run_start = None
        if lineno == -1:
            break
        if run_start is None:
            run_start, run_indent = lineno, indent
        prev_line = lineno
    return blocks


def _reflow_comment_block(lines: list[str], indent: int) -> list[str] | None:
    """Reflow one comment block's text, or `None` if any line in it is not prose.

    `lines` are the raw source lines (with their `#` marker and indent still attached) making up
    the block. Every line must carry a `#` followed by either nothing or exactly one space before
    its text, which is the convention `ruff format` already enforces on every comment in this
    repo.
    """
    texts: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n")[indent:]
        if not _is_prose_comment(stripped):
            return None
        body = stripped[1:]  # drop the `#`
        texts.append(body.removeprefix(" "))

    dedented, margin = _dedent_lines(texts)
    # `# ` costs 2 columns per line on top of the block's own indent, on every line including the
    # first — a comment has no docstring-style "unindented summary line" exemption.
    new_lines = _reflow_indented(dedented, margin, extra_indent=indent + 2)
    comment_indent = " " * indent
    return [f"{comment_indent}#{' ' + line if line else ''}\n" for line in new_lines]


def _reflow_comments_in(source: str) -> str:
    """Reflow every genuine multi-line prose comment block in `source`."""
    lines = source.splitlines(keepends=True)
    for first, last, indent in sorted(_comment_blocks(source), reverse=True):
        block = lines[first - 1 : last]
        new_block = _reflow_comment_block(block, indent)
        if new_block is None or new_block == block:
            continue
        lines[first - 1 : last] = new_block
    return "".join(lines)


def _flatten(text: str) -> str:
    """`text` with every whole-line `#` marker and all remaining whitespace stripped.

    Mirrors `markdown_wrap._flatten`'s treatment of a blockquote's `>`: stripping the marker as
    well as whitespace is what lets `reflow_python_prose`'s round-trip check tell a legitimate
    rewrap (which changes how many `#` markers a block prints) from a change to the words
    themselves.
    """
    lines = [COMMENT_MARKER.sub("", line) for line in text.split("\n")]
    return "".join("\n".join(lines).split())


def reflow_python_prose(source: str) -> str:
    """Reflow every docstring and prose comment block in `source` to `WIDTH`, leaving code as-is.

    Safety net: `source` and the result must carry the same words once every `#` comment marker
    and all whitespace is stripped from both — the same round-trip check
    `markdown_wrap.reflow_text` makes on a whole `.md` file, so a bug in this script can't
    silently drop or duplicate a word.
    """
    tree = ast.parse(source)
    with_docstrings = _reflow_docstrings_in(source, tree)
    result = _reflow_comments_in(with_docstrings)
    assert _flatten(result) == _flatten(source), "reflow changed the text, not just the wrapping"
    return result


def main(argv: list[str]) -> int:
    """Rewrap each `.py` file named in `argv`, in place; return the process exit code."""
    changed = 0
    for arg in argv:
        path = Path(arg)
        source = path.read_text()
        rewritten = reflow_python_prose(source)
        if rewritten != source:
            path.write_text(rewritten)
            changed += 1
            print(f"{path}: reflowed")
    print(f"{changed}/{len(argv)} files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
