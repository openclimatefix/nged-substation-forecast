"""Apply a sentence sweep's findings to the docstrings and comments of hard-wrapped Python files.

`apply_findings.py` is the markdown sibling of this script and cannot be used here: its re-wrap
works in markdown units and would strip the indentation a Python file depends on, and its
projection knows nothing about the `#` that opens every line of a comment block. The two scripts
share a shape and no code.

Every guard below exists because the alternative silently damages a file that then passes `ruff`,
`ty` and `pytest`:

- **A sub-agent reports the sentence on one line, and the file wraps it across several.** Matching
  runs against a projection of each prose unit whose whitespace is collapsed to single spaces, and
  which swallows the `#` and indent that open a comment's continuation line. The projection keeps an
  offset map back into the raw file, so the splice lands on real characters.
- **The agent quotes the sentence with the markdown stripped**, because a docstring is markdown and
  `` `write_nwp` `` comes back as `write_nwp`. A second projection drops the backticks, the `**`
  markers and the address half of a link, and is tried when the plain projection does not match.
- **Splicing the replacement in whole would delete every backtick the sentence carried**, since the
  replacement is written without markup too. Only the run that actually differs between the quote
  and the replacement is spliced, snapped out to whole words, so markup on either side survives.
- **A span reaching across a blank line would weld two paragraphs together**, because the
  replacement is a single line. Such a finding is refused rather than applied.
- **A quote can match in more than one place**, and guessing which one was meant is how a sweep
  rewrites the wrong sentence. Two matches is a refusal.
- **An edit outside a docstring or a comment is not a prose edit at all.** Only the prose units
  found by `ast` and `tokenize` are searched, so a quote whose words also appear in a runtime string
  or an identifier cannot reach them.
- **The splice leaves a line far over the 100-character limit**, so every edited file is re-wrapped
  by `scripts/reflow_python_prose.py`, which is the same engine that wrapped the file originally.
- **A splice can close a string early or comment out a line of code**, so each edited file is parsed
  afterwards and its syntax tree compared, with the string constants blanked, against the tree
  before the edit. `check_prose_only.py` holds that comparison and runs it again over the branch.

Usage::

    python3 apply_findings_py.py findings.json
    python3 apply_findings_py.py findings.json --apply
    python3 apply_findings_py.py findings.json --apply --merge-base 7657db1c

`findings.json` holds a list of objects, each with `file`, `quote` and `replacement`; any other key
is ignored, so a sweep's own `rule` and `why` fields can stay in the file. Nothing is written
without `--apply`. With `--merge-base`, a finding whose sentence already exists at that revision is
prose the branch did not write, so it is skipped and listed rather than applied. Findings naming a
file that is not Python are reported as `not python` and left for `apply_findings.py`.
"""

import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.markdown_wrap import WIDTH, _is_unwrappable, _tokenise, _wrap

REFLOW: Final[Path] = Path("scripts/reflow_python_prose.py")
"""The repo's own Python prose re-wrapper, re-run over every file this script edits."""

LINK: Final[re.Pattern[str]] = re.compile(r"\[([^\]]*)\]\([^)]*\)", re.DOTALL)
"""A markdown inline link, whose visible text survives the markup-stripped projection."""

BLANK_LINE: Final[re.Pattern[str]] = re.compile(r"\n[ \t]*#?[ \t]*\n")
"""A paragraph break inside a docstring or a comment block, which no splice may cross."""

NOT_PROSE: Final[re.Pattern[str]] = re.compile(r"#|  ")
"""A `#` or an interior run of two or more spaces in a comment's text — not plain prose.

Both mean the line is carrying something a re-wrap would destroy, and `reflow_python_prose.py`
declines the same two for the same reason. A `#` after the marker is a second, embedded
directive: `storage.py`'s sample-key comment ends `# noqa: E501`, and wrapping the line moves
that directive off the line it was suppressing, so the suppression stops applying. An interior
double space is hand alignment — that same comment's arrow diagram points at three substrings of
a sample key, and collapsing the padding to single spaces makes the arrows point at nothing.
"""

MissType = Literal["no match", "ambiguous"]
"""Why a quote could not be located in exactly one place."""

StatusType = Literal[
    "applied", "pre-existing", "crosses blank line", "not python", "no-op", "no match", "ambiguous"
]
"""What became of one finding."""


class Finding(TypedDict):
    """One reported sentence and the sentence proposed in its place."""

    file: str
    quote: str
    replacement: str


class _Unit(TypedDict):
    """One run of prose in a Python file, and where it sits in the raw text."""

    start: int
    end: int
    is_comment: bool


def prose_units(source: str) -> list[_Unit]:
    """Return the raw bounds of every docstring and comment block in `source`.

    A docstring contributes the span between its quotes. Consecutive whole-line `#` comments at the
    same indent contribute one span between them, so a sentence wrapped across three comment lines
    is one searchable run; a trailing comment on a line of code is its own span.

    Args:
        source: The full text of a Python file.

    Returns:
        The units, in the order they appear.
    """
    units: list[_Unit] = []
    lines = source.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    for node in ast.walk(ast.parse(source)):
        is_string_statement = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_string_statement and isinstance(node.value.value, str)):
            continue
        if node.end_lineno is None or node.end_col_offset is None:
            continue
        literal_start = starts[node.lineno - 1] + node.col_offset
        literal_end = starts[node.end_lineno - 1] + node.end_col_offset
        literal = source[literal_start:literal_end]
        quote = '"""' if literal.count('"""') >= 2 else ("'''" if literal.count("'''") >= 2 else "")
        opening = len(literal) - len(literal.lstrip("rbfuRBFU"))
        opening += len(quote) if quote else 1
        closing = len(quote) if quote else 1
        units.append(
            {"start": literal_start + opening, "end": literal_end - closing, "is_comment": False}
        )

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
            units.append({"start": run[0], "end": run[1], "is_comment": True})
            run = None
        if whole_line:
            run, run_indent, run_line = (start, end), token.start[1], token.start[0]
        else:
            units.append({"start": start, "end": end, "is_comment": True})
    if run is not None:
        units.append({"start": run[0], "end": run[1], "is_comment": True})
    return units


def project(
    *, raw: str, start: int, end: int, is_comment: bool, strip_markup: bool
) -> tuple[str, list[int]]:
    """Return one prose unit as a single line, with a map from each character back into `raw`.

    Args:
        raw: The full text of the file.
        start: Where the unit begins in `raw`.
        end: Where the unit ends in `raw`.
        is_comment: True for a comment block, whose continuation lines open with `#`.
        strip_markup: True to drop backticks, `**` markers and link addresses as well.

    Returns:
        The projected text, and a list holding the raw offset of each projected character.
    """
    out: list[str] = []
    index: list[int] = []
    position = start
    while position < end:
        character = raw[position]
        if character in " \t\n":
            ahead = position
            saw_newline = False
            while ahead < end and raw[ahead] in " \t\n":
                saw_newline = saw_newline or raw[ahead] == "\n"
                ahead += 1
            if is_comment and saw_newline and ahead < end and raw[ahead] == "#":
                ahead += 1
                while ahead < end and raw[ahead] in " \t":
                    ahead += 1
            if out and out[-1] != " ":
                out.append(" ")
                index.append(position)
            position = ahead
            continue
        if strip_markup:
            if raw.startswith("**", position):
                position += 2
                continue
            if character == "`":
                position += 1
                continue
            match = LINK.match(raw, position, end)
            if match:
                for inner in range(match.start(1), match.end(1)):
                    if raw[inner] not in " \t\n`*":
                        out.append(raw[inner])
                        index.append(inner)
                position = match.end()
                continue
        out.append(character)
        index.append(position)
        position += 1
    return "".join(out), index


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


def changed_run(*, quote: str, replacement: str) -> tuple[int, int, str]:
    """Return the span of `quote` that differs from `replacement`, snapped out to whole words.

    A finding usually rewrites a few words inside a sentence carrying backticks and links on either
    side. Splicing only the run that differs is what leaves that markup where the author put it.

    Args:
        quote: The sentence as the file reads it, projected to one line.
        replacement: The sentence proposed in its place.

    Returns:
        The start and end of the differing run within `quote`, and its replacement text.
    """
    head = 0
    while head < len(quote) and head < len(replacement) and quote[head] == replacement[head]:
        head += 1
    tail = 0
    while (
        tail < len(quote) - head
        and tail < len(replacement) - head
        and quote[len(quote) - 1 - tail] == replacement[len(replacement) - 1 - tail]
    ):
        tail += 1
    head = quote.rfind(" ", 0, head) + 1 if head < len(quote) else head
    stop = len(quote) - tail
    space = quote.find(" ", stop)
    stop = len(quote) if space == -1 else space
    return head, stop, replacement[head : len(replacement) - (len(quote) - stop)]


def _existing_at(*, merge_base: str, path: Path) -> str | None:
    """Return `path` at `merge_base`, projected to one line, or None where it did not exist.

    Args:
        merge_base: The git revision that predates the branch's prose.
        path: The file to read.

    Returns:
        The old text with whitespace and comment markers collapsed, or None.
    """
    completed = subprocess.run(
        ["git", "show", f"{merge_base}:{path}"], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        return None
    flat = normalise(completed.stdout)
    return flat.replace("`", "").replace("**", "")


def locate(*, raw: str, quote: str) -> tuple[int, int, str, list[int]] | MissType:
    """Find `quote` among the prose units of `raw`, or return why it could not be found.

    Args:
        raw: The full text of a Python file.
        quote: The sentence to find, projected to one line.

    Returns:
        The unit's projected text, its offset map, and the match's bounds within the projection;
        or a status string naming the reason no unique match was found.
    """
    for strip_markup in (False, True):
        hits: list[tuple[int, int, str, list[int]]] = []
        for unit in prose_units(raw):
            text, index = project(
                raw=raw,
                start=unit["start"],
                end=unit["end"],
                is_comment=unit["is_comment"],
                strip_markup=strip_markup,
            )
            needle = quote.replace("`", "").replace("**", "") if strip_markup else quote
            position = text.find(needle)
            while position != -1:
                hits.append((position, position + len(needle), text, index))
                position = text.find(needle, position + 1)
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return "ambiguous"
    return "no match"


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
        old = _existing_at(merge_base=merge_base, path=Path(finding["file"]))
        if old is not None and quote.replace("`", "").replace("**", "") in old:
            return raw, "pre-existing"

    found = locate(raw=raw, quote=quote)
    if isinstance(found, str):
        return raw, found
    start, stop, text, index = found

    head, tail, middle = changed_run(quote=text[start:stop], replacement=replacement)
    raw_start = index[start + head]
    raw_stop = index[start + tail - 1] + 1 if start + tail <= len(index) else index[-1] + 1
    if BLANK_LINE.search(raw[raw_start:raw_stop]):
        return raw, "crosses blank line"
    return raw[:raw_start] + middle + raw[raw_stop:], "applied"


def wrappable_lines(source: str) -> set[int]:
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
    wrappable: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        is_string = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_string and isinstance(node.value.value, str)):
            continue
        if node.end_lineno is None:
            continue
        wrappable.update(range(node.lineno, node.end_lineno - 1))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and token.line[: token.start[1]].strip() == "":
            wrappable.add(token.start[0] - 1)
    return wrappable


def wrap_overlong(path: Path) -> int:
    """Re-wrap every prose line left over `markdown_wrap.WIDTH` in `path`, and return how many.

    `scripts/reflow_python_prose.py` hands each docstring to `markdown_wrap.reflow_text`, which
    reads a Google-style ``Args:`` or ``Returns:`` body as an indented code block and therefore
    never re-flows it. A splice landing in one of those sections leaves a single long line that no
    other tool will wrap and that `ruff`'s `E501` then rejects. Wrapping runs forward from the long
    line to the end of its paragraph, so the lines above the splice keep the wrapping they had.

    A line whose overflow is one unbreakable token — a bare URL, which this repo writes in full —
    is left alone, because `E501` does not flag it either.

    Args:
        path: The Python file to wrap.

    Returns:
        How many lines were re-wrapped.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    prose_lines = wrappable_lines(source)

    def in_prose(index: int) -> bool:
        return index in prose_lines

    wrapped = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if len(line) <= WIDTH or not in_prose(index):
            index += 1
            continue
        indent = line[: len(line) - len(line.lstrip())]
        marker = "# " if line.lstrip().startswith("#") else ""
        prefix = f"{indent}{marker}"
        # `_is_unwrappable` has to see the text without its prefix. A docstring or a comment inside
        # a function body is indented four spaces or more, which markdown reads as a code block, so
        # testing the raw line marks every deeply-indented line unwrappable and nothing gets fixed.
        if _is_unwrappable(line[len(prefix) :]):
            index += 1
            continue
        if marker and NOT_PROSE.search(line[len(prefix) :]):
            index += 1
            continue
        block = [line]
        while index + len(block) < len(lines):
            nxt_index = index + len(block)
            nxt = lines[nxt_index]
            # `in_prose` has to be re-tested on every line the paragraph grows onto. Testing only
            # the first line lets the paragraph run off the end of a docstring, and the re-wrap
            # then pulls the closing quotes up into the prose and the code below into the string.
            if not nxt.strip() or not nxt.startswith(prefix):
                break
            if _is_unwrappable(nxt[len(prefix) :]):
                break
            if not in_prose(nxt_index):
                break
            block.append(nxt)
        words = _tokenise(" ".join(item[len(prefix) :] for item in block))
        # A paragraph carrying a bare URL is re-wrapped like any other: `_wrap` puts a word longer
        # than the width on a line of its own, which is where this repo already writes its URLs,
        # and `E501` does not flag a line whose overflow is one unbreakable token. Re-wrapping a
        # paragraph that was already correct reproduces it exactly, so the count below tracks the
        # paragraphs that really changed rather than the ones that were merely looked at.
        rewrapped = _wrap(words, initial_indent=prefix, subsequent_indent=prefix)
        if rewrapped != block:
            wrapped += 1
        lines[index : index + len(block)] = rewrapped
        index += len(rewrapped)

    rewritten = "\n".join(lines) + "\n"
    # A re-wrap that welds the closing quotes into a paragraph leaves a file that still looks
    # plausible and no longer parses, so the write is gated on the result parsing.
    ast.parse(rewritten)
    path.write_text(rewritten, encoding="utf-8")
    return wrapped


def reflow(paths: Sequence[Path]) -> None:
    """Re-wrap the prose of every path given, using the repo's own re-wrapper.

    Args:
        paths: The files that were edited.
    """
    if not paths:
        return
    subprocess.run(
        [sys.executable, str(REFLOW), *[str(path) for path in paths]],
        check=True,
        env={"PYTHONPATH": "scripts"},
    )
    for path in paths:
        wrapped = wrap_overlong(path)
        if wrapped:
            print(f"{path}: wrapped {wrapped} over-long line(s) the reflow could not reach")


def main() -> None:
    """Apply every finding in the file named on the command line, reporting each one's status."""
    arguments = sys.argv[1:]
    if not arguments:
        sys.exit(__doc__)
    write = "--apply" in arguments
    merge_base: str | None = None
    if "--merge-base" in arguments:
        merge_base = arguments[arguments.index("--merge-base") + 1]
    findings: list[Finding] = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))

    tally: dict[StatusType, int] = {}
    edited: dict[Path, str] = {}
    for finding in findings:
        path = Path(finding["file"])
        if path.suffix != ".py":
            tally["not python"] = tally.get("not python", 0) + 1
            print(f"{'NOT PYTHON':>18}  {finding['file']}  {finding['quote'][:64]}")
            continue
        raw = edited.get(path, path.read_text(encoding="utf-8"))
        updated, status = apply_one(raw=raw, finding=finding, merge_base=merge_base)
        tally[status] = tally.get(status, 0) + 1
        if status == "applied":
            edited[path] = updated
        else:
            print(f"{status.upper():>18}  {finding['file']}  {finding['quote'][:64]}")

    if write:
        for path, updated in edited.items():
            path.write_text(updated, encoding="utf-8")
        reflow(sorted(edited))

    print()
    for status, count in sorted(tally.items()):
        print(f"{count:>5}  {status}")
    if not write:
        print("\ndry run - nothing written; pass --apply to write")
    elif edited:
        print(f"\nwrote {len(edited)} file(s); run check_prose_only.py next")
    if tally.keys() - {"applied", "pre-existing", "not python", "no-op"}:
        sys.exit("\nFAIL - some findings could not be applied; fix or hand-edit them")


if __name__ == "__main__":
    main()
