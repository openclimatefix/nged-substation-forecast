"""Count the prose words in a set of files, on the one definition every wave of #706 has used.

A word count nobody can reproduce cannot be compared against the next package's, and these sweeps
run one package at a time over months. The definition is fixed here so that the before-and-after
figures in a pull-request body mean the same thing in wave 6 as they did in wave 1:

- For a Python file, the text of every docstring and every comment, and nothing else. Code,
  identifiers and string literals that are not docstrings are not prose.
- For a markdown file, the whole file.
- Alongside the word count, each path reports how many times it links to the rendered
  documentation site. A sweep that names a noun often has somewhere to link that noun to, so the
  link total is the second figure worth quoting in a pull-request body.
- Any other suffix is reported as `not prose` and counted as nothing, because a `.json` or a
  `.toml` file read as markdown would contribute its keys and its punctuation to the figure.

A word is a whitespace-separated token, which counts `**the**` and `the` alike. The point is
comparability between two revisions of the same text, not lexical accuracy.

The before figure and the after figure have to be the same kind of measurement, so anything that
would make one of them a different measurement — a directory named where a file was meant, a file
that does not parse, a file that is not UTF-8 — is reported per path and makes the run exit
non-zero, rather than being counted as nothing or raising a traceback over the whole batch. A path
genuinely absent at the revision is the one exception: it counts zero, which is what a file the
branch created should contribute to the before figure.

Usage::

    python3 count_prose_words.py <path> [<path> ...]
    python3 count_prose_words.py --rev HEAD <path> [<path> ...]

With `--rev`, each file is read from that git revision instead of from the working tree, so the
same command produces the before figure and the after figure. Each path is resolved against the
current directory either way.
"""

import argparse
import ast
import io
import subprocess
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

DOCS_SITE: Final[str] = "openclimatefix.github.io/nged-substation-forecast"
"""The rendered documentation site, whose mentions are counted as the link total."""

PROSE_SUFFIXES: Final[frozenset[str]] = frozenset({".md", ".py"})
"""The two kinds of file this counter has a definition of prose for."""

ProblemType = Literal["absent", "not a file", "not prose", "unparseable", "unreadable"]
"""Why one path contributed nothing to the figures."""

COUNTED_AS_ZERO: Final[frozenset[str]] = frozenset({"absent"})
"""The one problem that is legitimate: a file the branch created has no text at the merge base."""


def _python_prose(source: str) -> str:
    """Return every docstring and comment in `source`, joined by newlines.

    Args:
        source: The full text of a Python file.

    Returns:
        The prose text, with the code between it discarded.

    Raises:
        SyntaxError: If `source` is not a Python file this interpreter can parse.
    """
    pieces: list[str] = []
    for node in ast.walk(ast.parse(source)):
        is_string_statement = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if is_string_statement and isinstance(node.value.value, str):
            pieces.append(node.value.value)
    pieces.extend(
        token.string.lstrip("#")
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    )
    return "\n".join(pieces)


def prose_of(*, path: Path, source: str) -> str:
    """Return the prose of one file, chosen by its suffix.

    Args:
        path: The file's path, read only for its suffix.
        source: The full text of the file.

    Returns:
        The prose text to count.

    Raises:
        SyntaxError: If a `.py` file does not parse.
    """
    return _python_prose(source) if path.suffix == ".py" else source


def _read_working_tree(path: Path) -> tuple[str | None, ProblemType | None]:
    """Read `path` from the working tree, naming the reason rather than raising.

    Args:
        path: The file to read.

    Returns:
        The text and None, or None and the problem.
    """
    if not path.exists():
        return None, "absent"
    if not path.is_file():
        return None, "not a file"
    try:
        return path.read_text(encoding="utf-8"), None
    except OSError, UnicodeDecodeError:
        return None, "unreadable"


def _read_at_revision(*, path: Path, rev: str) -> tuple[str | None, ProblemType | None]:
    """Read `path` at `rev`, naming the reason rather than raising.

    `<rev>:./<path>` is git's cwd-relative form; without the `./` git resolves the path against the
    repository root, so every lookup from a subdirectory fails and the before figure is zero for a
    file that is really there. `git cat-file blob` rather than `git show` is what refuses a
    directory: `git show` prints a tree listing, whose words would then be counted as prose while
    the same path read from the working tree is a `not a file` — two figures of different kinds,
    which is the one thing this script exists to prevent.

    Args:
        path: The file to read, relative to the current directory.
        rev: The git revision to read it at.

    Returns:
        The text and None, or None and the problem.
    """
    kind = subprocess.run(
        ["git", "cat-file", "-t", f"{rev}:./{path}"], capture_output=True, text=True, check=False
    )
    if kind.returncode != 0:
        return None, "absent"
    if kind.stdout.strip() != "blob":
        return None, "not a file"
    blob = subprocess.run(
        ["git", "cat-file", "blob", f"{rev}:./{path}"], capture_output=True, check=False
    )
    try:
        return blob.stdout.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "unreadable"


def figures_for(*, path: Path, rev: str | None) -> tuple[int, int] | ProblemType:
    """The word count and the docs-link count for one path, or why it has neither.

    Args:
        path: The file to count.
        rev: A git revision, or None to read the working tree.

    Returns:
        `(words, links)`, or the problem that stopped the count.
    """
    source, problem = (
        _read_working_tree(path) if rev is None else _read_at_revision(path=path, rev=rev)
    )
    if source is None:
        assert problem is not None
        return problem
    if path.suffix not in PROSE_SUFFIXES:
        return "not prose"
    try:
        prose = prose_of(path=path, source=source)
    except SyntaxError:
        return "unparseable"
    return len(prose.split()), prose.count(DOCS_SITE)


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    """Read the command line, settling `--help`, a valueless `--rev` and an unknown flag alike.

    Args:
        argv: The arguments after the program name.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Count the prose words of a set of markdown and Python files."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="the files to count")
    parser.add_argument("--rev", help="read each file at this git revision instead")
    return parser.parse_args(list(argv))


def main() -> None:
    """Count and print the prose words of every path named on the command line."""
    arguments = _parse_arguments(sys.argv[1:])

    total_words = 0
    total_links = 0
    problems: list[str] = []
    for path in arguments.paths:
        figures = figures_for(path=path, rev=arguments.rev)
        if isinstance(figures, str):
            print(f"{0:>7} {0:>5}  {path}  ({figures})")
            if figures not in COUNTED_AS_ZERO:
                problems.append(f"{path}: {figures}")
            continue
        words, links = figures
        total_words += words
        total_links += links
        print(f"{words:>7} {links:>5}  {path}")
    print("-" * 30)
    print(f"{total_words:>7} {total_links:>5}  TOTAL ({len(arguments.paths)} files)")
    if problems:
        sys.exit("\nFAIL - these paths were not counted:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
