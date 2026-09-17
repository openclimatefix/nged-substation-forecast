"""Count the prose words in a set of files, on the one definition every wave of #706 has used.

A word count nobody can reproduce cannot be compared against the next package's, and these sweeps
run one package at a time over months. The definition is fixed here so that the before-and-after
figures in a pull-request body mean the same thing in wave 6 as they did in wave 1:

- For a Python file, the text of every docstring and every comment, and nothing else. Code,
  identifiers and string literals that are not docstrings are not prose.
- For a markdown file, the whole file.

A word is a whitespace-separated token, which counts `**the**` and `the` alike. The point is
comparability between two revisions of the same text, not lexical accuracy.

Usage::

    python3 count_prose_words.py <path> [<path> ...]
    python3 count_prose_words.py --rev HEAD <path> [<path> ...]

With `--rev`, each file is read from that git revision instead of from the working tree, so the
same command produces the before figure and the after figure. A path missing at that revision
counts zero, which is what a file the branch created should contribute to the before count.
"""

import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Final

DOCS_SITE: Final[str] = "openclimatefix.github.io/nged-substation-forecast"
"""The rendered documentation site, whose mentions are counted as the link total."""


def _python_prose(source: str) -> str:
    """Return every docstring and comment in `source`, joined by newlines.

    Args:
        source: The full text of a Python file.

    Returns:
        The prose text, with the code between it discarded.
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
    """
    return _python_prose(source) if path.suffix == ".py" else source


def _read(*, path: Path, rev: str | None) -> str | None:
    """Return the text of `path` at `rev`, or from the working tree when `rev` is None.

    Args:
        path: The file to read.
        rev: A git revision, or None to read the working tree.

    Returns:
        The file's text, or None where the file does not exist at that revision.
    """
    if rev is None:
        return path.read_text(encoding="utf-8") if path.exists() else None
    completed = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def main() -> None:
    """Count and print the prose words of every path named on the command line."""
    arguments = sys.argv[1:]
    if not arguments:
        sys.exit(__doc__)
    rev: str | None = None
    if "--rev" in arguments:
        index = arguments.index("--rev")
        rev = arguments[index + 1]
        arguments = arguments[:index] + arguments[index + 2 :]

    total_words = 0
    total_links = 0
    for name in arguments:
        path = Path(name)
        source = _read(path=path, rev=rev)
        if source is None:
            print(f"{0:>7} {0:>5}  {name}  (absent)")
            continue
        prose = prose_of(path=path, source=source)
        words = len(prose.split())
        links = prose.count(DOCS_SITE)
        total_words += words
        total_links += links
        print(f"{words:>7} {links:>5}  {name}")
    print(f"{'-' * 30}")
    print(f"{total_words:>7} {total_links:>5}  TOTAL ({len(arguments)} files)")


if __name__ == "__main__":
    main()
