"""Lint the markdown embedded in Python docstrings.

`mkdocstrings` renders every module/class/function docstring as markdown in the published API
docs (`docs/api/`). So a docstring with, say, a list missing its blank line renders badly there,
just as it would in a `.md` file. This script extracts each docstring via `ast` and scans it with
`pymarkdown`'s Python API, the same linter used on `README.md`/`docs/*.md`.
"""

import ast
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path
from typing import Final

from pymarkdown.api import PyMarkdownApi, PyMarkdownApiException, PyMarkdownScanFailure

PYMARKDOWN_CONFIG: Final[str] = ".pymarkdown-docstrings.json"
"""Config overrides layered on top of `pyproject.toml`'s `[tool.pymarkdown]` for docstring text.

Disables the two rules that only make sense for a whole document — requiring the first line to be a
heading, and requiring a trailing newline — since a docstring is a prose fragment, not a document.
"""

POOL_FILE_COUNT: Final[int] = 16
"""File count from which linting moves to a process pool.

A scan costs about 10 ms per docstring and releases no lock worth sharing, so the work is
CPU-bound and splits cleanly across processes. Starting the pool is not free: Python 3.14 defaults
to the `forkserver` start method, so every worker imports `pymarkdown` afresh, which costs about as
much as linting 16 files. Below that count the serial path finishes sooner than the pool can start.
The commit-time hook passes only the staged files and so almost always takes the serial path; a
`--all-files` sweep passes a few hundred and takes the pool.
"""


def _dedent_docstring(raw: str) -> str:
    """Dedent a raw (`clean=False`) docstring while preserving its exact line count.

    Mirrors `inspect.cleandoc`'s dedenting logic but skips its leading/trailing blank-line
    stripping, so line ``i`` of the result always corresponds to ``docstring_start_line + i`` in
    the original source. That mapping is what lets violations be reported against real file:line
    locations; `ast.get_docstring(node, clean=True)` would break it by also removing blank lines.

    Args:
        raw: The docstring exactly as it appears in the source, indentation included.

    Returns:
        The dedented docstring, with the same number of lines as `raw`.
    """
    lines = raw.expandtabs().split("\n")
    margin = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            margin = min(margin, len(line) - len(stripped))
    lines[0] = lines[0].lstrip()
    if margin < sys.maxsize:
        lines[1:] = [line[margin:] for line in lines[1:]]
    return "\n".join(lines)


def _iter_docstrings(source: str, path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(source_start_line, dedented_text)`` for every docstring in `path`.

    Covers module, class, function, and async function docstrings (`ast.walk` naturally reaches
    methods and nested definitions too). An attribute-level docstring is a bare string literal
    following an assignment, such as a `ClassVar` docstring. `ast.get_docstring` does not pick
    those docstrings up, so they are out of scope here.

    Args:
        source: The full text of the Python file.
        path: The file `source` came from, used only to name the file in `ast` syntax errors.

    Yields:
        The 1-based line the docstring starts on, and its dedented text.
    """
    tree = ast.parse(source, filename=str(path))
    docstring_nodes: list[ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = [
        tree,
        *(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        ),
    ]
    for node in docstring_nodes:
        raw = ast.get_docstring(node, clean=False)
        if raw is None:
            continue
        yield node.body[0].lineno, _dedent_docstring(raw)


@cache
def _scanner() -> PyMarkdownApi:
    """Build the configured scanner once per process, and reuse it for every docstring.

    Constructing the scanner re-reads `pyproject.toml` and `PYMARKDOWN_CONFIG`, which costs more
    than scanning a docstring does. One instance scans any number of strings and keeps no state
    between them. Each process pool worker builds its own, which is why the cache sits on a
    function rather than on a module-level constant evaluated at import.

    Returns:
        A `PyMarkdownApi` carrying this repo's docstring configuration.
    """
    return PyMarkdownApi().configuration_file_path(PYMARKDOWN_CONFIG)


def _format_violation(failure: PyMarkdownScanFailure, path: Path, start_line: int) -> str:
    """Render one scan failure as `path:line:col: RULE: description (rule-name)`.

    The line number `pymarkdown` reports is relative to the docstring it was handed, so it is
    shifted back onto the line the docstring occupies in `path`.

    Args:
        failure: One violation as `pymarkdown` reported it.
        path: The Python file the docstring came from.
        start_line: The 1-based line `path` starts that docstring on.

    Returns:
        A single violation line, in the same format `pymarkdown`'s own command line prints.
    """
    source_line = start_line + failure.line_number - 1
    return (
        f"{path}:{source_line}:{failure.column_number}: {failure.rule_id}: "
        f"{failure.rule_description}{failure.extra_error_information} ({failure.rule_name})"
    )


def _lint_file(path: Path) -> list[str]:
    """Return violation lines for every docstring in `path`.

    Args:
        path: The Python file to lint.

    Returns:
        One string per violation, empty if the file's docstrings are clean.
    """
    violations: list[str] = []
    source = path.read_text()
    for start_line, text in _iter_docstrings(source=source, path=path):
        try:
            result = _scanner().scan_string(text)
        except PyMarkdownApiException as exception:
            violations.append(f"{path}:{start_line}: pymarkdown failed: {exception}")
            continue
        violations.extend(
            _format_violation(failure=failure, path=path, start_line=start_line)
            for failure in result.scan_failures
        )
    return violations


def main(argv: list[str]) -> int:
    """Lint docstring markdown in each `.py` file path in `argv`; return the process exit code.

    Args:
        argv: The file paths to lint.

    Returns:
        1 if any docstring carries a violation, 0 otherwise.
    """
    paths = [Path(arg) for arg in argv]
    if len(paths) < POOL_FILE_COUNT:
        per_file = [_lint_file(path) for path in paths]
    else:
        with ProcessPoolExecutor() as pool:
            per_file = list(pool.map(_lint_file, paths))
    violations = [violation for file_violations in per_file for violation in file_violations]
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
