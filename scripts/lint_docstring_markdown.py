"""Lint the markdown embedded in Python docstrings.

`mkdocstrings` renders every module/class/function docstring as markdown in the published API
docs (`docs/api/`), so a docstring with e.g. a list missing its blank line renders badly there
just like it would in a `.md` file. This script extracts each docstring via `ast` and pipes it
through `pymarkdown scan-stdin`, the same linter used on `README.md`/`docs/*.md`.
"""

import ast
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

PYMARKDOWN_CONFIG: Final[str] = ".pymarkdown-docstrings.json"
"""Config overrides layered on top of `pyproject.toml`'s `[tool.pymarkdown]` for docstring text.

Disables rules that only make sense for a whole document (e.g. requiring the first line to be a
heading), since a docstring is a prose fragment, not a document.
"""


def _dedent_docstring(raw: str) -> str:
    """Dedent a raw (`clean=False`) docstring while preserving its exact line count.

    Mirrors `inspect.cleandoc`'s dedenting logic but skips its leading/trailing blank-line
    stripping, so line ``i`` of the result always corresponds to ``docstring_start_line + i`` in
    the original source. That mapping is what lets violations be reported against real file:line
    locations; `ast.get_docstring(node, clean=True)` would break it by also removing blank lines.
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
    methods and nested definitions too). Attribute-level docstrings (a bare string literal
    following an assignment, e.g. a `ClassVar` docstring) aren't picked up by `ast.get_docstring`
    and are out of scope here.
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


def _scan_docstring(text: str) -> tuple[str, str, int]:
    """Run `pymarkdown scan-stdin` against `text`, returning stdout, stderr, and the return code."""
    result = subprocess.run(
        ["uv", "run", "pymarkdown", "--config", PYMARKDOWN_CONFIG, "scan-stdin"],
        input=text,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def _remap_violation(line: str, path: Path, start_line: int) -> str:
    """Rewrite a `pymarkdown scan-stdin` violation's pseudo `stdin:line:col:...` to `path:line:col:...`."""
    _pseudo_file, _, rest = line.partition(":")
    docstring_line_str, _, remainder = rest.partition(":")
    source_line = start_line + int(docstring_line_str) - 1
    return f"{path}:{source_line}:{remainder}"


def _lint_file(path: Path) -> list[str]:
    """Return remapped violation lines for every docstring in `path`."""
    violations: list[str] = []
    source = path.read_text()
    for start_line, text in _iter_docstrings(source, path):
        stdout, stderr, returncode = _scan_docstring(text)
        if returncode not in (0, 1):
            violations.append(
                f"{path}:{start_line}: pymarkdown scan-stdin failed: {stderr.strip()}"
            )
            continue
        violations.extend(
            _remap_violation(line, path, start_line) for line in stdout.splitlines() if line.strip()
        )
    return violations


def main(argv: list[str]) -> int:
    """Lint docstring markdown in each `.py` file path in `argv`; return the process exit code."""
    violations = [violation for arg in argv for violation in _lint_file(Path(arg))]
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
