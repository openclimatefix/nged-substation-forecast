"""Check that every name a marimo notebook's cells reference is bound inside the notebook.

Marimo never executes a notebook's module-level statements: it rebuilds the notebook from the
`with app.setup:` block plus the `@app.cell` functions. A name bound at module level is therefore
invisible to every cell, and the notebook dies with a `NameError` the next time it is opened —
while `ruff`, `ty` and `pytest` all report success, because the file they were handed is perfectly
valid Python.

Two tools turn a working notebook into exactly that shape:

- `ruff check --fix` writes an import that an autofix needs into the file's *top-level* import
  block (`UP017` and `UP035` today, and any future fix that reaches for `itertools`). The
  pre-commit hooks are split so notebooks are never auto-fixed, but a hand-typed
  `uv run ruff check . --fix` is not covered by that split.
- `marimo check --fix` deletes such an import and rewrites the cell that used the name as
  `def _(name)`, which leaves the name as a cell input that nothing defines.

This script catches both, and a cell input left dangling by hand. It is a *static* name-binding
check: it reads `Cell.refs` and `Cell.defs` — marimo's documented public API for the names a cell
reads and the names it binds — without executing a single cell, so it needs none of the notebooks'
runtime dependencies. It cannot catch a notebook that binds every name and still fails inside a
Polars or Altair call.

Loading a notebook *without* executing it needs `marimo._ast`, which is private API. So that a
marimo release cannot quietly downgrade this check to a no-op, `unbound_cells` raises whenever a
file does not parse into at least one cell, and `tests/test_marimo_notebooks.py` keeps a positive
control that fails if the check ever stops detecting a real breakage.
"""

import builtins
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from marimo._ast.app import InternalApp
from marimo._ast.load import get_notebook_status, load_app
from marimo._ast.parse import MarimoFileError, NonMarimoPythonScriptError

BUILTIN_NAMES: Final[frozenset[str]] = frozenset(dir(builtins))
"""Names a cell may reference without any cell defining them."""


@dataclass(frozen=True, slots=True)
class UnboundCell:
    """One notebook cell, and the names it references that its notebook never binds."""

    lineno: int
    """1-indexed line of the cell's `@app.cell` decorator, or of its `with app.setup:` statement."""

    names: tuple[str, ...]
    """The unbound names, sorted."""


def unbound_cells(path: Path) -> list[UnboundCell]:
    """Return the cells of the marimo notebook at `path` that reference names it never binds.

    A name counts as bound if any cell defines it — cell order is irrelevant, because marimo
    derives execution order from the dependency graph rather than from position in the file — or
    if it is a Python builtin.

    Args:
        path: Path to a marimo notebook.

    Raises:
        MarimoFileError: if `path` is not a marimo notebook.
        NonMarimoPythonScriptError: if `path` is an ordinary Python script.
        ValueError: if `path` parses into no cells at all, or if marimo's two views of the
            notebook disagree about how many cells it has. Either means the private API this
            check rides on has moved, and is raised rather than reported as "no findings" so that
            the check fails loudly instead of silently passing everything.
    """
    app = load_app(str(path))
    load_result = get_notebook_status(str(path))
    if app is None or load_result.notebook is None:
        raise ValueError(f"{path}: holds no marimo app — expected a notebook.")
    cell_data = list(InternalApp(app).cell_manager.cell_data())
    serialized = load_result.notebook.cells
    if not cell_data:
        raise ValueError(f"{path}: parsed into zero marimo cells.")
    if len(cell_data) != len(serialized):
        raise ValueError(
            f"{path}: marimo reports {len(cell_data)} compiled cells but {len(serialized)} "
            "serialized ones, so cells can no longer be matched to line numbers."
        )
    # `cell` is None for a cell marimo could not compile; those carry no refs or defs to check.
    cells = [(source.lineno, data.cell) for data, source in zip(cell_data, serialized, strict=True)]
    defined = BUILTIN_NAMES.union(*(cell.defs for _, cell in cells if cell is not None))
    return [
        UnboundCell(lineno, tuple(sorted(cell.refs - defined)))
        for lineno, cell in cells
        if cell is not None and cell.refs - defined
    ]


def _check_file(path: Path) -> list[str]:
    """Return one human-readable finding per unbound-name cell in the notebook at `path`."""
    try:
        unbound = unbound_cells(path)
    except (MarimoFileError, NonMarimoPythonScriptError) as error:
        return [
            (
                f"{path}: not a marimo notebook ({error}). Every `.py` file directly inside "
                "`packages/notebooks/` or `packages/dashboard/` is taken to be one, both here and "
                "by the `ruff check --fix` exclusion in .pre-commit-config.yaml — so an ordinary "
                "module put there would silently stop being auto-fixed."
            )
        ]
    return [
        f"{path}:{cell.lineno}: cell references name(s) that no cell defines: "
        f"{', '.join(cell.names)}"
        for cell in unbound
    ]


def main(argv: list[str]) -> int:
    """Check each marimo notebook path in `argv`; return the process exit code."""
    findings = [finding for arg in argv for finding in _check_file(Path(arg))]
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
