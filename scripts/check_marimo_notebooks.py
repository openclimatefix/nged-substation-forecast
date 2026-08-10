"""Check that every name a marimo notebook's cells reference is bound inside the notebook.

Marimo never executes a notebook's module-level statements, so a name bound there is invisible to
every cell and the notebook dies with a `NameError` the next time it is opened — while ruff, ty and
pytest all pass, because the file they were handed is valid Python. `ruff check --fix` and
`marimo check --fix` each produce that shape from a working notebook. Full rationale, and what this
check can and cannot catch:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#marimo-notebooks-bind-every-name-their-cells-reference>

`Cell.refs` and `Cell.defs` are public marimo API; parsing a notebook without running it is not, so
this reads `marimo._ast`. The serialized form carries the line numbers and the compiled form
carries the names, so both are needed.
"""

import builtins
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from marimo._ast.app import InternalApp
from marimo._ast.cell import Cell
from marimo._ast.load import get_notebook_status, load_notebook_ir
from marimo._ast.parse import MarimoFileError

ALWAYS_BOUND: Final[frozenset[str]] = frozenset(dir(builtins)) | {
    "__builtin__",
    "__builtins__",
    "__file__",
}
"""Names a cell may reference without any cell defining them.

The three dunders are not builtins — they are the names marimo seeds the cell globals with, in
`marimo._runtime.patches.create_main_module`. A cell locating a data file relative to the notebook
(`__file__`) or sandboxing an `eval` (`__builtins__`) is correct code and must not be flagged. Take
any addition here from that function, not from what looks plausible: `__marimo__`, for instance, is
bound in the editor kernel but *not* when the notebook runs as a script, so it stays flagged.
"""


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
    if it is in `ALWAYS_BOUND`.

    Args:
        path: Path to a marimo notebook.

    Raises:
        MarimoFileError: if marimo cannot parse `path` at all.
        ValueError: if the notebook cannot be checked in full — it parses into no cells, or it
            holds a cell marimo cannot compile. Raised rather than reported as "no findings", so
            that the check cannot degrade into passing everything.
    """
    notebook = get_notebook_status(str(path)).notebook
    if notebook is None or not notebook.cells:
        raise ValueError(
            "did not parse into any marimo cells. Every `.py` file directly inside "
            "`packages/notebooks/` or `packages/dashboard/` is taken to be a notebook — an "
            "ordinary module put there silently stops being auto-fixed by the ruff pre-commit hook."
        )
    compiled = InternalApp(load_notebook_ir(notebook)).cell_manager.cell_data()
    cells: list[tuple[int, Cell]] = []
    # `load_notebook_ir` registers one compiled cell per serialized cell, so `strict` never fires
    # unless that stops being true — in which case the line numbers below would be attached to the
    # wrong cells, and a loud failure beats a misleading report.
    for data, source in zip(compiled, notebook.cells, strict=True):
        if data.cell is None:
            raise ValueError(f"marimo cannot compile the cell at line {source.lineno}")
        cells.append((source.lineno, data.cell))
    defined = ALWAYS_BOUND.union(*(cell.defs for _, cell in cells))
    return [
        UnboundCell(lineno, tuple(sorted(cell.refs - defined)))
        for lineno, cell in cells
        if cell.refs - defined
    ]


def _check_file(path: Path) -> list[str]:
    """Return one human-readable finding per unbound-name cell in the notebook at `path`.

    A file that cannot be checked at all is itself a finding, rather than a raised exception, so
    that one bad path cannot hide the findings for the others the hook was given.
    """
    try:
        unbound = unbound_cells(path)
    except (MarimoFileError, OSError, SyntaxError, ValueError) as error:
        return [f"{path}: cannot be checked: {error}"]
    return [
        f"{path}:{cell.lineno}: cell references name(s) that no cell defines: "
        f"{', '.join(cell.names)}"
        for cell in unbound
    ]


def main(argv: list[str]) -> int:
    """Check each marimo notebook path in `argv`; return the process exit code."""
    found = False
    for arg in argv:
        for finding in _check_file(Path(arg)):
            print(finding)
            found = True
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
