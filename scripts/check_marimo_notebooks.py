"""Check that every name a marimo notebook's cells reference is bound inside the notebook.

Marimo never executes a notebook's module-level statements, so a name bound there is invisible to
every cell and the notebook dies with a `NameError` the next time it is opened — while ruff, ty and
pytest all pass, because the file they were handed is valid Python. `ruff check --fix` and
`marimo check --fix` each produce that shape from a working notebook. Full rationale, and what this
check can and cannot catch:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#marimo-notebooks-bind-every-name-their-cells-reference>

`Cell.refs` and `Cell.defs` are public marimo API; loading a notebook without running it is not, so
this reads `marimo._ast`. Attaching line numbers costs a second private entry point
(`get_notebook_status`) on top of `load_app` — worth it for a hook whose output should be
clickable, but it is the first thing to drop if keeping up with marimo gets expensive. Nothing here
may fail quietly: every way of not reaching an answer raises, and `tests/test_marimo_notebooks.py`
keeps a positive control that fails if this stops detecting a real breakage.
"""

import builtins
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from marimo._ast.app import InternalApp
from marimo._ast.load import get_notebook_status, load_app
from marimo._ast.parse import MarimoFileError, NonMarimoPythonScriptError

ALWAYS_BOUND: Final[frozenset[str]] = frozenset(dir(builtins)) | {"__file__"}
"""Names a cell may reference without any cell defining them.

`__file__` is not a builtin: marimo injects it into the cell globals itself (`_runtime.runtime`
sets it from the notebook's filename), so a cell locating a data file relative to the notebook is
correct code and must not be flagged.
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
        MarimoFileError: if `path` is not a marimo notebook.
        NonMarimoPythonScriptError: if `path` is an ordinary Python script.
        ValueError: if the notebook cannot be checked in full — it holds no marimo app, it holds a
            cell marimo cannot compile, or marimo's two views of it disagree about how many cells
            it has (which would mean the private API this rides on has moved). Raised rather than
            reported as "no findings", so that the check cannot degrade into passing everything.
    """
    app = load_app(str(path))
    load_result = get_notebook_status(str(path))
    if app is None or load_result.notebook is None:
        raise ValueError(f"not a usable marimo notebook (marimo status: {load_result.status})")
    compiled = list(InternalApp(app).cell_manager.cell_data())
    serialized = load_result.notebook.cells
    if len(compiled) != len(serialized):
        raise ValueError(
            f"marimo reports {len(compiled)} compiled cells but {len(serialized)} serialized "
            "ones, so cells can no longer be matched to line numbers"
        )
    cells = [(source.lineno, data.cell) for data, source in zip(compiled, serialized, strict=True)]
    for lineno, cell in cells:
        if cell is None:
            raise ValueError(f"marimo cannot compile the cell at line {lineno}")
    defined = ALWAYS_BOUND.union(*(cell.defs for _, cell in cells if cell is not None))
    return [
        UnboundCell(lineno, tuple(sorted(cell.refs - defined)))
        for lineno, cell in cells
        if cell is not None and cell.refs - defined
    ]


def _check_file(path: Path) -> list[str]:
    """Return one human-readable finding per unbound-name cell in the notebook at `path`.

    A file that cannot be checked at all is itself a finding, rather than a raised exception, so
    that one bad path cannot hide the findings for the others the hook was given.
    """
    try:
        unbound = unbound_cells(path)
    except (MarimoFileError, NonMarimoPythonScriptError) as error:
        return [f"{path}: not a marimo notebook: {error}"]
    except (OSError, SyntaxError, ValueError) as error:
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
