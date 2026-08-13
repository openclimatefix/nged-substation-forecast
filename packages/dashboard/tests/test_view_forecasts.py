"""The Reload button in ``view_forecasts.py`` re-reads every Delta table the app shows.

Marimo re-runs the cells that *reference* a changed name, plus their transitive descendants, but
never the cell that defines it. So the button works by being referenced — a bare ``reload``
statement, carrying no value — in the one cell every Delta read descends from. That statement is
what this test protects: delete it, or move the definition into the same cell, and the button still
renders and still clicks while nothing is re-read.

Parsing a notebook without running it is not public marimo API; `scripts/check_marimo_notebooks.py`
documents the same dependency and why the repo takes it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from marimo._ast.app import InternalApp
from marimo._ast.load import get_notebook_status, load_notebook_ir

NOTEBOOK: Final[Path] = Path(__file__).parents[1] / "view_forecasts.py"
"""The app under test, one directory above this `tests/` directory."""

RELOAD: Final[str] = "reload"
"""The name bound to the Reload button, and referenced to take a dependency on it."""

DELTA_READ_CALLS: Final[tuple[str, ...]] = ("pl.scan_delta(", "DeltaTable(")
"""Calls that identify a cell as reading a Delta table.

Matched against the cell's own source, so a read reached through a helper in `src/dashboard/`
would not be found. Nothing in the notebook reads that way.
"""


@dataclass(frozen=True, slots=True)
class NotebookCell:
    """One marimo cell: its source, the names it binds, and the names it references."""

    code: str
    defs: frozenset[str]
    refs: frozenset[str]


def _notebook_cells() -> list[NotebookCell]:
    """Parse `NOTEBOOK` into its cells, without running it."""
    notebook = get_notebook_status(str(NOTEBOOK)).notebook
    assert notebook is not None, f"marimo could not parse {NOTEBOOK}"
    cells = []
    for data in InternalApp(load_notebook_ir(notebook)).cell_manager.cell_data():
        assert data.cell is not None, f"marimo could not compile a cell of {NOTEBOOK}"
        cells.append(NotebookCell(data.code, frozenset(data.cell.defs), frozenset(data.cell.refs)))
    return cells


def _cells_rerun_by(cells: list[NotebookCell], name: str) -> set[int]:
    """Indices of the cells marimo re-runs when the UI element bound to `name` changes.

    The roots are the cells referencing `name` minus the cells defining it, mirroring
    `marimo._runtime.runtime`; every cell reachable from a root through a defines/references edge
    follows. Rooting the walk at the *defining* cell instead would prove nothing here, because that
    cell also defines `source`, which the whole notebook already descends from.
    """
    rerun = {i for i, cell in enumerate(cells) if name in cell.refs and name not in cell.defs}
    queue = list(rerun)
    while queue:
        for defined in cells[queue.pop()].defs:
            for i, cell in enumerate(cells):
                if i not in rerun and defined in cell.refs:
                    rerun.add(i)
                    queue.append(i)
    return rerun


def test_reload_button_re_reads_every_delta_table():
    cells = _notebook_cells()

    defining = [cell for cell in cells if RELOAD in cell.defs]
    assert len(defining) == 1, f"expected exactly one cell to bind `{RELOAD}`"
    # An mo.ui.button renders and clicks identically but never changes value, so swapping one in
    # would leave the graph below intact and the button inert.
    assert "mo.ui.refresh(" in defining[0].code

    delta_reads = {
        i for i, cell in enumerate(cells) if any(call in cell.code for call in DELTA_READ_CALLS)
    }
    # Without this, a marimo release that changed what `cell_data()` yields would empty the set and
    # leave the assertion below passing over nothing.
    assert delta_reads, f"found no cell calling any of {DELTA_READ_CALLS}"

    assert delta_reads <= _cells_rerun_by(cells, RELOAD)
