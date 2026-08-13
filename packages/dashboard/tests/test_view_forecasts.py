"""The Reload button in ``view_forecasts.py`` re-reads every Delta table the app shows.

The button works by being *referenced* — a bare ``reload`` statement, carrying no value — in the
one cell every Delta read descends from. That statement is what this test protects: delete it, or
move the definition into the same cell, and the button still renders and still clicks while
nothing is re-read.

Parsing a notebook without running it is not public marimo API; `scripts/check_marimo_notebooks.py`
documents the same dependency and why the repo takes it.
"""

from pathlib import Path
from typing import Final

from marimo._ast.app import InternalApp
from marimo._ast.load import get_notebook_status, load_notebook_ir
from marimo._runtime import dataflow

NOTEBOOK: Final[Path] = Path(__file__).parents[1] / "view_forecasts.py"
"""The app under test, one directory above this `tests/` directory."""

DELTA_READ_CALLS: Final[tuple[str, ...]] = ("pl.scan_delta(", "DeltaTable(")
"""Calls that identify a cell as reading a Delta table.

Matched against the cell's own source, so a read reached through a helper in `src/dashboard/`
would not be found. Nothing in the notebook reads that way.
"""


def test_reload_button_re_reads_every_delta_table():
    notebook = get_notebook_status(str(NOTEBOOK)).notebook
    assert notebook is not None, f"marimo could not parse {NOTEBOOK}"
    graph = InternalApp(load_notebook_ir(notebook)).graph

    # marimo's own rule for what changing a UI element re-runs, from `marimo._runtime.runtime`:
    # the cells referencing the name, minus the cells defining it, then their descendants.
    roots = graph.get_referring_cells("reload", language="python") - graph.get_defining_cells(
        "reload"
    )
    delta_reads = {
        cell_id
        for cell_id, cell in graph.cells.items()
        if any(call in cell.code for call in DELTA_READ_CALLS)
    }
    # Without this, a marimo release that changed `graph.cells` would empty the set and leave the
    # assertion below passing over nothing.
    assert delta_reads, f"found no cell calling any of {DELTA_READ_CALLS}"

    assert delta_reads <= dataflow.transitive_closure(graph, roots)
