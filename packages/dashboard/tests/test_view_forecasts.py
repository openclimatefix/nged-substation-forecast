"""The Reload button in ``view_forecasts.py`` re-reads every Delta table the app shows.

The button works by being *referenced* — a bare ``reload`` statement, carrying no value — in the
one cell every Delta read descends from. That statement is what this test protects: delete it, or
move the definition into the same cell, and the button still renders and still clicks while
nothing is re-read.

Parsing a notebook without running it is not public marimo API;
`scripts/lint/check_marimo_notebooks.py` documents the same dependency and why the repo takes it.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Final
from unittest.mock import Mock

import marimo as mo
import polars as pl
import pytest
from contracts.typing_utils import typeddict_to_dict
from dashboard.forecast_chart import PLOT_HORIZON
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


@pytest.mark.parametrize("experiment", [None, "second", "missing"])
def test_comparison_reads_only_the_same_fold_series_and_run(
    monkeypatch: pytest.MonkeyPatch, experiment: str | None
) -> None:
    init_time = datetime(2026, 7, 4, 6, tzinfo=UTC)
    row = {
        "experiment_name": "second",
        "fold_id": "fold-1",
        "time_series_id": 24,
        "power_fcst_init_time": init_time,
        "valid_time": init_time + timedelta(hours=1),
        "power_fcst": 1.0,
        "ensemble_member": 0,
    }
    rows = [
        row,
        {**row, "experiment_name": "primary", "power_fcst": 2.0},
        {**row, "fold_id": "fold-2", "power_fcst": 3.0},
        {**row, "time_series_id": 25, "power_fcst": 4.0},
        {**row, "power_fcst_init_time": init_time - timedelta(days=1), "power_fcst": 5.0},
        {**row, "valid_time": init_time + PLOT_HORIZON + timedelta(minutes=30)},
    ]
    scan = Mock(return_value=pl.DataFrame(rows).lazy())
    monkeypatch.setattr(pl, "scan_delta", scan)
    notebook = get_notebook_status(str(NOTEBOOK)).notebook
    assert notebook is not None
    graph = InternalApp(load_notebook_ir(notebook)).graph
    cell = next(cell for cell in graph.cells.values() if "comparison_forecasts" in cell.defs)
    namespace = {
        "pl": pl,
        "mo": mo,
        "typeddict_to_dict": typeddict_to_dict,
        "PLOT_HORIZON": PLOT_HORIZON,
        "comparison_picker": SimpleNamespace(value=experiment),
        "fold_picker": SimpleNamespace(value="fold-1"),
        "series_picker": SimpleNamespace(value=24),
        "init_time": init_time,
        "settings": SimpleNamespace(power_forecasts_data_path="unused", storage_options={}),
    }
    assert cell.body is not None
    exec(cell.body, namespace)  # noqa: S102 — execute the notebook's compiled selection cell.
    comparison = namespace["comparison_forecasts"]
    if experiment == "second":
        assert isinstance(comparison, pl.DataFrame)
        assert comparison.get_column("power_fcst").to_list() == [1.0]
        assert namespace["comparison_message"] is None
    else:
        assert comparison is None
        assert (namespace["comparison_message"] is not None) == (experiment == "missing")
    assert scan.call_count == (experiment is not None)
