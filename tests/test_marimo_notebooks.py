"""Every marimo notebook binds every name its cells reference.

The check itself lives in `scripts/check_marimo_notebooks.py`, which is also a pre-commit hook;
its module docstring explains what the failure looks like and which tools produce it. These tests
run it over every notebook in the repo, and — because it rides on private marimo API — keep hand-
written notebooks either side of the line: two that it must flag, so a marimo release cannot
quietly turn it into a no-op, and one correct notebook that it must not.
"""

import ast
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

CHECKER_PATH: Final[Path] = REPO_ROOT / "scripts" / "check_marimo_notebooks.py"
"""The checker, imported by path below because `scripts/` is not an importable package."""

NOTEBOOK_DIRS: Final[tuple[str, ...]] = ("packages/notebooks", "packages/dashboard")
"""Directories whose top-level `.py` files are marimo notebooks.

The same set as the `exclude`/`files` regex shared by the two ruff pre-commit hooks. Only the
files sitting *directly* in each directory are notebooks: `packages/dashboard/src/` is ordinary
library code and `packages/*/tests/` is ordinary test code.
"""

BROKEN_NOTEBOOK: Final[str] = """import marimo

from datetime import UTC

__generated_with = "0.23.16"
app = marimo.App()

with app.setup:
    from datetime import datetime


@app.cell
def _():
    print(datetime(2026, 1, 1, tzinfo=UTC))
    return


if __name__ == "__main__":
    app.run()
"""
"""A notebook broken exactly as `ruff check --fix` breaks one, held as a string rather than as a
file under `tests/data/` so that ruff never sees a deliberately-broken notebook to "fix"."""

UNPARSABLE_NOTEBOOK: Final[str] = '''import marimo

__generated_with = "0.23.16"
app = marimo.App()

with app.setup:
    from datetime import datetime


app._unparsable_cell(
    r"""
    print(datetime(2026, 1, 1)
    """,
    name="_",
)

if __name__ == "__main__":
    app.run()
'''
"""What marimo writes when a cell is saved with a syntax error.

Its names cannot be analysed, so a checker that quietly skipped it would report a notebook holding
one as clean.
"""

DUNDER_FILE_NOTEBOOK: Final[str] = """import marimo

__generated_with = "0.23.16"
app = marimo.App()

with app.setup:
    import pathlib


@app.cell
def _():
    print(pathlib.Path(__file__).name)
    return


if __name__ == "__main__":
    app.run()
"""
"""A correct notebook that locates itself. `__file__` is not a builtin — marimo injects it into
the cell globals — so it is the one name that would otherwise look unbound in working code."""


def _load_checker() -> ModuleType:
    """Import `scripts/check_marimo_notebooks.py` by path."""
    spec = importlib.util.spec_from_file_location("check_marimo_notebooks", CHECKER_PATH)
    assert spec is not None, f"cannot import {CHECKER_PATH}"
    assert spec.loader is not None, f"no loader for {CHECKER_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebooks() -> list[Path]:
    """Every marimo notebook in the repo."""
    return sorted(path for d in NOTEBOOK_DIRS for path in (REPO_ROOT / d).glob("*.py"))


def _run_checker(*paths: Path) -> subprocess.CompletedProcess[str]:
    """Run the checker as the pre-commit hook runs it, over `paths`."""
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), *(str(path) for path in paths)],
        capture_output=True,
        text=True,
        check=False,
    )


CHECKER: Final[ModuleType] = _load_checker()
"""The checker module, loaded once for every test in this file."""


def test_every_directory_of_notebooks_holds_at_least_one():
    # Guards the parametrised test below: a glob that silently matched nothing would collect zero
    # cases and pass.
    for directory in NOTEBOOK_DIRS:
        assert list((REPO_ROOT / directory).glob("*.py")), f"no notebooks found in {directory}"


@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda path: path.name)
def test_notebook_binds_every_name_its_cells_reference(notebook: Path):
    unbound = CHECKER.unbound_cells(notebook)
    assert not unbound, (
        f"{notebook.name} references names no cell defines: {unbound}. A name bound at module "
        "level does not count — marimo never runs module-level statements."
    )


def test_checker_flags_an_import_hoisted_to_module_level(tmp_path: Path):
    # The positive control. `unbound_cells` reads private marimo API, so this is what fails if a
    # marimo release moves it and the check degrades to reporting nothing.
    notebook = tmp_path / "hoisted_import.py"
    notebook.write_text(BROKEN_NOTEBOOK)

    result = _run_checker(notebook)

    assert result.returncode == 1, result.stdout
    assert "UTC" in result.stdout


def test_checker_rejects_a_file_that_is_not_a_notebook(tmp_path: Path):
    # The second control: an ordinary module in a notebook directory is a finding, not a silent
    # pass, because the ruff pre-commit hooks would stop auto-fixing it.
    script = tmp_path / "ordinary_module.py"
    script.write_text("x = 1\n")

    result = _run_checker(script)

    assert result.returncode == 1, result.stdout
    assert "not a marimo notebook" in result.stdout


def test_checker_flags_a_cell_marimo_cannot_compile(tmp_path: Path):
    # Such a cell has no names to analyse, so skipping it would pass a notebook marimo itself
    # reports as broken.
    notebook = tmp_path / "unparsable_cell.py"
    notebook.write_text(UNPARSABLE_NOTEBOOK)

    result = _run_checker(notebook)

    assert result.returncode == 1, result.stdout
    assert "cannot compile" in result.stdout


def test_checker_allows_a_cell_to_reference_dunder_file(tmp_path: Path):
    notebook = tmp_path / "locates_itself.py"
    notebook.write_text(DUNDER_FILE_NOTEBOOK)

    assert not CHECKER.unbound_cells(notebook)


def test_checker_passes_over_every_notebook_at_once():
    # The only test of the exit-0 path: without it, a checker that rejected every notebook it was
    # given would still pass this whole file.
    result = _run_checker(*_notebooks())

    assert result.returncode == 0, result.stdout
    assert not result.stdout


def test_every_notebook_named_in_python_files_exists_and_holds_tests():
    # `python_files` reaches the notebook's own `test_*` functions by naming one exact path, so a
    # rename would silently stop collecting them — the same silent-drop failure this file exists
    # to prevent, one layer up.
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["pytest"]["ini_options"]["python_files"]
    named = [REPO_ROOT / pattern for pattern in patterns if "/" in pattern and "*" not in pattern]
    assert named, "expected python_files to name the notebook holding test cells"
    for path in named:
        assert path.is_file(), f"{path} is named in python_files but does not exist"
        tree = ast.parse(path.read_text())
        tests = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert any(n.name.startswith("test_") for n in tests), f"{path} defines no test_ function"
