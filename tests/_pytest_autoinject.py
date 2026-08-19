"""Parallelise a full-suite run with `pytest-xdist`; leave a targeted run serial.

This can't be a plain `addopts = "-n auto"` because worker start-up (each worker importing the
whole Dagster/Polars/XGBoost/MLflow stack) costs a few seconds flat, paid whether one test runs or
the whole suite does. That's a good trade for the full suite, but a bad one for the "single test"
loop CLAUDE.md documents (`uv run pytest path/to/test_foo.py::test_bar`), where it would be pure
overhead spread across workers for one test.

So: add `-n auto` only when the invocation names no explicit file or node id, i.e. a plain
`uv run pytest` (or one already carrying its own `-n`/`-k`/`-m`, which is left alone).

This has to be a real, separately-loaded plugin rather than a `pytest_load_initial_conftests` hook
in the root `conftest.py`. That hook fires as part of pytest's own bootstrap logic for loading the
initial conftest files — including the root `conftest.py` itself — so a hookimpl defined inside
that same file is never registered in time to run for that call. Registering this module instead
via `addopts = "-p _pytest_autoinject"` loads it earlier, during setuptools-entry-point/`-p`
plugin loading, which completes before `pytest_load_initial_conftests` fires.
"""


def pytest_load_initial_conftests(args: list[str]) -> None:
    has_explicit_numprocesses = any(
        arg == "-n" or arg.startswith(("-n=", "--numprocesses")) for arg in args
    )
    has_explicit_target = False
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-p", "--plugin"):
            skip_next = True
            continue
        if not arg.startswith("-"):
            has_explicit_target = True
    if not has_explicit_target and not has_explicit_numprocesses:
        args[:0] = ["-n", "auto"]
