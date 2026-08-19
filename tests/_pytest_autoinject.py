"""Parallelise a full-suite run with `pytest-xdist`; leave a targeted run serial.

This can't be a plain `addopts = "-n auto"` because worker start-up (each worker importing the
whole Dagster/Polars/XGBoost/MLflow stack) costs a few seconds flat, paid whether one test runs or
the whole suite does. That's a good trade for the full suite, but a bad one for the "single test"
loop CLAUDE.md documents (`uv run pytest path/to/test_foo.py::test_bar`), where it would be pure
overhead spread across workers for one test.

So: add `-n auto` only when the invocation names no explicit file or node id, i.e. a plain
`uv run pytest` (or one already carrying its own `-n`, which is left alone). A `-k`/`-m` selection
does *not* suppress the injection — a filtered run can still be many tests, and is worth
parallelising the same as an unfiltered one.

Reading `early_config.known_args_namespace` rather than hand-scanning `args` is deliberate: pytest
has already parsed `args` into that namespace by the time this hook fires (`Config.parse` calls
`parse_known_args` twice, and this hook runs after both), so `file_or_dir` and `numprocesses` are
exactly the two fields pytest itself considers "there's an explicit target" and "there's an
explicit worker count" — including every value-taking flag pytest recognises (`-W`, `-k`, `-m`,
`-o`, `--deselect`, …), not just the ones this file happens to name.

This has to be a real, separately-loaded plugin rather than a `pytest_load_initial_conftests` hook
in the root `conftest.py`. That hook fires as part of pytest's own bootstrap logic for loading the
initial conftest files — including the root `conftest.py` itself — so a hookimpl defined inside
that same file is never registered in time to run for that call. Registering this module instead
via `addopts = "-p _pytest_autoinject"` loads it during the `-p`-plugin-loading phase, which
completes before `pytest_load_initial_conftests` fires (`pytest-xdist` itself, an entry-point
plugin, is loaded even earlier, in the phase before that — which is why `xdist`'s own `-n` option
is already visible on `known_args_namespace` here).
"""

import pytest


def pytest_load_initial_conftests(early_config: pytest.Config, args: list[str]) -> None:
    if not early_config.pluginmanager.hasplugin("xdist"):
        # `-p no:xdist` or `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` leaves xdist unloaded; injecting `-n`
        # with no plugin to consume it is a pytest usage error, not a silent fallback to serial.
        return
    ns = early_config.known_args_namespace
    if not ns.file_or_dir and ns.numprocesses is None:
        args[:0] = ["-n", "auto"]
