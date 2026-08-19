"""Parallelise a full-suite run with `pytest-xdist`; leave a targeted run serial.

This can't be a plain `addopts = "-n auto"` because worker start-up (each worker importing the
whole Dagster/Polars/XGBoost/MLflow stack) costs a few seconds flat, paid whether one test runs or
the whole suite does. That's a good trade for the full suite, but a bad one for the "single test"
loop CLAUDE.md documents (`uv run pytest path/to/test_foo.py::test_bar`), where it would be pure
overhead spread across workers for one test.

So: add `-n auto` only when the invocation names no explicit file or node id, i.e. a plain
`uv run pytest` (or one already carrying its own `-n`, which is left alone). A `-k`/`-m` selection
does *not* suppress the injection — a filtered run can still be many tests, and is worth
parallelising the same as an unfiltered one. `-s`/`--capture=no` does suppress it: under xdist a
worker's captured-off stdout never reaches the controller, so injecting `-n auto` into a `-s` run
would silently drop the very output the developer asked to see — exactly the debug-print loop
CLAUDE.md's "single test" workflow exists for.

Reading `early_config.known_args_namespace` rather than hand-scanning `args` is deliberate: pytest
has already parsed `args` into that namespace by the time this hook fires, so `file_or_dir` and
`capture` are exactly the fields pytest itself considers "there's an explicit target" and "output
capture is off" — including every value-taking flag pytest recognises (`-W`, `-k`, `-m`, `-o`,
`--deselect`, …), not just the ones this file happens to name. The one gap: a *conftest-defined*
option that takes a value is not yet known to pytest when this hook's `known_args_namespace` was
last parsed, so its value could still be misread as a positional target. Harmless today — this
repo's only custom option, `--run-network`, is `store_true` — but a future value-taking conftest
option would need checking against this file.

There's no need to also check `numprocesses`: an injected `-n auto` is prepended (`args[:0]`), so
an explicit `-n`/`--numprocesses` anywhere else in `args` — CLI, `addopts`, `PYTEST_ADDOPTS` —
still comes later in the list, and argparse keeps the last value for a `store` action (which is
what xdist's `-n` uses). A `numprocesses is None` guard here would be redundant with that ordering,
not a second line of defence — confirmed by deleting it and watching the regression suite still
pass in full.

This has to be a real, separately-loaded plugin rather than a `pytest_load_initial_conftests` hook
in the root `conftest.py`. That hook fires as part of pytest's own bootstrap logic for loading the
initial conftest files — including the root `conftest.py` itself — so a hookimpl defined inside
that same file is never registered in time to run for that call. Registering this module instead
via `addopts = "-p _pytest_autoinject"` loads it during `Config.parse`'s `-p`-plugin-loading phase
(`consider_preparse`), which runs *before* `pytest-xdist` loads (`load_setuptools_entrypoints`,
the next phase). `Config.parse` then re-parses `args` a third time, now that xdist's own `-n`
option has been registered — that third parse is what makes `-n`/`--numprocesses` and `capture`
visible on `known_args_namespace` by the time `pytest_load_initial_conftests` fires here.
"""

import pytest


def pytest_load_initial_conftests(early_config: pytest.Config, args: list[str]) -> None:
    if not early_config.pluginmanager.hasplugin("xdist"):
        # `-p no:xdist` or `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` leaves xdist unloaded; injecting `-n`
        # with no plugin to consume it is a pytest usage error, not a silent fallback to serial.
        return
    ns = early_config.known_args_namespace
    if not ns.file_or_dir and ns.capture != "no":
        args[:0] = ["-n", "auto"]
