"""Repo-root pytest configuration.

Gates the ``network``-marked tests behind an explicit ``--run-network`` flag so a plain
``uv run pytest`` — local dev and the per-PR CI — never touches the real Dynamical.org catalog.

Why a collection hook rather than ``-m "not network"`` in ``addopts``: pytest keeps only the *last*
``-m`` it sees, so any developer-supplied marker expression (e.g. ``-m "not integration"``) silently
replaces an ``addopts`` ``-m "not network"`` and re-includes the network tests. A skip applied
during collection cannot be defeated that way — the gate holds regardless of what ``-m`` the caller
passes. Run the network tests with ``uv run pytest --run-network`` (add ``-m network`` to run *only*
them). See
<https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/>.
"""

import os
from collections.abc import Iterable

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Neutralise any real ``SENTRY_DSN``, and cap per-process thread pools, for the whole session.

    A developer's ``.env`` carries a live Sentry DSN (see the Sentry setup how-to), and pydantic
    reads it into ``Settings.sentry_dsn``. Importing the Dagster definitions module — which several
    tests do — then runs ``init_sentry`` at import with that live DSN, arming the SDK for the rest
    of the process. From then on any Sentry send in a test reaches the *real* project: for example
    the deliberate ``report_power_freshness`` error path in ``test_sentry.py`` logs at ``ERROR``,
    which the SDK's default log-to-event capture would ship (that capture is now also disabled in
    ``init_sentry``, but this env override is the belt-and-braces guard that holds even for a code
    path we haven't foreseen).

    Forcing the env var empty overrides the ``.env`` file value (env vars outrank the dotenv source
    in pydantic-settings), so every ``Settings`` built during the session sees an empty DSN and
    ``init_sentry`` stays a no-op. This runs before collection imports any test module, so it lands
    ahead of the import-time ``init_sentry`` call.

    ``OMP_NUM_THREADS``/``POLARS_MAX_THREADS`` are capped at 1 for the same reason `pytest-xdist`
    (see ``addopts`` in ``pyproject.toml``) is worth having at all: Polars and XGBoost each default
    to spawning one thread per logical core, so with `-n auto` spinning up one worker *process* per
    core too, the suite oversubscribes the machine by core-count², and every worker's thread pool
    fights the others for the same cores. Measured on a 16-core box: `-n auto` alone made the suite
    *slower* than serial (82s vs 40s); adding these caps brought it to 23s — process-level
    parallelism replaces thread-level parallelism instead of stacking on top of it. Set here rather
    than left to each developer's shell so a plain `uv run pytest` gets the fast path everywhere,
    including CI.
    """
    os.environ["SENTRY_DSN"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["POLARS_MAX_THREADS"] = "1"


def pytest_load_initial_conftests(args: list[str]) -> None:
    """Parallelise a full-suite run with `pytest-xdist`; leave a targeted run serial.

    This can't be a plain `addopts = "-n auto"` because worker start-up (each of `-n auto`'s ~8
    workers importing the whole Dagster/Polars/XGBoost/MLflow stack) costs a few seconds flat,
    paid whether one test runs or all 751 do. That's a good trade for the full suite — it drops
    from ~44s to ~24s — but a bad one for the "single test" loop CLAUDE.md documents alongside it
    (`uv run pytest path/to/test_foo.py::test_bar`), which this measured at 5.3s serial vs 9.4s
    parallel: pure overhead with only one test to spread across workers.

    So: add `-n auto` only when the invocation names no explicit file or node id, i.e. a plain
    `uv run pytest` (or one already carrying its own `-n`/`-k`/`-m`, which is left alone). Modifying
    `args` in place here is the standard way to inject options conditionally, before pytest's own
    argument parsing runs — an `addopts` string is static and can't see the invocation first.
    """
    has_explicit_target = any(not arg.startswith("-") for arg in args)
    has_explicit_numprocesses = any(
        arg == "-n" or arg.startswith(("-n=", "--numprocesses")) for arg in args
    )
    if not has_explicit_target and not has_explicit_numprocesses:
        args[:0] = ["-n", "auto"]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-network`` opt-in flag."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (hit the real Dynamical.org NWP catalog).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip every ``network``-marked test unless ``--run-network`` was passed."""
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(
        reason="hits the real Dynamical.org catalog; pass --run-network"
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
