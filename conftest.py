"""Repo-root pytest configuration.

Gates the ``network``-marked tests behind an explicit ``--run-network`` flag so a plain ``uv run
pytest`` — local dev and the per-PR CI — never touches the real Dynamical.org catalog.

Why a collection hook rather than ``-m "not network"`` in ``addopts``: pytest keeps only the
*last* ``-m`` it sees, so any developer-supplied marker expression (e.g. ``-m "not
integration"``) silently replaces an ``addopts`` ``-m "not network"`` and re-includes the network
tests. A skip applied during collection cannot be defeated that way — the gate holds regardless
of what ``-m`` the caller passes. Run the network tests with ``uv run pytest --run-network`` (add
``-m network`` to run *only* them). See
<https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/>.

``--run-studies`` gates the tests under ``packages/studies/tests`` the same way, selecting them by
path, except ``test_study_boundaries.py``, which always runs.
"""

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

# Must be set here, at import time, not inside `pytest_configure` below: Polars reads
# `POLARS_MAX_THREADS` once, the first time it is imported, and `tests/conftest.py` (loaded as one
# of the initial conftests, before `pytest_configure` runs) imports Polars transitively via the
# Dagster defs module. See [Running the suite in
# parallel](https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#running-the-suite-in-parallel)
# for why the cap exists and why it's 4, not 1.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"


def pytest_configure(config: pytest.Config) -> None:
    """Neutralise any real ``SENTRY_DSN`` for the whole session.

    A developer's ``.env`` carries a live Sentry DSN (see the Sentry setup how-to), and pydantic
    reads it into ``Settings.sentry_dsn``. Importing the Dagster definitions module — which
    several tests do — then runs ``init_sentry`` at import with that live DSN, arming the SDK for
    the rest of the process. From then on any Sentry send in a test reaches the *real* project:
    for example the deliberate ``report_power_freshness`` error path in ``test_sentry.py`` logs
    at ``ERROR``, which the SDK's default log-to-event capture would ship (that capture is now
    also disabled in ``init_sentry``, but this env override is the belt-and-braces guard that
    holds even for a code path we haven't foreseen).

    Forcing the env var empty overrides the ``.env`` file value (env vars outrank the dotenv
    source in pydantic-settings), so every ``Settings`` built during the session sees an empty
    DSN and ``init_sentry`` stays a no-op. This runs before collection imports any test module,
    so it lands ahead of the import-time ``init_sentry`` call.
    """
    os.environ["SENTRY_DSN"] = ""


_STUDIES_TESTS_DIR = Path(__file__).parent / "packages" / "studies" / "tests"
_UNGATED_STUDIES_TEST_FILE = "test_study_boundaries.py"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-network`` and ``--run-studies`` opt-in flags."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (hit the real Dynamical.org NWP catalog).",
    )
    parser.addoption(
        "--run-studies",
        action="store_true",
        default=False,
        help="Run the slow tests under packages/studies/tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    """Skip each gated group of tests unless its opt-in flag was passed."""
    skip_network = pytest.mark.skip(
        reason="hits the real Dynamical.org catalog; pass --run-network"
    )
    skip_studies = pytest.mark.skip(reason="slow studies test; pass --run-studies")
    for item in items:
        if "network" in item.keywords and not config.getoption("--run-network"):
            item.add_marker(skip_network)
        # A path check, not a `studies` keyword: the `packages/studies` directory node is itself
        # a keyword of every test beneath it, including the ungated one.
        if (
            item.path.is_relative_to(_STUDIES_TESTS_DIR)
            and item.path.name != _UNGATED_STUDIES_TEST_FILE
            and not config.getoption("--run-studies")
        ):
            item.add_marker(skip_studies)
