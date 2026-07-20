"""Repo-root pytest configuration.

Gates the ``network``-marked tests behind an explicit ``--run-network`` flag so a plain
``uv run pytest`` — local dev and the per-PR CI — never touches the real Dynamical.org catalog.

Why a collection hook rather than ``-m "not network"`` in ``addopts``: pytest keeps only the *last*
``-m`` it sees, so any developer-supplied marker expression (e.g. ``-m "not integration"``) silently
replaces an ``addopts`` ``-m "not network"`` and re-includes the network tests. A skip applied during
collection cannot be defeated that way — the gate holds regardless of what ``-m`` the caller passes.
Run the network tests with ``uv run pytest --run-network`` (add ``-m network`` to run *only* them).
See docs/architecture/testing.md.
"""

from collections.abc import Iterable

import pytest


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
