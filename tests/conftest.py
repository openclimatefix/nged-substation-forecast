"""Shared fixtures for the root integration tests.

The repo-root ``conftest.py`` (one level up) owns the ``--run-network`` gate; this one scopes
its fixtures to the ``tests/`` directory only, so nothing here touches the ``packages/*/tests``
unit suites.
"""

from collections.abc import Iterator

import pytest
from dagster import DagsterInstance


@pytest.fixture
def dagster_instance() -> Iterator[DagsterInstance]:
    """A fresh ephemeral ``DagsterInstance``, disposed when the test finishes.

    Every test here should take this fixture rather than calling ``DagsterInstance.ephemeral()``
    itself; outside the test suite the equivalent is to use the instance as a context manager. An
    ephemeral
    instance's in-memory run storage and event-log storage each open one SQLAlchemy connection
    against an in-memory SQLite database and hold it for the life of the instance. Nothing closes
    those two connections unless ``dispose()`` is called, and ``DagsterInstance`` has no
    finaliser — so an instance handed to the garbage collector can survive to interpreter
    shutdown, where SQLAlchemy's connection-pool finaliser may run *after* SQLite has closed the
    underlying database. That prints a bare ``Exception during reset or similar`` traceback ending
    in ``sqlite3.ProgrammingError: Cannot operate on a closed database`` — two of them, one per
    held connection — after pytest's own summary line, where no test owns it.

    Entering the instance as a context manager makes ``__exit__`` call ``dispose()``, which closes
    both connections and removes the instance's temporary artifact directory, at the end of the
    test that created it.
    """
    with DagsterInstance.ephemeral() as instance:
        yield instance
