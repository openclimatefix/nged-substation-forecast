"""Shared fixtures for the root integration tests.

The repo-root ``conftest.py`` (one level up) owns the ``--run-network`` gate; this one scopes
its fixtures to the ``tests/`` directory only, so nothing here touches the ``packages/*/tests``
unit suites.
"""

from collections.abc import Iterator
from typing import Any

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


@pytest.fixture(autouse=True)
def _fail_on_an_undisposed_dagster_instance(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail the test that leaves a ``DagsterInstance.ephemeral()`` instance for the collector.

    Being autouse, this fixture is set up before the ones a test asks for by name and so torn down
    after them, which is what lets it see ``dagster_instance`` dispose.

    It catches an instance that *survives* the test, not a call that forgot to own one: CPython
    frees an unreferenced context as soon as the invoking helper returns, so a missing
    ``instance=`` still passes green until something — a captured traceback, most often — pins the
    frame holding it. Instances made by any route other than ``ephemeral()``, such as
    ``local_temp()``, are invisible to it.
    """
    undisposed: set[DagsterInstance] = set()
    make_ephemeral = DagsterInstance.ephemeral
    dispose = DagsterInstance.dispose

    def _tracked_ephemeral(*args: Any, **kwargs: Any) -> DagsterInstance:
        instance = make_ephemeral(*args, **kwargs)
        undisposed.add(instance)
        return instance

    def _tracked_dispose(self: DagsterInstance) -> None:
        undisposed.discard(self)
        dispose(self)

    monkeypatch.setattr(DagsterInstance, "ephemeral", _tracked_ephemeral)
    monkeypatch.setattr(DagsterInstance, "dispose", _tracked_dispose)

    yield

    assert not undisposed, (
        f"{len(undisposed)} ephemeral DagsterInstance(s) outlived this test. Take the "
        "`dagster_instance` fixture, or enter the instance as a context manager."
    )
