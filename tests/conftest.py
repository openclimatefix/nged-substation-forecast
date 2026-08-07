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


@pytest.fixture(autouse=True)
def _dummy_nged_s3_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placeholder NGED source-bucket credentials.

    Every integration test here builds a ``Settings``-backed object store, which requires these
    three variables to be present — but none of the tests actually reach the real NGED bucket
    (they read from temp Delta tables). Setting dummy values once, for every test in this
    directory, removes the per-fixture boilerplate. A test that needs real values (e.g. the moto
    S3 test constructs ``Settings`` with explicit kwargs) overrides them regardless.
    """
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
