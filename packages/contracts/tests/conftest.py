"""Shared fixtures for the ``contracts`` tests."""

from collections.abc import Iterator

import pytest
from contracts.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Give every test a fresh ``get_settings()`` cache.

    ``get_settings`` is a process-global ``lru_cache``, so without this the first test to call it
    (e.g. a defaulted ``NwpMetaData.load()`` or ``Nwp.scan_delta()``) would freeze the ``Settings``
    instance for the rest of the session — a later test that changes the environment and expects a
    rebuilt ``Settings`` would silently read the stale one. Clearing before and after keeps each
    test independent of execution order.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
