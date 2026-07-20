"""Shared fixtures for the root integration tests.

The repo-root ``conftest.py`` (one level up) owns the ``--run-network`` gate; this one scopes
its fixtures to the ``tests/`` directory only, so nothing here touches the ``packages/*/tests``
unit suites.
"""

import pytest


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
