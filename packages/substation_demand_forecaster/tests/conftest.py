"""Fixtures of a synthetic step-5 bundle, written to a temporary directory."""

from collections.abc import Callable
from pathlib import Path

import pytest
from _step5_bundle_test_data import write_bundle
from substation_demand_forecaster.bundle import Bundle, load_bundle


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return load_bundle(write_bundle(tmp_path / "bundle"))


@pytest.fixture
def write_bundle_at() -> Callable[..., Path]:
    """`write_bundle`, for a test that edits the bundle before loading it."""
    return write_bundle
