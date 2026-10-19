"""Tests of the feature-tensor cache of the differentiable-physics series (`studies.dp_features`).

The fixtures in `conftest.py` build the cache from synthetic inputs with the real builder. These
tests need no torch, so they run where the optional `gpu` dependency group is not installed.
"""

from dataclasses import replace

import numpy as np
import pytest
from studies.dp_features import (
    FeatureCache,
    input_key,
    load_feature_cache,
    save_feature_cache,
)


def test_the_cache_key_changes_with_an_input_file(dp_synthetic_inputs, tmp_path):  # noqa: ANN001
    key = input_key(paths=dp_synthetic_inputs)
    copy = tmp_path / "sites.parquet"
    copy.write_bytes(dp_synthetic_inputs.sites.read_bytes() + b"x")
    changed = replace(dp_synthetic_inputs, sites=copy)
    assert input_key(paths=changed) != key
    assert input_key(paths=dp_synthetic_inputs) == key


def test_the_cache_survives_a_round_trip_through_a_file(dp_synthetic_cache: FeatureCache, tmp_path):  # noqa: ANN001
    path = tmp_path / "cache.npz"
    save_feature_cache(cache=dp_synthetic_cache, path=path)
    loaded = load_feature_cache(path=path)
    original = dp_synthetic_cache.sites["S1"]
    restored = loaded.sites["S1"]
    assert loaded.key == dp_synthetic_cache.key
    assert restored.table == original.table
    assert np.array_equal(restored.y, original.y, equal_nan=True)
    assert np.array_equal(
        restored.smoothed_temperature_c, original.smoothed_temperature_c, equal_nan=True
    )
    assert np.array_equal(loaded.calendar.local_date, dp_synthetic_cache.calendar.local_date)
    assert np.array_equal(loaded.end_times, dp_synthetic_cache.end_times)


def test_the_builder_converts_power_to_points_of_effective_capacity(
    dp_synthetic_cache: FeatureCache,
):
    features = dp_synthetic_cache.sites["S1"]
    reading = features.y[np.isfinite(features.y)]
    assert reading[0] == pytest.approx(100.0 * 5.0 / 10.0)
    assert dp_synthetic_cache.end_times.max() < np.datetime64("2026-07-01", "us")
