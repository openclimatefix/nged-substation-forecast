"""Tests for the validator changes in `studies/weather_downloads/validate_ukv_ceda.py`.

No test touches the network or `data/`. Each test fails on the bug it exists for: a cloud upper
bound tighter than the source's percentage packing allows, a gradient check that runs on too few
runs, a check that crashes instead of skipping when it has no data, and a validator that reads a
store written under the other product.
"""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import pytest

pytest.importorskip("icechunk")
pytest.importorskip("zarr")
pytest.importorskip("pyproj")

import zarr

SCRIPT_DIR: Final[Path] = Path(__file__).parent.parent / "studies" / "weather_downloads"


def _load(name: str) -> ModuleType:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch = _load("fetch_ukv_ceda")
validate = _load("validate_ukv_ceda")


@pytest.fixture(autouse=True)
def _restore_default_profile() -> Iterator[None]:
    fetch.set_profile(fetch.DEFAULT_PROFILE)
    yield
    fetch.set_profile(fetch.DEFAULT_PROFILE)


def _group_with_cloud_value(cloud_value: float) -> zarr.Group:
    """One run, one lead, two cells: every variable mid-range, the two cloud ones at a value."""
    group = zarr.group()
    for spec in fetch.FIELDS:
        low, high = validate.VALUE_RANGES[spec.variable]
        value = cloud_value if spec.variable in ("cloud_total", "cloud_high") else (low + high) / 2
        group.create_array(
            spec.variable, data=np.full((1, 1, 2), value, dtype=np.float32), chunks=(1, 1, 2)
        )
    return group


@pytest.mark.parametrize(("cloud_value", "passes"), [(100.05, True), (100.2, False)])
def test_cloud_upper_bound_is_100_point_1(cloud_value: float, passes: bool) -> None:
    assert validate.VALUE_RANGES["cloud_total"][1] == pytest.approx(100.1)
    assert validate.VALUE_RANGES["cloud_high"][1] == pytest.approx(100.1)
    group = _group_with_cloud_value(cloud_value)
    ok, _ = validate.check_value_ranges(group, np.array([0]))
    assert ok is passes


def test_north_south_gradient_skips_below_the_minimum_runs() -> None:
    few = np.arange(validate.MIN_RUNS_FOR_GRADIENT - 1)
    outcome, _ = validate.check_north_south_gradient(zarr.group(), few)
    assert outcome is None


def test_north_south_gradient_minimum_is_exactly_20_runs() -> None:
    assert validate.MIN_RUNS_FOR_GRADIENT == 20
    skipped, _ = validate.check_north_south_gradient(zarr.group(), np.arange(19))
    assert skipped is None
    with pytest.raises(KeyError):  # an empty group has no cell_latitude, so the check ran
        validate.check_north_south_gradient(zarr.group(), np.arange(20))


def test_adjacent_runs_differ_skips_with_no_pairs() -> None:
    outcome, _ = validate.check_adjacent_runs_differ(zarr.group(), np.array([], dtype=int))
    assert outcome is None


def test_temperature_continuity_skips_with_no_runs() -> None:
    outcome, _ = validate.check_temperature_continuity(zarr.group(), np.array([], dtype=int))
    assert outcome is None


def test_main_fails_for_a_store_of_the_wrong_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch.set_profile(fetch.T120_PROFILE)
    fetch.UkvStore.open(store_path=tmp_path / "store").initialise(
        fetch.CellGrid(
            row_start=0,
            col_start=0,
            n_rows=1,
            n_cols=2,
            latitude=np.array([52.0, 52.0]),
            longitude=np.array([-1.0, -0.98]),
        )
    )
    monkeypatch.setattr(
        sys, "argv", ["validate", "--product", "ukv-ceda", "--store-dir", str(tmp_path)]
    )
    assert validate.main() == 1
    assert "FAIL product" in capsys.readouterr().out


def test_main_returns_zero_when_every_check_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = fetch.UkvStore.open(store_path=tmp_path / "store")
    store.initialise(
        fetch.CellGrid(
            row_start=0,
            col_start=0,
            n_rows=1,
            n_cols=2,
            latitude=np.array([52.0, 52.0]),
            longitude=np.array([-1.0, -0.98]),
        )
    )
    init_time = fetch.active_profile().slot_epoch
    store.commit_run(
        fetch.RunResult(
            init_time=init_time,
            status=fetch.STATUS_COMPLETE,
            files_expected=7,
            files_received=7,
            blocks={},
        )
    )
    for name in (
        "check_run_spacing",
        "check_status_counts",
        "check_value_ranges",
        "check_nan_layout",
        "check_shortwave_diurnal_cycle",
        "check_north_south_gradient",
        "check_gust_max_is_a_maximum",
        "check_gaps",
    ):
        monkeypatch.setattr(validate, name, lambda *args, **kwargs: (None, {}))
    monkeypatch.setattr(
        sys, "argv", ["validate", "--product", "ukv-ceda", "--store-dir", str(tmp_path)]
    )
    assert validate.main() == 0
    output = capsys.readouterr().out
    assert "SKIP run_spacing" in output
    assert "FAIL" not in output
