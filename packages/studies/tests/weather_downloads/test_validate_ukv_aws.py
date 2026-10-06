"""Tests for `studies/weather_downloads/validate_ukv_aws.py` on small synthetic run files.

No test touches `data/`. Each is built to fail on the defect its check exists for: a truncated
file read as fine, a constant field passed as data, a shifted timestamp, a radiation budget that
does not close, and an unfinished day counted as committed.
"""

import datetime as dt
import importlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("polars")
pytest.importorskip("h5py")
pytest.importorskip("fsspec")
pytest.importorskip("aiohttp")
pytest.importorskip("pyproj")

RUN = "20241008T0000Z"
ROWS, COLS = 4, 5


@pytest.fixture(scope="module")
def validate() -> ModuleType:
    """Import the script as a module."""
    return importlib.import_module("validate_ukv_aws")


@pytest.fixture(scope="module")
def pilot() -> ModuleType:
    """Import the download script, which names the variables."""
    return importlib.import_module("fetch_ukv_aws_pilot")


def _write_run(
    *,
    pilot: ModuleType,
    path: Path,
    residual: float = 0.0,
    time_shift_s: int = 0,
    constant_temperature: bool = False,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    run_time = int(
        dt.datetime.strptime(path.stem, "%Y%m%dT%H%MZ").replace(tzinfo=dt.UTC).timestamp()
    )
    leads = len(pilot.LEADS_HOURS)
    arrays: dict[str, np.ndarray] = {
        "x": np.arange(COLS, dtype=np.float64),
        "y": np.arange(ROWS, dtype=np.float64),
        "lead_hours": np.array(pilot.LEADS_HOURS),
        "missing": np.array(json.dumps([])),
        "wire_bytes": np.array(0),
        "height_m": np.array([50.0, 75.0, 100.0, 150.0]),
    }
    for name in pilot.ALL_FILES:
        shape = (leads, 4, ROWS, COLS) if name in pilot.LEVEL_FILES else (leads, ROWS, COLS)
        low, high = {
            "temperature_at_screen_level": (280.0, 290.0),
            "wind_direction_at_10m": (0.0, 360.0),
            "wind_direction_on_height_levels": (0.0, 360.0),
        }.get(name, (1.0, 10.0))
        arrays[name] = rng.uniform(low, high, shape).astype(np.float32)
        arrays[f"{name}__time_s"] = run_time + time_shift_s + 3600 * np.arange(leads)
        arrays[f"{name}__time_bounds_s"] = np.full((leads, 2), pilot.NO_BOUNDS, dtype=np.int64)
        arrays[f"{name}__attributes"] = np.array(json.dumps({"units": "x"}))
    if constant_temperature:
        arrays["temperature_at_screen_level"][:] = 285.0
    direct = arrays["radiation_flux_in_shortwave_direct_downward_at_surface"]
    diffuse = arrays["radiation_flux_in_shortwave_diffuse_downward_at_surface"]
    arrays["radiation_flux_in_shortwave_total_downward_at_surface"] = direct + diffuse + residual
    np.savez_compressed(path, allow_pickle=False, **arrays)


def test_a_clean_run_has_no_findings(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path)
    record = validate.summarise_run(path=path)
    assert record["readable"]
    assert record["objects"] == 48
    assert record["time_ok"]
    assert record["lead_hours_ok"]
    assert not record["has_time_bounds"]
    assert record["y_increasing"]
    assert record["temperature_at_screen_level__nan"] == 0
    assert record["temperature_at_screen_level__outside"] == 0
    assert record["temperature_at_screen_level__stuck"] == 0
    assert record["residual_mean"] == pytest.approx(0.0, abs=1e-4)


def test_a_truncated_file_is_reported_unreadable(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path)
    path.write_bytes(path.read_bytes()[:200])
    record = validate.summarise_run(path=path)
    assert not record["readable"]


def test_a_shifted_timestamp_fails_the_time_check(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path, time_shift_s=3600)
    assert not validate.summarise_run(path=path)["time_ok"]


def test_a_constant_temperature_field_counts_as_stuck(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path, constant_temperature=True)
    assert validate.summarise_run(path=path)["temperature_at_screen_level__stuck"] == 6


def test_an_open_radiation_budget_shows_in_the_residual(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path, residual=25.0)
    record = validate.summarise_run(path=path)
    assert record["residual_mean"] == pytest.approx(25.0, rel=1e-3)
    assert record["residual_share_over"] == 1.0


def test_report_flags_a_day_with_too_few_runs_and_names_it(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    import polars as pl

    records = []
    for hour in range(3):
        path = tmp_path / f"20241008T{hour:02d}00Z.npz"
        _write_run(pilot=pilot, path=path, seed=hour)
        records.append(validate.summarise_run(path=path))
    frame = pl.DataFrame(records, infer_schema_length=None)
    days = [dt.date(2024, 10, 8)]
    assert validate.runs_missing_per_day(records=frame, days=days) == ["2024-10-08: 3 run files"]
    report, daily = validate.build_report(records=frame, days=days)
    assert daily["runs"].to_list() == [3]
    assert "2024-10-08: 3 run files" in report


def test_committed_days_ignores_unfinished_ledger_files(
    validate: ModuleType, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "20241006.json").write_text("{}")
    (ledger / "20241007.json.partial").write_text("{}")
    assert validate.committed_days(product_dir=tmp_path) == [dt.date(2024, 10, 6)]
