"""Tests for `studies/weather_downloads/validate_ukv_aws.py` on small synthetic run files.

No test touches `data/`. Each is built to fail on the defect its check exists for: a truncated
file read as fine, a constant field passed as data, a shifted timestamp, a radiation budget that
does not close, an unfinished day counted as committed, an absent object counted as data, and a
ledger that contradicts its run files.
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
    absent: tuple[tuple[str, int], ...] = (),
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
    for name, lead in absent:
        arrays[name][lead] = np.nan
        arrays[f"{name}__time_s"][lead] = pilot.NO_BOUNDS
    arrays["missing"] = np.array(json.dumps([f"{lead}:{name}" for name, lead in absent]))
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


DIRECT = "radiation_flux_in_shortwave_direct_downward_at_surface"
TEMPERATURE = "temperature_at_screen_level"


def test_an_absent_direct_object_does_not_make_the_residual_nan(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path, residual=25.0, absent=((DIRECT, 1),))
    record = validate.summarise_run(path=path)
    assert record["residual_mean"] == pytest.approx(25.0, rel=1e-3)
    assert record["residual_signed_mean"] == pytest.approx(25.0, rel=1e-3)
    assert record["daylight_values"] == 5 * ROWS * COLS


def test_an_absent_lead_keeps_the_time_check_and_the_fetched_nan_count_clean(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path, absent=((TEMPERATURE, 0),))
    record = validate.summarise_run(path=path)
    assert record["time_ok"]
    assert record["absent_objects"] == 1
    assert record["objects"] == 47
    assert record[f"{TEMPERATURE}__nan"] == ROWS * COLS
    assert record[f"{TEMPERATURE}__nan_fetched"] == 0


def test_a_nan_inside_a_fetched_lead_is_counted_as_fetched_nan(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[TEMPERATURE][2, 0, 0] = np.nan
    np.savez_compressed(path, allow_pickle=False, **arrays)
    assert validate.summarise_run(path=path)[f"{TEMPERATURE}__nan_fetched"] == 1


def _frame(validate: ModuleType, pilot: ModuleType, tmp_path: Path, specs: list[dict]) -> object:
    import polars as pl

    records = []
    for index, spec in enumerate(specs):
        path = tmp_path / f"20241008T{index:02d}00Z.npz"
        _write_run(pilot=pilot, path=path, **spec)
        records.append(validate.summarise_run(path=path))
    return pl.DataFrame(records, infer_schema_length=None)


def test_runs_with_an_absent_lead_0_temperature_are_not_duplicates(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    absent = {"absent": ((TEMPERATURE, 0),)}
    frame = _frame(validate, pilot, tmp_path, [{"seed": 1, **absent}, {"seed": 2, **absent}])
    findings = " ".join(validate.consistency_findings(records=frame))
    assert "Duplicates: 0 of 0 runs" in findings
    assert "absent (left out of the duplicate check): 2" in findings


def test_two_runs_with_the_same_lead_0_temperature_are_named_as_duplicates(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    frame = _frame(validate, pilot, tmp_path, [{"seed": 1}, {"seed": 1}, {"seed": 2}])
    findings = " ".join(validate.consistency_findings(records=frame))
    assert "Duplicates: 2 of 3 runs" in findings
    assert "20241008T0000Z, 20241008T0100Z" in findings


def test_hour_of_day_profile_labels_each_lead_with_its_valid_hour(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "20241008T2200Z.npz"
    _write_run(pilot=pilot, path=path)
    import polars as pl

    frame = pl.DataFrame([validate.summarise_run(path=path)], infer_schema_length=None)
    profile = validate.hour_of_day_profile(records=frame)
    assert profile["valid_hour"].to_list() == [0, 1, 2, 3, 22, 23]


def test_ledger_findings_flag_a_ledger_that_contradicts_its_run_files(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    import polars as pl

    runs = tmp_path / "runs"
    runs.mkdir()
    path = runs / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path, absent=((TEMPERATURE, 0),))
    frame = pl.DataFrame([validate.summarise_run(path=path)], infer_schema_length=None)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "20241008.json").write_text(json.dumps({"absent_objects": 0, "complete": True}))
    days = [dt.date(2024, 10, 8)]
    findings = validate.ledger_findings(
        product_dir=tmp_path, records=frame, days=days, run_days=["20241008", "20241009"]
    )
    assert len(findings) == 3
    assert "ledger says 0 absent objects, the run files record 1" in findings[0]
    assert "says complete" in findings[1]
    assert "20241009" in findings[2]
    (ledger / "20241008.json").write_text(json.dumps({"absent_objects": 1, "complete": False}))
    clean = validate.ledger_findings(
        product_dir=tmp_path, records=frame, days=days, run_days=["20241008"]
    )
    assert clean == [
        "Ledger against run files: 1 day(s) agree, and no run file lacks a ledger entry."
    ]


def test_a_nan_inside_a_fetched_lead_reaches_the_days_with_a_finding(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    import polars as pl

    def day_report(*, nan_in_run_0: bool) -> str:
        folder = tmp_path / str(nan_in_run_0)
        folder.mkdir()
        records = []
        for hour in range(24):
            path = folder / f"20241008T{hour:02d}00Z.npz"
            _write_run(pilot=pilot, path=path, seed=hour)
            if nan_in_run_0 and hour == 0:
                with np.load(path) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                arrays[TEMPERATURE][2, 0, 0] = np.nan
                np.savez_compressed(path, allow_pickle=False, **arrays)
            records.append(validate.summarise_run(path=path))
        frame = pl.DataFrame(records, infer_schema_length=None)
        return validate.build_report(records=frame, days=[dt.date(2024, 10, 8)])[0]

    assert "0 of 1 days" in day_report(nan_in_run_0=False)
    assert "1 of 1 days" in day_report(nan_in_run_0=True)


def test_the_era_split_puts_the_level_change_day_in_the_later_era(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    import polars as pl

    records = []
    for name, residual in (("20260121T0000Z", 25.0), ("20260122T0000Z", 0.0)):
        path = tmp_path / f"{name}.npz"
        _write_run(pilot=pilot, path=path, residual=residual)
        records.append(validate.summarise_run(path=path))
    frame = pl.DataFrame(records, infer_schema_length=None)
    eras = validate.radiation_by_era(records=frame)
    by_era = dict(zip(eras["era"], eras["mean_abs_residual_w_m2"], strict=True))
    assert by_era["before 2026-01-22"] == pytest.approx(25.0, rel=1e-3)
    assert by_era["from 2026-01-22"] == pytest.approx(0.0, abs=1e-3)
    profile = validate.hour_of_day_profile(records=frame)
    assert sorted(set(profile["era"])) == ["before 2026-01-22", "from 2026-01-22"]


def test_hour_of_day_total_averages_every_finite_cell_not_only_daylight_cells(
    validate: ModuleType, pilot: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / f"{RUN}.npz"
    _write_run(pilot=pilot, path=path)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    total = arrays["radiation_flux_in_shortwave_total_downward_at_surface"]
    total[:, 0, :] = 0.0
    np.savez_compressed(path, allow_pickle=False, **arrays)
    record = validate.summarise_run(path=path)
    assert record["hod_total_mean_0"] == pytest.approx(float(total[0].mean()), rel=1e-6)
