from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr
from contracts.weather_schemas import ECMWF_ENS_ENSEMBLE_MEMBERS, ECMWF_ENS_LEAD_TIME_HOURS
from measure_wind_storage import _digest, _scalar_dataset, _validate_run


def test_scalar_magnitudes_precede_spatial_reduction():
    dataset = xr.Dataset(
        {
            f"wind_{axis}_{height}": ("point", values)
            for height in ("10m", "100m")
            for axis, values in (("u", [3.0, -3.0]), ("v", [4.0, -4.0]))
        }
    )
    result = _scalar_dataset(dataset=dataset)
    for height in ("10m", "100m"):
        assert float(result[f"wind_scalar_mean_speed_{height}"].mean()) == 5.0
        assert (
            float(np.hypot(result[f"wind_u_{height}"].mean(), result[f"wind_v_{height}"].mean()))
            == 0
        )


def test_scalar_reduction_rejects_different_component_footprints():
    dataset = xr.Dataset(
        {
            f"wind_{axis}_{height}": ("point", values)
            for height in ("10m", "100m")
            for axis, values in (("u", [3.0, float("nan")]), ("v", [4.0, 4.0]))
        }
    )
    with pytest.raises(ValueError, match="unmatched"):
        _scalar_dataset(dataset=dataset)


def test_full_run_check_rejects_equal_cardinality_wrong_members():
    init_time = datetime(2025, 1, 15, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "init_time": init_time,
            "valid_time": init_time + timedelta(hours=hour),
            "ensemble_member": member,
            "h3_index": 1,
        }
        for member in ECMWF_ENS_ENSEMBLE_MEMBERS
        for hour in ECMWF_ENS_LEAD_TIME_HOURS
    )
    grid = pl.DataFrame({"h3_index": [1]})
    _validate_run(frame=frame, h3_grid=grid, date="2025-01-15")
    wrong = frame.with_columns(ensemble_member=pl.col("ensemble_member") + 1)
    with pytest.raises(ValueError, match="misidentified"):
        _validate_run(frame=wrong, h3_grid=grid, date="2025-01-15")


def test_content_digest_detects_same_length_raw_change(tmp_path: Path):
    (tmp_path / "values").write_bytes(b"123")
    original = _digest(tmp_path)
    (tmp_path / "values").write_bytes(b"124")
    assert _digest(tmp_path) != original


def test_unavailable_dates_use_earlier_before_later_and_reject_duplicates():
    from measure_wind_storage import _select_dates

    assert _select_dates(requested=["2025-01-15"], available={"2025-01-14", "2025-01-16"}) == [
        "2025-01-14"
    ]
    with pytest.raises(ValueError, match="duplicate"):
        _select_dates(requested=["2025-01-15", "2025-01-31"], available={"2025-01-14"})


def test_retry_refuses_unowned_table_and_recovers_owned_table(tmp_path: Path):
    from measure_wind_storage import _prepare_table

    destination = tmp_path / "2025-01-15"
    destination.mkdir()
    (destination / "important").write_text("preserve me")
    with pytest.raises(ValueError, match="unmarked"):
        _prepare_table(destination=destination, identity={"sample": "first"})
    assert (destination / "important").read_text() == "preserve me"
    (destination / "important").unlink()
    destination.rmdir()
    _prepare_table(destination=destination, identity={"sample": "first"})
    destination.mkdir()
    (destination / "unfinished").touch()
    _prepare_table(destination=destination, identity={"sample": "first"})
    assert not destination.exists()


def test_output_owner_rejects_existing_data(tmp_path: Path):
    from measure_wind_storage import _own_output

    (tmp_path / "existing").touch()
    with pytest.raises(ValueError, match="nonempty"):
        _own_output(output=tmp_path, identity={"sample": "first"})
