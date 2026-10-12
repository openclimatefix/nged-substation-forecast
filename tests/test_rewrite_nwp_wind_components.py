"""Archive migration preserves a pinned snapshot and resumes into a separate table."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
import pytest
from _nwp_test_data import nwp_records
from contracts.weather_schemas import Nwp
from delta_store.nwp import NWP_SIGNIFICAND_BITS
from delta_store.precision import round_to_significand_bits
from polars.testing import assert_frame_equal

SCRIPT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1] / "scripts/maintenance/rewrite_nwp_wind_components.py"
)
spec = importlib.util.spec_from_file_location("rewrite_wind", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def _old_partition(init_time: datetime, speed: float = 5.0) -> pl.DataFrame:
    records = nwp_records(
        cell=1, day=init_time, members=(0,), valid_times=[init_time + timedelta(hours=3)]
    )
    frame = (
        pl.DataFrame(records)
        .drop(
            [f"wind_{axis}_{height}" for height in ("10m", "100m") for axis in ("u", "v")],
            strict=False,
        )
        .with_columns(
            **{
                f"wind_{quantity}_{height}": pl.lit(value, dtype=pl.Float32)
                for height in ("10m", "100m")
                for quantity, value in (("speed", speed), ("direction", 180.0))
            }
        )
    )
    return frame.cast(
        {name: dtype for name, dtype in Nwp.dtypes.items() if name in frame.columns}
    ).with_columns(
        round_to_significand_bits(pl.col(name), keep_bits=NWP_SIGNIFICAND_BITS)
        for name in frame.columns
        if Nwp.dtypes.get(name, frame.schema[name]) == pl.Float32
    )


def test_rewrite_pins_snapshot_resumes_and_catches_up(tmp_path: Path) -> None:
    source, destination = tmp_path / "polar", tmp_path / "components"
    first = datetime(2025, 1, 15, tzinfo=UTC)
    old = pl.concat(
        [_old_partition(init_time=first), _old_partition(init_time=first + timedelta(days=1))]
    )
    old.write_delta(
        str(source), delta_write_options={"partition_by": ["nwp_model_id", "init_time"]}
    )
    third = _old_partition(init_time=first + timedelta(days=2), speed=7.0)
    third.write_delta(str(source), mode="append")

    migration.rewrite(source=source, destination=destination, source_version=0, limit=1)
    assert pl.read_delta(str(destination)).height == 1
    migration.rewrite(source=source, destination=destination, source_version=0, limit=None)
    assert pl.read_delta(str(destination)).height == 2
    migration.rewrite(source=source, destination=destination, source_version=1, limit=None)
    result = pl.read_delta(str(destination)).sort("init_time")
    assert result.height == 3
    assert result["wind_v_10m"].to_list() == [5.0, 5.0, 7.0]
    assert result["wind_u_100m"].to_list() == pytest.approx([0.0] * 3, abs=1e-10)
    assert not set(migration.POLAR_COLUMNS) & set(result.columns)
    assert_frame_equal(
        pl.read_delta(str(source)).sort("init_time"),
        pl.concat([old, third]).sort("init_time"),
        check_exact=True,
    )

    replacement = _old_partition(init_time=first, speed=9.0)
    replacement.write_delta(
        str(source),
        mode="overwrite",
        delta_write_options={"predicate": f"init_time = '{first.isoformat()}'"},
    )
    migration.rewrite(source=source, destination=destination, source_version=2, limit=None)
    assert pl.read_delta(str(destination)).sort("init_time")["wind_v_10m"].to_list() == [9, 5, 7]


def test_rewrite_reads_old_values_after_source_partition_replacement(tmp_path: Path) -> None:
    source, destination = tmp_path / "polar", tmp_path / "components"
    init_time = datetime(2025, 1, 15, tzinfo=UTC)
    _old_partition(init_time=init_time, speed=5.0).write_delta(
        str(source), delta_write_options={"partition_by": ["nwp_model_id", "init_time"]}
    )
    _old_partition(init_time=init_time, speed=9.0).write_delta(
        str(source),
        mode="overwrite",
        delta_write_options={"predicate": f"init_time = '{init_time.isoformat()}'"},
    )
    migration.rewrite(source=source, destination=destination, source_version=0, limit=None)
    assert pl.read_delta(str(destination))["wind_v_10m"].to_list() == [5.0]
    migration.rewrite(source=source, destination=destination, source_version=1, limit=None)
    assert pl.read_delta(str(destination))["wind_v_10m"].to_list() == [9.0]


def test_rewrite_rejects_source_overlap_and_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "polar"
    _old_partition(init_time=datetime(2025, 1, 15, tzinfo=UTC)).write_delta(str(source))
    with pytest.raises(ValueError, match="separate"):
        migration.rewrite(source=source, destination=source, source_version=0, limit=None)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "important").write_text("preserve")
    with pytest.raises(ValueError, match="absent destination"):
        migration.rewrite(source=source, destination=destination, source_version=0, limit=None)
    assert (destination / "important").read_text() == "preserve"


@pytest.mark.parametrize("bearing", [0.0, 90.0, 180.0, 270.0, 37.5, 359.9375])
@pytest.mark.parametrize("speed", [0.0, 5.0, 200.0])
def test_migration_round_trip_bounds(tmp_path: Path, bearing: float, speed: float) -> None:
    from delta_store.nwp import write_nwp

    old = _old_partition(init_time=datetime(2025, 1, 15, tzinfo=UTC), speed=speed).with_columns(
        wind_direction_10m=pl.lit(bearing, dtype=pl.Float32),
        wind_direction_100m=pl.lit(bearing, dtype=pl.Float32),
    )
    destination = tmp_path / "components"
    write_nwp(nwp=migration.convert_polar_wind(frame=old), table_uri=destination)
    result = pl.read_delta(str(destination))
    errors = migration._verify_wind_round_trip(old=old, readback=result)
    assert errors["10m_max_relative_speed_error"] <= 2**-13 + 2e-7
    assert errors["100m_max_circular_bearing_error_degrees"] <= 0.0081
