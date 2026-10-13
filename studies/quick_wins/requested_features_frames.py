"""Proposed immutable solar and reconstructed-wind augmentation of pinned frames.

Private study outputs contain coordinates and source identifiers. Nothing here fits
a model, fetches observations, rebuilds NWP boundaries or imputes missing weather.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import pvlib
import pyarrow as pa
import pyarrow.parquet as pq
from contracts.power_schemas import EffectiveCapacity, PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
from patito.exceptions import DataFrameValidationError
from polars.testing import assert_frame_equal
from pvlib import spa as pvlib_spa
from studies.baselines import haurwitz_w_m2
from studies.solar import azimuth, zenith

KEYS: Final[tuple[str, ...]] = (
    ("time_series_id"),
    ("power_fcst_init_time"),
    ("nwp_init_time"),
    ("valid_time"),
    ("ensemble_member"),
)
NONFEATURES: Final[frozenset[str]] = frozenset({*KEYS, "power"})
SOLAR_ADDED: Final[tuple[str, ...]] = ("sun_east", "sun_north", "sun_up", "clear_sky_irradiance")
WIND_ADDED: Final[tuple[str, ...]] = ("wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m")
CIRCULAR_ADDED: Final[tuple[str, ...]] = (
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "wind_direction_100m_sin",
    "wind_direction_100m_cos",
)
ADDED: Final[tuple[str, ...]] = (*SOLAR_ADDED, *WIND_ADDED, *CIRCULAR_ADDED)
SUPPORT: Final[frozenset[str]] = frozenset(
    {("metadata.parquet"), ("capacity.parquet"), ("bounded_power.parquet")}
)
THREADS: Final[tuple[str, ...]] = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
BATCH_ROWS: Final[int] = 65_536
SOLAR_BATCH_TIMES: Final[int] = 1024
ANGLE_TOLERANCE_DEGREES: Final[float] = 0.0001
VECTOR_NORM_TOLERANCE: Final[float] = 0.0000005
CONVERGENCE_TOLERANCE_W_M2: Final[float] = 0.25
WIND_MAGNITUDE_ABSOLUTE_TOLERANCE_M_S: Final[float] = 0.00001
WIND_MAGNITUDE_RELATIVE_TOLERANCE: Final[float] = 0.0000002
WIND_BEARING_TOLERANCE_DEGREES: Final[float] = 0.0001
ERA_BOUNDARY: Final[datetime] = datetime(2024, 11, 13, tzinfo=UTC)


def _require(condition: bool, message: str) -> None:
    """Reject an invalid invariant without exposing private values."""
    if not condition:
        raise ValueError(message)


def _plain(path: str | Path) -> Path:
    """Require an explicit canonical unlinked path."""
    path = Path(path)
    _require(
        condition=path.is_absolute() and path.resolve() == path,
        message=("Noncanonical or linked path"),
    )
    return path


def _sha(path: Path) -> str:
    """Hash one canonical file in bounded byte blocks."""
    _plain(path=path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _encoded(document: dict[str, Any]) -> bytes:
    """Serialize deterministic private JSON bytes."""
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def _write_bytes(path: Path, content: bytes) -> None:
    """Create one private file exclusively without overwriting."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)


def _copy_bytes(source: Path, destination: Path) -> None:
    """Copy support bytes into an exclusive private file."""
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with source.open("rb") as incoming, os.fdopen(fd, "wb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)


def _vector(zenith_degrees: np.ndarray, azimuth_degrees: np.ndarray) -> np.ndarray:
    """Convert apparent solar angles into east, north and up components."""
    z, a = np.radians(zenith_degrees), np.radians(azimuth_degrees)
    return np.column_stack((np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)))


def _reconstruct_wind(frame: pl.DataFrame) -> pl.DataFrame:
    """Append deterministic component and circular representations without filling nulls."""
    expressions, circular = [], []
    for height in ("10m", "100m"):
        speed = pl.col(f"wind_speed_{height}").cast(pl.Float64)
        radians = pl.col(f"wind_direction_{height}").cast(pl.Float64) * (np.pi / 180.0)
        expressions.extend(
            [
                (-speed * radians.sin()).cast(pl.Float32).alias(f"wind_u_{height}"),
                (-speed * radians.cos()).cast(pl.Float32).alias(f"wind_v_{height}"),
            ]
        )
        circular.extend(
            [
                radians.sin().cast(pl.Float32).alias(f"wind_direction_{height}_sin"),
                radians.cos().cast(pl.Float32).alias(f"wind_direction_{height}_cos"),
            ]
        )
    # Null arithmetic propagates either missing polar input. Nonfinite values
    # remain nonfinite; even a calm speed with a missing bearing stays missing.
    return frame.with_columns(*expressions, *circular)


def _wind_audit_frame(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Validate wind geometry and return aggregate support and roundtrip counts."""
    result = {}
    for height in ("10m", "100m"):
        speed_column, bearing_column = f"wind_speed_{height}", f"wind_direction_{height}"
        speed = frame[speed_column].cast(pl.Float64).to_numpy()
        bearing = frame[bearing_column].cast(pl.Float64).to_numpy()
        u_column, v_column = f"wind_u_{height}", f"wind_v_{height}"
        u, v = (
            frame[u_column].cast(pl.Float64).to_numpy(),
            frame[v_column].cast(pl.Float64).to_numpy(),
        )
        missing = (
            frame[speed_column].is_null().to_numpy() | frame[bearing_column].is_null().to_numpy()
        )
        _require(
            condition=np.array_equal(frame[u_column].is_null().to_numpy(), missing)
            and np.array_equal(frame[v_column].is_null().to_numpy(), missing),
            message="Wind component null propagation differs from polar inputs",
        )
        finite = ~missing & np.isfinite(speed) & np.isfinite(bearing)
        nonfinite = ~missing & ~finite
        _require(
            condition=np.isfinite(u[finite]).all() and np.isfinite(v[finite]).all(),
            message="Finite polar wind produced nonfinite components",
        )
        _require(
            condition=(~np.isfinite(u[nonfinite])).all() and (~np.isfinite(v[nonfinite])).all(),
            message="Nonfinite polar wind was converted to a finite component",
        )
        _require(
            condition=(speed[finite] >= 0).all()
            and ((bearing[finite] >= 0) & (bearing[finite] <= 360)).all(),
            message="Finite polar wind has invalid speed or bearing range",
        )
        magnitude = np.hypot(u[finite], v[finite])
        error = np.abs(magnitude - speed[finite])
        _require(
            condition=(
                error
                <= WIND_MAGNITUDE_ABSOLUTE_TOLERANCE_M_S
                + WIND_MAGNITUDE_RELATIVE_TOLERANCE * speed[finite]
            ).all(),
            message="Reconstructed wind magnitude exceeds Float32 tolerance",
        )
        noncalm = finite & (speed > 0)
        returned = np.mod(np.degrees(np.arctan2(-u[noncalm], -v[noncalm])), 360.0)
        bearing_error = np.abs(np.mod(returned - bearing[noncalm] + 180.0, 360.0) - 180.0)
        _require(
            condition=(bearing_error <= WIND_BEARING_TOLERANCE_DEGREES).all(),
            message="Reconstructed wind bearing exceeds Float32 tolerance",
        )
        calm = finite & (speed == 0)
        _require(
            condition=(u[calm] == 0).all() and (v[calm] == 0).all(),
            message=("Finite calm wind is not zero"),
        )
        sine_column, cosine_column = f"wind_direction_{height}_sin", f"wind_direction_{height}_cos"
        sine = frame[sine_column].cast(pl.Float64).to_numpy()
        cosine = frame[cosine_column].cast(pl.Float64).to_numpy()
        direction_missing = frame[bearing_column].is_null().to_numpy()
        direction_finite = ~direction_missing & np.isfinite(bearing)
        _require(
            condition=((bearing[direction_finite] >= 0) & (bearing[direction_finite] <= 360)).all(),
            message="Finite wind bearing range is invalid independently of speed",
        )
        direction_nonfinite = ~direction_missing & ~direction_finite
        _require(
            condition=np.array_equal(frame[sine_column].is_null().to_numpy(), direction_missing)
            and np.array_equal(frame[cosine_column].is_null().to_numpy(), direction_missing),
            message="Circular wind encoding null propagation differs from bearing",
        )
        _require(
            condition=np.isfinite(sine[direction_finite]).all()
            and np.isfinite(cosine[direction_finite]).all()
            and (~np.isfinite(sine[direction_nonfinite])).all()
            and (~np.isfinite(cosine[direction_nonfinite])).all(),
            message="Circular wind encoding changed finite/nonfinite bearing support",
        )
        circle_error = np.abs(np.hypot(sine[direction_finite], cosine[direction_finite]) - 1)
        _require(
            condition=(circle_error <= VECTOR_NORM_TOLERANCE).all(),
            message=("Circular wind unit norm failed"),
        )
        result[height] = {
            "rows": frame.height,
            "polar_input_null_rows": int(missing.sum()),
            "polar_input_nonfinite_nonnull_rows": int(nonfinite.sum()),
            "finite_polar_rows": int(finite.sum()),
            "finite_calm_rows": int(calm.sum()),
            "component_u_null_rows": frame[u_column].null_count(),
            "component_v_null_rows": frame[v_column].null_count(),
            "component_u_finite_rows": int(np.isfinite(u).sum()),
            "component_v_finite_rows": int(np.isfinite(v).sum()),
            "bearing_input_null_rows": int(direction_missing.sum()),
            "bearing_input_nonfinite_nonnull_rows": int(direction_nonfinite.sum()),
            "circular_sin_null_rows": frame[sine_column].null_count(),
            "circular_cos_null_rows": frame[cosine_column].null_count(),
            "circular_sin_finite_rows": int(np.isfinite(sine).sum()),
            "circular_cos_finite_rows": int(np.isfinite(cosine).sum()),
            "unit_circle_max_absolute_error": float(circle_error.max())
            if circle_error.size
            else None,
            "magnitude_roundtrip_max_absolute_error_m_s": float(error.max())
            if error.size
            else None,
            "bearing_roundtrip_max_absolute_error_degrees": float(bearing_error.max())
            if bearing_error.size
            else None,
            "finite_roundtrip_supported": bool(finite.any()),
            "noncalm_bearing_roundtrip_supported": bool(noncalm.any()),
            "null_nonfinite_calm_guards_passed": True,
        }
    return result


def _merge_wind_audits(
    total: dict[str, dict[str, Any]], addition: dict[str, dict[str, Any]]
) -> None:
    """Combine batch counts and maxima without averaging incompatible support."""
    for height, values in addition.items():
        if height not in total:
            total[height] = dict(values)
            continue
        for name, value in values.items():
            if name.endswith("_rows") or name == "rows":
                total[height][name] += value
            elif "max_absolute_error" in name:
                previous = total[height][name]
                total[height][name] = (
                    value
                    if previous is None
                    else previous
                    if value is None
                    else max(previous, value)
                )
            elif name.endswith("supported"):
                total[height][name] = total[height][name] or value
            else:
                _require(
                    condition=total[height][name] is True and value is True,
                    message=("Wind audit guard did not pass"),
                )


def _clear_mean(
    stamps: pl.Series, latitude: float, longitude: float, step_seconds: int = 60
) -> np.ndarray:
    """Average Haurwitz irradiance over bounded midpoint samples."""
    count = 1800 // step_seconds
    _require(condition=count * step_seconds == 1800, message="Unsupported integration step")
    offsets = [int((index + 0.5) * step_seconds - 1800) for index in range(count)]
    values = []
    for start in range(0, len(stamps), SOLAR_BATCH_TIMES):
        chunk = stamps.slice(start, SOLAR_BATCH_TIMES)
        samples = pl.concat([chunk.dt.offset_by(f"{offset}s") for offset in offsets])
        angles = zenith(stamps=samples, latitude=latitude, longitude=longitude)
        flux = haurwitz_w_m2(apparent_zenith_deg=angles).reshape(count, len(chunk))
        _require(
            condition=np.isfinite(flux).all() and (flux >= 0).all(),
            message=("Invalid clear-sky primitive output"),
        )
        values.append(flux.mean(axis=0))
    return np.concatenate(values) if values else np.empty(0)


def _analytic_guards() -> dict[str, Any]:
    """Exercise public synthetic geometry, irradiance and wind invariants."""
    actual = _vector(
        zenith_degrees=np.array([90.0, 90.0, 90.0, 90.0, 0.0]),
        azimuth_degrees=np.array([0.0, 90.0, 180.0, 270.0, 123.0]),
    )
    expected = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    _require(
        condition=np.allclose(actual, expected, atol=1e-12, rtol=0),
        message=("Cardinal solar-vector guard failed"),
    )
    wrapped = _vector(
        zenith_degrees=np.array([70.0, 70.0]), azimuth_degrees=np.array([0.000001, 359.999999])
    )
    _require(
        condition=np.max(np.abs(wrapped[0] - wrapped[1])) < 0.000001,
        message=("Azimuth wrap continuity failed"),
    )
    _require(
        condition=np.max(np.abs(np.linalg.norm(actual, axis=1) - 1)) < 1e-12,
        message=("Analytic vector norm failed"),
    )
    _require(
        condition=np.array_equal(
            haurwitz_w_m2(apparent_zenith_deg=np.array([90.0, 100.0, 180.0])), np.zeros(3)
        ),
        message="Haurwitz night guard failed",
    )
    stamps = pl.Series(
        "valid_time", [datetime(2025, 3, 20, hour, tzinfo=UTC) for hour in [0, 6, 12, 18]]
    )
    minute = _clear_mean(stamps=stamps, latitude=51.5, longitude=-0.12)
    half_minute = _clear_mean(stamps=stamps, latitude=51.5, longitude=-0.12, step_seconds=30)
    _require(
        condition=minute[0] == 0 and minute[2] > 100, message=("Synthetic night/noon guard failed")
    )
    difference = float(np.max(np.abs(minute - half_minute)))
    _require(
        condition=difference <= CONVERGENCE_TOLERANCE_W_M2,
        message=("Integration convergence guard failed"),
    )
    polar = {}
    for height in ("10m", "100m"):
        polar[f"wind_speed_{height}"] = pl.Series(
            [1.0, 1.0, 1.0, 1.0, 0.0, None, 1.0, float("nan"), 0.0, 0.0], dtype=pl.Float32
        )
        polar[f"wind_direction_{height}"] = pl.Series(
            [0.0, 90.0, 180.0, 270.0, 90.0, 0.0, None, 0.0, None, float("nan")], dtype=pl.Float32
        )
    wind = _reconstruct_wind(frame=pl.DataFrame(polar))
    _wind_audit_frame(frame=wind)
    expected_uv = np.array([[0.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    for height in ("10m", "100m"):
        _require(
            condition=np.allclose(
                wind.select(f"wind_u_{height}", f"wind_v_{height}").head(5).to_numpy(),
                expected_uv,
                atol=0.000001,
                rtol=0,
            ),
            message="Meteorological wind cardinal guard failed",
        )
        expected_circle = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]])
        _require(
            condition=np.allclose(
                wind.select(f"wind_direction_{height}_sin", f"wind_direction_{height}_cos")
                .head(5)
                .to_numpy(),
                expected_circle,
                atol=0.000001,
                rtol=0,
            ),
            message="Circular wind cardinal guard failed",
        )
    wrap = {f"wind_speed_{height}": [1.0, 1.0] for height in ("10m", "100m")}
    wrap.update({f"wind_direction_{height}": [0.000001, 359.999999] for height in ("10m", "100m")})
    wrap_frame = _reconstruct_wind(frame=pl.DataFrame(wrap))
    for column in [*WIND_ADDED, *CIRCULAR_ADDED]:
        _require(
            condition=abs(wrap_frame[column][0] - wrap_frame[column][1]) < 0.000001,
            message="Wind circular wrap continuity failed",
        )
    return {
        "cardinal_wrap_norm_passed": True,
        "synthetic_night_noon_passed": True,
        "wind_cardinal_magnitude_null_nonfinite_calm_and_circle_wrap_passed": True,
        "one_minute_vs_half_minute_max_absolute_w_m2": difference,
        "convergence_tolerance_w_m2": CONVERGENCE_TOLERANCE_W_M2,
        "synthetic_location": "Public synthetic London coordinate 51.5 N, 0.12 W",
        "synthetic_date": "2025-03-20",
        "synthetic_hours_utc": [0, 6, 12, 18],
    }


def _source_preflight(
    source: Path, pinned: str
) -> tuple[dict[str, Any], list[str], dict[str, datetime], str]:
    """Verify the pinned complete original inventory and inclusive fold window."""
    _require(condition=source.is_dir(), message="Missing source root")
    manifest_path = source / "manifest.json"
    _require(condition=_sha(path=manifest_path) == pinned, message="Source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    _require(
        condition=set(manifest)
        == {("complete"), ("context"), ("files"), ("frames"), ("input_fingerprint"), ("window")},
        message="Unknown source manifest schema",
    )
    _require(
        condition=manifest[("complete")] is True
        and manifest[("context")].get("pipeline_tag") == ("combined"),
        message="Only completed combined frames are supported",
    )
    window = manifest["window"]
    _require(
        condition=set(window) == {"fold_id", "train_start", "train_end", "val_start", "val_end"},
        message="Unknown fold-window schema",
    )
    times = {name: datetime.fromisoformat(window[name]) for name in window if name != "fold_id"}
    _require(
        condition=all(
            value.tzinfo is not None and value.utcoffset() == timedelta(0)
            for value in times.values()
        ),
        message="Fold dates must be explicit UTC",
    )
    _require(
        condition=times["train_start"] <= times["train_end"]
        and times["val_start"] <= times["val_end"]
        and (times["train_end"] < times["val_start"] or times["val_end"] < times["train_start"])
        and max(times.values()) < datetime(2026, 7, 1, tzinfo=UTC),
        message="Invalid bounded fold",
    )
    validation = []
    stamp = times["val_start"]
    _require(
        condition=stamp.hour == stamp.minute == stamp.second == stamp.microsecond == 0,
        message="Scheduled validation must start at midnight",
    )
    while stamp <= times["val_end"]:
        validation.append(f"val/init={stamp:%Y-%m-%d}.parquet")
        stamp += timedelta(days=7)
    expected = SUPPORT | {"train.parquet", *validation}
    _require(
        condition=set(manifest[("files")]) == expected,
        message=("Source scheduled input inventory differs"),
    )
    _require(
        condition={path.name for path in source.iterdir()}
        == SUPPORT | {"manifest.json", "context.json", "train.parquet", "val"},
        message="Unknown source root output",
    )
    _require(
        condition=_plain(path=source / "val").is_dir()
        and {str(path.relative_to(source)) for path in (source / "val").iterdir()}
        == set(validation),
        message="Source validation inventory differs",
    )
    for relative in sorted(expected):
        _require(
            condition=_sha(path=source / relative) == manifest[("files")][relative],
            message=("Source input hash differs"),
        )
    context_bytes = _plain(path=source / "context.json").read_bytes()
    _require(
        condition=json.loads(context_bytes) == manifest[("context")],
        message=("Source context differs from manifest"),
    )
    _require(
        condition=hashlib.sha256(
            json.dumps(manifest[("context")], sort_keys=True).encode()
        ).hexdigest()
        == manifest["input_fingerprint"],
        message="Source context fingerprint differs",
    )
    return manifest, validation, times, hashlib.sha256(context_bytes).hexdigest()


def _make_lookup(
    paths: list[Path], metadata: pl.DataFrame, output: Path
) -> tuple[list[Path], float, float]:
    """Build private solar lookups once per series and unique valid time."""
    lookups, max_angle_error = [], 0.0
    max_azimuth_error = 0.0
    scan = pl.scan_parquet(paths)
    source_id_dtype = scan.collect_schema()["time_series_id"]
    for series_id, latitude, longitude in metadata.select(
        "time_series_id", "latitude", "longitude"
    ).iter_rows():
        _require(
            condition=latitude is not None
            and longitude is not None
            and np.isfinite([latitude, longitude]).all()
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180,
            message="Invalid private coordinate",
        )
        selected = scan.filter(pl.col("time_series_id") == series_id).select(
            "valid_time", "solar_zenith", "solar_azimuth"
        )
        _require(
            condition=not selected.select(
                pl.any_horizontal(
                    pl.all().exclude("valid_time").is_null()
                    | ~pl.all().exclude("valid_time").is_finite()
                ).any()
            )
            .collect()
            .item(),
            message="Original solar angles are missing or nonfinite",
        )
        values = selected.unique().sort("valid_time").collect()
        _require(
            condition=values.height > 0
            and not values["valid_time"].is_duplicated().any()
            and values["valid_time"].null_count() == 0,
            message="Solar angles depend on member or run",
        )
        z, a = values["solar_zenith"].to_numpy(), values["solar_azimuth"].to_numpy()
        _require(
            condition=((z >= 0) & (z <= 180) & (a >= 0) & (a <= 360)).all(),
            message="Solar angle units or range invalid",
        )
        recomputed = zenith(
            stamps=values["valid_time"].dt.offset_by("-15m"),
            latitude=float(latitude),
            longitude=float(longitude),
        )
        error = float(np.max(np.abs(z.astype(np.float64) - recomputed)))
        _require(
            condition=error <= ANGLE_TOLERANCE_DEGREES,
            message=("Stored midpoint zenith differs from solar primitive"),
        )
        max_angle_error = max(max_angle_error, error)
        recomputed_azimuth = azimuth(
            stamps=values["valid_time"].dt.offset_by("-15m"),
            latitude=float(latitude),
            longitude=float(longitude),
        )
        azimuth_error = float(
            np.max(np.abs(np.mod(a.astype(np.float64) - recomputed_azimuth + 180.0, 360.0) - 180.0))
        )
        _require(
            condition=azimuth_error <= ANGLE_TOLERANCE_DEGREES,
            message="Stored midpoint azimuth differs from solar primitive",
        )
        max_azimuth_error = max(max_azimuth_error, azimuth_error)
        directions = _vector(
            zenith_degrees=z.astype(np.float64), azimuth_degrees=a.astype(np.float64)
        ).astype(np.float32)
        _require(
            condition=np.isfinite(directions).all()
            and np.max(np.abs(np.linalg.norm(directions.astype(np.float64), axis=1) - 1))
            <= VECTOR_NORM_TOLERANCE,
            message="Float32 unit-vector guard failed",
        )
        clear = _clear_mean(
            stamps=values[("valid_time")], latitude=float(latitude), longitude=float(longitude)
        ).astype(np.float32)
        _require(
            condition=np.isfinite(clear).all() and (clear >= 0).all(),
            message=("Invalid Float32 clear-sky feature"),
        )
        lookup = (
            values.select("valid_time")
            .with_columns(
                pl.lit(series_id, dtype=source_id_dtype).alias("time_series_id"),
                *[
                    pl.Series(name, directions[:, index], dtype=pl.Float32)
                    for index, name in enumerate(SOLAR_ADDED[:3])
                ],
                pl.Series("clear_sky_irradiance", clear, dtype=pl.Float32),
            )
            .select("time_series_id", "valid_time", *SOLAR_ADDED)
        )
        path = output / "_lookups" / f"series={series_id}.parquet"
        _write_bytes(path=path, content=b"")
        lookup.write_parquet(path)
        path.chmod(mode=0o600)
        assert_frame_equal(lookup, pl.read_parquet(path), check_exact=True)
        lookups.append(path)
    return lookups, max_angle_error, max_azimuth_error


def _frame_details(path: Path) -> dict[str, Any]:
    """Summarize actual ordered keys, rows, features and schema."""
    frame = pl.scan_parquet(path)
    key_frame = frame.select(KEYS).collect().rechunk()
    _require(
        condition=not key_frame.is_duplicated().any()
        and key_frame.null_count().sum_horizontal().item() == 0,
        message="Invalid original forecast key",
    )
    buffer = io.BytesIO()
    key_frame.write_ipc(buffer)
    schema = frame.collect_schema()
    return {
        "rows": key_frame.height,
        "key_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "key_hash_semantics": "Polars IPC of original five ordered key columns, rechunked",
        "valid_start": key_frame["valid_time"].min().isoformat(),
        "valid_end": key_frame["valid_time"].max().isoformat(),
        "features": [name for name in schema.names() if name not in NONFEATURES],
        "schema": {name: str(dtype) for name, dtype in schema.items()},
    }


def _assert_original_equal(original: pl.DataFrame, restored: pl.DataFrame) -> None:
    """Require exact original values, nulls and valid Float32 bit patterns."""
    assert_frame_equal(original, restored, check_exact=True)
    # Numeric equality treats signed zeros and NaN payloads as equivalent. Preserve
    # valid Float32 bit patterns explicitly; null storage slots have no value.
    for column, dtype in original.schema.items():
        if dtype == pl.Float32:
            valid = ~original[column].is_null().to_numpy()
            left = original[column].to_numpy().view(np.uint32)
            right = restored[column].to_numpy().view(np.uint32)
            _require(
                condition=np.array_equal(left[valid], right[valid]),
                message=("Original Float32 valid bits changed"),
            )


def _assert_arrow_original_bits(original: pa.RecordBatch, restored: pa.RecordBatch) -> None:
    """Compare original Float32 buffers directly, including signed zero and NaN payloads."""
    for index, field in enumerate(original.schema):
        if pa.types.is_float32(field.type):
            left = original.column(index)
            right = restored.column(restored.schema.get_field_index(field.name))
            _require(
                condition=pa.types.is_float32(right.type),
                message=("Original Arrow Float32 type changed"),
            )
            left_valid = left.is_valid().to_numpy(zero_copy_only=False)
            right_valid = right.is_valid().to_numpy(zero_copy_only=False)
            _require(
                condition=np.array_equal(left_valid, right_valid),
                message=("Original Arrow validity changed"),
            )
            left_bits = np.frombuffer(
                left.buffers()[1], dtype=np.uint32, count=len(left), offset=left.offset * 4
            )
            right_bits = np.frombuffer(
                right.buffers()[1], dtype=np.uint32, count=len(right), offset=right.offset * 4
            )
            _require(
                condition=np.array_equal(left_bits[left_valid], right_bits[right_valid]),
                message="Original on-disk Float32 bits changed",
            )


def _augment(
    source: Path, destination: Path, lookup: pl.DataFrame
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Append twelve inputs in bounded batches and verify every original field."""
    original_schema = pl.scan_parquet(source).collect_schema()
    _require(
        condition=not set(ADDED) & set(original_schema.names()),
        message=("New feature already exists"),
    )
    writer = None
    wind_counts = {}
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=BATCH_ROWS):
            original = pl.from_arrow(batch)
            _require(
                condition=original.schema == original_schema,
                message=("Arrow conversion changed original schema"),
            )
            extended = original.join(
                lookup,
                on=["time_series_id", "valid_time"],
                how="left",
                validate="m:1",
                maintain_order="left",
            )
            extended = _reconstruct_wind(frame=extended)
            _require(
                condition=extended.height == original.height
                and extended.columns == original.columns + list(ADDED),
                message="Augmentation changed row count or column order",
            )
            _assert_original_equal(original=original, restored=extended.select(original.columns))
            _require(
                condition=not extended.select(
                    pl.any_horizontal(
                        pl.col(SOLAR_ADDED).is_null() | ~pl.col(SOLAR_ADDED).is_finite()
                    ).any()
                ).item(),
                message="Solar features are missing or nonfinite",
            )
            _merge_wind_audits(total=wind_counts, addition=_wind_audit_frame(frame=extended))
            table = extended.to_arrow()
            if writer is None:
                _write_bytes(path=destination, content=b"")
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=BATCH_ROWS)
    finally:
        if writer is not None:
            writer.close()
    _require(condition=writer is not None, message="Empty original frame")
    destination.chmod(mode=0o600)
    source_batches = pq.ParquetFile(source).iter_batches(batch_size=BATCH_ROWS)
    output_batches = pq.ParquetFile(destination).iter_batches(batch_size=BATCH_ROWS)
    for old, new in zip(source_batches, output_batches, strict=True):
        _assert_arrow_original_bits(original=old, restored=new)
        original = pl.from_arrow(old)
        reread = pl.from_arrow(new)
        _assert_original_equal(original=original, restored=reread.select(original_schema.names()))
        expected = original.join(
            lookup,
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        expected = _reconstruct_wind(frame=expected)
        assert_frame_equal(expected, reread, check_exact=True)
    details = _frame_details(path=destination)
    _require(
        condition=details[("key_sha256")] == _frame_details(path=source)[("key_sha256")],
        message=("Re-read key hash changed"),
    )
    return details, wind_counts


def _precipitation_contract_bounds(path: Path) -> dict[str, dict[str, float]]:
    """Read descriptive precipitation bounds from the reviewed contract source."""
    tree = ast.parse(path.read_text())
    model = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Nwp"
    )
    expected = {
        "precipitation_surface": {"ge": 0.0, "le": 0.01},
        "categorical_precipitation_type_surface": {"ge": 0.0, "le": 255.0},
    }
    found = {}
    for node in model.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in expected
        ):
            _require(
                condition=isinstance(node.value, ast.Call),
                message=("Unexpected precipitation contract declaration"),
            )
            found[node.target.id] = {
                keyword.arg: float(ast.literal_eval(keyword.value))
                for keyword in node.value.keywords
                if keyword.arg in {"ge", "le"}
            }
    _require(
        condition=found == expected,
        message=("Pinned precipitation contract bounds differ from reviewed design"),
    )
    return found


def _precipitation_audit(
    paths: list[Path],
    metadata: pl.DataFrame,
    source_manifest_hash: str,
    source_file_hashes: dict[str, str],
    contract_path: Path,
    contract_bounds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Count missingness and suspicious values in fixed noncausal audit strata."""
    pv = metadata.filter(pl.col("time_series_type").cast(pl.String) == "PV")["time_series_id"]
    result = []
    for subset, selected_paths in (("train", paths[:1]), ("validation", paths[1:])):
        frame = pl.scan_parquet(selected_paths).filter(pl.col("time_series_id").is_in(pv.implode()))
        for era in ("before_precip", "after_precip"):
            chosen = frame.filter(
                pl.col("nwp_init_time") < pl.lit(ERA_BOUNDARY)
                if era == "before_precip"
                else pl.col("nwp_init_time") >= pl.lit(ERA_BOUNDARY)
            )
            for lead_axis, init_column in (
                ("weather_lead", "nwp_init_time"),
                ("served_forecast_lead", "power_fcst_init_time"),
            ):
                lead = (pl.col("valid_time") - pl.col(init_column)).dt.total_seconds() / 3600
                strata = [
                    ("all", pl.lit(True)),
                    ("less_than_72h", lead < 72),
                    ("72h_to_less_than_264h", (lead >= 72) & (lead < 264)),
                    ("264h_or_more", lead >= 264),
                ]
                for band, predicate in strata:
                    bounded = chosen.filter(predicate)
                    for column, bounds in contract_bounds.items():
                        lower, upper = bounds["ge"], bounds["le"]
                        finite = pl.col(column).is_not_null() & pl.col(column).is_finite()
                        counts = (
                            bounded.select(
                                rows=pl.len(),
                                missing=pl.col(column).is_null().sum(),
                                nonfinite=(
                                    pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                                ).sum(),
                                zeros=(finite & (pl.col(column) == 0)).sum(),
                                negative=(finite & (pl.col(column) < 0)).sum(),
                                above_contract_upper=(finite & (pl.col(column) > upper)).sum(),
                                lead_zero_rows=(lead == 0).sum(),
                                negative_lead_rows=(lead < 0).sum(),
                            )
                            .collect()
                            .row(0, named=True)
                        )
                        result.append(
                            {
                                "subset": subset,
                                "source_era": era,
                                "column": column,
                                "lead_axis": lead_axis,
                                "lead_band": band,
                                "contract_range_lower": lower,
                                "contract_range_upper": upper,
                                "supported": counts["rows"] > 0,
                                "lead_zero_present": counts["lead_zero_rows"] > 0,
                                **counts,
                            }
                        )
    return {
        "source_manifest_sha256": source_manifest_hash,
        "source_frame_file_sha256": source_file_hashes,
        "weather_contract_sha256": _sha(path=contract_path),
        "weather_contract_bounds": contract_bounds,
        "source_run_boundary_utc": ERA_BOUNDARY.isoformat(),
        "rows": result,
        ("semantics"): (
            "PV frame-row audit strata include repeated ensemble members and run/lead "
            "combinations. Weather lead is valid minus NWP init; served forecast lead "
            "is valid minus power forecast init. These strata are not predictors. "
            "Empty era/bin is unsupported. Absent engineered lead zero does not show "
            "the raw-archive lead-zero missingness issue is absent. No imputation, "
            "filtering of model rows, causal claim or independent-case weighting. "
            "Pinned descriptive contract bounds are rate [0,0.01] and type [0,255]."
        ),
    }


STUDY_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
KEY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "time_series_id": pl.Int32,
        "power_fcst_init_time": pl.Datetime("us", "UTC"),
        "nwp_init_time": pl.Datetime("us", "UTC"),
        "valid_time": pl.Datetime("us", "UTC"),
        "ensemble_member": pl.Int8,
        "power": pl.Float32,
    }
)
LAG_HOURS: Final[tuple[int, ...]] = (24, 48, 168, 336)
RUNTIME_VERSION_NAMES: Final[frozenset[str]] = frozenset(
    {("numpy"), ("polars"), ("pyarrow"), ("pvlib"), ("pandas"), ("patito")}
)
GEOMETRY_DEFAULTS: Final[dict[str, str | float]] = {
    "method": "nrel_numpy",
    "altitude_m": 0.0,
    "pressure_pa": 101325.0,
    "temperature_c": 12.0,
}


def _runtime_sources() -> dict[str, Path]:
    """Locate exact primitive, contract and underlying SPA source implementations."""
    return {
        "solar_wrapper": Path(inspect.getfile(zenith)).resolve(),
        "baselines_haurwitz": Path(inspect.getfile(haurwitz_w_m2)).resolve(),
        "pvlib_solarposition": Path(inspect.getfile(pvlib.solarposition)).resolve(),
        "pvlib_spa": Path(inspect.getfile(pvlib_spa)).resolve(),
        "weather_contract": Path(inspect.getfile(Nwp)).resolve(),
        "power_contract": Path(inspect.getfile(PowerTimeSeries)).resolve(),
    }


def _check_runtime(*, approved: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    """Enforce reviewed versions, exact source bindings and existing geometry defaults."""
    _require(
        condition=set(approved) == {("versions"), ("sources"), ("geometry_defaults")},
        message=("Unknown runtime schema"),
    )
    _require(
        condition=set(approved[("versions")]) == RUNTIME_VERSION_NAMES,
        message=("Incomplete runtime version binding"),
    )
    actual_versions = {
        name: importlib.metadata.version(name) for name in sorted(RUNTIME_VERSION_NAMES)
    }
    _require(
        condition=actual_versions == approved["versions"]
        and actual_versions["pvlib"] == parent["context"]["dependencies"]["pvlib"],
        message="Runtime versions differ from approved or parent pvlib convention",
    )
    sources = _runtime_sources()
    _require(
        condition=set(approved[("sources")]) == set(sources),
        message=("Incomplete reviewed source binding"),
    )
    actual_sources = {}
    _require(
        condition=parent["context"]["code_sha256"].get(str(sources["solar_wrapper"]))
        == approved[("sources")][("solar_wrapper")][("sha256")],
        message=("Solar wrapper differs from historical parent"),
    )
    for name, path in sources.items():
        binding = approved["sources"][name]
        _require(
            condition=set(binding) == {("path"), ("sha256")}
            and _plain(path=binding[("path")]) == path,
            message="Reviewed runtime source path differs",
        )
        actual_sources[name] = {"path": str(path), "sha256": _sha(path=path)}
        _require(
            condition=actual_sources[name] == binding,
            message=("Reviewed runtime source hash differs"),
        )
    defaults = inspect.signature(pvlib.solarposition.get_solarposition).parameters
    _require(
        condition=approved["geometry_defaults"] == GEOMETRY_DEFAULTS
        and defaults[("method")].default == ("nrel_numpy")
        and defaults[("temperature")].default == 12.0
        and defaults["altitude"].default is None
        and defaults["pressure"].default is None,
        message="Reviewed implicit geometry defaults differ",
    )
    return {
        ("versions"): actual_versions,
        ("sources"): actual_sources,
        ("geometry_defaults"): GEOMETRY_DEFAULTS,
    }


def _support_semantics(
    *, metadata: pl.DataFrame, capacity: pl.DataFrame, power: pl.DataFrame, expected_count: int
) -> dict[str, bool]:
    """Validate exact support contracts while retaining early history and nullable targets."""
    _require(
        condition=dict(metadata.schema) == TimeSeriesMetadata.dtypes
        and dict(capacity.schema) == EffectiveCapacity.dtypes
        and dict(power.schema) == PowerTimeSeries.dtypes,
        message=("Frozen support schema differs"),
    )
    TimeSeriesMetadata.validate(dataframe=metadata)
    EffectiveCapacity.validate(dataframe=capacity)
    ids = set(metadata["time_series_id"].to_list())
    _require(
        condition=metadata.height == expected_count
        and len(ids) == expected_count
        and capacity.height == expected_count
        and capacity[("time_series_id")].n_unique() == expected_count
        and set(capacity["time_series_id"].to_list()) == ids
        and not capacity.filter(
            ~pl.col("effective_capacity_mw").is_finite() | (pl.col("effective_capacity_mw") <= 0)
        ).height,
        message="Capacity or metadata population/positivity differs",
    )
    _require(
        condition=power.height > 0
        and power.select(("time_series_id"), ("time")).null_count().sum_horizontal().item() == 0
        and not power.select("time_series_id", "time").is_duplicated().any()
        and set(power["time_series_id"].unique().to_list()) == ids,
        message="Bounded observations have invalid keys or population",
    )
    _require(
        condition=power.equals(power.sort("time_series_id", "time"))
        and not power.filter(
            (pl.col("time") >= pl.lit(STUDY_END))
            | ~pl.col("time").dt.minute().is_in([0, 30])
            | (pl.col("time").dt.second() != 0)
            | (pl.col("time").dt.microsecond() != 0)
            | (pl.col("power").is_not_null() & ~pl.col("power").is_finite())
        ).height,
        message="Bounded observations exceed boundary or contain invalid values/times",
    )
    _bounded_power_boundary_guard(power=power)
    # No lower time cutoff: early observations remain available to cleaning/lags.
    # Capacity's full-history timestamp may be later; it is a scoring denominator.
    return {
        "support_schemas": True,
        "support_population": True,
        "positive_finite_capacity": True,
        "bounded_power_unique_sorted_finite_nonnull_before_study_end": True,
        "early_history_preserved": True,
        "capacity_future_history_is_scoring_only": True,
    }


def _frame_key_semantics(*, keys: pl.DataFrame, training: bool) -> None:
    """Reject wrong ordered keys and incomplete native member support."""
    _require(
        condition=keys.height > 0
        and keys.null_count().sum_horizontal().item() == 0
        and not keys.is_duplicated().any()
        and keys.equals(keys.sort(KEYS)),
        message="Frozen forecast keys are not unique, nonnull and sorted",
    )
    groups = keys.group_by(*KEYS[:-1]).agg(members=pl.col("ensemble_member").sort())
    expected = [0] if training else list(range(51))
    _require(
        condition=groups.filter(pl.col("members") != pl.lit(expected)).is_empty(),
        message="Frozen member support differs",
    )


def _frame_batch_semantics(
    *,
    frame: pl.DataFrame,
    power: pl.DataFrame,
    population: set[int],
    start: datetime,
    end: datetime,
    scheduled_init: datetime | None,
) -> None:
    """Check issuance, source leads, lag masks and every nonnull frozen target."""
    _require(
        condition=all(frame.schema.get(name) == dtype for name, dtype in KEY_SCHEMA.items()),
        message="Frozen key/target dtypes differ",
    )
    _require(
        condition=set(frame[("time_series_id")].unique().to_list()) <= population,
        message=("Unknown frame population"),
    )
    _require(
        condition=not frame.filter(
            ~pl.col("valid_time").is_between(start, end)
            | (pl.col("valid_time") >= pl.lit(STUDY_END))
            | (pl.col("valid_time") <= pl.col("power_fcst_init_time"))
            | (pl.col("nwp_init_time") > pl.col("valid_time"))
            | (pl.col("power_fcst_init_time") != pl.col("nwp_init_time") + pl.duration(hours=9))
            | (pl.col("nwp_init_time").dt.hour() != 0)
            | (pl.col("nwp_init_time").dt.minute() != 0)
            | (pl.col("nwp_init_time").dt.second() != 0)
            | (pl.col("nwp_init_time").dt.microsecond() != 0)
            | ~pl.col("valid_time").dt.minute().is_in([0, 30])
            | (pl.col("valid_time").dt.second() != 0)
            | (pl.col("valid_time").dt.microsecond() != 0)
        ).height,
        message="Frozen source/issuance/valid-time convention differs",
    )
    if scheduled_init is not None:
        _require(
            condition=not frame.filter(pl.col("nwp_init_time") != pl.lit(scheduled_init)).height,
            message="Validation source initialization differs from scheduled filename",
        )
    if "nwp_lead_time_hours" in frame.columns:
        expected = (pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_seconds() / 3600
        _require(
            condition=not frame.filter(
                pl.col("nwp_lead_time_hours").is_null()
                | ~pl.col("nwp_lead_time_hours").is_finite()
                | ((pl.col(("nwp_lead_time_hours")) - expected).abs() > 0.000001)
            ).height,
            message="Stored weather lead differs from source timestamps",
        )
    for hours in LAG_HOURS:
        column = f"power_lag_{hours}h"
        _require(
            condition=column in frame.columns and frame.schema[column] == pl.Float32,
            message=("Frozen lag schema differs"),
        )
        unavailable = pl.col("valid_time") - pl.duration(hours=hours) >= pl.col(
            "power_fcst_init_time"
        )
        _require(
            condition=not frame.filter(unavailable & pl.col(column).is_not_null()).height,
            message="Power lag contains a value unavailable strictly before issuance",
        )
    targets = power.rename({"time": "valid_time", "power": "bounded_target"})
    checked = frame.select("time_series_id", "valid_time", "power").join(
        targets,
        on=[("time_series_id"), ("valid_time")],
        how=("left"),
        validate=("m:1"),
        maintain_order=("left"),
    )
    _require(
        condition=not checked.filter(
            pl.col("power").is_not_null()
            & (
                ~pl.col("power").is_finite()
                | pl.col("bounded_target").is_null()
                | (pl.col("power") != pl.col("bounded_target"))
            )
        ).height,
        message="Nonnull frozen target differs from bounded observations",
    )


def _frozen_frame_semantics(
    *,
    root: Path,
    parent: dict[str, Any],
    validation: list[str],
    window: dict[str, datetime],
    metadata: pl.DataFrame,
    power: pl.DataFrame,
    augmented: bool,
) -> dict[str, Any]:
    """Audit each complete frozen source/output frame with bounded feature batches."""
    population = set(metadata["time_series_id"].to_list())
    summaries = {}
    for relative in ["train.parquet", *validation]:
        path = root / relative
        training = relative == "train.parquet"
        schema = pl.scan_parquet(path).collect_schema()
        expected_schema = dict(parent["frames"][relative]["schema"])
        if augmented:
            expected_schema.update(dict.fromkeys(ADDED, "Float32"))
        _require(
            condition={name: str(dtype) for name, dtype in schema.items()} == expected_schema,
            message="Actual frozen frame schema differs from declared parent/augmentation",
        )
        keys = pl.scan_parquet(path).select(KEYS).collect().rechunk()
        _frame_key_semantics(keys=keys, training=training)
        _require(
            condition=set(keys["time_series_id"].unique().to_list()) == population
            and keys.height == parent[("frames")][relative][("rows")],
            message=("Frozen frame population/rows differ"),
        )
        start, end = (
            (window["train_start"], window["train_end"])
            if training
            else (window["val_start"], window["val_end"])
        )
        scheduled = (
            None
            if training
            else datetime.strptime(Path(relative).stem.removeprefix("init="), ("%Y-%m-%d")).replace(
                tzinfo=UTC
            )
        )
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
            _frame_batch_semantics(
                frame=pl.from_arrow(batch),
                power=power,
                population=population,
                start=start,
                end=end,
                scheduled_init=scheduled,
            )
        summaries[relative] = {
            ("rows"): keys.height,
            ("schema_keys_members_sorted_issuance_lags_targets_passed"): True,
            "nullable_targets_preserved": True,
        }
    return summaries


def _expect_rejection(*, operation: Callable[[], Any], reason: str) -> None:
    """Require malformed public synthetic data to fail through a semantic guard."""
    try:
        operation()
    except ValueError:
        return
    raise ValueError(reason)


def _semantic_guards(
    *, approved_runtime: dict[str, Any], parent: dict[str, Any]
) -> dict[str, bool]:
    """Exercise bounded malformed members, lags, targets, init, boundary and runtime cases."""
    init = datetime(2025, 7, 1, tzinfo=UTC)
    valid = init + timedelta(hours=9, minutes=30)
    frame = pl.DataFrame(
        {
            "time_series_id": [1] * 51,
            "nwp_init_time": [init] * 51,
            "power_fcst_init_time": [init + timedelta(hours=9)] * 51,
            "valid_time": [valid] * 51,
            "ensemble_member": list(range(51)),
            ("power"): [1.0] * 51,
            **{f"power_lag_{hours}h": [None] * 51 for hours in LAG_HOURS},
        },
        schema={**dict(KEY_SCHEMA), **{f"power_lag_{hours}h": pl.Float32 for hours in LAG_HOURS}},
    ).sort(KEYS)
    power = pl.DataFrame(
        {("time_series_id"): [1], ("time"): [valid], ("power"): [1.0]},
        schema=PowerTimeSeries.dtypes,
    )
    _frame_key_semantics(keys=frame.select(KEYS), training=False)

    def _check_batch(candidate: pl.DataFrame) -> None:
        """Validate one synthetic frame under the same semantic guard as real data."""
        _frame_batch_semantics(
            frame=candidate,
            power=power,
            population={1},
            start=init,
            end=init + timedelta(days=3),
            scheduled_init=init,
        )

    _check_batch(candidate=frame)
    _expect_rejection(
        operation=lambda: _frame_key_semantics(keys=frame.head(50).select(KEYS), training=False),
        reason=("Missing-member guard failed"),
    )
    _expect_rejection(
        operation=lambda: _frame_key_semantics(
            keys=frame.filter(pl.col("ensemble_member") == 1).select(KEYS), training=True
        ),
        reason=("Training-member guard failed"),
    )
    _expect_rejection(
        operation=lambda: _frame_key_semantics(
            keys=frame.with_columns(pl.lit(51, dtype=pl.Int8).alias("ensemble_member")).select(
                KEYS
            ),
            training=False,
        ),
        reason=("Wrong-member guard failed"),
    )
    _expect_rejection(
        operation=lambda: _check_batch(
            candidate=frame.with_columns(power=pl.lit(2.0, dtype=pl.Float32))
        ),
        reason=("Wrong-target guard failed"),
    )
    _expect_rejection(
        operation=lambda: _check_batch(
            candidate=frame.with_columns(nwp_init_time=pl.lit(init + timedelta(days=1)))
        ),
        reason=("Wrong-init guard failed"),
    )
    leaky = frame.with_columns(
        valid_time=pl.lit(init + timedelta(hours=33)),
        power=pl.lit(None, dtype=pl.Float32),
        power_lag_24h=pl.lit(1.0, dtype=pl.Float32),
    )
    _expect_rejection(
        operation=lambda: _check_batch(candidate=leaky), reason=("Unavailable-lag guard failed")
    )
    _expect_rejection(
        operation=lambda: _check_batch(candidate=frame.with_columns(valid_time=pl.lit(STUDY_END))),
        reason=("Study-boundary guard failed"),
    )
    bad_power = power.with_columns(time=pl.lit(STUDY_END))
    _expect_rejection(
        operation=lambda: _bounded_power_boundary_guard(power=bad_power),
        reason=("Support-boundary guard failed"),
    )
    invalid_bearing = pl.DataFrame(
        {
            f"wind_speed_{height}": pl.Series([None], dtype=pl.Float32)
            for height in (("10m"), ("100m"))
        }
    )
    invalid_bearing = invalid_bearing.with_columns(
        *[
            pl.lit(400.0, dtype=pl.Float32).alias(f"wind_direction_{height}")
            for height in (("10m"), ("100m"))
        ]
    )
    _expect_rejection(
        operation=lambda: _wind_audit_frame(frame=_reconstruct_wind(frame=invalid_bearing)),
        reason=("Missing-speed independent-bearing guard failed"),
    )
    mismatch = json.loads(json.dumps(approved_runtime))
    mismatch["versions"]["pvlib"] = "deliberately-unapproved-synthetic-version"
    _expect_rejection(
        operation=lambda: _check_runtime(approved=mismatch, parent=parent),
        reason=("Runtime mismatch guard failed"),
    )
    return {
        ("missing_wrong_members"): True,
        ("wrong_target"): True,
        ("wrong_init"): True,
        ("unavailable_lag"): True,
        ("frame_support_study_boundary"): True,
        ("missing_speed_invalid_bearing"): True,
        ("runtime_mismatch"): True,
    }


def _bounded_power_boundary_guard(*, power: pl.DataFrame) -> None:
    """Reject observed outcomes at or beyond the held-out study boundary."""
    _require(
        condition=not power.filter(pl.col("time") >= pl.lit(STUDY_END)).height,
        message=("Bounded observations reach held-out boundary"),
    )


def _arguments() -> argparse.Namespace:
    """Parse explicit pinned source, runtime, proof and absent output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--runtime-spec", required=True, type=Path)
    parser.add_argument("--runtime-spec-sha256", required=True)
    parser.add_argument("--proof-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _write_completion(
    *,
    output: Path,
    proof_output: Path,
    parent: dict[str, Any],
    parent_manifest_sha256: str,
    context: dict[str, Any],
    files: dict[str, str],
    details: dict[str, Any],
) -> None:
    """Finalize the frame manifest, then write its external no-cycle proof handshake."""
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    manifest = {
        "input_fingerprint": fingerprint,
        "window": parent["window"],
        "context": context,
        "files": files,
        "frames": details,
        "parent_provenance": {
            "manifest_sha256": parent_manifest_sha256,
            "historical_parent_manifest": parent,
            "original_frames": parent["frames"],
            ("boundary_semantics"): (
                "Inherited parent native-input and boundary checks only; this builder d"
                "oes not re-engineer NWP"
            ),
        },
        "complete": True,
    }
    # Completion manifest is the last write inside the frame root. Failed outputs remain.
    _write_bytes(path=output / "manifest.json", content=_encoded(document=manifest))
    # Separate external handshake is written after manifest finalization; the
    # manifest fingerprints augmentation-audit.json, so no proof hash cycle exists.
    proof = {
        "schema_version": 1,
        "status": "complete",
        "original_manifest_sha256": parent_manifest_sha256,
        "augmented_manifest_sha256": _sha(path=output / "manifest.json"),
        "builder_sha256": context["augmentation_code_sha256"]["builder"],
        "original_columns_bit_exact": True,
        "copied_artifacts_byte_exact": True,
    }
    _write_bytes(path=proof_output, content=_encoded(document=proof))


def _initialize_output(
    *,
    output: Path,
    source: Path,
    context: dict[str, Any],
    runtime_bytes: bytes,
    parent: dict[str, Any],
) -> dict[str, str]:
    """Create private auxiliary files and copy exact frozen support bytes."""
    output.mkdir(mode=0o700, exist_ok=False)
    (output / "val").mkdir(mode=0o700)
    (output / "_lookups").mkdir(mode=0o700)
    _write_bytes(path=output / "context.json", content=_encoded(document=context))
    _write_bytes(path=output / "runtime-spec.json", content=runtime_bytes)
    files = {}
    for name in sorted(SUPPORT):
        _copy_bytes(source=source / name, destination=output / name)
        _require(
            condition=_sha(path=output / name) == parent[("files")][name],
            message=("Byte-copy support file changed"),
        )
        files[name] = _sha(path=output / name)
    return files


def main() -> None:
    """Augment only after approved runtime and complete frozen-frame semantic preflight."""
    args = _arguments()
    _require(
        condition=all(os.environ.get(name) == "8" for name in THREADS),
        message="Caller must set all seven thread caps to eight",
    )
    source, output = _plain(path=args.source), _plain(path=args.output)
    proof_output = _plain(path=args.proof_output)
    _require(
        condition=not output.exists()
        and not output.is_symlink()
        and output.parent.is_dir()
        and not output.is_relative_to(source)
        and not source.is_relative_to(output),
        message="Output must be new and separate",
    )
    _require(
        condition=not proof_output.exists()
        and not proof_output.is_symlink()
        and proof_output.parent.is_dir()
        and not proof_output.is_relative_to(source)
        and not proof_output.is_relative_to(output)
        and proof_output != _plain(path=args.runtime_spec),
        message=("Proof must be an absent separate private file"),
    )
    parent, validation, window, context_hash = _source_preflight(
        source=source, pinned=args.source_manifest_sha256
    )
    runtime_path = _plain(path=args.runtime_spec)
    runtime_bytes = runtime_path.read_bytes()
    _require(
        condition=hashlib.sha256(runtime_bytes).hexdigest() == args.runtime_spec_sha256,
        message=("Approved runtime spec changed"),
    )
    approved_runtime = json.loads(runtime_bytes)
    actual_runtime = _check_runtime(approved=approved_runtime, parent=parent)
    guards = _analytic_guards()
    malformed_guards = _semantic_guards(approved_runtime=approved_runtime, parent=parent)
    paths = [source / "train.parquet", *[source / name for name in validation]]
    metadata = pl.read_parquet(source / "metadata.parquet")
    capacity = pl.read_parquet(source / "capacity.parquet")
    power = pl.read_parquet(source / "bounded_power.parquet")
    support_proof = _support_semantics(
        metadata=metadata, capacity=capacity, power=power, expected_count=31
    )
    source_proof = _frozen_frame_semantics(
        root=source,
        parent=parent,
        validation=validation,
        window=window,
        metadata=metadata,
        power=power,
        augmented=False,
    )
    code_paths = {"builder": Path(__file__).resolve(), **_runtime_sources()}
    contract_bounds = _precipitation_contract_bounds(path=code_paths["weather_contract"])
    code_hashes = {name: _sha(path=path) for name, path in code_paths.items()}
    context = {
        **parent["context"],
        "parent_manifest_sha256": args.source_manifest_sha256,
        "parent_context_sha256": context_hash,
        "parent_input_fingerprint": parent["input_fingerprint"],
        "approved_runtime_spec_sha256": args.runtime_spec_sha256,
        "approved_and_actual_runtime": {"approved": approved_runtime, "actual": actual_runtime},
        "window": parent["window"],
        ("augmentation"): (
            "Twelve appended features: four deterministic time-and-coordinate solar "
            "features, four reconstructed post-interpolation wind components, and "
            "four circular wind-bearing encodings; original engineered columns "
            "unchanged"
        ),
        "new_features": ADDED,
        ("vector_semantics"): (
            "Float32 east=sin(z)*sin(a), north=sin(z)*cos(a), up=cos(z); original "
            "apparent angles at valid_time minus 15 minutes; no night clipping"
        ),
        ("clear_sky_semantics"): (
            "Haurwitz global horizontal W/m2 averaged at 30 one-minute midpoints of "
            "the half-hour ending valid_time; offsets seconds -1770,-1710,...,-30; "
            "model estimate, not measured or tilted-panel irradiance"
        ),
        ("wind_component_semantics"): (
            "Reconstructed eastward u=-speed*sin(bearing_rad), northward "
            "v=-speed*cos(bearing_rad), at 10m and 100m, using corrected "
            "POST-interpolation meteorological speed and bearing. Float64 arithmetic "
            "then Float32. These are not native upstream components and add no "
            "weather information. Null inputs propagate; nonfinite inputs are not "
            "replaced by zero; finite calm components are zero."
        ),
        ("wind_direction_circular_semantics"): (
            "Sine and cosine of corrected post-interpolation meteorological bearings "
            "in degrees, at 10m and 100m; Float64 radians and trigonometry then "
            "Float32. Null bearings propagate independently of speed; nonfinite "
            "bearings stay nonfinite. These are alternate representations and add no "
            "weather information."
        ),
        "augmentation_code_sha256": code_hashes,
        "augmentation_python": sys.version,
        "augmentation_dependencies": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "polars", "pyarrow", "pvlib", "pandas"]
        },
        "thread_caps": {name: os.environ[name] for name in THREADS},
        "analytic_tolerances": {
            "angle_degrees": ANGLE_TOLERANCE_DEGREES,
            "unit_norm": VECTOR_NORM_TOLERANCE,
            "integration_w_m2": CONVERGENCE_TOLERANCE_W_M2,
            "wind_magnitude_absolute_m_s": WIND_MAGNITUDE_ABSOLUTE_TOLERANCE_M_S,
            "wind_magnitude_relative": WIND_MAGNITUDE_RELATIVE_TOLERANCE,
            "wind_bearing_degrees": WIND_BEARING_TOLERANCE_DEGREES,
        },
    }
    files = _initialize_output(
        output=output, source=source, context=context, runtime_bytes=runtime_bytes, parent=parent
    )
    lookup_paths, angle_error, azimuth_error = _make_lookup(
        paths=paths, metadata=metadata, output=output
    )
    lookup = pl.concat([pl.read_parquet(path) for path in lookup_paths])
    _require(
        condition=not lookup.select("time_series_id", "valid_time").is_duplicated().any(),
        message="Duplicate solar lookup key",
    )
    details, wind_counts = {}, {}
    for relative in ["train.parquet", *validation]:
        details[relative], wind_counts[relative] = _augment(
            source=source / relative, destination=output / relative, lookup=lookup
        )
        _require(
            condition=details[relative]["rows"] == parent["frames"][relative]["rows"],
            message="Augmented frame differs from inherited row count",
        )
        start, end = (
            (window["train_start"], window["train_end"])
            if relative == "train.parquet"
            else (window["val_start"], window["val_end"])
        )
        _require(
            condition=datetime.fromisoformat(details[relative]["valid_start"]) >= start
            and datetime.fromisoformat(details[relative]["valid_end"]) <= end,
            message="Frame exceeds inherited window",
        )
        files[relative] = _sha(path=output / relative)
    output_proof = _frozen_frame_semantics(
        root=output,
        parent=parent,
        validation=validation,
        window=window,
        metadata=metadata,
        power=power,
        augmented=True,
    )
    audit = {
        "analytic_guards": guards,
        "malformed_synthetic_guards": malformed_guards,
        "support_semantic_proof": support_proof,
        "source_frame_semantic_proof": source_proof,
        "augmented_frame_semantic_proof": output_proof,
        "maximum_stored_midpoint_zenith_error_degrees": angle_error,
        "maximum_stored_midpoint_azimuth_error_degrees": azimuth_error,
        "original_columns_rows_keys_targets_preserved": True,
        "support_files_byte_identical": True,
        "wind_reconstruction": {
            "per_frame_aggregate_counts_and_roundtrip": wind_counts,
            "semantics": context["wind_component_semantics"],
            "circular_semantics": context["wind_direction_circular_semantics"],
            ("count_semantics"): (
                "Frame rows include repeated weather members and run/lead combinations. "
                "No private polar/component values are reported. Zero finite support is "
                "explicitly unsupported."
            ),
        },
        "precipitation": _precipitation_audit(
            paths=paths,
            metadata=metadata,
            source_manifest_hash=args.source_manifest_sha256,
            source_file_hashes={
                name: parent[("files")][name] for name in [("train.parquet"), *validation]
            },
            contract_path=code_paths["weather_contract"],
            contract_bounds=contract_bounds,
        ),
    }
    _write_bytes(path=output / "augmentation-audit.json", content=_encoded(document=audit))
    for path in [
        output / ("context.json"),
        output / ("runtime-spec.json"),
        output / ("augmentation-audit.json"),
        *lookup_paths,
    ]:
        files[str(path.relative_to(output))] = _sha(path=path)
    final_parent, _, _, final_context_hash = _source_preflight(
        source=source, pinned=args.source_manifest_sha256
    )
    _require(
        condition=_sha(path=runtime_path) == args.runtime_spec_sha256
        and _check_runtime(approved=approved_runtime, parent=final_parent) == actual_runtime,
        message="Approved runtime changed before completion",
    )
    _require(
        condition=final_parent == parent
        and final_context_hash == context_hash
        and all(_sha(path=path) == code_hashes[name] for name, path in code_paths.items()),
        message="Source or builder changed",
    )
    for relative, expected in files.items():
        _require(
            condition=_sha(path=output / relative) == expected,
            message=("Completed augmentation input changed"),
        )
    _write_completion(
        output=output,
        proof_output=proof_output,
        parent=parent,
        parent_manifest_sha256=args.source_manifest_sha256,
        context=context,
        files=files,
        details=details,
    )
    print(
        "Completed immutable solar and reconstructed-wind augmentation; original "
        "inputs preserved; no models fitted"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        AssertionError,
        StopIteration,
        pl.exceptions.PolarsError,
        pa.ArrowException,
        DataFrameValidationError,
    ):
        raise SystemExit(
            "Solar augmentation failed; any partial private output is preserved; no automatic retry"
        ) from None
