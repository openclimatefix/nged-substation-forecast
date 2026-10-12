"""Build immutable, version-pinned feature frames for the fixed-pipeline comparisons.

Choose the installed feature implementation through PYTHONPATH before starting Python. This
script never redirects imports or writes to the published quick-wins output directory. All output
is private: source identifiers, metadata and observations must not be copied into public reports.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import patito as pt
import polars as pl
import quick_wins_inputs as inputs
from build_frames import ENGINEERED_FEATURES, KEY_COLUMNS
from contracts.power_schemas import EffectiveCapacity, PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import Nwp
from deltalake import DeltaTable
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer
from studies.power import CLEANED_POWER_DELTA_URI, scan_power
from studies.solar import azimuth, zenith

STUDY_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Exclusive upper bound on every observed outcome used in this study."""
SORT_KEYS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "nwp_init_time",
    "valid_time",
    "ensemble_member",
)
SIGMA: Final[float] = 5.670374419e-8
"""Stefan-Boltzmann constant in W m⁻² K⁻⁴."""
CHUNK_DAYS: Final[int] = 7
"""Initialisation-date span of one bounded training engineering job."""


def _digest_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_frame(*, frame: pl.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _digest_context(*, context: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def _atomic_json(*, path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_parquet(*, path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def _version(*, uri: str, settings: Settings) -> int:
    return DeltaTable(
        table_uri=uri, storage_options=typeddict_to_dict(settings.storage_options)
    ).version()


def _power(*, settings: Settings, version: int, series_ids: list[int]) -> pl.DataFrame:
    if str(Path(settings.cleaned_power_time_series_data_path).resolve()) != str(
        Path(CLEANED_POWER_DELTA_URI).resolve()
    ):
        raise ValueError("Settings and studies.power resolve different cleaned-power tables")
    if _version(uri=CLEANED_POWER_DELTA_URI, settings=settings) != version:
        raise ValueError("Cleaned-power version differs from the requested snapshot")
    # Keep early history for first-live-date cleaning and pre-window power lags.
    power = pl.DataFrame(
        scan_power()
        .filter(pl.col("time_series_id").is_in(series_ids), pl.col("time") < STUDY_END)
        .collect()
    ).sort("time_series_id", "time")
    if _version(uri=CLEANED_POWER_DELTA_URI, settings=settings) != version:
        raise ValueError("Cleaned-power table changed while observations were materialised")
    if power.is_empty() or power.select("time_series_id", "time").is_duplicated().any():
        raise ValueError("Bounded observations are empty or have duplicate keys")
    return power


def _solar(*, frame: pl.DataFrame, metadata: pt.DataFrame[TimeSeriesMetadata]) -> pl.DataFrame:
    parts = []
    for series_id, latitude, longitude in metadata.select(
        "time_series_id", "latitude", "longitude"
    ).iter_rows():
        stamps = frame.filter(pl.col("time_series_id") == series_id)["valid_time"].unique().sort()
        if stamps.is_empty():
            continue
        centre = stamps.dt.offset_by("-15m")
        parts.append(
            pl.DataFrame(
                {
                    "time_series_id": pl.Series(
                        [series_id] * len(stamps), dtype=frame.schema["time_series_id"]
                    ),
                    "valid_time": stamps,
                    "solar_zenith": zenith(
                        stamps=centre, latitude=float(latitude), longitude=float(longitude)
                    ),
                    "solar_azimuth": azimuth(
                        stamps=centre, latitude=float(latitude), longitude=float(longitude)
                    ),
                }
            )
        )
    return frame.join(pl.concat(parts), on=["time_series_id", "valid_time"], how="left")


def _native_scan(
    *,
    args: argparse.Namespace,
    settings: Settings,
    metadata: pt.DataFrame[TimeSeriesMetadata],
    init_start: datetime,
    init_end: datetime,
    training: bool,
) -> pl.LazyFrame:
    native = pl.scan_delta(
        args.archive,
        version=args.archive_version,
        storage_options=typeddict_to_dict(settings.storage_options),
    ).filter(
        pl.col("init_time") >= init_start,
        pl.col("init_time") < init_end,
        pl.col("h3_index").is_in(metadata["h3_res_5"].unique().to_list()),
        pl.col("nwp_model_id") == "ECMWF_ENS_0_25_degree",
    )
    columns = set(native.collect_schema().names())
    expected = (
        {"wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"}
        if args.pipeline_tag in {"wind", "combined"}
        else {"wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m"}
    )
    if not expected <= columns:
        raise ValueError("Archive wind representation does not match the pipeline treatment")
    if training:
        native = native.filter(pl.col("ensemble_member") == 0)
    return native


def _engineer(
    *,
    native: pl.LazyFrame,
    power: pl.DataFrame,
    metadata: pt.DataFrame[TimeSeriesMetadata],
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    # Do not cut native valid times: radiation needs complete intervals and PCHIP neighbours.
    engineered = (
        TabularFeatureEngineer()
        .engineer(
            selected_features=set(ENGINEERED_FEATURES),
            power_time_series=pt.LazyFrame.from_existing(power.lazy()).set_model(PowerTimeSeries),
            time_series_metadata=metadata,
            nwp=pt.LazyFrame.from_existing(native).set_model(Nwp),
        )
        .filter(pl.col("valid_time") >= start, pl.col("valid_time") <= end)
        .collect()
    )
    frame = pl.DataFrame(engineered).select(*KEY_COLUMNS, *sorted(ENGINEERED_FEATURES))
    if frame.is_empty():
        raise ValueError("Engineering produced an empty frame")
    if set(frame["time_series_id"].unique()) != set(metadata["time_series_id"]):
        raise ValueError("Engineering omitted one or more eligible series")
    frame = _solar(frame=frame, metadata=metadata).with_columns(
        longwave_minus_sigma_t4=pl.col("downward_long_wave_radiation_flux_surface")
        - SIGMA * (pl.col("temperature_2m") + 273.15).pow(4)
    )
    # Enum physical positions are the same encoding used by the existing quick-wins fitter.
    features = [name for name in frame.columns if name not in KEY_COLUMNS]
    frame = frame.with_columns(
        [pl.col(name).to_physical().cast(pl.Float32).alias(name) for name in features]
    ).sort(*SORT_KEYS)
    if frame.select(*SORT_KEYS).is_duplicated().any():
        raise ValueError("Engineered frame has duplicate forecast keys")
    if frame.filter(
        (pl.col("valid_time") >= STUDY_END)
        | (pl.col("valid_time") <= pl.col("power_fcst_init_time"))
    ).height:
        raise ValueError("Engineered frame has illegal forecast dates")
    return frame


def _check_members(*, frame: pl.DataFrame, training: bool) -> None:
    expected = [0] if training else list(range(51))
    groups = frame.group_by(*[key for key in SORT_KEYS if key != "ensemble_member"]).agg(
        members=pl.col("ensemble_member").sort()
    )
    if groups.filter(pl.col("members") != pl.lit(expected)).height:
        raise ValueError("Forecast keys have incomplete or unexpected ensemble members")


def _native_info(*, native: pl.LazyFrame) -> dict[str, Any]:
    keys = ["nwp_model_id", "init_time", "valid_time", "ensemble_member", "h3_index"]
    rows = native.select(*keys).sort(*keys).collect()
    if rows.is_duplicated().any():
        raise ValueError("Native archive contains duplicate input keys")
    return {"rows": rows.height, "key_sha256": _digest_frame(frame=rows)}


def _iso_stamp(*, value: object) -> str:
    if not isinstance(value, datetime):
        raise TypeError("Frame date bounds must be nonnull timestamps")
    return value.isoformat()


def _frame_info(*, frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "key_sha256": _digest_frame(frame=frame.select(*SORT_KEYS)),
        "valid_start": _iso_stamp(value=frame["valid_time"].min()),
        "valid_end": _iso_stamp(value=frame["valid_time"].max()),
        "features": [name for name in frame.columns if name not in KEY_COLUMNS],
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
    }


def _combine_training(*, parts: list[Path], output: Path) -> dict[str, Any]:
    train = pl.scan_parquet(parts).sort(*SORT_KEYS)
    temporary = output / "train.parquet.tmp"
    train.sink_parquet(temporary)
    temporary.replace(output / "train.parquet")
    # Summarise the training keys without loading its feature matrix a second time.
    keys = pl.scan_parquet(output / "train.parquet").select(*SORT_KEYS).collect()
    if keys.is_duplicated().any():
        raise ValueError("Concatenated training frames have duplicate forecast keys")
    details = {
        "rows": keys.height,
        "key_sha256": _digest_frame(frame=keys),
        "valid_start": _iso_stamp(value=keys["valid_time"].min()),
        "valid_end": _iso_stamp(value=keys["valid_time"].max()),
        "features": [name for name in train.collect_schema().names() if name not in KEY_COLUMNS],
        "schema": {name: str(dtype) for name, dtype in train.collect_schema().items()},
    }
    for part in parts:
        part.unlink()
    (output / "_train_parts").rmdir()
    return details


def _build(
    *,
    args: argparse.Namespace,
    settings: Settings,
    window: inputs.FoldWindow,
    metadata: pt.DataFrame[TimeSeriesMetadata],
    power: pl.DataFrame,
) -> tuple[dict[str, Any], dict[str, str]]:
    output = args.output
    details: dict[str, Any] = {}
    files: dict[str, str] = {}
    start = window.train_start - inputs.MAX_NWP_LEAD
    parts = []
    boundary_checked = False
    native_details: dict[str, Any] = {}
    while start <= window.train_end:
        end = min(start + timedelta(days=CHUNK_DAYS), window.train_end + timedelta(microseconds=1))
        native = _native_scan(
            args=args,
            settings=settings,
            metadata=metadata,
            init_start=start,
            init_end=end,
            training=True,
        )
        # Initial chunks may precede archive coverage; they contribute no fitting rows.
        if native.limit(1).collect().is_empty():
            start = end
            continue
        native_details[f"train/init={start:%Y-%m-%d}"] = _native_info(native=native)
        frame = _engineer(
            native=native,
            power=power,
            metadata=metadata,
            start=window.train_start,
            end=window.train_end,
        )
        _check_members(frame=frame, training=True)
        if not boundary_checked:
            init_time = frame["nwp_init_time"].max()
            alone = _engineer(
                native=native.filter(pl.col("init_time") == init_time),
                power=power,
                metadata=metadata,
                start=window.train_start,
                end=window.train_end,
            )
            if not alone.equals(frame.filter(pl.col("nwp_init_time") == init_time)):
                raise ValueError("Full-run engineering depends on neighbouring run chunks")
            boundary_checked = True
        part = output / "_train_parts" / f"init={start:%Y-%m-%d}.parquet"
        _atomic_parquet(path=part, frame=frame)
        parts.append(part)
        start = end
    if not parts:
        raise ValueError("No training frames were produced")
    details["train.parquet"] = _combine_training(parts=parts, output=output)
    files["train.parquet"] = _digest_file(path=output / "train.parquet")
    runs = inputs.sample_init_times(window=window)
    if args.max_validation_runs is not None:
        runs = runs[: args.max_validation_runs]
    for init_time in runs:
        native = _native_scan(
            args=args,
            settings=settings,
            metadata=metadata,
            init_start=init_time,
            init_end=init_time + timedelta(microseconds=1),
            training=False,
        )
        native_details[f"val/init={init_time:%Y-%m-%d}"] = _native_info(native=native)
        frame = _engineer(
            native=native,
            power=power,
            metadata=metadata,
            start=window.val_start,
            end=min(window.val_end, init_time + inputs.MAX_NWP_LEAD),
        )
        _check_members(frame=frame, training=False)
        name = f"val/init={init_time:%Y-%m-%d}.parquet"
        _atomic_parquet(path=output / name, frame=frame)
        files[name] = _digest_file(path=output / name)
        details[name] = _frame_info(frame=frame)
    details["native_inputs"] = native_details
    details["boundary_equivalence"] = {"checked": boundary_checked, "mode": "bulk-run-isolation"}
    return details, files


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", choices=[inputs.FOLD_A, inputs.FOLD_B], required=True)
    parser.add_argument(
        "--pipeline-tag", choices=["old", "radiation", "wind", "combined"], required=True
    )
    parser.add_argument("--pipeline-revision", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-version", type=int, required=True)
    parser.add_argument("--power-version", type=int, required=True)
    parser.add_argument("--capacity-version", type=int, required=True)
    parser.add_argument("--max-validation-runs", type=int)
    args = parser.parse_args()
    if args.max_validation_runs is not None and args.max_validation_runs < 1:
        parser.error("--max-validation-runs must be positive")
    if not os.environ.get("DATA_PATH_INTERNAL") or not os.environ.get("DATA_PATH_DELIVERY"):
        parser.error("Explicit DATA_PATH_INTERNAL and DATA_PATH_DELIVERY are required")
    return args


def main() -> None:
    """Build private frames without fitting or scoring any forecast model."""
    args = _arguments()
    settings = Settings()
    window = inputs.window_for(fold_id=args.fold)
    if max(window.train_end, window.val_end) >= STUDY_END:
        raise ValueError("Configured fold crosses the study outcome boundary")
    options = typeddict_to_dict(settings.storage_options)
    population_version = _version(uri=settings.eligible_time_series_data_path, settings=settings)
    population = (
        pl.read_delta(
            settings.eligible_time_series_data_path,
            version=population_version,
            storage_options=options,
        )
        .filter(pl.col("fold_id") == inputs.FOLD_A)
        .select("time_series_id")
        .sort("time_series_id")
    )
    if population.height != 31 or population["time_series_id"].n_unique() != 31:
        raise ValueError("The pinned quick-wins population must contain exactly 31 unique series")
    series_ids = population["time_series_id"].to_list()
    metadata = inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    metadata = pt.DataFrame(metadata.sort("time_series_id")).set_model(TimeSeriesMetadata)
    if metadata.height != 31:
        raise ValueError("Metadata must have exactly one row per eligible series")
    capacity = (
        pl.read_delta(
            settings.effective_capacity_data_path,
            version=args.capacity_version,
            storage_options=options,
        )
        .filter(pl.col("time_series_id").is_in(series_ids))
        .sort("time_series_id")
    )
    EffectiveCapacity.validate(capacity)
    if capacity.height != 31 or capacity["time_series_id"].n_unique() != 31:
        raise ValueError("Capacity must have exactly one row per eligible series")
    power = _power(settings=settings, version=args.power_version, series_ids=series_ids)
    engineer_file = Path(inspect.getfile(TabularFeatureEngineer)).resolve()
    code_hashes = {
        str(path): _digest_file(path=path) for path in sorted(engineer_file.parent.rglob("*.py"))
    }
    for path in [
        Path(__file__),
        Path(inspect.getfile(inputs)),
        Path(inspect.getfile(zenith)),
        inputs.PRODUCTION_MODEL_CONFIG,
    ]:
        code_hashes[str(path.resolve())] = _digest_file(path=path)
    context = {
        "window": {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in window._asdict().items()
        },
        "engineered_features": sorted(ENGINEERED_FEATURES),
        "key_columns": list(KEY_COLUMNS),
        "pipeline_tag": args.pipeline_tag,
        "pipeline_revision": args.pipeline_revision,
        "archive": args.archive,
        "archive_version": args.archive_version,
        "power_version": args.power_version,
        "capacity_version": args.capacity_version,
        "population_version": population_version,
        "max_validation_runs": args.max_validation_runs,
        "study_end_exclusive": STUDY_END.isoformat(),
        "code_sha256": code_hashes,
        "metadata_source": settings.metadata_path,
        "capacity_source": settings.effective_capacity_data_path,
        "population_source": settings.eligible_time_series_data_path,
        "power_source": CLEANED_POWER_DELTA_URI,
        "data_path_internal": settings.data_path_internal,
        "data_path_delivery": settings.data_path_delivery,
        "input_hashes": {
            "metadata": _digest_frame(frame=pl.DataFrame(metadata)),
            "capacity": _digest_frame(frame=capacity),
            "population": _digest_frame(frame=population),
            "bounded_power": _digest_frame(frame=power),
        },
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("polars", "patito", "deltalake", "pvlib", "numpy")
        },
    }
    fingerprint = _digest_context(context=context)
    output = args.output.resolve()
    args.output = output
    context_path = output / "context.json"
    if output.exists() and any(output.iterdir()):
        if not context_path.exists() or json.loads(context_path.read_text()) != context:
            raise ValueError("Output directory has a different or missing immutable context")
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["input_fingerprint"] != fingerprint or any(
                _digest_file(path=output / name) != digest
                for name, digest in manifest["files"].items()
            ):
                raise ValueError("Completed output does not match its manifest")
            print("Verified existing immutable feature frames")
            return
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(path=context_path, content=context)
    for name, frame in {
        "metadata.parquet": pl.DataFrame(metadata),
        "capacity.parquet": capacity,
        "bounded_power.parquet": power,
    }.items():
        _atomic_parquet(path=output / name, frame=frame)
    details, files = _build(
        args=args,
        settings=settings,
        window=window,
        metadata=metadata,
        power=power,
    )
    for name in ("metadata.parquet", "capacity.parquet", "bounded_power.parquet"):
        files[name] = _digest_file(path=output / name)
    _atomic_json(
        path=output / "manifest.json",
        content={
            "input_fingerprint": fingerprint,
            "window": context["window"],
            "context": context,
            "files": files,
            "frames": details,
            "complete": args.max_validation_runs is None,
        },
    )
    print("Completed immutable feature frames; no models fitted")


if __name__ == "__main__":
    main()
