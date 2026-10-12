"""Rewrite a pinned local polar-wind archive into a separate component-wind Delta table.

The source remains read-only. A sibling manifest pins the source snapshot and records completed
partitions; rerunning replaces any interrupted partition through the production writer. Keep the
source and its matching code for rollback. Cutover and catch-up are described at
<https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/>.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Final, cast

import patito as pt
import polars as pl
from contracts.weather_schemas import Nwp
from delta_store.nwp import write_nwp
from deltalake import DeltaTable
from ml_core.features import wind_feature_expressions
from polars.testing import assert_frame_equal

POLAR_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"wind_{quantity}_{height}" for height in ("10m", "100m") for quantity in ("speed", "direction")
)
KEY: Final[tuple[str, ...]] = ("init_time", "valid_time", "ensemble_member", "h3_index")


def convert_polar_wind(frame: pl.DataFrame) -> pt.DataFrame[Nwp]:
    """Recover signed eastward/northward components from archived meteorological bearings."""
    expressions = []
    for height in ("10m", "100m"):
        speed = pl.col(f"wind_speed_{height}").cast(pl.Float64)
        bearing = pl.col(f"wind_direction_{height}").cast(pl.Float64).radians()
        expressions.extend(
            (
                (-speed * bearing.sin()).cast(pl.Float32).alias(f"wind_u_{height}"),
                (-speed * bearing.cos()).cast(pl.Float32).alias(f"wind_v_{height}"),
            )
        )
    return Nwp.validate(frame.with_columns(expressions).drop(POLAR_COLUMNS))


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def _verify_wind_round_trip(*, old: pl.DataFrame, readback: pl.DataFrame) -> dict[str, float]:
    """Check speed and circular-bearing error after the production writer's quantization."""
    original = old.sort(KEY)
    reconstructed = readback.sort(KEY).with_columns(
        expression
        for height in ("10m", "100m")
        for expression in wind_feature_expressions(height=height)
    )
    errors = {}
    for height in ("10m", "100m"):
        speed = original[f"wind_speed_{height}"].cast(pl.Float64)
        relative = (
            (reconstructed[f"wind_speed_{height}"] - speed).abs() / speed.replace(0, None)
        ).max()
        circular = (
            (
                (
                    reconstructed[f"wind_direction_{height}"]
                    - original[f"wind_direction_{height}"]
                    + 180
                )
                % 360
                - 180
            )
            .abs()
            .filter(speed > 0)
            .max()
        )
        relative = cast(float | None, relative)
        circular = cast(float | None, circular)
        # Independent 13-bit component rounding bounds the vector's relative error by 2^-13.
        # The corresponding angular bound is asin(2^-13 / (1 - 2^-13)), below 0.008 degrees.
        # Float32 derivation adds a small allowance; exact calm has no defined bearing.
        if relative is not None and relative > 2**-13 + 2e-7:
            raise ValueError(f"{height} speed round-trip exceeds writer precision: {relative}")
        if circular is not None and circular > 0.0081:
            raise ValueError(f"{height} bearing round-trip exceeds writer precision: {circular}")
        errors[f"{height}_max_relative_speed_error"] = float(relative or 0)
        errors[f"{height}_max_circular_bearing_error_degrees"] = float(circular or 0)
    return errors


def rewrite(*, source: Path, destination: Path, source_version: int, limit: int | None) -> None:
    """Copy a pinned snapshot, resuming or catching up to an explicitly newer snapshot."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-overlapping directories")
    snapshot = DeltaTable(str(source), version=source_version)
    manifest_path = destination.with_name(destination.name + ".wind-migration.json")
    identity = {
        "source": str(source),
        "destination": str(destination),
        "target_schema": {name: str(dtype) for name, dtype in Nwp.dtypes.items()},
    }
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity or manifest["source_version"] > source_version:
            raise ValueError(
                "Migration manifest belongs to a different source, schema or newer snapshot"
            )
        if manifest["completed"] and not DeltaTable.is_deltatable(str(destination)):
            raise ValueError("Previously verified destination table is missing")
    else:
        if destination.exists():
            raise ValueError("A new migration requires an absent destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"identity": identity, "source_version": source_version, "completed": {}}
        _save_manifest(path=manifest_path, manifest=manifest)

    actions = pl.DataFrame(snapshot.get_add_actions(flatten=True))
    partitions = (
        actions.group_by("partition.nwp_model_id", "partition.init_time")
        .agg(pl.col("path").sort().alias("source_files"))
        .sort("partition.nwp_model_id", "partition.init_time")
    )
    inventory = {
        f"{model_id}/{init_time.isoformat()}" for model_id, init_time, _ in partitions.iter_rows()
    }
    if set(manifest["completed"]) - inventory:
        raise ValueError("Source snapshot removed a migrated partition; reconcile before catch-up")
    manifest["source_version"] = source_version
    _save_manifest(path=manifest_path, manifest=manifest)
    selected = partitions if limit is None else partitions.head(limit)
    for model_id, init_time, source_files in selected.iter_rows():
        partition_id = f"{model_id}/{init_time.isoformat()}"
        previous = manifest["completed"].get(partition_id)
        if previous is not None and previous["source_files"] == source_files:
            print(f"{partition_id}: already verified", flush=True)
            continue
        predicate = (pl.col("nwp_model_id") == model_id) & (pl.col("init_time") == init_time)
        old = pl.scan_delta(str(source), version=source_version).filter(predicate).collect()
        converted = convert_polar_wind(frame=old)
        write_nwp(nwp=converted, table_uri=destination)
        readback = pl.scan_delta(str(destination)).filter(predicate).collect()
        Nwp.validate(readback)
        unchanged = [name for name in old.columns if name not in POLAR_COLUMNS]
        assert_frame_equal(
            old.select(unchanged).sort(KEY), readback.select(unchanged).sort(KEY), check_exact=True
        )
        manifest["completed"][partition_id] = {
            "rows": readback.height,
            "source_files": source_files,
            "wind_round_trip": _verify_wind_round_trip(old=old, readback=readback),
        }
        _save_manifest(path=manifest_path, manifest=manifest)
        print(f"{partition_id}: {readback.height} rows verified", flush=True)


def main() -> None:
    """Build a separate component archive; never modify the serving source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-version", type=int, required=True)
    parser.add_argument("--limit", type=int, help="Pilot: maximum number of source partitions")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    rewrite(
        source=args.source,
        destination=args.destination,
        source_version=args.source_version,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
