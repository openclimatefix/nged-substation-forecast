"""Measure wind representations with identical raw runs and the production NWP writer.

Run ``fetch`` once, then ``write`` in each experimental schema worktree. The ``report`` action
requires all three arms and refuses unequal run inventories. Outputs are local to the study;
no action writes to the serving archive. Dates are frozen in ``manifest.json``.
"""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, get_args

import dynamical_catalog
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import (
    ECMWF_ENS_ENSEMBLE_MEMBERS,
    ECMWF_ENS_LEAD_TIME_HOURS,
    Nwp,
    WeatherFeature,
)
from delta_store.nwp import NWP_SIGNIFICAND_BITS, NWP_SORT_COLS, write_nwp
from deltalake import DeltaTable
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe,
)
from dynamical_data.ecmwf_ens.download import download_ecmwf_ens_data, open_ecmwf_ens_run

RepresentationType = Literal["polar", "components", "scalar"]
REPRESENTATIONS: Final[tuple[RepresentationType, ...]] = ("polar", "components", "scalar")
HEIGHTS: Final[tuple[str, ...]] = ("10m", "100m")
KEY: Final[tuple[str, ...]] = ("init_time", "valid_time", "ensemble_member", "h3_index")


def _save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _identity(manifest: Path, grid: Path) -> dict[str, str]:
    return {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "grid_sha256": hashlib.sha256(grid.read_bytes()).hexdigest(),
    }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in paths:
        digest.update(str(file.relative_to(path) if path.is_dir() else "file").encode())
        with file.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _select_dates(requested: list[str], available: set[str]) -> list[str]:
    resolved = []
    for date in requested:
        candidates = sorted(day for day in available if day[:7] == date[:7])
        earlier = sorted((day for day in candidates if day <= date), reverse=True)
        later = [day for day in candidates if day > date]
        if not earlier and not later:
            raise ValueError(f"{date}: no available 00Z run in requested month")
        resolved.append((earlier + later)[0])
    if len(set(resolved)) != len(resolved):
        raise ValueError("Availability substitution produced duplicate runs")
    return resolved


def _resolved_dates(output: Path, requested: list[str], identity: dict[str, str]) -> list[str]:
    path = output / "resolved-manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity or saved["requested"] != requested:
            raise ValueError("Resolved sample belongs to different requested inputs")
        if saved["dates"] != _select_dates(requested=requested, available=set(saved["available"])):
            raise ValueError("Cached availability mapping is invalid")
        return saved["dates"]
    catalog = dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None)
    available = {
        str(value)[:10] for value in catalog.init_time.values if str(value)[11:19] == "00:00:00"
    }
    resolved = _select_dates(requested=requested, available=available)
    _save_json(
        path,
        {
            "identity": identity,
            "requested": requested,
            "dates": resolved,
            "available": sorted(available),
            "resolved_at": datetime.now(tz=UTC).isoformat(),
        },
    )
    return resolved


def _source_metadata() -> dict[str, Any]:
    import contracts.weather_schemas
    import delta_store.nwp
    import delta_store.precision
    import dynamical_data.ecmwf_ens.convert_to_polars

    modules = (
        contracts.weather_schemas,
        delta_store.nwp,
        delta_store.precision,
        dynamical_data.ecmwf_ens.convert_to_polars,
    )
    files = {module.__name__: Path(module.__file__).resolve() for module in modules}
    root = Path(__file__).resolve().parents[2]
    return {
        "module_paths": {name: str(path) for name, path in files.items()},
        "source_hashes": {name: _digest(path) for name, path in files.items()},
        "driver_hash": _digest(Path(__file__)),
        "lock_hash": _digest(root / "uv.lock"),
        "dirty": bool(subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"], text=True)),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def _validate_arm(frame: pl.DataFrame, representation: RepresentationType) -> None:
    polar = {f"wind_{name}_{height}" for name in ("speed", "direction") for height in HEIGHTS}
    components = {f"wind_{name}_{height}" for name in ("u", "v") for height in HEIGHTS}
    scalar = {f"wind_scalar_mean_speed_{height}" for height in HEIGHTS}
    expected_wind = polar if representation == "polar" else components
    if representation == "scalar":
        expected_wind |= scalar
    if {name for name in frame.columns if name.startswith("wind_")} != expected_wind:
        raise ValueError("Wind columns do not match the exact experimental arm")
    expected = (set(get_args(WeatherFeature)) - polar) | expected_wind | set(KEY) | {"nwp_model_id"}
    if set(frame.columns) != expected:
        raise ValueError("Experimental arm has missing or extra non-wind columns")
    if any(frame.schema[name] != pl.Float32 for name in expected_wind):
        raise ValueError("Wind storage fields must all be Float32")


def _validate_run(frame: pl.DataFrame, h3_grid: pl.DataFrame, date: str) -> None:
    init_time = datetime.fromisoformat(date).replace(tzinfo=UTC)
    cells = set(h3_grid["h3_index"].to_list())
    times = {init_time + timedelta(hours=hour) for hour in ECMWF_ENS_LEAD_TIME_HOURS}
    members = set(ECMWF_ENS_ENSEMBLE_MEMBERS)
    expected = len(cells) * len(times) * len(members)
    if (
        set(frame["init_time"].unique().to_list()) != {init_time}
        or set(frame["valid_time"].unique().to_list()) != times
        or set(frame["ensemble_member"].unique().to_list()) != members
        or set(frame["h3_index"].unique().to_list()) != cells
        or frame.height != expected
        or frame.select(*KEY).n_unique() != expected
    ):
        raise ValueError(f"{date}: incomplete, misidentified, or duplicated run")


def _own_output(output: Path, identity: dict[str, str]) -> None:
    marker = output / "wind-storage-owner.json"
    expected = {"study": "issue-525-wind-storage", **identity}
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError("Output directory belongs to different study inputs")
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing to adopt a nonempty unmarked output directory")
        _save_json(marker, expected)


def _prepare_table(destination: Path, identity: dict[str, str]) -> None:
    if destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError("Study table and arm directories must not be symlinks")
    marker = destination.with_suffix(".started.json")
    expected = {"destination": str(destination.resolve()), **identity}
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise ValueError("Refusing to remove an unmarked or differently owned table")
        shutil.rmtree(destination)
    _save_json(marker, expected)


def _fetch(output: Path, grid: Path, dates: list[str], identity: dict[str, str]) -> None:
    h3_grid = H3GridWeights.validate(pl.read_parquet(grid))
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    stamp = raw / "identity.json"
    if stamp.exists() and json.loads(stamp.read_text()) != identity:
        raise ValueError("Cached raw runs belong to a different grid or sample manifest")
    _save_json(stamp, identity)
    for date in dates:
        path = raw / f"{date}.zarr"
        if path.exists():
            saved = json.loads((raw / f"{date}.json").read_text())
            if saved["digest"] != _digest(path):
                raise ValueError(f"{date}: raw cache content changed")
            print(f"{date}: verified raw cache", flush=True)
            continue
        init_time = datetime.fromisoformat(date).replace(tzinfo=UTC)
        dataset = open_ecmwf_ens_run(nwp_init_time=init_time, h3_grid=h3_grid)
        print(f"{date}: downloading complete run {dict(dataset.sizes)}", flush=True)
        downloaded = download_ecmwf_ens_data(ds_sliced=dataset)
        temporary = raw / f"{date}.zarr.partial"
        downloaded.to_zarr(temporary, mode="w", zarr_format=2)
        _save_json(
            raw / f"{date}.json",
            {
                "digest": _digest(temporary),
                "fetched_at": datetime.now(tz=UTC).isoformat(),
                "source": "Dynamical ECMWF IFS ENS 15-day 0.25-degree",
                "resolved_date": date,
                "attributes": downloaded.attrs,
            },
        )
        temporary.rename(path)
        print(f"{date}: cached", flush=True)


def _scalar_dataset(dataset: xr.Dataset) -> xr.Dataset:
    # Different footprints would confound the vector/scalar comparison. Fail rather than report
    # a gap between two different spatial populations; a paired-footprint variant is needed then.
    result = dataset.copy()
    for height in HEIGHTS:
        u, v = dataset[f"wind_u_{height}"], dataset[f"wind_v_{height}"]
        if not bool((u.isnull() == v.isnull()).all()):
            raise ValueError(f"{height}: unmatched u/v null footprints; paired reduction required")
        result[f"wind_scalar_mean_speed_{height}"] = np.hypot(u, v)
    return result


def _column_bytes(table: Path) -> tuple[int, int, dict[str, int]]:
    columns: defaultdict[str, int] = defaultdict(int)
    rows = file_bytes = 0
    for filename in DeltaTable(str(table)).file_uris():
        path = Path(filename)
        file_bytes += path.stat().st_size
        metadata = pq.read_metadata(path)
        rows += metadata.num_rows
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                chunk = group.column(column_index)
                columns[chunk.path_in_schema] += chunk.total_compressed_size
    return rows, file_bytes, dict(columns)


def _write(
    output: Path,
    grid: Path,
    dates: list[str],
    identity: dict[str, str],
    representation: RepresentationType,
) -> None:
    raw = output / "raw"
    if json.loads((raw / "identity.json").read_text()) != identity:
        raise ValueError("Raw cache identity differs from requested inputs")
    h3_grid = H3GridWeights.validate(pl.read_parquet(grid))
    expected_rows = (
        h3_grid["h3_index"].n_unique()
        * len(ECMWF_ENS_ENSEMBLE_MEMBERS)
        * len(ECMWF_ENS_LEAD_TIME_HOURS)
    )
    arm = output / representation
    if arm.is_symlink():
        raise ValueError("Study arm directory must not be a symlink")
    arm.mkdir(parents=True, exist_ok=True)
    metadata = {
        **identity,
        "representation": representation,
        **_source_metadata(),
        "schema": {name: str(dtype) for name, dtype in Nwp.dtypes.items()},
        "significand_bits": NWP_SIGNIFICAND_BITS,
        "sort_columns": NWP_SORT_COLS,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("polars", "patito", "deltalake", "pyarrow", "xarray")
        },
    }
    if metadata["dirty"]:
        raise ValueError("Commit experimental variant before measurement")
    for date in dates:
        raw_metadata = json.loads((raw / f"{date}.json").read_text())
        if raw_metadata["digest"] != _digest(raw / f"{date}.zarr"):
            raise ValueError(f"{date}: raw cache differs from recorded content")
        metadata_for_run = {**metadata, "raw": raw_metadata}
        destination = arm / date
        result_path = arm / f"{date}.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous["metadata"] != json.loads(json.dumps(metadata_for_run)):
                raise ValueError(f"{date}: previous write has different provenance")
            if _digest(destination) != previous["table_digest"]:
                raise ValueError(f"{date}: measured table changed")
            if (
                representation == "scalar"
                and _digest(arm / f"{date}-gaps.parquet") != previous["gap_digest"]
            ):
                raise ValueError(f"{date}: scalar gaps changed")
            print(f"{representation} {date}: verified existing measurement", flush=True)
            continue
        with xr.open_zarr(raw / f"{date}.zarr", chunks=None) as source:
            dataset = source.load()
            if representation == "scalar":
                dataset = _scalar_dataset(dataset=dataset)
            frame = convert_nwp_xarray_dataset_to_polars_dataframe(ds=dataset, h3_grid=h3_grid)
        _validate_arm(frame=frame, representation=representation)
        _validate_run(frame=frame, h3_grid=h3_grid, date=date)
        _prepare_table(destination=destination, identity=identity)
        if representation == "scalar":
            gaps = frame.select(
                "h3_index",
                "valid_time",
                "ensemble_member",
                **{
                    f"vector_{height}": (
                        pl.col(f"wind_u_{height}") ** 2 + pl.col(f"wind_v_{height}") ** 2
                    ).sqrt()
                    for height in HEIGHTS
                },
                **{
                    f"scalar_{height}": pl.col(f"wind_scalar_mean_speed_{height}")
                    for height in HEIGHTS
                },
            ).with_columns(
                **{
                    f"gap_{height}": pl.col(f"scalar_{height}") - pl.col(f"vector_{height}")
                    for height in HEIGHTS
                },
                **{
                    f"relative_gap_{height}": pl.when(pl.col(f"vector_{height}") > 0)
                    .then(pl.col(f"scalar_{height}") / pl.col(f"vector_{height}") - 1)
                    .otherwise(None)
                    for height in HEIGHTS
                },
            )
            gaps.write_parquet(arm / f"{date}-gaps.parquet")
        write_nwp(nwp=frame, table_uri=destination)
        readback = pl.read_delta(str(destination))
        _validate_arm(frame=readback, representation=representation)
        _validate_run(frame=readback, h3_grid=h3_grid, date=date)
        Nwp.validate(readback)
        rows, file_bytes, columns = _column_bytes(table=destination)
        result = {
            "metadata": metadata_for_run,
            "table_digest": _digest(destination),
            "gap_digest": _digest(arm / f"{date}-gaps.parquet")
            if representation == "scalar"
            else None,
            "rows": rows,
            "key_hash": frame.select(*KEY).sort(*KEY).hash_rows().sum(),
            "file_bytes": file_bytes,
            "delta_log_bytes": sum(
                p.stat().st_size for p in (destination / "_delta_log").rglob("*") if p.is_file()
            ),
            "column_bytes": columns,
        }
        if rows != expected_rows:
            raise ValueError(f"{date}: writer changed row count")
        _save_json(result_path, result)
        print(f"{representation} {date}: {rows} rows, {file_bytes} file bytes", flush=True)


def _validate_record(
    output: Path, date: str, arm: RepresentationType, item: dict[str, Any], identity: dict[str, str]
) -> None:
    if any(item["metadata"][key] != value for key, value in identity.items()):
        raise ValueError(f"{date}: record differs from requested manifest/grid")
    if item["metadata"]["representation"] != arm:
        raise ValueError(f"{date}: wrong recorded representation")
    if _digest(output / arm / date) != item["table_digest"]:
        raise ValueError(f"{date}: measured table changed")
    if arm == "scalar" and _digest(output / arm / f"{date}-gaps.parquet") != item["gap_digest"]:
        raise ValueError(f"{date}: scalar gaps changed")


def _paired_records(
    output: Path, dates: list[str], identity: dict[str, str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for date in dates:
        paired = [
            json.loads((output / arm / f"{date}.json").read_text()) for arm in REPRESENTATIONS
        ]
        for arm, item in zip(REPRESENTATIONS, paired, strict=True):
            _validate_record(output=output, date=date, arm=arm, item=item, identity=identity)
        if len({(item["rows"], item["key_hash"]) for item in paired}) != 1:
            raise ValueError(f"{date}: arms have different keys or row counts")
        for key in (
            "manifest_sha256",
            "grid_sha256",
            "packages",
            "significand_bits",
            "sort_columns",
            "raw",
            "driver_hash",
            "lock_hash",
        ):
            if len({json.dumps(item["metadata"][key]) for item in paired}) != 1:
                raise ValueError(f"{date}: mismatched {key}")
        non_wind_schemas = [
            {
                name: dtype
                for name, dtype in item["metadata"]["schema"].items()
                if not name.startswith("wind_")
            }
            for item in paired
        ]
        if len({json.dumps(schema, sort_keys=True) for schema in non_wind_schemas}) != 1:
            raise ValueError(f"{date}: mismatched non-wind schema")
        for module in ("delta_store.nwp", "delta_store.precision"):
            if len({item["metadata"]["source_hashes"][module] for item in paired}) != 1:
                raise ValueError(f"{date}: differing production storage source")
        for arm, item in zip(REPRESENTATIONS, paired, strict=True):
            records.append({"date": date, "arm": arm, **item})
    for arm in REPRESENTATIONS:
        metadata = [item["metadata"] for item in records if item["arm"] == arm]
        signatures = [
            {key: value for key, value in item.items() if key != "raw"} for item in metadata
        ]
        if len({json.dumps(item, sort_keys=True) for item in signatures}) != 1:
            raise ValueError(f"{arm}: provenance changes across selected runs")
    return records


def _report(output: Path, dates: list[str], identity: dict[str, str], pilot: bool) -> None:
    records = _paired_records(output=output, dates=dates, identity=identity)
    if any(item["metadata"]["driver_hash"] != _digest(Path(__file__)) for item in records):
        raise ValueError("Reporting driver differs from measurement driver")
    lines = [
        "# Wind storage measurement",
        "",
        "Pilot only; not a seasonal result."
        if pilot
        else "Full-run seasonal sample; not an archive census.",
        "Selected runs: " + ", ".join(dates),
        "Availability resolution: " + (output / "resolved-manifest.json").read_text(),
        "",
    ]
    lines.extend(["| Run | Polar bytes | u/v bytes | u/v + scalar bytes |", "|---|---:|---:|---:|"])
    for index, date in enumerate(dates):
        sizes = [str(records[index * 3 + offset]["file_bytes"]) for offset in range(3)]
        lines.append(f"| {date} | {' | '.join(sizes)} |")
    lines.extend(["", "| Arm | Column | Compressed bytes/row |", "|---|---|---:|"])
    for arm in REPRESENTATIONS:
        items = [item for item in records if item["arm"] == arm]
        total_rows = sum(int(item["rows"]) for item in items)
        column_bytes: defaultdict[str, int] = defaultdict(int)
        for item in items:
            for column, size in item["column_bytes"].items():
                column_bytes[column] += size
        for column, size in sorted(column_bytes.items()):
            lines.append(f"| {arm} | {column} | {size / total_rows:.6f} |")
    lines.extend(
        ["", "| Arm | Total rows | File bytes | Delta-log bytes |", "|---|---:|---:|---:|"]
    )
    for arm in REPRESENTATIONS:
        items = [item for item in records if item["arm"] == arm]
        lines.append(
            f"| {arm} | {sum(item['rows'] for item in items)} | "
            f"{sum(item['file_bytes'] for item in items)} | "
            f"{sum(item['delta_log_bytes'] for item in items)} |"
        )
    gap_frames = pl.concat(
        [
            pl.scan_parquet(output / "scalar" / f"{date}-gaps.parquet").with_columns(
                month=pl.lit(int(date[5:7])), run=pl.lit(date)
            )
            for date in dates
        ]
    ).with_columns(season=((pl.col("month") % 12) // 3).cast(pl.Int8))
    statistics: list[pl.Expr] = []
    for height in HEIGHTS:
        for prefix in ("gap", "relative_gap"):
            column = f"{prefix}_{height}"
            statistics.extend(
                pl.col(column).quantile(q).alias(f"{column}_p{q * 100:g}") for q in (0.5, 0.9, 0.99)
            )
            statistics.extend(
                [
                    pl.col(column).mean().alias(f"{column}_mean"),
                    pl.col(column).max().alias(f"{column}_max"),
                ]
            )
        statistics.extend(
            [
                (pl.col(f"vector_{height}") == 0).sum().alias(f"calm_{height}"),
                pl.col(f"gap_{height}").count().alias(f"valid_pairs_{height}"),
                pl.col(f"gap_{height}").null_count().alias(f"null_pairs_{height}"),
                pl.col(f"relative_gap_{height}").null_count().alias(f"excluded_relative_{height}"),
            ]
        )
    overall = gap_frames.select(*statistics).collect()
    seasonal = gap_frames.group_by("season").agg(*statistics).collect().sort("season")
    cell_season = gap_frames.group_by("season", "h3_index").agg(*statistics).collect()
    cell_season.write_parquet(output / "scalar-gap-by-cell-season.parquet")
    lines.extend(
        [
            "",
            "Pre-rounding scalar-minus-vector gaps (m/s), relative gaps, and calm counts.",
            "",
            overall.write_json(),
            "",
            "Season codes: 0 DJF, 1 MAM, 2 JJA, 3 SON.",
            "",
            seasonal.write_json(),
            "",
            "Provenance for each arm:",
            "",
        ]
    )
    lines.extend(json.dumps(item["metadata"], indent=2) for item in records)
    _save_json(output / "measurements.json", records)
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Fetch raw runs, measure one schema variant, or assemble the paired report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "write", "report"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="polar")
    parser.add_argument(
        "--limit", type=int, help="Pilot only; report requires the same selected dates"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    requested = json.loads(args.manifest.read_text())["dates"]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    identity = _identity(manifest=args.manifest, grid=args.grid)
    _own_output(output=args.output, identity=identity)
    dates = _resolved_dates(output=args.output, requested=requested, identity=identity)
    if args.limit is not None:
        dates = dates[: args.limit]
    if args.action == "fetch":
        _fetch(output=args.output, grid=args.grid, dates=dates, identity=identity)
    elif args.action == "write":
        _write(
            output=args.output,
            grid=args.grid,
            dates=dates,
            identity=identity,
            representation=args.representation,
        )
    else:
        _report(output=args.output, dates=dates, identity=identity, pilot=args.limit is not None)


if __name__ == "__main__":
    main()
