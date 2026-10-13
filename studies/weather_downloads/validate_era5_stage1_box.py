"""Validate the ERA5 stage 1 box download against the held CDS tables and the physics.

Reads the combined parquet files that `fetch_era5_stage1_box_arco.py` wrote, runs the
`data-validation` checklist on each, and compares the cells it shares with the tables already held
from the Copernicus Climate Data Store (CDS):

- `ERA5/solar_variables/<name>.parquet` (d2m, sp, strd, tcc, tp, u10, v10), on 20 cells;
- the `ERA5/beam_diffuse/*.zip` archives (t2m, ssrd, fdir), on the same 20 cells;
- `ERA5-WIND-2019-2023/wind_era5_cds.parquet` (u10, v10, u100, v100), on 90 cells to 2023-12.

The report goes to `validation.json` beside the data. No coordinate is printed or written to the
report: cells are named by integer quarter-degree indices only inside this process.

Run command (from `studies/weather_downloads/`):

    uv run --with netCDF4 python validate_era5_stage1_box.py
"""

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xarray as xr
from fetch_era5_stage1_box_arco import (
    OUTPUT_DIR,
    VARIABLES,
    Stage1Variable,
    expected_hour_count,
)
from fetch_era5_stage1_box_arco import (
    ROUNDING_TOLERANCE as ROUNDING_SLACK,
)
from studies.sources import (
    ERA5_PRODUCT_DIR,
    ERA5_SOLAR_VARIABLES_DIR,
    ERA5_WIND_2019_2023_PRODUCT_DIR,
    SCRATCH_DIR,
)

KEYS: Final[list[str]] = ["time", "latitude", "longitude"]
QUARTER_DEGREES: Final[float] = 0.25
HELD_SOLAR_VARIABLES: Final[tuple[str, ...]] = ("d2m", "sp", "strd", "tcc", "tp", "u10", "v10")
HELD_ARCHIVE_VARIABLES: Final[tuple[str, ...]] = ("t2m", "ssrd", "fdir")
HELD_WIND_VARIABLES: Final[tuple[str, ...]] = ("u10", "v10", "u100", "v100")
FLOAT_ROUNDING: Final[float] = 1e-5
"""The relative difference that counts as float rounding when comparing two copies of ERA5.

CDS stores `tcc` as a packed GRIB field, which differs from ARCO by up to 2**-17.
"""
ERA5T_BOUNDARY_MONTH: Final[str] = "2026-07"
"""The last month ARCO and the held CDS tables both label final ERA5 (expver 0001)."""


def held_solar_table(*, name: str) -> pl.LazyFrame:
    """Return a held `solar_variables` table: `time`, `latitude`, `longitude`, `value`, `expver`."""
    return pl.scan_parquet(ERA5_SOLAR_VARIABLES_DIR / f"{name}.parquet")


def held_archive_tables() -> dict[str, pl.DataFrame]:
    """Read the CDS archives holding t2m, ssrd, and fdir into one long frame per variable."""
    archives = sorted((ERA5_PRODUCT_DIR / "beam_diffuse").glob("era5_*.zip"))
    frames: dict[str, list[pl.DataFrame]] = {name: [] for name in HELD_ARCHIVE_VARIABLES}
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH_DIR) as scratch:
        for archive in archives:
            member_dir = Path(scratch) / archive.stem
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(member_dir)
            datasets = [xr.open_dataset(path) for path in sorted(member_dir.glob("*.nc"))]
            merged = xr.merge([ds.drop_vars("expver", errors="ignore") for ds in datasets])
            times = merged.valid_time.values.astype("datetime64[us]")
            time_grid, lat_grid, lon_grid = np.meshgrid(
                times, merged.latitude.values, merged.longitude.values, indexing="ij"
            )
            for name in HELD_ARCHIVE_VARIABLES:
                frames[name].append(
                    pl.DataFrame(
                        {
                            "time": time_grid.ravel(),
                            "latitude": lat_grid.ravel().astype(np.float64),
                            "longitude": lon_grid.ravel().astype(np.float64),
                            "value": merged[name].values.ravel().astype(np.float32),
                        }
                    ).with_columns(pl.col("time").cast(pl.Datetime("us", "UTC")))
                )
            for dataset in datasets:
                dataset.close()
    return {
        name: pl.concat(parts).unique(subset=KEYS, keep="first") for name, parts in frames.items()
    }


def held_wind_table(*, name: str) -> pl.DataFrame:
    """Return a held native-wind variable as `time`, `latitude`, `longitude`, `value`."""
    cells = pl.read_parquet(ERA5_WIND_2019_2023_PRODUCT_DIR / "cells.parquet")
    return (
        pl.scan_parquet(ERA5_WIND_2019_2023_PRODUCT_DIR / "wind_era5_cds.parquet")
        .select("cell_id", "time", value=pl.col(name))
        .collect()
        .join(cells, on="cell_id")
        .select(
            "time",
            latitude=pl.col("lat_q") * QUARTER_DEGREES,
            longitude=pl.col("lon_q") * QUARTER_DEGREES,
            value="value",
        )
    )


def compare_with_held(*, new: pl.DataFrame, held: pl.DataFrame) -> dict[str, object]:
    """Compare the new values with a held copy over the cells and hours both hold.

    Args:
        new: The new table: `time`, `latitude`, `longitude`, `value`.
        held: The held table with the same four columns.

    Returns:
        The number of rows compared, the share of the held rows found in the new table, the
        largest absolute difference, the largest difference relative to the variable's largest
        absolute value, and how many rows differ by more than float rounding.
    """
    joined = new.select(*KEYS, "value").join(
        held.select(*KEYS, "value"), on=KEYS, suffix="_held", how="inner"
    )
    if joined.is_empty():
        return {"rows_compared": 0}
    difference = (joined["value"] - joined["value_held"]).abs()
    scale = float(joined["value_held"].abs().max())  # ty: ignore[invalid-argument-type]
    tolerance = FLOAT_ROUNDING * scale
    return {
        "rows_compared": joined.height,
        "held_rows": held.height,
        "share_of_held_rows_found": joined.height / held.height,
        "max_abs_difference": float(difference.max()),  # ty: ignore[invalid-argument-type]
        "max_abs_difference_relative_to_maximum": float(difference.max()) / scale,  # ty: ignore[invalid-argument-type]
        "rows_beyond_float_rounding": int((difference > tolerance).sum()),
    }


def checklist(*, frame: pl.DataFrame, variable: Stage1Variable) -> dict[str, object]:
    """Run the data-validation checks that need only the new table."""
    cell_count = frame.select(pl.struct("latitude", "longitude").n_unique()).item()
    latitudes = np.sort(frame["latitude"].unique().to_numpy())
    longitudes = np.sort(frame["longitude"].unique().to_numpy())
    grid_is_regular = bool(
        np.allclose(np.diff(latitudes), QUARTER_DEGREES)
        and np.allclose(np.diff(longitudes), QUARTER_DEGREES)
    )
    first_latitude_block = frame.head(cell_count)["latitude"].to_numpy()
    descending_latitude_in_rows = bool(first_latitude_block[0] > first_latitude_block[-1])
    per_cell_hours = frame.group_by("latitude", "longitude").agg(hours=pl.col("time").n_unique())
    monthly = (
        frame.with_columns(month=pl.col("time").dt.truncate("1mo"))
        .group_by("month")
        .agg(mean=pl.col("value").mean(), std=pl.col("value").std())
        .sort("month")
    )
    hourly_profile = (
        frame.group_by(hour=pl.col("time").dt.hour())
        .agg(mean=pl.col("value").mean())
        .sort("hour")["mean"]
        .to_list()
    )
    runs = (
        frame.sort("latitude", "longitude", "time")
        .with_columns(
            run_id=(pl.col("value") != pl.col("value").shift(1))
            .cum_sum()
            .over("latitude", "longitude")
        )
        .group_by("latitude", "longitude", "run_id")
        .agg(run_length=pl.len(), value=pl.col("value").first())
        .sort("run_length", descending=True)
        .head(1)
        .select("run_length", "value")
    )
    boundary = monthly.filter(
        pl.col("month").dt.strftime("%Y-%m").is_in(["2026-05", "2026-06", "2026-07"])
    )
    month_means = monthly["mean"].to_numpy()
    month_to_month = np.abs(np.diff(month_means))
    return {
        "variable": variable.short_name,
        "expected_rows": expected_hour_count() * cell_count,
        "rows": frame.height,
        "cells": cell_count,
        "every_cell_has_every_hour": bool((per_cell_hours["hours"] == expected_hour_count()).all()),
        "duplicate_keys": int(frame.select(pl.struct(KEYS).is_duplicated().sum()).item()),
        "nan": int(frame["value"].is_nan().sum()),
        "null": int(frame["value"].null_count()),
        "minimum": float(frame["value"].min()),  # ty: ignore[invalid-argument-type]
        "maximum": float(frame["value"].max()),  # ty: ignore[invalid-argument-type]
        "outside_limits": int(
            (
                (frame["value"] < variable.minimum - ROUNDING_SLACK * variable.maximum)
                | (frame["value"] > variable.maximum + ROUNDING_SLACK * variable.maximum)
            ).sum()
        ),
        "grid_is_regular_quarter_degree": grid_is_regular,
        "latitude_descends_within_each_hour": descending_latitude_in_rows,
        "hour_of_day_mean_utc": hourly_profile,
        "longest_unchanged_run": runs.to_dicts(),
        "largest_month_to_month_change_of_mean": float(month_to_month.max()),
        "typical_month_to_month_change_of_mean": float(np.median(month_to_month)),
        "means_around_the_final_to_preliminary_boundary": boundary.to_dicts(),
        "boundary_month": ERA5T_BOUNDARY_MONTH,
    }


def wind_speed_check(*, tables: dict[str, pl.DataFrame]) -> dict[str, object]:
    """Compare hypot(u, v) from the new tables with the held speed, per height.

    The held speed series is the 3-by-3 block around each wind site in `wind_native_cds.parquet`,
    which carries no cell coordinates, so the speed is compared on the cells held in
    `wind_era5_cds.parquet` instead: hypot of the held u and v against hypot of the new u and v.
    """
    report: dict[str, object] = {}
    for height in ("10", "100"):
        new_speed = (
            tables[f"u{height}"]
            .join(tables[f"v{height}"], on=KEYS, suffix="_v")
            .select(*KEYS, value=(pl.col("value") ** 2 + pl.col("value_v") ** 2).sqrt())
        )
        held_speed = (
            held_wind_table(name=f"u{height}")
            .join(held_wind_table(name=f"v{height}"), on=KEYS, suffix="_v")
            .select(*KEYS, value=(pl.col("value") ** 2 + pl.col("value_v") ** 2).sqrt())
        )
        report[f"speed_{height}m"] = compare_with_held(new=new_speed, held=held_speed)
    return report


def main() -> None:
    """Validate every variable and write `validation.json`."""
    tables = {
        variable.short_name: pl.read_parquet(OUTPUT_DIR / f"{variable.short_name}.parquet")
        for variable in VARIABLES
    }
    report: dict[str, object] = {
        "checklist": {
            v.short_name: checklist(frame=tables[v.short_name], variable=v) for v in VARIABLES
        }
    }
    held_comparison: dict[str, object] = {}
    for name in HELD_SOLAR_VARIABLES:
        held = held_solar_table(name=name).collect()
        held_comparison[f"{name}_vs_solar_variables"] = compare_with_held(
            new=tables[name], held=held
        )
    for name, held in held_archive_tables().items():
        held_comparison[f"{name}_vs_beam_diffuse_archives"] = compare_with_held(
            new=tables[name], held=held
        )
    for name in HELD_WIND_VARIABLES:
        held_comparison[f"{name}_vs_wind_2019_2023"] = compare_with_held(
            new=tables[name], held=held_wind_table(name=name)
        )
    report["held_comparison"] = held_comparison
    report["wind_speed"] = wind_speed_check(tables=tables)
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {"held_comparison": held_comparison, "wind_speed": report["wind_speed"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
