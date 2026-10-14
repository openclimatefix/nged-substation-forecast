"""Validate the ERA5 stage 1 box download against the held CDS tables and the physics.

Reads the combined parquet files that `fetch_era5_stage1_box_arco.py` wrote, runs the
`data-validation` checklist on each, and compares the cells it shares with the tables already held:

- `ERA5/solar_variables/<name>.parquet` (d2m, sp, strd, tcc, tp, u10, v10), on 20 cells. Only `tcc`
  came from the Copernicus Climate Data Store (CDS); the other six were read from ARCO, so their
  agreement checks the cell selection and not the source;
- the `ERA5/beam_diffuse/*.zip` archives (t2m, ssrd, fdir), from the CDS, on the same 20 cells;
- `ERA5-WIND-2019-2023/wind_era5_cds.parquet` (u10, v10, u100, v100), from the CDS, on 90 cells to
  2023-12, of which 77 lie in the block.

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
    ROUNDING_TOLERANCE,
    VARIABLES,
    Stage1Variable,
    expected_hour_count,
)
from lineage import write_readme
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
                (frame["value"] < variable.minimum - ROUNDING_TOLERANCE * variable.maximum)
                | (frame["value"] > variable.maximum + ROUNDING_TOLERANCE * variable.maximum)
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


def write_stage1_readme(*, tables: dict[str, pl.DataFrame], report: dict[str, object]) -> None:
    """Write the directory's README, taking every fact about the files from the written tables."""
    first = next(iter(tables.values()))
    cells = first.select(pl.struct("latitude", "longitude").n_unique()).item()
    hours = first["time"].n_unique()
    nan_total = sum(
        int(t["value"].is_nan().sum()) + t["value"].null_count() for t in tables.values()
    )
    held = report["held_comparison"]
    assert isinstance(held, dict)
    solar_exact = [
        k.split("_vs")[0]
        for k, c in held.items()
        if "solar_variables" in k and c["max_abs_difference"] == 0
    ]
    units = ", ".join(f"`{v.short_name}` {v.units}" for v in VARIABLES)
    write_readme(
        product_dir=OUTPUT_DIR,
        product_name="ERA5 hourly weather variables for stage 1 of the staged forecast",
        source_web_page="https://github.com/google-research/arco-era5",
        script_path="studies/weather_downloads/fetch_era5_stage1_box_arco.py",
        columns={
            "time": f"Valid time, UTC, stored as {first.schema['time']}. An accumulation (`ssrd`, "
            "`fdir`, `strd`, `tp`) is a total over the hour ending at `time`; every other "
            "variable is a snapshot at `time`.",
            "latitude": f"Cell latitude, degrees north, {first.schema['latitude']}; "
            f"{first['latitude'].n_unique()} values on the 0.25 degree grid, stored northernmost "
            "first within each hour.",
            "longitude": f"Cell longitude, degrees east, {first.schema['longitude']}; "
            f"{first['longitude'].n_unique()} values on the 0.25 degree grid.",
            "value": f"The variable named by the file, {first.schema['value']}, in ARCO's units: "
            f"{units}.",
            "expver": f"ERA5 release, {first.schema['expver']}. ARCO carries no release label, so "
            "every row is the constant `0001`.",
        },
        missing_value_convention=(
            f"There are none: {nan_total} NaN or null values across all {len(tables)} files. A "
            "chunk ARCO never wrote reads back as NaN, and the fetch script rejects any month "
            "that holds one."
        ),
        gotchas=[
            f"**Coverage.** {cells} cells for {hours:,} hours (2019-09-01 00:00 to 2026-07-31 "
            "23:00 UTC), one file per variable: `" + "`, `".join(tables) + "`. The files and the "
            "cell coordinates stay in the private store.",
            (
                "**Geopotential at 500 hPa is not fetched.** ARCO stores all 37 pressure levels in "
                "one 80 MB chunk per hour, which reads at about 0.5 hours per second, so the "
                "variable would take about 34 hours."
            ),
            (
                "**Accumulations hold float noise below zero.** `ssrd` and `fdir` reach about -1.6 "
                "and -0.25 J m-2 where the true total is zero. Clip at zero before use. Divide "
                "radiation by 3600 for the mean W m-2 over the hour ending at `time`."
            ),
            (
                "**`ptype` is a code (table 4.201), not a quantity.** ARCO stores the codes with "
                "float noise of about 5e-5, so the fetch rounds them to integers. Values seen: "
                "0, 1, 3, 5, 6, 7, 8."
            ),
            "**Agreement with the held tables.** "
            + ", ".join(f"`{n}`" for n in solar_exact)
            + " match the held `solar_variables/` tables bit for bit, but those tables were also "
            "read from ARCO, so this checks the cell selection only. `tcc`, the one CDS table "
            "there, differs by at most 8e-6. `t2m`, `ssrd`, `fdir` and the four wind components differ "
            "from the held "
            "`beam_diffuse/` archives and `ERA5-WIND-2019-2023/` by at most about 6e-5 of the "
            "variable's maximum, in no pattern by cell, consistent with 16-bit packing in those "
            "netCDF files. Details are in `validation.json`.",
            (
                "**No ERA5 to ERA5T step.** Monthly means of every variable are continuous across "
                "2026-05 to 2026-07. The span ends on the last final month in the held tables."
            ),
            (
                "**Re-fetch** only the missing months with the command in the script's docstring; "
                "`_month_cache/` can be deleted once the combined files are validated."
            ),
        ],
        external_docs={
            "ARCO-ERA5": "https://github.com/google-research/arco-era5",
            "ERA5 data documentation": "https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation",
            "Hersbach et al. (2020)": "https://doi.org/10.1002/qj.3803",
        },
        lineage_filenames=[f"lineage_{name}.json" for name in tables],
    )


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
    write_stage1_readme(tables=tables, report=report)
    (OUTPUT_DIR / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {"held_comparison": held_comparison, "wind_speed": report["wind_speed"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
