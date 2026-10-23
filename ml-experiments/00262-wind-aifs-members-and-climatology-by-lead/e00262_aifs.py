"""Build the AIFS ensemble's member rows at the three farms, on the scored cases' half-hours.

Reads the AIFS ensemble store of Dynamical.org (ECMWF AIFS ENS, CC BY 4.0), takes each farm's
area-weighted mean over the 0.25 degree cells under the farm's H3 resolution-5 cell on the east and
north wind components (as `dynamical_data`'s H3 aggregation does for the ECMWF ensemble), and
interpolates the 6-hourly steps linearly to half-hours before turning the components into speed and
from-direction sine and cosine. Two run choices are written, each for every Tuesday run of the
scored cases: the AIFS run of 00 UTC on the run's date (`same_day`, read as available at 09:00 UTC)
and the run of 18 UTC the day before (`previous_run`, the conservative choice).

Output, in `work/`: `aifs_<choice>.parquet`, one row per (farm, run, member, half-hour) with the six
`XGBOOST_FEATURES` and `temperature_c`, and `aifs_checks.json`. Farm coordinates and the cell
weights never leave the process.

    uv run --no-sync python ml-experiments/00262-*/e00262_aifs.py
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00262_common as common
import h3.api.basic_int as h3
import numpy as np
import polars as pl
from geo.h3 import compute_h3_grid_weights
from studies.sources import STUDIES_DATA_DIR

AIFS_DIR: Final[Path] = STUDIES_DATA_DIR / "downloads" / "NWP" / "ECMWF-AIFS-ENS"
H3_RESOLUTION: Final[int] = 5
GRID_DEGREES: Final[float] = 0.25
WEIGHT_TOLERANCE: Final[float] = 1e-6
STEP_HOURS: Final[int] = 6
MAX_HOURS: Final[float] = 360.0
"""The last AIFS step; later half-hours of the previous run have no AIFS forecast."""
CHOICES: Final[dict[str, timedelta]] = {
    "same_day": timedelta(hours=0),
    "previous_run": timedelta(hours=-6),
}
"""Offset from the ENS run's 00 UTC to the AIFS run's start."""
VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)


def cell_weights(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return `time_series_id`, `lat_index`, `lon_index`, and `weight` for every farm."""
    cells = {
        int(row["time_series_id"]): h3.latlng_to_cell(
            row["latitude"], row["longitude"], H3_RESOLUTION
        )
        for row in sites.iter_rows(named=True)
    }
    grid = pl.read_parquet(AIFS_DIR / "_grid_cells.parquet")
    h3_weights = compute_h3_grid_weights(
        nwp_grid_size_degrees=GRID_DEGREES, h3_index=sorted(set(cells.values()))
    )
    by_cell = h3_weights.select(
        "h3_index",
        "proportion",
        nwp_lat=pl.col("nwp_lat").cast(pl.Float64).round(4),
        nwp_lon=pl.col("nwp_lon").cast(pl.Float64).round(4),
    ).join(
        grid.select(
            "lat_index",
            "lon_index",
            nwp_lat=pl.col("latitude").cast(pl.Float64).round(4),
            nwp_lon=pl.col("longitude").cast(pl.Float64).round(4),
        ),
        on=["nwp_lat", "nwp_lon"],
    )
    farms = pl.DataFrame(
        {"time_series_id": list(cells), "h3_index": list(cells.values())},
        schema={"time_series_id": pl.Int32, "h3_index": pl.UInt64},
    )
    weights = farms.join(by_cell, on="h3_index").select(
        "time_series_id", "lat_index", "lon_index", weight=pl.col("proportion").cast(pl.Float64)
    )
    sums = weights.group_by("time_series_id").agg(total=pl.col("weight").sum())
    if sums.filter((pl.col("total") - 1.0).abs() > WEIGHT_TOLERANCE).height or sums.height != len(
        cells
    ):
        msg = "a farm's H3 weights over the AIFS crop do not sum to 1"
        raise ValueError(msg)
    return weights


def read_runs(*, weights: pl.DataFrame, inits: list[datetime]) -> pl.DataFrame:
    """Return the weighted-mean values per farm, run, member, and step for the given runs."""
    wanted = [i.replace(tzinfo=None) for i in inits]
    return (
        pl.scan_parquet(AIFS_DIR / "ECMWF-AIFS-ENS.parquet")
        .filter(pl.col("init_time").is_in(wanted))
        .join(weights.lazy(), on=["lat_index", "lon_index"])
        .group_by("time_series_id", "init_time", "ensemble_member", "lead_time")
        .agg(
            *(
                ((pl.col(c) * pl.col("weight")).sum() / pl.col("weight").sum()).alias(c)
                for c in VALUE_COLUMNS
            ),
            n_cells=pl.len(),
        )
        .collect()
    )


def interpolate(
    *, steps: pl.DataFrame, init: datetime, series: int, targets: np.ndarray
) -> pl.DataFrame | None:
    """Return one farm's members on `targets` (hours from `init`), or None if the run is absent.

    Args:
        steps: `read_runs` rows.
        init: The AIFS run's start.
        series: The farm's `time_series_id`.
        targets: Hours after `init` at which to read, ascending.

    Returns:
        One row per member and target with the six features and `temperature_c`.
    """
    rows = steps.filter(
        (pl.col("time_series_id") == series) & (pl.col("init_time") == init.replace(tzinfo=None))
    )
    if rows.is_empty():
        return None
    rows = rows.with_columns(hours=pl.col("lead_time").dt.total_hours()).sort(
        "ensemble_member", "hours"
    )
    members = rows["ensemble_member"].unique(maintain_order=True).to_list()
    hours = rows.filter(pl.col("ensemble_member") == members[0])["hours"].to_numpy()
    if (np.diff(hours) != STEP_HOURS).any() or targets.max() > hours.max() or targets.min() < 0:
        msg = f"farm {series}: the AIFS steps of the run of {init} do not cover the targets"
        raise ValueError(msg)
    pieces = []
    for member in members:
        part = rows.filter(pl.col("ensemble_member") == member)
        values = {
            c: np.interp(targets, hours, part[c].to_numpy().astype(np.float64))
            for c in VALUE_COLUMNS
        }
        u10, v10, u100, v100 = (values[c] for c in VALUE_COLUMNS[1:])
        direction_10 = np.arctan2(-u10, -v10)
        direction_100 = np.arctan2(-u100, -v100)
        pieces.append(
            pl.DataFrame(
                {
                    "ensemble_member": np.full(targets.size, member, dtype=np.int8),
                    "wind_speed_10m": np.hypot(u10, v10),
                    "wind_speed_100m": np.hypot(u100, v100),
                    "wind_dir_sin_10m": np.sin(direction_10),
                    "wind_dir_cos_10m": np.cos(direction_10),
                    "wind_dir_sin_100m": np.sin(direction_100),
                    "wind_dir_cos_100m": np.cos(direction_100),
                    "temperature_c": values["temperature_2m"],
                }
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Write both run choices' member tables and a short check file."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    weights = cell_weights(sites=sites)
    cases = common.cases_table(sites=sites).select(*common.KEY_COLUMNS)
    runs = sorted(cases["power_fcst_init_time"].unique().to_list())
    checks: dict = {"runs_scored": len(runs)}
    for choice, offset in CHOICES.items():
        inits = [datetime(r.year, r.month, r.day, tzinfo=UTC) + offset for r in runs]
        steps = read_runs(weights=weights, inits=inits)
        short = steps.filter(pl.col("n_cells") != steps["n_cells"].max()).height
        frames, missing = [], []
        for run, init in zip(runs, inits, strict=True):
            run_cases = cases.filter(pl.col("power_fcst_init_time") == run)
            for series in sorted(run_cases["time_series_id"].unique().to_list()):
                farm_cases = run_cases.filter(pl.col("time_series_id") == series).sort("valid_time")
                targets = (
                    farm_cases["valid_time"].to_numpy()
                    - np.datetime64(init.replace(tzinfo=None), "us")
                ) / np.timedelta64(1, "h")
                covered = targets <= MAX_HOURS
                farm_cases, targets = farm_cases.filter(pl.Series(covered)), targets[covered]
                members = interpolate(steps=steps, init=init, series=series, targets=targets)
                if members is None:
                    missing.append(run.isoformat())
                    break
                n_members = members["ensemble_member"].n_unique()
                if n_members != common.MEMBERS:
                    msg = f"{init}: {n_members} members"
                    raise ValueError(msg)
                keys = pl.concat([farm_cases] * common.MEMBERS)
                frames.append(keys.hstack(members))
        frame = (
            pl.concat(frames)
            .sort(*common.KEY_COLUMNS, "ensemble_member")
            .with_columns(pl.col(common.WEATHER_COLUMNS).cast(pl.Float32))
        )
        frame.write_parquet(common.WORK_DIR / f"aifs_{choice}.parquet")
        checks[choice] = {
            "rows": frame.height,
            "runs_missing": sorted(set(missing)),
            "cells_short_groups": short,
            "wind_speed_100m_mean_by_month": {
                row["month"]: row["mean"]
                for row in frame.group_by(pl.col("valid_time").dt.strftime("%Y-%m").alias("month"))
                .agg(mean=pl.col("wind_speed_100m").mean())
                .sort("month")
                .iter_rows(named=True)
            },
        }
        print(
            choice,
            checks[choice]["rows"],
            "missing runs",
            checks[choice]["runs_missing"],
            flush=True,
        )
    (common.WORK_DIR / "aifs_checks.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
