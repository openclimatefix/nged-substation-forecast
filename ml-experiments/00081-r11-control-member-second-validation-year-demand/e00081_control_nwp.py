"""Read one control-member backfill file as rows in the archive's schema, for chosen H3 cells.

The archive that experiments 48 and 49 read holds one row per (H3 cell, member, valid time), with
ECMWF's fields in the units of the `Nwp` contract: temperatures in degrees C, accumulated fields as
average rates per second, and each cell the area-weighted mean of the 0.25 degree grid points that
overlap it (`h3_grid_weights.parquet`). The backfill files hold the raw grid in ECMWF's units. This
module applies the same conversions, so that `_upsample_nwp_to_half_hourly` and the frame builder
of experiment 48 can read the output unchanged. The backfill has no precipitation type, which no
experiment's final feature list uses.

    uv run python ml-experiments/00081-*/e00081_control_nwp.py --smoke
"""

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.deaccumulation import (
    PRECIPITATION_INVALID_BELOW_MM_PER_S,
    RADIATION_INVALID_BELOW_W_PER_M2,
    deaccumulate_to_rates,
)
from studies.sources import REPO_DATA_DIR

BACKFILL_DIR: Final[Path] = Path("/mnt/wd_18tb/data/nwp-archive/ECMWF-ENS-BACKFILL-CONTROL/control")
GRID_WEIGHTS_PATH: Final[Path] = REPO_DATA_DIR / "h3_grid_weights.parquet"
KELVIN: Final[float] = 273.15
INSTANTANEOUS: Final[dict[str, str]] = {
    "2t": "temperature_2m",
    "2d": "dew_point_temperature_2m",
    "10u": "wind_u_10m",
    "10v": "wind_v_10m",
    "100u": "wind_u_100m",
    "100v": "wind_v_100m",
    "sp": "pressure_surface",
    "msl": "pressure_reduced_to_mean_sea_level",
}
"""Backfill variable names and the `Nwp` column each fills; the two temperatures need K to C."""
ACCUMULATED: Final[dict[str, tuple[str, float]]] = {
    "tp": ("precipitation_surface", 1000.0),
    "strd": ("downward_long_wave_radiation_flux_surface", 1.0),
    "ssrd": ("downward_short_wave_radiation_flux_surface", 1.0),
}
"""Backfill variable names, the `Nwp` column each fills, and the factor to the contract's unit
(`tp` arrives in metres and the contract holds kilograms per square metre per second)."""
_INVALID_BELOW: Final[dict[str, float]] = {
    "tp": PRECIPITATION_INVALID_BELOW_MM_PER_S,
    "strd": RADIATION_INVALID_BELOW_W_PER_M2,
    "ssrd": RADIATION_INVALID_BELOW_W_PER_M2,
}


def cell_weights(*, cells: list[int]) -> pl.DataFrame:
    """Return the grid-point weights of the chosen H3 cells, with the grid indices appended."""
    return pl.read_parquet(GRID_WEIGHTS_PATH).filter(pl.col("h3_index").is_in(cells))


def control_rows(*, run: date, cells: list[int]) -> pl.DataFrame:
    """Return one control-member run as archive-schema rows for the chosen cells.

    Args:
        run: The 00:00 UTC run date.
        cells: The H3 cells (resolution 5) to keep.

    Returns:
        One row per cell and step with `h3_index`, `ensemble_member` (0), `valid_time`, and the
        `Nwp` weather columns it can fill. Step 0 carries null accumulated rates, as the archive
        does.
    """
    with np.load(BACKFILL_DIR / f"{run:%Y-%m-%d}.npz") as archive:
        values = archive["values"][:, 0]
        names = [str(name) for name in archive["variables"]]
        steps = archive["steps_hours"].astype(np.int64)
        latitudes = archive["latitudes"]
        longitudes = np.mod(archive["longitudes_degrees_east"], 360.0)
    weights = cell_weights(cells=cells)
    row_index = np.abs(latitudes[None, :] - weights["nwp_lat"].to_numpy()[:, None]).argmin(axis=1)
    column_index = np.abs(
        longitudes[None, :] - np.mod(weights["nwp_lon"].to_numpy(), 360.0)[:, None]
    ).argmin(axis=1)
    proportion = weights["proportion"].to_numpy().astype(np.float64)
    cell_of_point = weights["h3_index"].to_numpy()
    unique_cells = np.unique(cell_of_point)

    def to_cells(*, field: np.ndarray) -> np.ndarray:
        """Area-weighted mean of a (step, row, column) field over each cell's grid points."""
        points = field[:, row_index, column_index].astype(np.float64)
        out = np.empty((field.shape[0], unique_cells.size))
        for position, cell in enumerate(unique_cells):
            mask = cell_of_point == cell
            out[:, position] = (points[:, mask] * proportion[mask]).sum(axis=1) / proportion[
                mask
            ].sum()
        return out

    elapsed = steps * 3600.0
    columns: dict[str, np.ndarray] = {}
    for variable, column in INSTANTANEOUS.items():
        field = to_cells(field=values[names.index(variable)])
        columns[column] = field - KELVIN if variable in ("2t", "2d") else field
    for variable, (column, factor) in ACCUMULATED.items():
        field = to_cells(field=values[names.index(variable)]) * factor
        columns[column] = deaccumulate_to_rates(
            accumulations=field,
            elapsed_seconds=elapsed,
            invalid_below_rate=_INVALID_BELOW[variable],
        ).rates
    init = datetime(run.year, run.month, run.day, tzinfo=UTC)
    valid = [init + timedelta(hours=int(hours)) for hours in steps]
    frame = pl.DataFrame(
        {
            "h3_index": np.tile(unique_cells, steps.size).astype(np.uint64),
            "ensemble_member": np.zeros(steps.size * unique_cells.size, dtype=np.int64),
            "valid_time": np.repeat(np.array(valid, dtype="datetime64[us]"), unique_cells.size),
            **{column: field.reshape(-1) for column, field in columns.items()},
        }
    )
    return frame.with_columns(
        pl.col("valid_time").dt.replace_time_zone("UTC"),
        pl.col("h3_index").cast(pl.UInt64),
        pl.col("ensemble_member").cast(pl.Int8),
    )


def main() -> None:
    """With `--smoke`, convert one run for 3 cells and print ranges against physical limits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.parse_args()
    cells = pl.read_parquet(GRID_WEIGHTS_PATH)["h3_index"].unique().sort().to_list()[1000:1003]
    frame = control_rows(run=date(2023, 7, 1), cells=[int(cell) for cell in cells])
    print(frame.shape)
    print(frame.describe())


if __name__ == "__main__":
    main()
