"""H3-related utilities for geospatial operations."""

import patito as pt
import polars as pl
import polars_h3 as plh3
from contracts.data_schemas import H3GridWeights


def compute_h3_grid_weights(
    df: pl.DataFrame, grid_size: float, child_res: int = 7
) -> pt.DataFrame["H3GridWeights"]:
    """Computes the proportion mapping for H3 grid cells to a regular lat/lng grid.

    This function takes a DataFrame containing H3 indices and calculates how many
    child H3 cells at a finer resolution (`child_res`) fall into each cell of a
    regular lat/lng grid of size `grid_size`.

    The regular grid is assumed to be perfectly aligned to 0.0 (e.g., 0.0, `grid_size`,
    `grid_size * 2`).

    Args:
        df: A Polars DataFrame containing an 'h3_index' column (UInt64).
        grid_size: The size of the regular lat/lng grid in degrees (e.g., 0.25).
        child_res: The H3 resolution to use for the underlying points. Must be
            strictly greater than the resolution of the input 'h3_index' column.
            Defaults to 7. The `+2` heuristic (e.g., res 5 -> res 7) provides ~49
            sample points per H3 cell (7^2), which is a sufficient balance between
            spatial precision (for area-weighting against a 0.25-degree grid) and
            computation time/memory overhead. Increasing it further could cause
            an exponential explosion in the number of child cells and potentially
            trigger OOM errors.

    Returns:
        A Polars DataFrame with columns:
            - h3_index: The original H3 index.
            - nwp_lat: The latitude of the regular grid cell.
            - nwp_lng: The longitude of the regular grid cell.
            - len: The number of child H3 cells in this grid cell.
            - total: The total number of child H3 cells for this h3_index.
            - proportion: The proportion of the H3 cell that falls into this grid cell.
    """
    if df.is_empty():
        raise ValueError("Input DataFrame is empty.")

    # Ensure grid_size is strictly positive to avoid division by zero
    # or nonsensical snapping.
    if grid_size <= 0:
        raise ValueError("grid_size must be strictly positive")

    # Check resolution of all H3 indices to ensure consistency.
    # Using polars_h3.get_resolution for vectorized check.
    h3_res_unique = df.select(plh3.get_resolution("h3_index")).unique()
    if h3_res_unique.height > 1:
        raise ValueError("All H3 indices must have the same resolution.")
    h3_res = h3_res_unique.item()

    if child_res <= h3_res:
        raise ValueError(f"child_res ({child_res}) must be strictly greater than h3_res ({h3_res})")

    weights_df = (
        df.with_columns(child_h3=plh3.cell_to_children("h3_index", child_res))
        .explode("child_h3")
        .with_columns(
            # GRID SNAPPING FORMULA:
            # The half-grid offset binning `((lat + grid_size/2) / grid_size).floor() * grid_size`
            # ensures that points are snapped to the *closest* grid center rather than
            # the bottom-left corner of the grid cell. Adding `grid_size/2` before
            # flooring shifts the bin boundaries so that the grid points (0, 0.25, 0.5, etc.)
            # are at the center of each bin.
            nwp_lat=((plh3.cell_to_lat("child_h3") + (grid_size / 2)) / grid_size).floor()
            * grid_size,
            nwp_lng=((plh3.cell_to_lng("child_h3") + (grid_size / 2)) / grid_size).floor()
            * grid_size,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lng"])
        .len()
        .with_columns(
            total=pl.col("len").sum().over("h3_index"),
        )
        .with_columns(
            # Ensure len and total are UInt32 as per contract
            len=pl.col("len").cast(pl.UInt32),
            total=pl.col("total").cast(pl.UInt32),
            proportion=pl.col("len") / pl.col("total"),
        )
    )

    return H3GridWeights.validate(weights_df, drop_superfluous_columns=True)
