"""H3-related utilities for geospatial operations."""

import polars as pl
import polars_h3 as plh3
from contracts.data_schemas import H3GridWeights


def compute_h3_grid_weights(df: pl.DataFrame, grid_size: float, child_res: int = 7) -> pl.DataFrame:
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
            finer than or equal to the resolution of the input 'h3_index' column.
            Defaults to 7.

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
        return pl.DataFrame(
            schema={
                "h3_index": pl.UInt64,
                "nwp_lat": pl.Float64,
                "nwp_lng": pl.Float64,
                "len": pl.UInt32,
                "total": pl.UInt32,
                "proportion": pl.Float64,
            }
        )

    # We use child_res to sample the H3 cell. child_res must be finer than the
    # resolution of the input h3_index.
    # Note: We don't explicitly check the resolution of the input h3_index here
    # for performance reasons, but downstream users should ensure child_res is
    # appropriate.

    weights_df = (
        df.with_columns(child_h3=plh3.cell_to_children("h3_index", child_res))
        .explode("child_h3")
        .with_columns(
            nwp_lat=((plh3.cell_to_lat("child_h3") + (grid_size / 2)) / grid_size).floor()
            * grid_size,
            nwp_lng=((plh3.cell_to_lng("child_h3") + (grid_size / 2)) / grid_size).floor()
            * grid_size,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lng"])
        .len()
        .with_columns(
            total=pl.col("len").sum().over("h3_index"),
            # Ensure len and total are UInt32 as per contract
            len=pl.col("len").cast(pl.UInt32),
        )
        .with_columns(
            total=pl.col("total").cast(pl.UInt32),
            proportion=pl.col("len") / pl.col("total"),
        )
    )

    return H3GridWeights.validate(weights_df, drop_superfluous_columns=True)
