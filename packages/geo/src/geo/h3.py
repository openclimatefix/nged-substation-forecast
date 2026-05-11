"""H3-related utilities for geospatial operations."""

import logging

import h3.api.basic_int as h3
import patito as pt
import polars as pl
import polars_h3 as plh3
from contracts.geo_schemas import H3GridWeights
from shapely.geometry.base import BaseGeometry

_LOG = logging.getLogger(__name__)


def compute_h3_grid_weights_for_boundary(
    boundary: BaseGeometry,
    nwp_grid_size_degrees: float,
    h3_res: int,
    child_h3_res: int | None = None,
) -> pt.DataFrame[H3GridWeights]:
    """Computes the H3 grid weights for a geospatial boundary.

    Attributes:
        boundary: The geospatial boundary to find H3 cells for.
        nwp_grid_size_degrees: The size of the regular lat/lon grid in degrees.
        h3_res: The H3 resolution to use for the grid.
        child_h3_res: The H3 resolution to use for the underlying points. If None,
            it defaults to h3_res + 2.

    Generates the spatial mapping between the hexagonal H3 grid and the regular lat/lon NWP grid.

    The mapping is calculated by sampling each H3 cell with finer-resolution child cells and
    determining which regular grid cell each child falls into. The `grid_size` parameter is used to
    snap high-resolution H3 cells to the nearest regular NWP grid points.
    """

    _LOG.info(f"Generating H3 cells at resolution {h3_res}...")

    h3_index = h3.geo_to_cells(geo=boundary, res=h3_res)
    if not h3_index:
        raise ValueError(
            f"No H3 cells found for the given boundary at resolution {h3_res}."
            " Check if the boundary geometry is valid and covers the expected area."
        )

    return compute_h3_grid_weights(
        nwp_grid_size_degrees=nwp_grid_size_degrees, h3_index=h3_index, child_h3_res=child_h3_res
    )


def compute_h3_grid_weights(
    nwp_grid_size_degrees: float, h3_index: list[int], child_h3_res: int | None = None
) -> pt.DataFrame[H3GridWeights]:
    """Computes the proportion mapping for H3 grid cells to a regular lat/lng grid.

    This function takes a DataFrame containing H3 indices and calculates how many
    child H3 cells at a finer resolution (`child_res`) fall into each cell of a
    regular lat/lng grid of size `grid_size`.

    The regular grid is assumed to be perfectly aligned to 0.0 (e.g., 0.0, `grid_size`,
    `grid_size * 2`).

    Args:
        h3_index: List of 64-bit H3 discrete spatial indices.
        nwp_grid_size_degrees: The size of the regular NWP lat/lng grid in degrees (e.g., 0.25).
        child_h3_res: The H3 resolution to use for the underlying points. Must be
            strictly greater than the resolution of the input 'h3_index' column.
    """
    _LOG.info(
        f"Computing H3 grid weights for grid size {nwp_grid_size_degrees} with child_res {child_h3_res}"
        f" for {len(h3_index)} H3 indicies..."
    )

    if len(h3_index) == 0:
        raise ValueError("h3_index is empty.")

    # Ensure grid_size is strictly positive to avoid division by zero or nonsensical snapping.
    if nwp_grid_size_degrees <= 0:
        raise ValueError(
            f"nwp_grid_size_degrees must be strictly positive, not {nwp_grid_size_degrees}."
        )

    df = pl.DataFrame({"h3_index": h3_index}, schema={"h3_index": pl.UInt64}).sort("h3_index")

    # Check resolution of all H3 indices to ensure consistency.
    h3_res_unique = df.select(plh3.get_resolution("h3_index")).unique()
    if h3_res_unique.height > 1:
        raise ValueError(f"All H3 indices must have the same resolution. {h3_res_unique=}")
    h3_res = h3_res_unique.item()

    # The `+2` heuristic provides ~49 sample points per H3 cell (7^2), which is a sufficient balance
    # between spatial precision (for area-weighting against a 0.25-degree grid) and computation
    # time/memory overhead. Increasing it further could cause an exponential explosion in the number
    # of child cells and potentially trigger OOM errors.
    child_h3_res = h3_res + 2 if child_h3_res is None else child_h3_res

    if child_h3_res <= h3_res:
        raise ValueError(f"{child_h3_res=} must be strictly greater than {h3_res=}.")

    half_grid_size = nwp_grid_size_degrees / 2

    def _snap_to_grid(lat_or_lon: pl.Expr) -> pl.Expr:
        """Grid snapping formula: The half-grid offset binning `((lat + grid_size/2) /
        grid_size).floor() * grid_size` ensures that points are snapped to the *closest* grid
        center rather than the bottom-left corner of the grid cell. Adding `grid_size/2`
        before flooring shifts the bin boundaries so that the grid points (0, 0.25, 0.5, etc.)
        are at the center of each bin."""
        return (
            (lat_or_lon + half_grid_size) / nwp_grid_size_degrees
        ).floor() * nwp_grid_size_degrees

    weights_df = (
        df.with_columns(child_h3=plh3.cell_to_children("h3_index", child_h3_res))
        .explode("child_h3")
        .with_columns(
            nwp_lat=_snap_to_grid(plh3.cell_to_lat("child_h3")),
            nwp_lon=_snap_to_grid(plh3.cell_to_lng("child_h3")),
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lon"])
        .len(name="h3_children_in_nwp_grid_box")
        .with_columns(
            h3_children_per_h3_parent=pl.col("h3_children_in_nwp_grid_box").sum().over("h3_index"),
        )
        .with_columns(
            proportion=pl.col("h3_children_in_nwp_grid_box") / pl.col("h3_children_per_h3_parent")
        )
    )

    return pt.DataFrame(weights_df).set_model(H3GridWeights).drop().cast().validate()
