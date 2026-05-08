"""H3-related utilities for geospatial operations."""

import importlib.resources
import logging

import h3.api.basic_int as h3
import patito as pt
import polars as pl
import polars_h3 as plh3
import shapely
from contracts.geo_schemas import H3GridWeights
from shapely.geometry.base import BaseGeometry

_LOG = logging.getLogger(__name__)


def h3_grid_weights_for_boundary(
    boundary: BaseGeometry, h3_res: int, grid_size: float, child_h3_res: int | None = None
) -> pt.DataFrame[H3GridWeights]:
    """Computes the H3 grid weights for a geospatial boundary.

    Attributes:
        boundary: The geospatial boundary to find H3 cells for.
        h3_res: The H3 resolution to use for the grid.
        grid_size: The size of the regular lat/lon grid in degrees.
        child_res: The H3 resolution to use for the underlying points. If None,
            it defaults to h3_res + 2.

    Generates the spatial mapping between the hexagonal H3 grid and the regular lat/lon NWP grid.

    The mapping is calculated by sampling each H3 cell with finer-resolution child cells and
    determining which regular grid cell each child falls into. The `grid_size` parameter is used to
    snap high-resolution H3 cells to the nearest regular NWP grid points.
    """

    _LOG.info(f"Generating H3 cells at resolution {h3_res}...")

    # Note: In h3-py v4+, `h3.geo_to_cells` is the generic entry point for any object implementing
    # `__geo_interface__` (like shapely.Polygon). `h3.polygon_to_cells` is an alias for
    # `h3shape_to_cells` and expects an internal H3Shape object, which would fail here.
    h3_index = h3.geo_to_cells(uk_boundary, res=h3_res)
    if not h3_index:
        raise ValueError(
            f"No H3 cells found for the given boundary at resolution {h3_res}."
            " Check if the boundary geometry is valid and covers the expected area."
        )

    df_with_counts = compute_h3_grid_weights(
        h3_index=h3_index, grid_size=grid_size, child_h3_res=child_h3_res
    )

    return H3GridWeights.validate(df_with_counts)


def compute_h3_grid_weights(
    h3_index: list[int], grid_size: float, child_h3_res: int | None = None
) -> pt.DataFrame[H3GridWeights]:
    """Computes the proportion mapping for H3 grid cells to a regular lat/lng grid.

    This function takes a DataFrame containing H3 indices and calculates how many
    child H3 cells at a finer resolution (`child_res`) fall into each cell of a
    regular lat/lng grid of size `grid_size`.

    The regular grid is assumed to be perfectly aligned to 0.0 (e.g., 0.0, `grid_size`,
    `grid_size * 2`).

    Args:
        h3_index: List of 64-bit integers.
        grid_size: The size of the regular lat/lng grid in degrees (e.g., 0.25).
        child_h3_res: The H3 resolution to use for the underlying points. Must be
            strictly greater than the resolution of the input 'h3_index' column.

    Returns:
        A Polars DataFrame with columns:
            - h3_index: The original H3 index.
            - nwp_lat: The latitude of the regular grid cell.
            - nwp_lng: The longitude of the regular grid cell.
            - len: The number of child H3 cells in this NWP grid cell.
            - total: The total number of child H3 cells for this h3_index hexagon.
            - proportion: The proportion of the H3 cell that falls into this grid cell.
    """
    _LOG.info(
        f"Computing H3 grid weights for grid size {grid_size} with child_res {child_h3_res}"
        f" for {len(h3_index)} H3 indicies..."
    )

    if len(h3_index):
        raise ValueError("h3_index is empty.")

    # Ensure grid_size is strictly positive to avoid division by zero or nonsensical snapping.
    if grid_size <= 0:
        raise ValueError(f"grid_size must be strictly positive, not {grid_size}.")

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

    half_grid_size = grid_size / 2

    weights_df = (
        df.with_columns(child_h3=plh3.cell_to_children("h3_index", child_h3_res))
        .explode("child_h3")
        .with_columns(
            # GRID SNAPPING FORMULA: The half-grid offset binning `((lat + grid_size/2) /
            # grid_size).floor() * grid_size` ensures that points are snapped to the *closest* grid
            # center rather than the bottom-left corner of the grid cell. Adding `grid_size/2`
            # before flooring shifts the bin boundaries so that the grid points (0, 0.25, 0.5, etc.)
            # are at the center of each bin.
            nwp_lat=((plh3.cell_to_lat("child_h3") + half_grid_size) / grid_size).floor()
            * grid_size,
            nwp_lon=((plh3.cell_to_lng("child_h3") + half_grid_size) / grid_size).floor()
            * grid_size,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lon"])
        .len(name="n_h3_children_in_nwp_grid_box")
        .with_columns(
            total_children_per_h3_hexagon=pl.col("n_h3_children_in_nwp_grid_box")
            .sum()
            .over("h3_index"),
        )
        .with_columns(
            proportion=pl.col("n_h3_children_in_nwp_grid_box")
            / pl.col("total_children_per_h3_hexagon"),
        )
    )

    return pt.DataFrame(weights_df).set_model(H3GridWeights).cast().drop().validate()


def uk_boundary() -> BaseGeometry:
    """Loads the UK boundary geometry from a local GeoJSON file.

    The boundary is buffered to ensure that coastal substations and nearby islands
    are included in the resulting H3 grid without spatial distortion.
    """
    geojson_path = importlib.resources.files("geo").joinpath(
        "../../assets/england_scotland_wales.geojson"
    )

    _LOG.info(f"Loading UK boundary from {geojson_path}")
    file_contents = geojson_path.read_text()
    shape: BaseGeometry = shapely.from_geojson(file_contents)

    # Buffer by 25,000 meters (25km) to ensure that even the most coastal H3 cells
    # will have at least one overlapping NWP grid cell (given the 0.25-degree
    # resolution, which is ~28km at UK latitudes). This takes about 30 seconds.
    return shape.buffer(distance=0.25)
