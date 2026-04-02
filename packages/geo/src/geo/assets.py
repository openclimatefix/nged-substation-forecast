import importlib.resources

import h3.api.basic_int as h3
import patito as pt
import polars as pl
import pyproj
import shapely
from dagster import AssetExecutionContext, Config, asset
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from contracts.data_schemas import H3GridWeights
from geo.h3 import compute_h3_grid_weights


class H3GridConfig(Config):
    """Configuration for the H3 grid weights computation.

    Attributes:
        h3_res: The H3 resolution to use for the grid (default 5).
        grid_size: The size of the regular lat/lng grid in degrees (default 0.25).
        child_res: The H3 resolution to use for the underlying points. If None,
            it defaults to h3_res + 2.
    """

    h3_res: int = 5
    grid_size: float = 0.25
    child_res: int | None = None


@asset(group_name="reference_data")
def uk_boundary(context: AssetExecutionContext) -> BaseGeometry:
    """Loads the UK boundary geometry from a local GeoJSON file.

    The boundary is projected to EPSG:27700 (British National Grid) and buffered
    by 25,000 meters to ensure that coastal substations and nearby islands are
    included in the resulting H3 grid without spatial distortion.
    """
    geojson_path = importlib.resources.files("geo").joinpath(
        "assets/england_scotland_wales.geojson"
    )

    context.log.info(f"Loading UK boundary from {geojson_path}")
    file_contents = geojson_path.read_text()

    shape: BaseGeometry = shapely.from_geojson(file_contents)

    # Project to OSGB36 (EPSG:27700) for metric buffering
    project_to_osgb = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:27700", always_xy=True
    ).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(
        "EPSG:27700", "EPSG:4326", always_xy=True
    ).transform

    shape_osgb = transform(project_to_osgb, shape)
    # Buffer by 25000 meters (25km) to catch islands/coasts without spatial distortion
    shape_osgb_buffered = shape_osgb.buffer(25000)
    shape_buffered = transform(project_to_wgs84, shape_osgb_buffered)

    return shape_buffered


@asset(group_name="reference_data")
def gb_h3_grid_weights(
    context: AssetExecutionContext, config: H3GridConfig, uk_boundary: BaseGeometry
) -> pt.DataFrame[H3GridWeights]:
    """Computes the H3 grid weights for Great Britain based on the UK boundary.

    This asset dynamically generates the spatial mapping between the hexagonal H3 grid
    and the regular lat/lng NWP grid. It acts as the foundational reference data
    for downstream weather data ingestion (e.g., ECMWF ENS forecasts), ensuring
    weather variables are correctly area-weighted to the H3 cells.

    The mapping is calculated by sampling each H3 cell with finer-resolution child
    cells and determining which regular grid cell each child falls into. The
    `grid_size` parameter is used to snap high-resolution H3 cells to the nearest
    regular NWP grid points (FLAW-4).
    """
    h3_res = config.h3_res
    grid_size = config.grid_size
    child_res = config.child_res if config.child_res is not None else h3_res + 2

    if child_res <= h3_res:
        raise ValueError(f"child_res ({child_res}) must be strictly greater than h3_res ({h3_res})")

    context.log.info(f"Generating H3 cells at resolution {h3_res}...")

    # Note: In h3-py v4+, `h3.geo_to_cells` is the generic entry point for any object
    # implementing `__geo_interface__` (like shapely.Polygon). `h3.polygon_to_cells`
    # is an alias for `h3shape_to_cells` and expects an internal H3Shape object,
    # which would fail here.
    cells = h3.geo_to_cells(uk_boundary, res=h3_res)
    if not cells:
        raise ValueError(
            f"No H3 cells found for the given boundary at resolution {h3_res}. "
            "Check if the boundary geometry is valid and covers the expected area."
        )

    df = pl.DataFrame({"h3_index": list(cells)}, schema={"h3_index": pl.UInt64}).sort("h3_index")

    context.log.info(
        f"Computing H3 grid weights for grid size {grid_size} with child_res {child_res}..."
    )
    df_with_counts = compute_h3_grid_weights(df, grid_size=grid_size, child_res=child_res)

    return H3GridWeights.validate(df_with_counts)
