import patito as pt
import polars as pl


class H3GridWeights(pt.Model):
    """Schema for the pre-computed H3 grid weights.

    This contract defines the mapping between H3 hexagons and a regular latitude/longitude grid.
    It is used to ensure type safety when passing spatial mapping data from generic geospatial
    utilities (like `packages/geo`) to dataset-specific ingestion pipelines (like `packages/dynamical_data`).
    """

    h3_index: int = pt.Field(dtype=pl.UInt64)
    nwp_lat: float = pt.Field(
        dtype=pl.Float32, ge=-90, le=90, description="The latitude of the NWP grid box."
    )
    nwp_lon: float = pt.Field(
        dtype=pl.Float32, ge=-180, le=180, description="The longitude of the NWP grid box."
    )

    proportion: float = pt.Field(
        dtype=pl.Float32,
        ge=0,
        le=1,
        description="The proportion of this H3 hexagon that falls into this NWP grid box.",
    )
