import patito as pt
import polars as pl


class H3GridWeights(pt.Model):
    """Schema for the pre-computed H3 grid weights.

    This contract defines the mapping between H3 hexagons and a regular latitude/longitude grid.
    It is used to ensure type safety when passing spatial mapping data from generic geospatial
    utilities (like `packages/geo`) to dataset-specific ingestion pipelines (like `packages/dynamical_data`).
    """

    h3_index: int = pt.Field(dtype=pl.UInt64)
    nwp_lat: float = pt.Field(dtype=pl.Float64, ge=-90, le=90)
    nwp_lng: float = pt.Field(dtype=pl.Float64, ge=-180, le=180)
    len: int = pt.Field(dtype=pl.UInt32)
    total: int = pt.Field(dtype=pl.UInt32)
    proportion: float = pt.Field(dtype=pl.Float64)
