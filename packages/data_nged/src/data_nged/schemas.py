from typing import Any

import patito as pt
import polars as pl
from pydantic import BaseModel, ConfigDict


class RawSubstationLocations(pt.Model):
    # NGED has 192,000 substations.
    substation_number: int = pt.Field(dtype=pl.Int64, unique=True, lt=1_000_000)
    # The min and max string lengths are actually 3 and 48 chars, respectively.
    # Note that there are two "Park Lane" substations, with different locations and different
    # substation numbers.
    substation_name: str = pt.Field(dtype=pl.String, min_length=2, max_length=64)
    substation_type: str = pt.Field(dtype=pl.String)
    easting: float = pt.Field(dtype=pl.Float64)
    northing: float = pt.Field(dtype=pl.Float64)
    latitude: float = pt.Field(dtype=pl.Float64)
    longitude: float = pt.Field(dtype=pl.Float64)


class PackageSearchResult(BaseModel):
    model_config = ConfigDict(strict=True)

    count: int  # The number of results
    facets: dict
    results: list[dict[str, Any]]
    sort: str  # e.g. "score desc, metadata_modified desc"
    search_facets: dict
