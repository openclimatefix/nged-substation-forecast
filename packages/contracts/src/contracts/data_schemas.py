"""Data schemas for the NGED substation forecast project."""

from datetime import datetime

import patito as pt
import polars as pl


class SubstationFlows(pt.Model):
    """A single measurement from a substation."""

    substation_name: str = pt.Field(dtype=pl.Categorical)
    timestamp: datetime = pt.Field(dtype=pl.Datetime)

    # Primary substations usually have flows in the tens of MW.
    # We'll set a loose range for now to catch extreme errors.
    # If we want to reduce storage space we could store kW and kVAr as Int16.

    # Active power:
    MW: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Apparent power:
    MVA: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Reactive power:
    MVAr: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)


class SubstationLocations(pt.Model):
    """Metadata for a substation."""

    substation_name: str
    latitude: float | None = pt.Field(dtype=pl.Float32, ge=49, le=61)  # UK latitude range
    longitude: float | None = pt.Field(dtype=pl.Float32, ge=-9, le=2)  # UK longitude range
