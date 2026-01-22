"""Data schemas for the NGED substation forecast project."""

from __future__ import annotations

from typing import TYPE_CHECKING

import patito as pt

if TYPE_CHECKING:
    from datetime import datetime


class SubstationMeasurement(pt.Model):
    """A single measurement from a substation."""

    substation_id: str
    timestamp: datetime
    # Primary substations usually have flows in the tens of MW.
    # We'll set a loose range for now to catch extreme errors.
    mw: float | None = pt.Field(None, ge=-1000, le=1000)
    mvar: float | None = pt.Field(None, ge=-1000, le=1000)


class SubstationMetadata(pt.Model):
    """Metadata for a substation."""

    substation_id: str
    substation_name: str
    latitude: float | None = pt.Field(None, ge=49, le=61)  # UK latitude range
    longitude: float | None = pt.Field(None, ge=-9, le=2)  # UK longitude range
    voltage: float | None = pt.Field(None, ge=0, le=400)
    region: str | None = None
