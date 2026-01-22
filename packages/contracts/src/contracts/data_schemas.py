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
    mw: float | None = None
    mvar: float | None = None


class SubstationMetadata(pt.Model):
    """Metadata for a substation."""

    substation_id: str
    substation_name: str
    latitude: float | None = None
    longitude: float | None = None
    voltage: float | None = None
    region: str | None = None
