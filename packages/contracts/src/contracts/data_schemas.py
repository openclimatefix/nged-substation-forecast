"""Data schemas for the NGED substation forecast project."""

from collections.abc import Sequence
from datetime import datetime

import patito as pt
import polars as pl


class SubstationFlows(pt.Model):
    substation_name: str = pt.Field(dtype=pl.Categorical)
    timestamp: datetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))

    # Primary substations usually have flows in the tens of MW.
    # We'll set a loose range for now to catch extreme errors.
    # If we want to reduce storage space we could store kW and kVAr as Int16.

    # Active power:
    MW: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Apparent power:
    MVA: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    # Reactive power:
    MVAr: float | None = pt.Field(dtype=pl.Float32, allow_missing=True, ge=-1_000, le=1_000)

    @classmethod
    def validate(
        cls,
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame["SubstationFlows"]:
        """Validate the given dataframe, ensuring either MW or MVA is present."""
        if "MW" not in dataframe.columns and "MVA" not in dataframe.columns:
            raise ValueError(
                "SubstationFlows dataframe must contain at least one of 'MW' or 'MVA' columns."
            )
        return super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )


class SubstationLocations(pt.Model):
    # NGED has 192,000 substations.
    substation_number: int = pt.Field(dtype=pl.Int32, unique=True, gt=0, lt=1_000_000)

    # The min and max string lengths are actually 3 and 48 chars, respectively.
    # Note that there are two "Park Lane" substations, with different locations and different
    # substation numbers.
    substation_name: str = pt.Field(dtype=pl.String, min_length=2, max_length=64)

    substation_type: str = pt.Field(dtype=pl.Categorical)
    latitude: float | None = pt.Field(dtype=pl.Float32, ge=49, le=61)  # UK latitude range
    longitude: float | None = pt.Field(dtype=pl.Float32, ge=-9, le=2)  # UK longitude range
