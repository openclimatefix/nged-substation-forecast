from typing import Any, Final, Type

import patito as pt
import polars as pl
from patito.exceptions import (
    ColumnDTypeError,
    DataFrameValidationError,
    ErrorWrapper,
    MissingColumnsError,
)

# Define our standard datetime type for all schemas
UTC_DATETIME_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")

DELIVERY_QUANTILES: Final[tuple[float, ...]] = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
)
"""The thirteen quantile levels agreed with NGED for the delivery tables.

Deliberately tail-heavy: NGED is far more interested in the tails than the shoulders. This
tuple is the single source of truth for every quantile-indexed artefact — the pinball-loss
``metric_param`` labels today, and the percentile columns of the delivery-table
representations (Representations 2 and 3) when those land in v0.5.
"""


def quantile_label(quantile: float) -> str:
    """Return the canonical ``p{level}`` label for a quantile, e.g. ``0.05`` → ``"p5"``.

    The label format matches the percentile column names agreed with NGED for the delivery
    tables (see ``DELIVERY_QUANTILES``).
    """
    return f"p{round(quantile * 100)}"


def _get_time_series_id_dtype(**kwargs) -> Any:
    return pt.Field(
        dtype=pl.Int32,
        description=(
            "Provided by NGED. This is the primary key for identifying the time series."
            " There's _almost_ a one-to-one mapping between time_series_id and the"
            " asset ID, so you can think of time_series_id as the asset ID"
            " (where an 'asset' is a physical asset like a substation or PV farm)"
        ),
        **kwargs,
    )


def validate_schema(model: Type[pt.Model], df: pl.DataFrame | pl.LazyFrame) -> None:
    """Validates that the schema of a Polars DataFrame or LazyFrame matches the schema defined in a
    Patito model, raising DataFrameValidationError on failure. On LazyFrames, this function doesn't
    materialize any data, it just calls `collect_schema()`.
    """
    # Get actual schema
    if isinstance(df, pl.LazyFrame):
        actual_schema = dict(df.collect_schema())
    else:
        actual_schema = dict(df.schema)

    # Check for missing columns
    missing_cols = set(model.dtypes.keys()) - set(actual_schema.keys())
    if missing_cols:
        error = ErrorWrapper(
            MissingColumnsError(f"Missing columns: {missing_cols}"), loc=tuple(missing_cols)
        )
        raise DataFrameValidationError([error], model)

    # Check for dtype mismatches
    errors = []
    for col, expected_dtype in model.dtypes.items():
        if col in actual_schema:
            actual_dtype = actual_schema[col]
            if actual_dtype != expected_dtype:
                errors.append(
                    ErrorWrapper(
                        ColumnDTypeError(
                            f"Column '{col}' expected {expected_dtype}, got {actual_dtype}"
                        ),
                        loc=(col,),
                    )
                )

    # Raise the native Patito exception if errors were found
    if errors:
        raise DataFrameValidationError(errors, model)
