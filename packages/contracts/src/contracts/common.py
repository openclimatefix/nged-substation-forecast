from typing import Any, Type

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

    expected_dtypes = model.dtypes
    errors = []

    # Check for missing columns
    missing = set(expected_dtypes.keys()) - set(actual_schema.keys())
    if missing:
        # Wrap the error with the location of the missing columns
        errors.append(
            ErrorWrapper(MissingColumnsError(f"Missing columns: {missing}"), loc=tuple(missing))
        )

    # Check for dtype mismatches
    for col, expected_dtype in expected_dtypes.items():
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
