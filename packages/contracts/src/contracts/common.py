from datetime import UTC, datetime
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

MIN_PLAUSIBLE_DATETIME: Final[datetime] = datetime(2000, 1, 1, tzinfo=UTC)
"""The earliest timestamp any datetime column in any contract may carry.

NGED telemetry cannot predate the instrumentation that produced it, and the ECMWF archive we
ingest begins later still, so no legitimate row is older than this. The bound is also deliberately
far later than 1847: ``Europe/London`` ran on local mean time at UTC−0:01:15 until then, so a
pre-1848 timestamp produces a sub-minute UTC offset and a nonsensical value for every local-time
feature. Ruling those timestamps out at the contract boundary is what lets the local-time features
in ``ml_core`` express the UTC offset in whole minutes.
"""

MAX_PLAUSIBLE_DATETIME: Final[datetime] = datetime(2100, 1, 1, tzinfo=UTC)
"""The latest timestamp any datetime column in any contract may carry.

This is a fixed date rather than an offset from the current time, so validation never depends on
the wall clock: a frame that validated when it was written still validates when it is read back
years later, and tests need no clock control. A fixed far-future bound still catches the failure
this check exists for — the classic epoch-unit mix-up, where Unix milliseconds read as seconds land
tens of thousands of years in the future. It deliberately does not catch a small clock skew that
ships tomorrow's data as today's; that is a monitoring concern, not a contract one.
"""


def check_datetime_bounds(dataframe: pl.DataFrame, column: str, *more_columns: str) -> None:
    """Raise ``ValueError`` if any timestamp lies outside the plausible-datetime range.

    Call this from a Patito model's ``validate`` override, after ``super().validate()``. It exists
    because Patito **silently ignores** ``ge``/``le`` on a datetime field: Patito derives its bounds
    checks from the Pydantic JSON schema's ``minimum``/``maximum`` keywords, which JSON Schema
    defines for numbers only, so a datetime field's ``Ge``/``Le`` metadata never reaches the JSON
    schema and no check is ever generated. (``ge``/``le`` on a *numeric* field works normally, which
    is why ``PowerTimeSeries.power`` can state its bounds on the field itself.)

    Args:
        dataframe: An already-validated frame. Every named column must be a datetime column.
        column: Name of a datetime column to bound.
        *more_columns: Names of any further datetime columns to bound.

    Raises:
        ValueError: If any value is before :data:`MIN_PLAUSIBLE_DATETIME` or after
            :data:`MAX_PLAUSIBLE_DATETIME`. Nulls are ignored — absence is not malformedness — and
            an empty frame always passes.
    """
    columns = (column, *more_columns)
    extremes = dataframe.select(
        *(pl.col(name).min().alias(f"min_{name}") for name in columns),
        *(pl.col(name).max().alias(f"max_{name}") for name in columns),
    ).row(0, named=True)
    for name in columns:
        _raise_if_outside_plausible_range(name, extremes[f"min_{name}"], extremes[f"max_{name}"])


def _raise_if_outside_plausible_range(
    column: str, earliest: datetime | None, latest: datetime | None
) -> None:
    """Raise ``ValueError`` if ``earliest`` or ``latest`` falls outside the plausible range.

    ``earliest`` and ``latest`` are the column's min and max; both are ``None`` when the column is
    empty or entirely null, in which case there is nothing to reject.
    """
    if earliest is not None and earliest < MIN_PLAUSIBLE_DATETIME:
        raise ValueError(
            f"`{column}` is outside the plausible datetime range: its earliest value is {earliest},"
            f" which is before MIN_PLAUSIBLE_DATETIME ({MIN_PLAUSIBLE_DATETIME}). A timestamp this"
            " old indicates a corrupt feed or an epoch-unit mix-up, not a real reading."
        )
    if latest is not None and latest > MAX_PLAUSIBLE_DATETIME:
        raise ValueError(
            f"`{column}` is outside the plausible datetime range: its latest value is {latest},"
            f" which is after MAX_PLAUSIBLE_DATETIME ({MAX_PLAUSIBLE_DATETIME}). A timestamp this"
            " far in the future indicates a corrupt feed or an epoch-unit mix-up, not a real"
            " reading."
        )


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
