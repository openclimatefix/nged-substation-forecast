from datetime import datetime, timezone

import patito as pt
import polars as pl
import pytest
from contracts.common import (
    MAX_PLAUSIBLE_DATETIME,
    MIN_PLAUSIBLE_DATETIME,
    split_by_datetime_plausibility,
    validate_schema,
)
from patito.exceptions import DataFrameValidationError


class SimpleModel(pt.Model):
    a: int = pt.Field(dtype=pl.Int64)
    b: str = pt.Field(dtype=pl.String)


def test_validate_schema_success():
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    validate_schema(SimpleModel, df)


def test_validate_schema_missing_columns():
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises(DataFrameValidationError, match="Missing columns"):
        validate_schema(SimpleModel, df)


def test_validate_schema_dtype_mismatch():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(DataFrameValidationError, match="expected Int64, got Float64"):
        validate_schema(SimpleModel, df)


def test_validate_schema_lazyframe():
    df = pl.LazyFrame({"a": [1, 2], "b": ["x", "y"]})
    validate_schema(SimpleModel, df)


def test_split_by_datetime_plausibility_partitions_by_bound():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "t": [
                datetime(1840, 1, 1, tzinfo=timezone.utc),  # before MIN_PLAUSIBLE_DATETIME
                MIN_PLAUSIBLE_DATETIME,  # inclusive lower bound
                MAX_PLAUSIBLE_DATETIME,  # inclusive upper bound
                datetime(3000, 1, 1, tzinfo=timezone.utc),  # after MAX_PLAUSIBLE_DATETIME
            ],
        }
    )

    plausible, implausible = split_by_datetime_plausibility(df, "t")

    assert plausible["id"].to_list() == [2, 3]
    assert implausible["id"].to_list() == [1, 4]


def test_split_by_datetime_plausibility_treats_nulls_as_plausible():
    df = pl.DataFrame({"id": [1], "t": pl.Series([None], dtype=pl.Datetime(time_zone="UTC"))})

    plausible, implausible = split_by_datetime_plausibility(df, "t")

    assert plausible.height == 1
    assert implausible.is_empty()
