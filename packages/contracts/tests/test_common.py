import patito as pt
import polars as pl
import pytest
from contracts.common import validate_schema
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
