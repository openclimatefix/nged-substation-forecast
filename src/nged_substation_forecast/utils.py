from typing import Any, Type

import patito as pt
import polars as pl
from dagster import DagsterType, TypeCheck, TypeCheckContext


def create_dagster_type_from_patito_model(model: Type[pt.Model]) -> DagsterType:
    """
    Creates a DagsterType that validates a Polars DataFrame against a Patito model.

    WHY:
    We want to ensure data quality at the boundaries of our Dagster assets.
    By integrating Patito with Dagster's type system, we get automatic runtime
    validation and rich metadata in the Dagster UI whenever an asset is materialized.
    This prevents bad data from silently propagating downstream.

    HOW:
    This factory function takes a Patito model and returns a DagsterType.
    The returned DagsterType uses a `type_check_fn` to validate the output
    DataFrame against the model's schema. We handle both DataFrame and LazyFrame:
    - For eager DataFrames, we validate the actual data.
    - For LazyFrames, we only validate the schema (using an empty dummy DataFrame)
      to avoid triggering expensive computations during type checking.
    """

    def type_check_fn(context: TypeCheckContext, value: Any) -> TypeCheck:
        if not isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            return TypeCheck(
                success=False,
                description=f"Expected Polars DataFrame or LazyFrame, got {type(value)}",
            )

        try:
            if isinstance(value, pl.LazyFrame):
                # For LazyFrames, we only validate the schema to avoid materialization.
                # We create a dummy empty DataFrame with the same schema to validate.
                dummy_df = pl.DataFrame(schema=value.collect_schema())
                model.validate(dummy_df)
            else:
                # For eager DataFrames, we validate the actual data.
                model.validate(value)

            return TypeCheck(success=True)
        except Exception as e:
            return TypeCheck(
                success=False,
                description=f"Patito validation failed for {model.__name__}: {str(e)}",
            )

    return DagsterType(
        name=f"{model.__name__}DagsterType",
        type_check_fn=type_check_fn,
        description=f"A Polars DataFrame validated against the Patito model {model.__name__}.",
    )
