---
status: "draft"
version: "v3.4"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/utils.py", "src/nged_substation_forecast/defs/"]
---

# Implementation Plan: Dagster-Patito Integration

## Objective
Integrate Patito models with Dagster's type system using a factory function. This will provide automatic runtime validation and rich metadata in the Dagster UI whenever an asset is materialized, without compromising static type checking.

## Step 1: Create `create_dagster_type_from_patito_model` in `utils.py`

**File:** `src/nged_substation_forecast/utils.py`

Add the following utility function. It includes liberal comments explaining the *why* and *how*, ensuring junior engineers can understand the rationale.

```python
from typing import Type, Any
from dagster import DagsterType, TypeCheck, TypeCheckContext
import patito as pt
import polars as pl

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
                description=f"Expected Polars DataFrame or LazyFrame, got {type(value)}"
            )

        try:
            if isinstance(value, pl.LazyFrame):
                # For LazyFrames, we only validate the schema to avoid materialization.
                # We create a dummy empty DataFrame with the same schema to validate.
                dummy_df = pl.DataFrame(schema=value.collect_schema())
                pt.DataFrame(dummy_df).set_model(model).validate()
            else:
                # For eager DataFrames, we validate the actual data.
                pt.DataFrame(value).set_model(model).validate()

            return TypeCheck(success=True)
        except Exception as e:
            return TypeCheck(
                success=False,
                description=f"Patito validation failed for {model.__name__}: {str(e)}"
            )

    return DagsterType(
        name=f"{model.__name__}DagsterType",
        type_check_fn=type_check_fn,
        description=f"A Polars DataFrame validated against the Patito model {model.__name__}."
    )
```

## Step 2: Generate ADR `001-dagster-patito-integration.md`

**File:** `docs/architecture_decision_records/001-dagster-patito-integration.md`

Create the ADR with the following content:

```markdown
# ADR 001: Dagster-Patito Integration

## Status
Accepted

## Context
We use Patito to define strict data schemas for our Polars DataFrames. We also use Dagster to orchestrate our data pipelines. Currently, our Dagster assets return `pt.DataFrame[...]` or `pl.LazyFrame`, but Dagster is unaware of the Patito schemas. This means we don't get runtime validation at the Dagster boundary, nor do we get rich metadata in the Dagster UI.

## Alternatives Considered
1. **TypeAlias:** Using a simple TypeAlias (e.g., `PowerTimeSeriesType = pt.DataFrame[PowerTimeSeries]`). This satisfies static type checkers but provides no runtime validation or Dagster UI integration.
2. **Side-effect Assets:** Manually calling `df.set_model(Model).validate()` inside every asset before returning. This is boilerplate-heavy, prone to being forgotten, and doesn't integrate with Dagster's type system (e.g., no type descriptions in the UI).
3. **Returning raw Polars DataFrames:** Returning `pl.DataFrame` and relying entirely on downstream assets to validate inputs. This violates the principle of failing fast and makes it harder to trace where bad data originated.

## Decision
We will create a factory function `create_dagster_type_from_patito_model` that generates a `DagsterType` for a given Patito model. We will use this `DagsterType` in the `dagster_type` argument of the `@asset` decorator.

## Consequences
- **Fails Fast:** Validates data exactly at the boundary where it is produced.
- **Zero Boilerplate in Assets:** The validation logic is abstracted away into the Dagster type system.
- **Rich UI:** The Dagster UI will display the schema name and description.
- **Static Typing:** By using `dagster_type=...` in the decorator, we can keep the Python return type annotation as `pt.DataFrame[Model]`, satisfying mypy/pyright.
- **LazyFrame Support:** The factory function intelligently handles `LazyFrame` by only validating the schema (via an empty dummy DataFrame) to avoid triggering expensive computations during type checking.
```

## Step 3: Update Assets

Update the following assets to use the new `DagsterType` factory.

**How to update:**
1. Import the factory function and the relevant Patito model.
2. Create the DagsterType instance at the module level (e.g., `PowerTimeSeriesDagsterType = create_dagster_type_from_patito_model(PowerTimeSeries)`).
3. Pass it to the `@asset` decorator via the `dagster_type` argument.
4. Ensure the Python return type annotation is `pt.DataFrame[Model]` or `pt.LazyFrame[Model]`.

**Files to update:**

1. **`src/nged_substation_forecast/defs/data_cleaning_assets.py`**
   - Asset: `cleaned_power_time_series`
   - Model: `PowerTimeSeries`
   - Update: Add `dagster_type=PowerTimeSeriesDagsterType` to `@dg.asset`.

2. **`src/nged_substation_forecast/defs/weather_assets.py`**
   - Asset: `all_nwp_data`
   - Model: `Nwp`
   - Update: Change return type annotation from `pl.LazyFrame` to `pt.LazyFrame[Nwp]`. Add `dagster_type=NwpDagsterType` to `@asset`.
   - Asset: `processed_nwp_data`
   - Model: `ProcessedNwp`
   - Update: Change return type annotation from `pl.LazyFrame` to `pt.LazyFrame[ProcessedNwp]`. Add `dagster_type=ProcessedNwpDagsterType` to `@asset`.

3. **`src/nged_substation_forecast/defs/metrics_assets.py`**
   - Asset: `metrics`
   - Model: `Metrics`
   - Update: Change return type annotation from `pl.DataFrame` to `pt.DataFrame[Metrics]`. Add `dagster_type=MetricsDagsterType` to `@dg.asset`.

## Review Responses & Rejections

* **FLAW-001 (Tester):** REJECTED. The tester suggested using `typing.Annotated[pt.DataFrame[Model], DagsterType]` for the return type annotation. While this is supported in newer Dagster versions for standard Python types, it fails when the second argument is an instance of `DagsterType` rather than a class, causing `DagsterInvalidDefinitionError`. We have chosen to use the `dagster_type` argument in the `@asset` decorator instead, which works flawlessly and keeps the Python type hint clean for static type checkers.
