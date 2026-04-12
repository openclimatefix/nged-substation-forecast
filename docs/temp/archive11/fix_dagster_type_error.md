---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/weather_assets.py"]
---
# Implementation Plan: Fix DagsterInvalidDefinitionError for all_nwp_data

## Context
The test suite is failing with a `DagsterInvalidDefinitionError` at line 58 of `src/nged_substation_forecast/defs/weather_assets.py`. The error states: `Dagster cannot resolve the type patito.polars.LazyFrame[contracts.data_schemas.Nwp]`. 

Dagster uses Python type hints to infer the Dagster types of assets. While `patito.LazyFrame[Nwp]` is useful for static type checking and documenting the expected schema, Dagster's type resolution system does not natively understand this generic type from the `patito` library, leading to the definition error.

## Proposed Changes

1. **Update Return Type Hint in `weather_assets.py`:**
   - Modify the `all_nwp_data` asset definition in `src/nged_substation_forecast/defs/weather_assets.py`.
   - Change the return type hint from `pt.LazyFrame[Nwp]` to `pl.LazyFrame`.
   - This aligns with the downstream asset `processed_nwp_data`, which already expects `all_nwp_data: pl.LazyFrame`.

2. **Maintain Schema Documentation:**
   - Keep the `cast(pt.LazyFrame[Nwp], ...)` calls inside the function body. Since `pt.LazyFrame` is a subclass of `pl.LazyFrame`, this remains valid for static type checkers (like `mypy` or `pyright`) while avoiding Dagster's runtime type resolution issues.
   - Ensure the docstring clearly states that the returned `LazyFrame` adheres to the `Nwp` schema.

## Code Modifications

**File:** `src/nged_substation_forecast/defs/weather_assets.py`

```python
# Current:
@asset(deps=[ecmwf_ens_forecast])
def all_nwp_data(settings: ResourceParam[Settings]) -> pt.LazyFrame[Nwp]:
    """Provides a LazyFrame scanning all downloaded NWP data."""
    # ...

# Proposed:
@asset(deps=[ecmwf_ens_forecast])
def all_nwp_data(settings: ResourceParam[Settings]) -> pl.LazyFrame:
    """Provides a LazyFrame scanning all downloaded NWP data.
    
    The returned LazyFrame adheres to the contracts.data_schemas.Nwp schema.
    """
    # ...
```

## Rationale
- **Why `pl.LazyFrame`?** Dagster knows how to handle standard Polars types.
- **Why keep `cast`?** It preserves the developer intent and static typing benefits provided by `patito` without interfering with Dagster's runtime asset definition parsing.
