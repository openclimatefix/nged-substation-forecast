## In `packages/contracts/src/contracts/data_schemas.py`

### In `SubstationFlows`

- The data contract says that several columns are allowed to be missing.
Please check the remainder of the code. Are we *sure* that we still want to allow these columns to
be missing? I think it's possible that the code now guarantees that all columns will be present, but
that they might have null values.
- Line 70: Can we remove this `if` block, because we check if "MW" or "MVA" are in columns
later in the method.
-
