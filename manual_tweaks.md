## In `packages/contracts/src/contracts/data_schemas.py`

### In `class SubstationFlows`

- The data contract says that several columns are allowed to be missing.
Please check the remainder of the code. Are we *sure* that we still want to allow these columns to
be missing? I think it's possible that the code now guarantees that all columns will be present, but
that they might have null values. Please check if we can remove the `allow_missing=True`).
- Line 70: Can we remove this `if` block, because we check if "MW" or "MVA" are in columns
later in the method?
- Line 77 - 79: This NOTE comment makes no sense to me. Don't reference Flaw-004 in the code,
because these reference IDs are temporary.
- Line 80: Do we really need to check `if isinstance(dataframe, pl.DataFrame)`? In our code,
`dataframe` will never be a Pandas dataframe.
- Lines 81 & 82: If the data contract guarantees that MW and MVA columns will always be present,
then remove the `in dataframe.columns` checks.

## General

- Throughout the code, please ensure that H3 discrete indices are ALWAYS 64-bit unsigned integers
(e.g. all columns named `h3_index` or `h3_res_5`). Make sure that these data types are retained
throughout joins.
-

# New:


## In `packages/contracts/src/contracts/data_schemas.py`

### In `SubstationTargetMap`
- let's use an Enum or Literal for `target_col`, as `target_col` can only be `MW` or
  `MVA`.
- And let's also ensure that `choose_power_column` returns this stricter type.
- Rename `target_col` to `power_col`
- What unit is `peak_capacity` in? MW? MVA? Either? At the very least, let's add a comment. Even
better, let's include the unit in the field name, even if it's `peak_capacity_MW_or_MVA`.
- Update the class docstring to explain where `SubstationTargetMap` is used, and to note that it
also contains peak capacity.

### In `SimplifiedSubstationFlows`
- Add a class docstring to explain where this is used.


### In `InferenceParams`
- Please check the accuracy of the comment for `forecast_time`.

### In `class Nwp`
- Why do we need the block `if not isinstance(validated_df, pl.DataFrame)`? In fact, is this a bug?
  Shouldn't we _always_ continue through this function to check all the cols?

### In `class NwpColumns`
- Why is this necessary? If it's dead code then remove it. If it's not dead code then add a class
docstring explain why it's necessary, and where it's used.

### In `class ProcessedNwp` and `class SubstationFeatures`
- Why are we using `Float32` for the weather variables? I thought we wanted to use `uint8` whenever
  possible to save RAM, and only convert to `float32` if/when we need to do any maths on these
values (e.g. computing windchill)

## In `packages/contracts/src/contracts/settings.py`

### In `class Settings`
- Please add more description explaining `nwp_ensemble_member` and `ml_model_ensemble_size`. Is
`nwp_ensemble_member` only used when training (i.e. do we use this ensemble member to train the ML
model?) Is `ml_model_ensemble_size` only used at inference time?

## In `packages/contracts/tests/test_data_schemas.py`

### In `def test_nwp_validation`
- There are two places where the code uses a long `with_columns` block just to cast to `pl.UInt8`.
Please replace these `.with_columns` blocks with `.cast(nwp_vars_to_uint8)`, where
`nwp_vars_to_uint8` is a dict that you programmatically define once, to map from NWP variable names
to `pl.UInt8`.

### In `def test_substation_flows_property_based`
- Replace the `.with_columns` with `.cast`. It's more readable.

## In `packages/dynamical_data/src/dynamical_data/assets/ecmwf_scaling_params.csv`
- Why do we need a scaling param for `categorical_precipitation_type_surface`? Surely this is
already an integer, which - if I remember correctly - is always in the range of roughly [0, 12].

## In `packages/dynamical_data/`
- Please remove `src/dynamical_data/example_data` from git.
- I think it makes more sense for `packages/dynamical_data/src/dynamical_data/assets/` and `packages/dynamical_data/src/dynamical_data/example_data/` to be moved to `packages/dynamical_data/assets/` and `packages/dynamical_data/example_data/` because this stuff isn't _source_ code (so shouldn't go into the `src` directory)
- Please add some docs (maybe in `packages/dynamical_data/assets/README.md`) to explain where all the files
  in the `assets/` directory come from.
