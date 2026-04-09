The current git branch is all about implementing the ingestion of power data from NGED's JSON files, and the removal of the old code which ingested power data from NGED's CKAN data portal (as CSV files).

The new JSON data is in a much more convenient format, and allows us to significantly simplify our
code. In some ways, one of the main aims of this refactor is to simplify our code.

We have already done a lot of work towards these aims. We've already completed several implementation
plans, and reviewed those plans, and implemented most of the ideas in these plans. The main
two plans to review are `docs/temp/implementation_plan_v1_merged.md` and `docs/temp/implementation_plan_v2_fix_imports.md`

But, in the final stages of the implementation, the `custom_build` agent was getting confused, and the code is currently
in an inconsistent state. A summary of the most recent state of the `custom_build` is in
`docs/temp/session_handoff.md`

# Tasks
- Review the original implementation plans, and compare those plans to what has _actually_ been done
  so far. To find what has already been done, look at the recent git commits in this branch and
  also look at the code itself.
- Write a fresh implementation plan for the steps that still need to be completed.

# Overview of the ultimate aims of this large refactor:

- Remove `packages/nged_data`
- Remove the NGED CKAN Dagster assets
- Remove the code in the XGBoost model (or maybe it's in the ml_core package) that downsamples the 5-minutely power data from CKAN to half hourly. The new JSON data is always half-hourly.

## Data contracts:

Make these changes to the data contracts in packages/contracts. And update all downstream code and
tests that use these contracts:

- Delete `SubstationPowerFlows`. 
- Rename `NgedJsonPowerFlows` to `PowerTimeSeries`. This should replace all previous uses of
  `SubstationPowerFlows` and `SimplifiedSubstationPowerFlows`.
- Delete `SubstationLocations` (which was just for validating the raw substation locations datasets from CKAN, which we're no longer using) and delete `SubstationLocationsWithH3`.
- In the `PowerForecast` contract, rename `MW_or_MVA` to `power_fcst`. And replace `substation_number` with `time_series_id`, which is now the canonical way to refer to the objects we're forecasting for.
- Rename `SubstationMetadata` to just `Metadata` (because we're now storing metadata for single-site solar PV, and things like that).

### In the `Metadata` contract, remove the following fields:
- substation_name_in_location_table
- substation_name_in_live_primaries
- url
- preferred_power_col

### `SubstationFeatures`
- Rename `SubstationFeatures` to `XGBoostInputFeatures`.
- replace `substation_number` with `time_series_id`.
- rename `MW_or_MVA` with `power`. 
- add a categorical feature for `time_series_type`.

### Simplify MW versus MVA
The code no longer needs to worry about figuring out whether to use MW or MVA columns because the JSON data only reports a single "value", and the metadata records the unit (MW or MVA). This can make our code much simpler! Specifically:
- delete the consts `POWER_MW`, `POWER_MVA`, and `POWER_MW_OR_MVA`, and `PowerColumn`. 
- All downstream code that uses any of these constants can be simplified.
- Delete `MissingcorePowerVariablesError` (that concept doesn't exist any more!)
- Delete `SimplifiedSubstationPowerFlows`. All code that used that can now just use `PowerFlows` directory.
- Delete `SubstationTargetMap`. This concept doesn't exist any more!
- Delete the Dagster asset that figured out which column to use.
