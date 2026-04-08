The current git branch is all about implementing the ingestion of power data from NGED's JSON files. We've already implemented a first pass at this, in this git branch, following plan_for_ingesting_NGEDs_new_S3_data.md 

We had originally decided to keep the old, deprecated code that downloads data from NGED's CKAN portal. i.e. the code in `packages/nged_data`, and the related data contracts and processing code.

But I think it's making the code far too complex to have the old code co-exist with the new code. So let's work on completely removing the code that processes data from NGED's CKAN portal. 

Please ask write an implementation plan to remove the CKAN processing code, including the following (but I may have missed some points):

- Remove `packages/nged_data`
- Remove the NGED CKAN Dagster assets
- The code in the XGBoost model (or maybe it's in the ml_core package) that downsamples the 5-minutely power data from CKAN to half hourly. The new JSON data is always half-hourly.

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
