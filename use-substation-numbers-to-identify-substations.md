The code, as it stands right now, uses `substation_name` to identify each of NGED's ~800 primary
substations. The aim of this PR is to move to using `substation_number` as the primary way to
identify substations.

`substation_name` is a poor choice primarily because different NGED datasets use slightly different substation names!

But, note that the NGED `live_primary_flows` dataset does *not* use `substation_number`.

The `substation_number` can be found in the substation locations dataset.

But the substation locations dataset uses a slightly different naming convention compared to the
live primary flows dataset!  But, fear not, the function
`nged_data.substation_names.align.join_location_table_to_live_primaries` knows how to "normalise"
these two naming conventions, and to join the substation locations dataset to the list of live primaries CkanResources.

I'd propose that the implementation plan should be something like this, but I'd be keen to hear your
critique:

Let's modify the `substation_locations` Dagster asset. As well as grabbing the substation locations
from CKAN, and computing the `h3_res_5` column, the asset should also grab the list CKAN resources for
live primary flows. It then merges the two dataframes using `join_location_table_to_live_primaries`.
(See @packages/dashboard/main.py for an example of how to call
`join_location_table_to_live_primaries`.) Save the full output of
`join_location_table_to_live_primaries` to disk as Parquet, with a filename something like
`substation_metadata.parquet`. This file replaces the old `substation_locations.parquet`.
Maybe rename the `substation_locations` Dagster asset to `substation_metadata`.

Please also create a data contract for `primary_substation_metadata`, and modify
`join_location_table_to_live_primaries` to return a `pt.DataFrame` with the new type.

Change the `live_primary_flows` Dagster asset to depend on `primary_substation_metadata`.

I'm imagining that we'd re-run `primary_substation_metadata` every day, and *update*
`substation_metadata.parquet`. This _might_ be overkill. But it's fairly fast (a few seconds), and
means that we can dynamically adapt to NGED adding new substations. But please never _delete_
substations from `primary_substation_metadata` to protect against NGED accidentally truncating their
lists of substations.

In the Delta Lake schema for `live_primary_flows`, replace `substation_name` with
`substation_number`. (i.e. that Delta Table won't store the `substation_name` at all. It'll
partition on `substation_number`.)

Please write a throw-away script to convert the existing Delta Lake table on disk to the new schema.

Please update all the downstream code, for example:

- @packages/dashboard/main.py (remember that this is a Marimo notebook!)
- @packages/xgboost_forecaster
- any relevant unit tests

And please look for other downstream code that might need changing.

Finally, please add comments to the code to explain why we have to do this little dance of
normalising the substation names before joining them.
