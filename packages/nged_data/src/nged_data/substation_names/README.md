NGED uses slightly different naming conventions in their different datasets.

For example, the NGED substation locations table uses the name `Alliance And Leicester 33 11kv S
Stn`, whilst the NGED live primary flows dataset uses the name `Alliance & Leicester Primary
Transformer Flows`.

The code in `align.py` does its best to simplify the substation names (e.g. removing
"Primary Flows" and "11kV") to make it easier to automatically match substation names. But it's not
perfect, and some substation names fail to match, even after the names have been simplified. And so
the CSV files in this directory contain manual mappings of NGED's substation names that fail to
match after the automatic alignment. Note that the names in these manual mappings are the substation
names _after_ the function `simplify_substation_names` has been applied.

## How the CSVs were created

1. Simplify the substation names in the two datasets to be joined (using
   `align.simplify_substation_names`). See the function body of
   `align.join_location_table_to_live_primaries` for how to do this.
2. Perform an inner dataframe join on the `simple_name` columns. If the joined dataframe is the same
   length as the smaller of the two input dataframes then you're done!
3. But if the joined dataframe is too short then some names failed to match. Perform join with
   `how='anti'` to returns rows from the left table that have no match in the right table.
   And manually work through these names. Hopefully there will only be a few tens of names to
   manually match.
