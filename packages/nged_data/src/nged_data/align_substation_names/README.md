NGED uses slightly different naming conventions in their different datasets.

For example, the NGED substation locations table uses the name `Alliance And Leicester 33 11kv S
Stn`, whilst the NGED live primary flows dataset uses the name `Alliance & Leicester Primary
Transformer Flows`.

The code in `simplify_names.py` does its best to simplify the substation names (e.g. removing
"Primary Flows" and "11kV") to make it easier to automatically match substation names. But it's not
perfect, and some substation names fail to match, even after the names have been simplified. And so
the CSV files in this directory contain manual mappings of NGED's substation names that fail to
match after the automatic alignment. Note that the names in these manual mappings are the substation
names _after_ the function `simplify_names` (in `src/substation_names.py`) has been applied.
