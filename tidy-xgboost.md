## Aims

- Significantly tidy up the XGBoost code.
- Remove all the "silent failures" like `if not weather_dfs: return pl.DataFrame()`. Fail early.
Fail loudly! If data is missing then I want to know about it, and I want to know about the failure
as early in the call chain as possible.
- Remove any code paths that are left over from previous experiments. Only keep code paths that are
  actively used.
- Whenever a function accepts or returns a Polars DataFrame, it *must* use a Patito DataFrame with
  a specific type.
- Write Python like you're writing Rust: Invalid states must be unrepresentable. 
- Whenever a function accepts a string that can be one of a small set of values, e.g. `weather_ens_member_selection` in the
`prepare_data_for_substation` function, replace that string with an `Enum`.
- Functions should, in general, be no more than 50 lines of code. Break large functions into
smaller, easy-to-test, easy-to-understand units. Each function should do one well-defined task.

## Specific tasks

- I don't understand why `load_weather_data` is so complex. Instead of having `start_date` and
`end_date` arguments, why doesn't it just load one NWP run at a time (which can be fully determined
from the `init_time`)? (The weather data is stored as one Parquet file per NWP run).
