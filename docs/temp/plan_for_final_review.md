This git branch represents a huge refactor. The primary aims of this refactor are:

1. To replace the "old way" of ingesting power data (from NGED's CKAN portal, as CSV files) with the
   "new way" of ingesting power data (from JSON files).
2. To remove any code that tries to select between "MW" or "MVA" columns in the "old" CSV files. The
   JSON files only contain a single value per time series. So we no longer need the complexity of
   switching between MV or MVA.
3. To generalise the code to forecast for _any_ power time series (e.g. from a substation, or from a
   solar farm, or from a wind farm, or any other distributed energy resource). This means moving
   away from using `substation_number` to identify time series, and instead using `time_series_id`
   as the canonical way to identify time series. (However, we do still want to _record_
   `substation_number` in the `TimeSeriesMetadata`.

Please use your `code-review-loop` skill to thoroughly review the code in this branch. But, before
implementing any findings from each reviewer station, please pause to get my approval of the
reviewer's comments. For now, let's assume we'll just do a single review loop. Please ask the
reviewer agents to focused especially on:

- Make sure the aims (listed above) are achieved.
- Make sure the code is as simple as possible.
- Look for dead code left over from the "old way".
- Simplify the unit tests.
