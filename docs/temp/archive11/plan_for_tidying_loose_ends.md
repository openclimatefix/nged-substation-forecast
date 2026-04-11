The current git branch is all about implementing the ingestion of power data from NGED's JSON files, and the complete removal of the old code which ingested power data from NGED's CKAN data portal (as CSV files).

The new JSON data is in a much more convenient format, and allows us to significantly simplify our code. In some ways, one of the main aims of this refactor is to simplify our code.

We have almost completed all the tasks but it looks like there's some dead code left which is no
longer needed.

- Can we delete `src/nged_substation_forecast/ingestion/helpers.py`? It looks like all the classes and functions in `helpers.py` are left over from when we ingested CSV files of just substations.
- Can we delete  `src/nged_substation_forecast/ingestion/cleaning.py`? It appears to be concerned
  with cleaning individual substations. But we're now working with `PowerTimeSeries` which are
  more general than just substations.
- Please search the codebase for any more mentions of `substation` or `csv` and give me a report on
  whether any of that code could be removed. Pause for me review of the plan before proceeding.

