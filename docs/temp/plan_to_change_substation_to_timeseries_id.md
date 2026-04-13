The current git branch is all about implementing the ingestion of power data from NGED's JSON files, and the complete removal of the old code which ingested power data from NGED's CKAN data portal (as CSV files). This git branch also focused on generalising the energy forecasting pipeline. We previously only created forecasts for **substations**. We now create forecasts for any **time_series_id**, which might be a substation, or a solar power farm, or a wind farm, or any other distributed energy resource.

The new JSON data is in a much more convenient format, and allows us to significantly simplify our code. In some ways, one of the main aims of this refactor is to simplify our code.

Crucially, we now use `time_series_id` as **the** primary identifier for all time series. We no
longer use `substation_number` or `substation_name` as the primary identifier (although we still
track these attributes in the `TimeSeriesMetadata` data contract, and we still want the ability to
display these attributes when we plot forecasts.)

We have almost completed all the tasks but it looks like there's some code that still refers to
`substation_number` instead of `time_series_id`.

## Tasks

1. Please search the codebase for any remaining uses of `substation` or `substation_number`. Please
   give me a detailed report of where these are still used. Pause for my review.
2. Come up with a detailed plan for changing `substation_number` to `time_series_id`. This plan
   should change all variable names, and function arguments. (But do **not** change
   `TimeSeriesMetadata`. We **want** to store `substation_number` in `TimeSeriesMetadata`). Pause
   for my review of this plan.
3. Implement the plan using your `implement` skill.
