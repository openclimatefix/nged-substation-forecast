- Please ensure that the code that creates `time_series_metadata.parquet` correctly populates the
  `h3_res_5` column. The code will need to calculate the H3 discrete spatial index from the latitude
  and longitude. Look at how this is done in the main git branch.
- In your work just now, you modified the processed_nwp_data asset to load both time_series_metadata.parquet and substation_metadata.parquet. That is **incorrect**! Do NOT use `substation_metadata.parquet` ANYWHERE in the code!
-
