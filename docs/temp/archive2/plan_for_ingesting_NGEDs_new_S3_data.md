NGED have substantially changed how they deliver their power data to us. We had previously been getting data as CSV files from NGED's CKAN data portal. Now the data will be available as JSON files in an S3 bucket.

## NGED's new JSON data

The new data on S3 has a very different file format to the CSV files we've been ingesting so far. All the data we need for a given substation is included in each JSON file (e.g. the geographical location, and the substation name, and the power data). The metadata will be repeated when the new JSON files are published by NGED every 6 hours.

A one-off dump of archival data for 33 time series can be found at
~/dev/python/nged-substation-forecast/data/NGED/from_NGED_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/*.json

Here's the first 50 lines of one JSON file:

```json
{
  "TimeSeriesID": 1,
  "TimeSeriesName": "BOSTON 33kV S STN",
  "TimeSeriesType": "Raw Flow",
  "Units": "MW",
  "LicenceArea": "EMids",
  "SubstationNumber": 900004,
  "SubstationType": "BSP",
  "Latitude": 52.966444,
  "Longitude": -0.014139518,
  "Information": null,
  "Area": {
    "WKT": NaN,
    "GeometryType": "MultiPolygon",
    "SRID": 4326,
    "IsValid": true,
    "CenterLat": NaN,
    "CenterLon": NaN
  },
  "data": [
    {
      "value": 0.0,
      "startTime": "2016-01-01 00:30:00+0000",
      "endTime": "2016-01-01 01:00:00+0000"
    },
    {
      "value": 0.0,
      "startTime": "2016-01-01 01:00:00+0000",
      "endTime": "2016-01-01 01:30:00+0000"
    },
    {
      "value": 0.0,
      "startTime": "2016-01-01 01:30:00+0000",
      "endTime": "2016-01-01 02:00:00+0000"
    }
  ]
}
```

Note that this new data is not just about _primary substations_. This new data is _mostly_ about primary substations. But also includes data from bulk supply points and grid supply points and customer data (e.g. large metered PV sites). NGED now use a `TimeSeriesID` to uniquely identify each time series. For example, one time series might be the power flow for a specific primary substation. Another time series might be the power flow for a customer's PV site.

There are separate archival files (which contain several years of data) and live files (which contain only something like a rolling 2 weeks of data). The live files will be updated every 6 hours.

Note that we currently only have access to a one-off dump of archival JSON files. In a week or so
we'll have access to NGED's S3 bucket with the live-updating data. I believe the live-updating data
will consists of a new directory named `<start_unix_epoch>_<end_unix_epoch>/`, where `start UNIX
epoch` and `end UNIX epoch` mark the start and end of a approximately 2-week rolling window of data,
that will shift forwards in time by 6 hours on every update.

## How to store the NGED data locally

Let's keep our local data structure the same as when we were reading from CKAN: i.e. we'll have a separate Parquet for the substation metadata, and we'll store the power data in a Delta table. All our local will still be strictly controlled by our data contracts.

The new JSON data includes the start and end time of each power reading. Let's simplify this by only storing the end time of each recording, and making sure that we document that our data is period-ending.

### Storing metadata about each time series.

The workflow could be something like this:
1. When we first download the multi-year archive JSONs, we populate our local metadata Parquet
   files.
2. Then, when we download the updated data every 6 hours, we check the new metadata against our old metadata. If an update is necessary then go ahead and update our local metadata, and notify the user somehow. Let's assume that any updates to the metadata will be insignificant, and won't require any downstream actions (like re-training a model).

## Ultimate aims

We have to create forecasts for all these different types of assets. And we should probably switch
away from using `substation_number` as the main way we refer to timeseries, to using `TimeSeriesID`.

We will, ultimately, stop using CKAN. But let's not throw away the CKAN code just yet. Let's
deprecate it, and switch over to using the new S3 data as the default. But let's make it easy to
fall back to using CKAN if necessary. Eventually, once we're happy the S3 pipeline is stable, we'll
delete the CKAN code to keep our codebase as simple as possible.

## How to load the JSON data into Polars

The most elegant way I can find to process the JSON data in Polars goes like this:

```python
df = pl.read_json(filename)

# Load metadata
metadata_columns = [
    "TimeSeriesID",
    "TimeSeriesName",
    "TimeSeriesType",
    "Units",  # The physical unit, e.g. "MW" or "MVA"
    "LicenceArea",   # The DNO license area.
    "SubstationNumber",
    "SubstationType",  # e.g. "primary" or "BSP"
    "Latitude",
    "Longitude",
    "Information",
    "Area",
]

metadata = df[metadata_columns]

# Load time series
timeseries = (
    df["data"]
    .explode()
    .struct.unnest()
    .select(["value", "endTime"])
    .with_columns(endTime=pl.col("endTime").str.to_datetime(time_zone="UTC"))
)

# `timeseries` is now a DataFrame with two columns: `value` and `endTime`.
# The data is half-hourly.

```

## Next steps

Ask the @architect agent to create an implementation plan. The plan should include:

### Deprecating the existing CKAN code
- Rename `packages/nged_data` to `packages/nged_ckan_data`.
- Update any downstream code that refers to the `nged_data` package (e.g. the Dagster code in
  src/nged_substation_forecast/defs/nged_assets.py)
- Move the NGED dagster assets into a "NGED CKAN" group so the deprecated assets don't clutter the
  UI.
- In a month or so, we will fully remove the NGED CKAN code. But let's keep it as a backup for now.

### New code to import JSON data
- Create a new package, `packages/nged_json_data`. In this package, implement the following
  functionality:
- Create a new data contract for NGED's JSON data. (Please tell me if you think we should verify the raw JSON
  data. Or if we should convert the JSON data to a Polars DataFrame, and verify the DataFrame using
  a Patito data contract). Please look through all 33 JSON files in the example directory to find
  valid values for the metadata keys. (You only need to read the first 25 lines of each JSON file to
  find the metadata).
- Load an NGED JSON file into a Polars DataFrame. Extract the metadata and time series data.
- Write the time series data to our local Delta table, ensuring that we don't save any duplicate
  data, and that the data is sorted a free of nulls.
- Update the existing `SubstationMetadata` data contract to handle the metadata from the NGED JSON
  files. Note that I want to use `snake_case` for our internal data.
- If we don't have any metadata stored yet then create new local metadata using the metadata
  extracted from the JSON. If we do already have a local metadata store then check the new metadata
  values against what we already have stored. If they're the same then there's nothing to do. If
  they differ then update our local metadata and somehow alert the user.
- Implement checks for the data. The archival power data starts as all zeros, and then moves on to data with tiny values, before it moves onto correct data. We need to remove that bad data before appending it to our local Delta table. I do mean we need to *fully* remove the data. I don't want our Delta table to have years and years of nulls.

It's not 100% clear to me how we should partition the new NGED JSON asset. Ultimately, on S3, there will be one large "archival" JSON file per time series ID that we'll only need to download once (when we first initialise our system). Then, every 6 hours, there will be a "live" JSON file per time series ID. It'd be nice if we could have a Dagster partition that can be set to "grab the archives", or set to grab a live update. Note that the files that are updated every 6 hours will contain about 2 weeks of data.

During this design phase, you are allowed to write "test scripts" in exploration_scripts/. But don't
modify any other code yet!

Note that, in a week or so, we'll have access to live data on S3. We don't have access to live data
yet. So let's not worry about loading from S3 today. Let's instead focus on loading the example JSON
data we have on disk.

Show me the architect's plan, along with any clarifying questions the architect may have.

Once I'm happy with the plan then proceed with the `plan-complex-architecture` skill to ask other agents to
review the plan. Then pause again for my review of the final plan before implementation.
