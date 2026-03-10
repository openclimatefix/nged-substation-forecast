Let's significantly simplify the way that Dagster ingests new data from the NGED CKAN data portal.
We currently have an over-complicated system where we dynamically partition by substation (there are
over 800 substations), and have separate Dagster assets for downloading the CSV files, and
converting to Parquet.

Let's work through the following to-do list, step by step. Do NOT commit to git at the end of each
step, I will do that manually.

- [ ] don't partition by substation. (the ML assets won't partition by substation either)
- [ ] only partition the Dagster NGED download asset by *access* date. (We currently partition by the "modified
      date", which we get from CKAN. But this is over-complicated. Let's just partition by the day
      that we access the data. Let's use Dagster's DailyPartition.)
- [ ] don't save the NGED CSVs to disk. Collapse the "save CSV" and "save parquet" assets into one asset which just grabs the data from the network, and saves as Parquet. Ensure no duplication in the saved Parquet. If the saving fails then log the first few lines of the CSV.
- [ ] use a python ThreadPoolExecutor to download multiple NGED CSV files at once. Perhaps up to a
      max of 10 concurrent connections.
- [ ] Still tell Dagster about failures in individual substations, perhaps using Dagster's asset
      checks.
- [ ] The NGED downloading & conversion asset should accept an argument to only process a subset of substations (for the use-case where a subset have failed, and we want to re-run only those).
- [ ] The NGED downloading & conversion asset should still dynamically load the list of available
substations from NGED's CKAN portal when the asset starts up, similar to how the code works now, but
perhaps collapse this all into a single Dagster asset that gets the list of substations, downloads
the CSVs, and updates our Parquet files.
