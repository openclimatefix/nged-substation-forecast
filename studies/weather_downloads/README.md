# Throwaway downloads: the weather products for the past-weather and forecast studies

Everything in this directory is a one-off. It is not part of the Dagster asset graph, nothing else
in the repo imports it, it adds no package and it changes no data contract. It downloads the weather
products the fact-checked survey (`docs/background/weather-products-survey.md`) ranks for the two
past-weather studies (#809) and the forecast study (#810), recorded in [issue
841](https://github.com/openclimatefix/nged-substation-forecast/issues/841). This directory covers
the downloads only; #809 and #810 read what lands under `data/studies/weather/<PRODUCT>/` and are
out of scope here.

## The trial-area box

**Every gridded product is cut to a box covering the NGED trial area, with a margin of a few grid
cells, taken from the private generator roster.**
`studies.trial_area.write_trial_area_box_from_roster` derives the box once from `TimeSeriesMetadata`
and writes it to `data/studies/weather/_trial_area_box.json`, a file under the gitignored `data/`
tree that is never read outside this process's own working state.
`studies.trial_area.load_trial_area_box` reads it back as a `TrialAreaBox`, held in memory only.
**Generator locations must never appear in anything published**: not a request URL, not a log line,
not a filename, not this repository's git history. Every fetch script keys its output on an integer
index (`point_id`, `cell_id`, `y_index`/`x_index`) rather than a coordinate.

## Lineage notes

`lineage.write_lineage_note` writes a lineage JSON file into each product directory
(`data/studies/weather/<PRODUCT>/lineage.json`, or `lineage_<variable>.json` where a script fetches
several variables into one directory, as `fetch_cerra.py` and `fetch_icon_dream.py` do). Each note
records the source address, what was requested, the variables kept, and the retrieval time — the one
format every fetch script here reuses, per `packages/studies/src/studies/sources.py`'s convention.

## Running a fetch script

Each script in this directory is one product (or a small family of related products served the same
way):

- `fetch_open_meteo_grid.py` — ECMWF IFS HRES 9 km, DMI and KNMI HARMONIE-AROME, and Meteo-France
  ARPEGE Europe, all served the same way by Open-Meteo's historical-forecast API.
- `fetch_open_meteo_previous_runs.py` — 11 models' Previous Runs (lead days 0 to 7) at the nine
  anonymised sites, from Open-Meteo's Previous Runs API.
- `fetch_open_meteo_ensemble_means.py` and `validate_open_meteo_ensemble_means.py` — the ensemble
  mean and spread of MOGREPS-UK (`ukmo_uk_ensemble_mean_2km`), ICON-D2-EPS
  (`dwd_icon_d2_eps_ensemble_mean`), ICON-EU-EPS (`dwd_icon_eu_eps_ensemble_mean`), and ECMWF IFS
  ENS at 0.25 degrees (`ecmwf_ifs025_ensemble_mean`), at the nine anonymised sites, from
  Open-Meteo's Ensemble API. All four are served as one stitched series (no `init_time`) from
  2026-06-25, measured on 2026-09-26 with `--probe-first-date`, so a later re-run may find a moved
  start. The plain `icon_d2_eps` and `icon_eu_eps` models serve members only. The 100 m wind is all
  null for MOGREPS-UK and ICON-EU-EPS. The Historical Forecast and Previous Runs APIs serve the
  MOGREPS-UK mean as all null, and every Single Runs request for the mean models returned "run not
  available". The full fetch is about 504 weighted calls (4 products x 7 windows x 9 sites x 2).
- `fetch_cerra.py` — CERRA solar radiation and wind speed (10 m from the single-levels dataset, 50,
  75, 100, and 150 m from the height-levels dataset), from the Copernicus Climate Data Store, needs
  `uv run --with cdsapi --with netCDF4`. The flag `--wind-direction` fetches only wind direction at
  the same five heights, into files named `wind_direction_<height>_m.parquet` and
  `10m_wind_direction_surface.parquet`. `fetch_cerra_grid.py` writes the whole-domain latitude and
  longitude of every grid cell, which is how the cropped files' `y_index` and `x_index` map to a
  location.
- `fetch_era5_wind.py` — native ERA5 10 m and 100 m wind from the Climate Data Store at a 3 x 3
  block of cells around each of the three wind sites, needs `uv run --with cdsapi --with netCDF4`.
  The data on disk covers only 2024-01-01 to 2026-09-20 (the newest 6 chunks, about 2.7 years,
  fetched with `--chunks 6`), because the comparison with Open-Meteo's ERA5 wind needs only a couple
  of years. Running without `--chunks` fetches the older chunks back to 2019-09; cached chunks are
  skipped, so extending past 2026-09-20 means deleting `era5_wind_2026_07_09.zip` first.
  `compare_era5_wind_openmeteo_cds.py` is that comparison.
- `fetch_midas_open.py` and `validate_midas_open.py` — Met Office MIDAS Open station observations.
- `fetch_icon_dream.py` — DWD's ICON-DREAM-EU, whole-domain monthly GRIB cropped to the box then
  deleted, needs `uv run --with cfgrib --with eccodes --with requests`.
- `fetch_nora3.py` and `validate_nora3.py` — NORA3 hourly wind at 50 m and 100 m over OPeNDAP, cut
  server-side to the box, from the aggregated dataset and then MET Norway's monthly files, needs `uv
  run --with pydap`. The flag `--height-10m` fetches the 10 m level instead, into its own folder
  `NORA3_10m/`, so the two sets never share a month cache.
- `fetch_ukv_ceda.py` and `validate_ukv_ceda.py` — the Met Office UKV 2 km archive held at CEDA
  (four runs a day from 2019-09-01), whole GRIB files cropped to the box and written to a local
  Icechunk store, one commit per run, needs `uv run --with icechunk --with zarr --with eccodes` and
  the `CEDA_TOKEN` environment variable. The licence is CC BY-NC-SA 4.0. The flag `--product
  ukv-ceda-t120` archives the 03 and 15 UTC runs (leads 0 to 120 hours) into its own store
  `UKV-CEDA-T120`, and `--newest-first` works from the newest run back to `--start`. A crashed
  download can leave a `.partial` file in the product's `_scratch/` directory, which is safe to
  delete.
- `fetch_weathernext3.py` and `validate_weathernext3.py` — WeatherNext 3 ensemble-mean runs (00, 06,
  12, and 18 UTC) from a Google Cloud Storage bucket that is not Requester Pays, cropped to a wide
  United Kingdom box (49.0 to 61.5 degrees north, 10.0 degrees west to 3.5 degrees east, which is
  public and unrelated to the private trial-area box) and written to an Icechunk store. The Zarr
  chunks are whole-globe, so a run reads about 50 GB and keeps about 15 MB. Run the fetch only on a
  Compute Engine machine in us-east1, because reads from elsewhere may be billed as egress
  (`--dry-run` prints the estimate first). No billing project is needed, but the bucket cannot be
  read anonymously: access must be requested from Google (see its [access
  guide](https://developers.google.com/weathernext/guides/access-forecast)), and the script then
  reads with the Google account's application default credentials. The [survey
  page](https://openclimatefix.github.io/nged-substation-forecast/background/weather-products-survey/#what-we-learnt-about-weathernexts-precomputed-statistics-store)
  describes the bucket and the statistics it holds beyond the mean.
    - **Arguments.** `--bucket` (a Cloud Storage bucket in us-east1) or `--local-store` (a
      directory, for tests) names the output; exactly one is required. `--start-date` and
      `--end-date` give the window, and `--end-date` is required when the repository is created,
      because it fixes the end of the `init_time` axis for good. Later invocations clip their window
      to that stored range. `--init-hours` (any of `0 6 12 18`, default all four), `--workers`
      (default 16), and `--max-external-gb` (default 5.0) complete the arguments. A test of one 00
      UTC run from outside us-east1 needs `--max-external-gb 60` (about $6 of egress). The validator
      then needs the same `--start-date`, `--end-date` and `--init-hours 0`, or its check that
      skipped runs cover every unwritten slot fails.
    - **Outputs.** One Icechunk repository with seven `Float32` arrays (one per variable, dimensions
      `(init_time, lead_time, latitude, longitude)`, one shard per run), the arrays `run_written`
      and `source_init_time`, and the lineage and skipped runs in the group attributes. The fetch
      makes one commit per run on the `staging` branch and never moves `main`. Re-running the fetch
      skips every run already written.
    - **Validating and publishing.** `validate_weathernext3.py` checks `staging` and prints one PASS
      or FAIL line per check. `--compare-source` also compares sampled runs with the source store,
      and `--publish` moves `main` to the validated snapshot. Run it in us-east1 as well, because it
      reads every shard.

Every script resolves `data/` the way `sources.REPO_DATA_DIR` does — the main checkout's `data/`,
shared by every worktree, not a per-worktree copy — so run each script once, from whichever worktree
is doing the download, and every other worktree sees the result.
