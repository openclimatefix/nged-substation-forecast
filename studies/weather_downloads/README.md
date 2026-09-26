# Throwaway downloads: the weather products for the past-weather and forecast studies

Everything in this directory is a one-off. It is not part of the Dagster asset graph, nothing else
in the repo imports it, it adds no package and it changes no data contract. It downloads the weather
products the fact-checked survey (`docs/background/weather-products-survey.md`) ranks for the two
past-weather studies (#809) and the forecast study (#810), recorded in
[issue 841](https://github.com/openclimatefix/nged-substation-forecast/issues/841). This directory
covers the downloads only; #809 and #810 read what lands under `data/studies/weather/<PRODUCT>/`
and are out of scope here.

## The trial-area box

**Every gridded product is cut to a box covering the NGED trial area, with a margin of a few grid
cells, taken from the private generator roster.** `paths.write_trial_area_box_from_roster` derives
the box once from `TimeSeriesMetadata` and writes it to
`data/studies/weather/_trial_area_box.json`, a file under the gitignored `data/` tree that is never
read outside this process's own working state. `paths.load_trial_area_box` reads it back as a
`TrialAreaBox`, held in memory only. **Generator locations must never appear in anything
published**: not a request URL, not a log line, not a filename, not this repository's git history.
Every fetch script keys its output on an integer index (`point_id`, `cell_id`, `y_index`/`x_index`)
rather than a coordinate.

## Lineage notes

`lineage.write_lineage_note` writes a lineage JSON file into each product directory
(`data/studies/weather/<PRODUCT>/lineage.json`, or `lineage_<variable>.json` where a script fetches
several variables into one directory, as `fetch_cerra.py` and `fetch_icon_dream.py` do). Each note
records the source address, what was requested, the variables kept, and the retrieval time — the one
format every fetch script here reuses, per `studies/beam_diffuse_split/sources.py`'s convention.

## Running a fetch script

Each script in this directory is one product (or a small family of related products served the same
way):

- `fetch_open_meteo_grid.py` — ECMWF IFS HRES 9 km, DMI and KNMI HARMONIE-AROME, and Meteo-France
  ARPEGE Europe, all served the same way by Open-Meteo's historical-forecast API.
- `fetch_open_meteo_previous_runs.py` — 11 models' Previous Runs (lead days 0 to 7) at the nine
  anonymised sites, from Open-Meteo's Previous Runs API.
- `fetch_open_meteo_ensemble_means.py` and `validate_open_meteo_ensemble_means.py` — the ensemble mean
  and spread of MOGREPS-UK (`ukmo_uk_ensemble_mean_2km`), ICON-D2-EPS (`dwd_icon_d2_eps_ensemble_mean`),
  ICON-EU-EPS (`dwd_icon_eu_eps_ensemble_mean`), and ECMWF IFS ENS at 0.25 degrees
  (`ecmwf_ifs025_ensemble_mean`), at the nine anonymised sites, from Open-Meteo's Ensemble API. All
  four are served as one stitched series (no `init_time`) from 2026-06-25, measured on 2026-09-26
  with `--probe-first-date`, so a later re-run may find a moved start. The plain `icon_d2_eps` and
  `icon_eu_eps` models serve members only. The 100 m wind is all null for MOGREPS-UK and ICON-EU-EPS.
  The Historical Forecast and Previous Runs APIs serve the MOGREPS-UK mean as all null, and every
  Single Runs request for the mean models returned "run not available". The full fetch is about 504
  weighted calls (4 products x 7 windows x 9 sites x 2).
- `fetch_cerra.py` — CERRA solar radiation and wind, from the Copernicus Climate Data Store, needs
  `uv run --with cdsapi --with netCDF4`.
- `fetch_era5_wind.py` — native ERA5 10 m and 100 m wind from the Climate Data Store at a 3 x 3 block
  of cells around each of the three wind sites, needs `uv run --with cdsapi --with netCDF4`. The
  data on disk covers only 2024-01-01 to 2026-09-20 (the newest 6 chunks, about 2.7 years, fetched
  with `--chunks 6`), because the comparison with Open-Meteo's ERA5 wind needs only a couple of years.
  Running without `--chunks` fetches the older chunks back to 2019-09; cached chunks are skipped,
  so extending past 2026-09-20 means deleting `era5_wind_2026_07_09.zip` first.
  `compare_era5_wind_openmeteo_cds.py` is that comparison.
- `fetch_midas_open.py` and `validate_midas_open.py` — Met Office MIDAS Open station observations.
- `fetch_icon_dream.py` — DWD's ICON-DREAM-EU, whole-domain monthly GRIB cropped to the box then
  deleted, needs `uv run --with cfgrib --with eccodes --with requests`.
- `fetch_nora3.py` and `validate_nora3.py` — NORA3 hourly wind at 50 m and 100 m over OPeNDAP, cut
  server-side to the box, from the aggregated dataset and then MET Norway's monthly files, needs
  `uv run --with pydap`.

Every script resolves `data/` the way `sources.REPO_DATA_DIR` does — the main checkout's `data/`,
shared by every worktree, not a per-worktree copy — so run each script once, from whichever worktree
is doing the download, and every other worktree sees the result.

## Not covered here

**WeatherNext 3** has no fetch script: whether a colleague's existing archive can be reused instead
of a fresh access request is a decision for the maintainer.
