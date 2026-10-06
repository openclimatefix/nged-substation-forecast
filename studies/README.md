# Studies

**Code in this directory is held to a lower standard than the rest of the repository, and it is
kept anyway because the findings it produced are cited elsewhere.** A study answers a question once.
The answer goes into `docs/`, and a reader who doubts the answer needs the code that produced it, so
the code stays where they can find and re-run it.

**"Study" rather than "experiment", because `experiment` already names a column.** `PowerForecast`
carries `experiment_name` and `ml_flow_experiment_id`, and the forecasts Delta table is partitioned
by `experiment_name`, so in this repository an experiment is one MLflow-tracked run of the
production pipeline. A study of whether a weather product's published field carries information is a
different kind of thing.

**A script under `studies/` may import from its own folder, from `studies.*` (`packages/studies`),
and from the other reviewed packages in `packages/*`.** `src/` and every package under `packages/`
except `packages/studies` must never import `studies` or a study script, because humans review that
code and the study code is fast-moving and agent-written. Code that two study folders need lives in
`packages/studies/src/studies/`. `packages/studies/tests/test_study_boundaries.py` enforces both
halves.

## What this tier promises, and what it does not

| | `packages/` and `src/` | `studies/` |
|---|---|---|
| Runs in production | yes | never |
| Has tests | yes | the machinery does, and `packages/studies/tests/<folder>/` tests some scripts; the study as a whole does not |
| Maintained as the repository changes | yes | no |
| Backwards compatibility | within reason | none |
| Linted by CI | yes | yes |
| Validated against a Patito contract | yes | no |

**The tested half is `packages/studies/`, and the split is deliberate.** A study's arms, charts and
write-up answer one question and are then done, so tests on them would have no second reader. The
machinery those arms call is different: it is used by every arm, it will be used by the next study,
and its failures are silent — a centred rolling window off by one step, a timestamp assigned to the
wrong hour, a site label derived two different ways. Code moves into `packages/studies/` when a
study has already got it wrong once, or when getting it wrong would produce a plausible-looking
number rather than an error.

Linting is the one row where both columns agree, because a script nobody can read is no more
auditable than a script nobody kept. Everything else is deliberately weaker.

**The rule that earns a study its place: a study whose findings reach `docs/` has to merge.** A page
on `main` citing a number that only an unmerged branch reproduces is an unverifiable claim. Merging
the code is what keeps the claim checkable, and it is the whole of the argument for this directory
existing.

**A study that produced nothing worth citing does not belong here.** Delete it, or leave it on a
branch. The directory is not an attic.

## What to expect when reading one

- **Nothing here is imported by production code.** No study touches a Patito contract or enters the
  Dagster asset graph, and nothing in `src/` or `packages/` imports one. A study that needs to do
  any of that has stopped being a study.
- **Each folder holds the scripts of one family of pages, and its README maps every script to the
  page it feeds.** A script runs with only its own folder on `sys.path`: it never reaches into
  another folder, and code that two folders share is in `packages/studies/src/studies/`.
- **Paths may have rotted.** Every study's data lives under `data/studies/`, in the directory
  `DATA_PATH_INTERNAL` names — the same variable `contracts.Settings` reads. Downloads sit in
  `downloads/`, filed by what the data is, so that a later study can reuse them; what one study
  builds from them sits in `<name>/`. [Where data lives](#where-data-lives) lists the folders. None
  of it is in version control, and a data directory that has been cleaned out will not refill
  itself. `data/NGED/` and `data/NWP/` are the pipeline's own, and a study reads them rather than
  writing to them.
- **A run command in a module docstring is the checked way to run that script.** Each one runs
  against the workspace environment, and names with `--with` only what the lockfile does not carry.
- **Read the study's own README first.** Each directory has one, covering what the study measured,
  what the arms are, and which readings the result does not support.

## Where data lives

**Every download is filed under `data/studies/downloads/` by what it is, never by the study that
first fetched it.** Each product has one folder. A product's frames cut or fetched at each site's
coordinates sit in a `site_points/` subfolder of that folder, beside the gridded download. The names
of the folders are constants in `packages/studies/src/studies/sources.py`, and scripts never spell a
folder name themselves.

| Folder under `data/studies/downloads/` | What it holds | Written by |
|---|---|---|
| `NWP/OPEN-METEO-PREVIOUS-RUNS/<model>/` | Open-Meteo's Previous Runs and historical-forecast downloads, one folder for each of 11 models; `site_points/` holds the frames at each site | `weather_downloads/fetch_open_meteo_previous_runs.py`, `weather_downloads/fetch_open_meteo_grid.py`, `past_weather/fetch_open_meteo_point.py`, `past_weather/fetch_wind_point.py` |
| `NWP/ECMWF-AIFS/`, `NWP/ECMWF-AIFS-ENS/`, `NWP/GEFS/`, `NWP/GFS/` | Dynamical.org's copies of ECMWF AIFS, ECMWF AIFS ensemble, GEFS, and GFS over the trial area | `weather_downloads/fetch_dynamical_zarr.py` |
| `NWP/windows/<name>/` | A download of one model over a bounded window of dates | `weather_downloads/fetch_dynamical_zarr.py`, `weather_downloads/fetch_open_meteo_previous_runs.py` |
| `NWP/ECMWF-IFS-SINGLE-RUNS/` | Open-Meteo's Single Runs archive of ECMWF's high-resolution forecast | `weather_downloads/fetch_open_meteo_single_runs.py` |
| `NWP/OPEN-METEO-ENSEMBLE-MEANS/` | Open-Meteo's ensemble-mean products | `weather_downloads/fetch_open_meteo_ensemble_means.py` |
| `NWP/WeatherNext3/` | The local copy of WeatherNext 3 over the trial area | `nwp_forecast_comparison/build_wn3_inputs.py` |
| `NWP/ENS_SITE_EXTRACT/` | The per-site extract of ECMWF ENS, built from `data/NWP`; the frames at each site are in `site_points/` | `beam_diffuse_split/fetch_ens_point.py`, `beam_diffuse_split/fetch_ens_point_wind.py` |
| `reanalysis/ERA5/`, `reanalysis/ERA5-WIND-2019-2023/` | ERA5 irradiance and wind from Open-Meteo's mirror and from the Copernicus Climate Data Store; `ERA5/site_points/` holds the frames at each site | `beam_diffuse_split/fetch_era5.py`, `beam_diffuse_split/fetch_era5_open_meteo.py`, `weather_downloads/fetch_era5_wind.py`, `weather_downloads/fetch_era5_wind_2019_2023.py` |
| `reanalysis/CAMS/` | The CAMS radiation service's satellite retrieval at each site, and its yearly CSV downloads | `beam_diffuse_split/fetch_cams.py` |
| `reanalysis/CERRA/` | The CERRA regional reanalysis | `weather_downloads/fetch_cerra.py`, `weather_downloads/fetch_cerra_grid.py` |
| `reanalysis/NORA3/`, `reanalysis/NORA3_10m/` | The NORA3 reanalysis wind | `weather_downloads/fetch_nora3.py` |
| `reanalysis/ICON-DREAM-EU/` | The ICON-DREAM-EU reanalysis; `site_points/` holds the frame at each site | `weather_downloads/fetch_icon_dream.py`, `past_weather/extract_site_series.py` |
| `observations/MIDAS-OPEN/` | The Met Office's MIDAS Open station observations | `weather_downloads/fetch_midas_open.py` |
| `observations/SARAH-3/` | The SARAH-3 satellite retrieval, ordered by hand from CM SAF; `site_points/` holds the frame at each site | `past_weather/extract_site_series.py` |
| `observations/NGED-ANM/` | NGED's active network management setpoint exports, and the export-cap parquet derived from each | `beam_diffuse_split/anm_setpoints.py` (the exports come from NGED) |

Every other folder directly under `data/studies/`, apart from `per_study/`, is one study's own
inputs and results. The `UKV-CEDA*` stores and the trial-area box are in `data/studies/weather/`.

**Each study that has moved keeps one folder under `data/studies/per_study/`.**

| Folder under `data/studies/per_study/` | What it holds | Written by |
|---|---|---|
| `cerra_wind/<study>/` | The four CERRA wind studies: `direction`, `levels`, `levels_post_hoc`, and `shear` | `past_weather/cerra_wind_direction.py`, `past_weather/cerra_wind_levels.py` |
| `nwp_forecast_comparison/original/` | The published fit of the NWP forecast comparison | `nwp_forecast_comparison/nwp_forecast_comparison.py` |
| `nwp_forecast_comparison/<batch>/` | One folder for each of the 22 later batches of fits, such as `aifs_blends`, `leads_day10`, and `product_blends`; a batch's `superseded/` folder holds its earlier outputs | the `build_*.py` and `fit_*.py` scripts of `nwp_forecast_comparison/` |

## The studies

| Directory | Question it answered | Where the answer lives |
|---|---|---|
| `beam_diffuse_split/` | Does a weather product's own beam/diffuse split carry information a PV forecast can use, beyond the global horizontal irradiance alone? | [Does a weather product's beam/diffuse split help a PV forecast?](https://openclimatefix.github.io/nged-substation-forecast/studies/beam-diffuse-split/) |
| `past_weather/` | Which weather product best describes past sunshine and past wind, and does blending products beat the best single product? | The pages under [Past weather](https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/); the folder README maps each script to its page |
| `nwp_forecast_comparison/` | Which forecast product, or which blend of products, gives the most accurate power forecast at the day-ahead lead the live service delivers, and at the days around it? How accurate is an ECMWF ENS-driven forecast at each horizon? Does adding the Met Office's UKV, read from the CEDA archive, to the ECMWF ENS mean lower the error? | The Forecasts pages listed on the [studies index](https://openclimatefix.github.io/nged-substation-forecast/studies/); the folder README maps each script to its page |
| `open_meteo_ensemble_means/` | How well do Open-Meteo's ensemble-mean products for MOGREPS-UK, ICON-D2-EPS, ICON-EU-EPS, and ECMWF IFS ENS predict solar and wind power, beside CAMS and ERA5? | [How do Open-Meteo's ensemble-mean products compare for solar and wind power?](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ensemble-means/) |
| `weather_downloads/` | Which weather products can the studies download, and what does each download hold? | No page: the downloads feed the pages above, and the folder README describes each fetch script |
| `ens_backfill_pilot/` | Can ECMWF's control-member forecasts for 2021-03-21 to 2024-03-31 be rebuilt from the GRIB files that Dynamical.org stages on Source Cooperative? | No page: `studies/ens_backfill_pilot/report.md` holds the pilot's result |
| `era_fold_design/` | How many scored hours have a calendar month held out of every training row under the era folds, and how do planned contrasts move when the folds cover every month? | No page: `studies/era_fold_design/report.md` holds the measurement behind the plan in PR #906 |
