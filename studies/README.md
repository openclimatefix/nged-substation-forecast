# Studies

**Code in this directory is held to a lower standard than the rest of the repository, and it is kept
anyway because the findings it produced are cited elsewhere.** A study answers a question once. The
answer goes into `docs/`, and a reader who doubts the answer needs the code that produced it, so the
code stays where they can find and re-run it.

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

## A study's leaderboard number comes only from `score_study.py`

**A study that reports a forecast skill number hands a predictions file to
`scripts/forecasting/score_study.py`, and the `metrics` asset produces the number.** The script
takes a file of `PowerForecast` rows, a study name, and a leaderboard fold. The script stores the
rows in `power_forecasts` under the experiment name `study/<study name>`, and scores them in
leaderboard scope. Before the script writes anything, it refuses a file whose `(time_series_id,
power_fcst_init_time, valid_time)` keys differ from the keys of the reference experiment that
`conf/cv/default.yaml` names for the same fold, or whose rows carry more than one
`power_fcst_model_name`. A study therefore cannot raise its score by leaving out the rows it
forecasts worst. The `metrics` asset repeats both checks when it scores the study. Every leaderboard
skill number on a study page must trace to a `forecast_metrics` row. The `study/` prefix keeps a
study's forecasts out of the promotion candidates, and lets a reader filter the study experiments
out of the leaderboard.

**A study reads observed power through `studies.power.scan_power`.** The function returns cleaned
power before `final_test_start` in `conf/cv/default.yaml`, the date from which the `metrics` asset
refuses to score unless the maintainer sets `NGED_FINAL_TEST=1`.

## What to expect when reading one

- **Nothing here is imported by production code.** No study touches a Patito contract or enters the
  Dagster asset graph, and nothing in `src/` or `packages/` imports one. A study that needs to do
  any of that has stopped being a study. The one route from a study to the leaderboard is
  `scripts/forecasting/score_study.py`, described above.
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
| `NWP/UKV-CEDA/`, `NWP/UKV-CEDA-part2/`, `NWP/UKV-CEDA-part3/` | Three Icechunk stores of the Met Office's UKV archive on CEDA, each holding the 00, 06, 12, and 18 UTC runs to 54 hours, kept as three stores and never merged | `weather_downloads/fetch_ukv_ceda.py` |
| `NWP/UKV-CEDA-T120/` | The Icechunk store of the 03 and 15 UTC runs of the UKV archive on CEDA, to 120 hours | `weather_downloads/fetch_ukv_ceda.py --product ukv-ceda-t120` |
| `NWP/ENS_SITE_EXTRACT/` | The extracts of ECMWF ENS built from `data/NWP`: the frames at each site in `site_points/` and the per-member extract `ens_members.parquet` with its day-4 supplement `ens_members_day4.parquet` | `beam_diffuse_split/fetch_ens_point.py`, `beam_diffuse_split/fetch_ens_point_wind.py`, `nwp_forecast_comparison/fetch_ens_forecast_horizons.py`, `nwp_forecast_comparison/fetch_ens_day4_supplement.py` |
| `reanalysis/ERA5/`, `reanalysis/ERA5-WIND-2019-2023/` | ERA5 irradiance and wind from Open-Meteo's mirror and from the Copernicus Climate Data Store; `ERA5/site_points/` holds the frames at each site | `beam_diffuse_split/fetch_era5.py`, `beam_diffuse_split/fetch_era5_open_meteo.py`, `weather_downloads/fetch_era5_wind.py`, `weather_downloads/fetch_era5_wind_2019_2023.py` |
| `reanalysis/CAMS/` | The CAMS radiation service's satellite retrieval at each site, and its yearly CSV downloads | `beam_diffuse_split/fetch_cams.py` |
| `reanalysis/CERRA/` | The CERRA regional reanalysis | `weather_downloads/fetch_cerra.py`, `weather_downloads/fetch_cerra_grid.py` |
| `reanalysis/NORA3/`, `reanalysis/NORA3_10m/` | The NORA3 reanalysis wind | `weather_downloads/fetch_nora3.py` |
| `reanalysis/ICON-DREAM-EU/` | The ICON-DREAM-EU reanalysis; `site_points/` holds the frame at each site | `weather_downloads/fetch_icon_dream.py`, `past_weather/extract_site_series.py` |
| `observations/MIDAS-OPEN/` | The Met Office's MIDAS Open station observations | `weather_downloads/fetch_midas_open.py` |
| `observations/SARAH-3/` | The SARAH-3 satellite retrieval, ordered by hand from CM SAF; `site_points/` holds the frame at each site | `past_weather/extract_site_series.py` |
| `observations/NGED-ANM/` | NGED's active network management setpoint exports, and the export-cap parquet derived from each | `beam_diffuse_split/anm_setpoints.py` (the exports come from NGED) |
| `market/<source>/` | GB electricity prices (NESO N2EX day-ahead, Elexon system prices, Elexon APX market index), the national Carbon Intensity series, the Elexon BMU register with its storage-candidate list, and the bid-offer acceptance volumes, cashflows, and levels of the listed BMUs (`fetch_gb_prices.py`, `fetch_bmu_dispatch.py`); the physical notifications and maximum export and import limits of the listed BMUs (`fetch_bmu_notifications.py`); NESO's Enduring Auction Capability (EAC) results for response and reserve, with a table linking auction units to BMUs (`fetch_neso_eac.py`); and the bid-offer prices of the listed BMUs with system-wide series — frequency, demand outturn, generation by fuel type, demand and wind forecasts, balancing services adjustments, loss of load probability and de-rated margin, and system warnings (`fetch_system_series.py`); Elexon's indicated demand and generation sums of the final physical notifications for the national total and 17 boundaries (`fetch_system_series.py --sources elexon_inddem elexon_indgen`); the settled half-hourly energy each of the 14 GSP groups takes from the transmission system, from Elexon's Open Settlement Data (`fetch_agv.py`); and Sheffield Solar's PV_Live solar generation for NGED's four licence areas (`fetch_pv_live.py`); each folder holds a `README.md` and a `lineage.json` | `market_downloads/fetch_gb_prices.py`, `market_downloads/fetch_bmu_dispatch.py`, `market_downloads/fetch_bmu_notifications.py`, `market_downloads/fetch_neso_eac.py`, `market_downloads/fetch_system_series.py`, `market_downloads/fetch_agv.py`, `market_downloads/fetch_pv_live.py` |

`data/studies/_private/trial_area_box.json` holds the trial-area box, derived from the private list
of generators.

**Each study keeps one folder under `data/studies/per_study/`.**

| Folder under `data/studies/per_study/` | What it holds | Written by |
|---|---|---|
| `cerra_wind/<study>/` | The four CERRA wind studies: `direction`, `levels`, `levels_post_hoc`, and `shear` | `past_weather/cerra_wind_direction.py`, `past_weather/cerra_wind_levels.py` |
| `beam_diffuse_split/` | The beam/diffuse study's results and figures; `inputs/` holds the 13 joined `beam_diffuse_dataset_<source>.parquet` frames, and `past_weather_v2/` holds the past-weather studies' results | `beam_diffuse_split/build_dataset.py`, the `run_*.py` and `report_results.py` scripts, `past_weather/*.py` |
| `ens_forecast_horizons/` | The ENS horizons study's results; `era_covered/` holds the re-run on rows from 2024-12-01 | `nwp_forecast_comparison/ens_forecast_horizons.py` |
| `open_meteo_ensemble_means/`, `open_meteo_ens_gap/` | The Open-Meteo ensemble-means study, and its comparison of a local ensemble with ENS | `open_meteo_ensemble_means/ensemble_means_mae.py`, `open_meteo_ensemble_means/local_ens_gap.py` |
| `icon_eu_compare/`, `era5_wind_compare/` | The ICON-EU comparison of Dynamical.org with Open-Meteo, and the ERA5 wind comparison of Open-Meteo with the Climate Data Store | `weather_downloads/compare_icon_eu_dynamical_openmeteo.py`, `weather_downloads/compare_era5_wind_openmeteo_cds.py` |
| `ens_backfill_pilot/` | The checkpoint files of the ENS backfill pilot | `ens_backfill_pilot/fetch_pilot.py` |
| `ukv_ceda_blends/` | The UKV-on-CEDA blends study's inputs and results; `run15/` holds the inputs and results of the run on the 15 UTC cycle | `nwp_forecast_comparison/build_ukv_ceda_inputs.py`, `nwp_forecast_comparison/fit_ukv_ceda_blends.py` |
| `ukv_ceda_vs_openmeteo/` | The study of UKV from CEDA against UKV from Open-Meteo: the built frames, the model-free comparison, the per-row losses, the reports, and `superseded/` for earlier outputs. The published page is [Do CEDA's and Open-Meteo's archives of UKV give the same power forecasts?](https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/ukv-ceda-vs-openmeteo/) | `past_weather/ukv_ceda_vs_openmeteo_build.py`, `_compare.py`, `_wind_steps.py`, `_extra_reads.py`, `_fit.py`, and `_charts.py` |
| `solar_bmu_census/` | The solar-BMU census: `inputs/` holds the Elexon, NESO, and REPD downloads with their lineage note, `classes.parquet` and `solar_bmus.parquet` hold the classes and the census table, and `report.md` holds the numbers the page quotes. | `solar_bmu_census/fetch_sources.py`, `classify.py`, `collate.py`, `recall_check.py`, `report.py` |
| `nwp_forecast_comparison/original/` | The published fit of the NWP forecast comparison | `nwp_forecast_comparison/nwp_forecast_comparison.py` |
| `nwp_forecast_comparison/<batch>/` | One folder for each of the 22 later batches of fits, such as `aifs_blends`, `leads_day10`, and `product_blends`; a batch's `superseded/` folder holds its earlier outputs | the `build_*.py` and `fit_*.py` scripts of `nwp_forecast_comparison/` |

## The studies

| Directory | Question it answered | Where the answer lives |
|---|---|---|
| `beam_diffuse_split/` | Does a weather product's own beam/diffuse split carry information a PV forecast can use, beyond the global horizontal irradiance alone? | [Does a weather product's beam/diffuse split help a PV forecast?](https://openclimatefix.github.io/nged-substation-forecast/studies/beam-diffuse-split/) |
| `past_weather/` | Which weather product best describes past sunshine and past wind, and does blending products beat the best single product? | The pages under [Past weather](https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/); the folder README maps each script to its page |
| `nwp_forecast_comparison/` | Which forecast product, or which blend of products, gives the most accurate power forecast at the day-ahead lead the live service delivers, and at the days around it? How accurate is an ECMWF ENS-driven forecast at each horizon? Does adding the Met Office's UKV, read from the CEDA archive, to the ECMWF ENS mean lower the error? | The Forecasts pages listed on the [studies index](https://openclimatefix.github.io/nged-substation-forecast/studies/); the folder README maps each script to its page |
| `open_meteo_ensemble_means/` | How well do Open-Meteo's ensemble-mean products for MOGREPS-UK, ICON-D2-EPS, ICON-EU-EPS, and ECMWF IFS ENS predict solar and wind power, beside CAMS and ERA5? | [How do Open-Meteo's ensemble-mean products compare for solar and wind power?](https://openclimatefix.github.io/nged-substation-forecast/studies/forecasts/ensemble-means/) |
| `solar_bmu_census/` | How many Balancing Mechanism Units in Great Britain are solar, and what does each published capacity figure give them? | [How many Balancing Mechanism Units in Great Britain are solar?](https://openclimatefix.github.io/nged-substation-forecast/studies/solar-bmu-census/) |
| `weather_downloads/` | Which weather products can the studies download, and what does each download hold? | No page: the downloads feed the pages above, and the folder README describes each fetch script |
| `ens_backfill_pilot/` | Can ECMWF's control-member forecasts for 2021-03-21 to 2024-03-31 be rebuilt from the GRIB files that Dynamical.org stages on Source Cooperative? | No page: `studies/ens_backfill_pilot/report.md` holds the pilot's result |
| `era_fold_design/` | How many scored hours have a calendar month held out of every training row under the era folds, and how do planned contrasts move when the folds cover every month? | No page: `studies/era_fold_design/report.md` holds the measurement behind the plan in PR #906 |
