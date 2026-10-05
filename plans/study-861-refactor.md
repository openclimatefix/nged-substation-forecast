# Plan: give each study family its own folder and move shared study code into `packages/studies` (#861)

**The problem is that `studies/beam_diffuse_split/` holds 79 scripts for about ten different study
pages, and the scripts reach each other, and other folders, by bare imports.** The issue counted 44
scripts; `main` now holds 79 in that folder, 128 under `studies/` in all (excluding
`era_fold_design/scripts`), and 92,483 lines. An AST scan of the imports found 25 pairs of
(importing folder, imported module) that cross a folder boundary once the scripts are sorted by the
folder map below. Ten scripts call `sys.path.insert`, and nine of those hide a crossing, which
`ruff` and `ty` cannot check (`stamp_alignment.py` inserts its own folder). One more script,
`ens_hres_past_wind.py`, loads `weather_downloads/paths.py` with `spec_from_file_location`, which no
import scan sees. Published pages cite script paths on 81 lines in 11 files in `docs/`, 58 of them
naming a script that moves, not the 30 the issue estimated.

**The plan is to sort the scripts into folders that mirror the docs page families, and to make the
crossings impossible rather than merely rare.** A script may import only from its own folder and
from `studies.*` (`packages/studies`). Everything a second folder needs moves into the package, in
two layers. One AST test, added in the last commit, enforces the rule over `studies/` and over
`packages/studies/src`. Tests of study scripts move to one tree. The study data under `data/studies/`
is reorganised in a separate, gated sequence of renames after the code lands (see "Data"). The
general Fractions Skill Score and paired block bootstrap are not extracted here (issues #805 and #808):
they wait for their production callers.

## Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.** The preconditions (#858, #859)
have merged. This plan is written against `main`. Branch `study-1016-ukv-blends` (#1016) is running
a study and is unmerged. It edits `nwp_forecast_comparison/fit_aifs.py`,
`nwp_forecast_comparison.py` and `packages/studies/src/studies/ifs_single_runs.py`, and adds
`studies/ukv_ceda_blends/` (five scripts that reach `nwp_forecast_comparison/`,
`beam_diffuse_split/` and `weather_downloads/` through `sys.path` inserts). The implementer starts
only after #1016 merges, so that the work is not rebased over a moving folder, and re-runs the
import scan in step 0 on the merged `main`. The scan of the branch shows what it adds: about 59
layer-2 symbols, 50 of them from `ukv_ceda_blends/` into `nwp_forecast_comparison/` (from `fit_aifs`
21, `nwp_forecast_comparison` 15, `nwp_forecast_charts` 12, `fit_extra_leads` 2,
`weather_downloads/fetch_ukv_ceda` 7, and `paths`). The folder map below therefore folds
`ukv_ceda_blends/` into `nwp_forecast_comparison/`, and the layer-2 re-scan after the fold is a
named step.

**Size: complex.** The five triggers:

- **What gets stored:** no Patito model, Delta table or Dagster asset changes. The study files under
  `data/studies/` do change path, by rename on one device with every file hashed before and after.
  The one Dagster-adjacent step is step G, which deletes a stale copy of the Dagster tables
  (`data.old`) after a gate and the maintainer's go-ahead.
- **Production serving path:** no. Nothing in `src/` or `packages/` imports `studies` today, and a
  new test pins that.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The folder grouping, the number of extraction layers,
  the test location, and the #805/#808 boundary each have a defensible alternative.
- **Callers not nameable without searching:** yes. 128 scripts, 49 files in `tests/` (17 of them
  test scripts) and 41 in `packages/studies/tests/`, and 81 docs lines.

Either of those two triggers alone makes the issue complex: it gets this plan and both plan reviews
(run by the caller, not by this planner), and both diff reviews. The maintainer asked for one Opus
diff review; this plan keeps that review and adds the mutation pass the study skill already requires
whenever `packages/studies` changes, limited to the tests this work adds.

**Departures from the issue body:**

- **Folders follow the docs page families, not one folder per page.** The issue's five folders
  become two new ones. Layer 2 stays small because pages in one family share helpers freely. Each
  folder README carries a table mapping every script to the page it feeds.
- **"Move the shared plumbing" is done in two layers, not one.** Layer 1 is code every study uses.
  Layer 2 is the small set of helpers that remain shared across the new folders.
- **Tests go to `packages/studies/tests/<study>/`, not into each study folder.**
- **#805 and #808's extraction is deferred**, with a guard test now.

## The folder map

**Two new folders mirror the docs.** `past_weather/` mirrors `docs/studies/past-weather/`, and
`nwp_forecast_comparison/` already mirrors the Forecasts pages. Existing folders keep their names:
`beam_diffuse_split/`, `era_fold_design/`, `weather_downloads/`, `open_meteo_ensemble_means/` and
`ens_backfill_pilot/`. The one exception is `ukv_ceda_blends/` (from #1016), which folds into
`nwp_forecast_comparison/` because it feeds a Forecasts page and the rule is that folders mirror
page families. Its five scripts, its tests (`packages/studies/tests/test_ukv_ceda_*.py`, which
use `sys.path.insert` today) and its `extra-paths` line in `pyproject.toml` move with it, and its
`sys.path.insert` calls are removed.

| Folder | Scripts |
|---|---|
| `past_weather/` (38) | Solar (15): `weather_products`\*\*, `cerra_past_solar`, `ens_past_solar`, `ens_past_solar_charts`, `station_past_solar`, `station_past_solar_charts`, `past_solar_leaderboard`, `past_solar_leaderboard_charts`, `weather_product_charts`, `check_new_products`, `extract_site_series`, `verify_icon_lineage`, `verify_ukv_lineage`, `check_station_page_numbers`, `fetch_open_meteo_point`\*. Wind (14): `wind_products`, `fetch_wind_point`, `wind_product_charts`, `wind_icon_dream`, `wind_icon_dream_charts`, `station_wind_arms`, `station_wind_arms_charts`, `ens_hres_past_wind`, `ens_hres_past_wind_charts`, `reanalysis_past_wind`, `reanalysis_past_wind_charts`, `past_wind_leaderboard`, `past_wind_leaderboard_charts`, `check_served_wind`. CERRA wind (5): `cerra_wind_levels`, `cerra_wind_levels_charts`, `cerra_wind_levels_shear`, `cerra_wind_direction`, `cerra_wind_direction_figures`. Blending (4): `blend_products`\*\*, `blend_product_charts`, `blend_satellites`, `blend_satellites_charts` |
| `nwp_forecast_comparison/` (17 + 4) | The 17 scripts already there, plus the four ENS-horizons scripts: `fetch_ens_forecast_horizons`, `fetch_ens_day4_supplement`, `ens_forecast_horizons`\*\*, `ens_forecast_charts` |
| `beam_diffuse_split/` (36) | `era5_grid`\*, `fetch_era5`, `fetch_era5_open_meteo`, `verify_era5_sources`, `fetch_cams`, `build_dataset`\*, `run_experiment`\*, `physics_model`\*, `run_physics_experiment`, `run_hybrid_experiment`, `report_results`, `compare_sources`, `fractions_skill_score`, `elevation_breakdown`, `make_chart`, `make_figures`, `sky_conditions`, `inverter_clipping`, `restart_basins`, `shared_geometry`, `capacity_denominator`, `oracle_capacity`, `multi_nwp`, `ens_horizons`, `fetch_ens_point`, `fetch_ens_point_wind`, `anm_curtailment`, `anm_setpoints`, `export_cap`\*, `commissioning`\*, `sources`\*, `site_e_commissioning`, and the four scripts that feed no past-weather page: `weather_product_domains` (with the `domains/` outlines), `nwp_horizons`, `check_page_numbers`, `stamp_alignment` |
| the package, no script left | `figure_numbers` (read by `past_weather/` and by `nwp_forecast_comparison/leaderboard_by_day.py`) |

A single asterisk marks a module that moves wholesale to the package in layer 1 and is deleted from
`studies/` (the starred scripts that remain in `beam_diffuse_split/` are listed there only until
their commit). A double asterisk marks a script from which named helpers move to the package in
layer 2 (the script stays, shorter). `weather_downloads/fetch_era5_wind.py` keeps its folder and
loses its `sys.path.insert` into `beam_diffuse_split/`, because the helpers it reads come from
layer 1. The scripts of `past_weather/` that today read each other, such as the 13 symbols
`cerra_wind_direction` takes from `cerra_wind_levels`, become same-folder imports and need no move.

**The four orphans stay in `beam_diffuse_split/` for now.** They feed the data-sources page, the
page-number checker and the timestamp-lag check, and the scan decides whether any of them crosses a
folder. If one does, the crossing is a layer-1 or layer-2 symbol like any other.

**`era_fold_design/scripts/` is left untouched, except for the absolute paths in three `.py` files
and nine shell scripts.** `common.py` puts a different worktree
(`.claude/worktrees/era-fold-design`) and a frozen scratch copy of the ENS script on `sys.path`, so
those scripts do not import this repository's scripts at all. The worktree is gone and the frozen
copy holds only the ENS script, so `common.py` cannot be imported, and nine of the other ten `.py`
files import it. Only `two_shares.py` and `saved_cov.py` still run: they are standalone readers of
`data/studies`. Repointing any script at `studies.*` would change the code the recorded numbers
came from, so the `ruff`, `ty` and boundary-test configurations exclude the folder.

The maintainer's decision supersedes "untouched" for portability only. `common.py` (4 literals),
`saved_cov.py` (1) and `two_shares.py` (1) spell `/home/jack/...` six times, and nine `run*.sh`
files each have one `cd /home/jack/...` line. Each Python literal becomes
`Path("~/...").expanduser()`, except the `sys.path.insert` literal at `common.py:9`, which stays a
`str` because `sys.path` ignores a non-string entry (use `os.path.expanduser`). The shell lines
become `cd ~/...`. The repository has no helper for this, so `Path.expanduser()` is used directly.
The change makes the files portable. It does not make them re-runnable, and
`era_fold_design/README.md` gains one sentence saying that `common.py`, the `part*` scripts and the
`run*.sh` scripts cannot be re-run, because the worktree and the frozen code they load are gone.
`two_shares.py`'s `FILES` list holds subpaths under `beam_diffuse_split/` (`past_weather_v2/...`,
`beam_diffuse_wind_products/...`), so the wave that moves a file also updates its entry. The data
folders those literals name move in the data sequence below. The reproduction check is under
"Reproduction".

## What moves to `packages/studies`

**Layer 1 moves code every study reads; each module is one commit, with tests.** Underscore-private
names lose their underscore on the way, because a name imported from a package is public. Two
rules apply to every moved definition. First, a definition that uses `__file__` is rewritten
against `PROJECT_ROOT` or stays in the script: `weather_product_charts.ASSETS_DIR` is
`Path(__file__).resolve().parents[2] / "docs" / ...`, which from `packages/studies/src/studies/`
resolves to `packages/studies` and would write charts to the wrong directory without an error.
Second, `logging.getLogger("<old script name>")` calls in moved code are updated.

| New module | Comes from | What it holds |
|---|---|---|
| `studies.sources` | `sources.py`, wholesale | `SourceType`, `SOURCE_CHOICES`, `PER_SITE_SOURCES`, `OpenMeteoModel` and its registry, `point_output_path_for`, and the path constants (`REPO_DATA_DIR`, `STUDIES_DATA_DIR`, `WEATHER_DATA_DIR`, `ANM_DATA_DIR`). `weather_downloads/paths.py` imports `REPO_DATA_DIR` from here instead of defining it a second time. |
| `studies.pv_dataset` | `build_dataset.py` | The site rosters (`pv_sites`, `wind_sites`), `solar_hourly_power` (from `build_dataset._hourly_power`, which builds a period-ending hour), `read_era5`, `nearest_era5_cell`, `read_cams`, the outage and false-zero filters, the separation-model columns. The build's command line stays in `beam_diffuse_split/build_dataset.py`. `wind_products._hourly_power` is a different function (it shifts every stamp by 30 minutes), becomes `wind_hourly_power` in the layer-2 commit, and a test asserts that the two give different timestamps for the same input, so a swapped import is caught. `ensemble_means_mae` and `ens_forecast_horizons` import both under aliases. |
| `studies.arm_runner` | `run_experiment.py` | `Job`, `run_all`, `MAX_CONCURRENT_FITS`, `SHARED_FEATURES`, `add_time_features`, `dataset_path_for`. |
| `studies.commissioning`, `studies.export_cap`, `studies.physics_model`, `studies.era5_grid`, `studies.figure_numbers` | the scripts of the same name | Whole modules. |
| `studies.open_meteo_point` | `fetch_open_meteo_point.py` | `fetch_point_frame`, `HOURLY_VARIABLES`, the timestamp-convention checks. The `--model` command line stays. It moves because `weather_downloads/` and `beam_diffuse_split/` scripts import it as well as `past_weather/`. |

**Layer 2 is about 1,000 lines of code, not about 28 symbols, because a package module cannot import
a script.** The scan counts 29 symbols in 8 (importing folder, imported module) pairs on `main`.
Each symbol must take its whole transitive call closure into the package, though. For example,
`blend_products._solar_frame` calls `weather_products.joined`, `common_rows`, `with_eras` and
`with_irradiance_context`, `wind_products.joined`, `common_rows` and `with_wind_context`, and
`_with_blend_columns` (`blend_products.py:1886-1915`).

| Candidate module | Closure measured on `main` | Used by |
|---|---|---|
| `studies.product_frames` | The nine symbols (`solar_frame`, `wind_frame`, `SOLAR`, `WIND`, `Domain`, `PERMUTATION_GROUPS` and neighbours) reach about 77 definitions and about 1,040 lines, from `blend_products`, `weather_products`, `wind_products` and `fetch_wind_point` | `nwp_forecast_comparison/`, `open_meteo_ensemble_means/` |
| `studies.ens_members` | About 14 definitions and 235 lines, reaching into `blend_products` as well | `past_weather/` |
| `studies.wn3_fetch` | Only the eight constants of `weather_downloads/fetch_weathernext3.py`. A wholesale move would pull `gcsfs`, `icechunk`, `zarr` and `delta_store` into `packages/studies`, which declares none of them | `nwp_forecast_comparison/build_wn3_inputs.py` |
| `_power_version` | Stays in the script: it imports `deltalake`, which `packages/studies` does not declare | |
| singletons | The scan lists them | |

**Step 0 measures call closures, not import names**, using a transitive call-graph scan over the
layer-2 symbols (saved as `closure.py` in the session scratch, with the import scan `edges.py`).
It records, per (importing folder, imported module) pair, the symbols, the definitions in the
closure, the lines, and the third-party imports the closure needs. After the `ukv_ceda_blends/`
fold lands, the scan runs again, because the fold turns most of the 59 new symbols into same-folder
imports and the remainder are listed as layer-2 candidates.

**Decision rule when a closure is too large to move safely.** The implementer moves a closure only
if it needs no third-party package that `packages/studies` lacks, and if its callers can be given a
characterisation test first. If a closure fails either test, its pair stays as a documented
exception: the importing script keeps a single cross-folder import, listed by name in the
boundary test's one-line allowlist and in the folder README. The cost is that the boundary rule has
a named exception and the two folders stay coupled for that pair. The implementer reports each such
pair to the maintainer before the layer-2 commits start, with the line count it would have moved.
The expected outcome is that `studies.product_frames` moves (the 1,040 lines are about 1 percent
of the code under `studies/`) and the others are decided by the scan.

**Departure from "every function carries tests" for layer 2, with its reason.** A moved
report-line builder is covered by a stronger check than a unit test: the saved-loss reports must
come out byte for byte identical (see "Reproduction"). Row builders, fold cutters and anything that
changes a number get a characterisation test written *before* the move, against the unmoved code,
and moved with it. That covers `solar_frame`, `wind_frame`, `add_time_features`, the site rosters,
`with_export_cap`, and the two hourly-power functions. Existing tests of moved code (`test_weather_products_served_lead`,
`test_cerra_past_solar`, `test_reanalysis_past_wind` and others) move with it and change only their
import line.

**Not moved: study-only code.** Each `main()`, each study's own arm lists and `PLANNED_CONTRASTS`,
and each page's own tables stay in the script.

## Boundary with #805 and #808: defer the extraction

**Recommendation: do not create `packages/evaluation/` in this work.** The Fractions Skill Score
(`studies/fractions_skill_score.py`, 115 lines) is built on an hourly grid and calendar-month
components, and the bootstrap (`studies/bootstrap.py`, 642 lines) is mostly study-shaped intervals
(`YearInterval`, `YearChangeInterval`, `BlendVerdict`). #805 needs half-hourly cadence and an
explicit grouping; #808 needs resampling of whole time blocks across every series at once. Neither
production caller exists, so an API designed now would guess both. Moving the code twice (into an
interim package, then into the production one) doubles the reproduction risk.

- **A guard test, `test_production_does_not_import_studies`,** scans `src/` and `packages/*/src/`
  (excluding `packages/studies`) for `studies` imports and asserts there are none. It passes on
  `main` today and fails the day #805 or #808 imports `studies.bootstrap` for convenience.
- **`studies.bootstrap` and `studies.fractions_skill_score` stay put and import only `studies.*`
  and third-party code.** When #805 or #808 starts, it creates `packages/evaluation`, writes the
  production-quality functions with their own tests, and then switches the study callers. That change
  runs the same reproduction check as this one.
- **Open question:** whether the maintainer prefers the extraction now. If so, it is a separate
  Opus-reviewed PR after the folder move, and #805 and #808 are updated to reference it.

## Tests

**One convention: every test of study code lives under `packages/studies/tests/`, with the tests of
a study's scripts in `packages/studies/tests/<study>/`.** Library tests stay at the top of that
directory. Today 17 files in `tests/` and 12 in `packages/studies/tests/` test scripts, using two
styles: `spec_from_file_location` with a copied `SCRIPT_DIR`, and `sys.path.insert` at module top
with `ty` `extra-paths` to match.

- **Each study folder is listed once in pytest `pythonpath` and once in `ty` `extra-paths`.** Tests
  then keep static lines such as `from wind_products import ...`, which `ty` checks. Only `ty`
  `extra-paths` lists study folders today (`beam_diffuse_split`, `nwp_forecast_comparison`,
  `weather_downloads`, and `tests`); pytest `pythonpath` lists only `tests`. An entry for a folder
  is added in the commit where that folder is complete, never earlier. Once every folder is on the
  path, a module under `packages/studies/src` that still does `from build_dataset import ...`
  would pass `pytest` and `ty` and fail only when a script in another folder runs, which is why
  the boundary test also scans `packages/studies/src` (below). The entries are fine to keep for
  folders that hold no test-imported script, because the boundary test guarantees basenames are
  unique across `studies/`.
- **Pytest discovery needs no other change.** It already collects `packages/*/tests/`, and importlib
  mode (`addopts`) lets two test files share a basename. The one rule is that no script under
  `studies/` is named `test_*.py`.
- **The study skill's per-script review rule is unchanged.** It covers `studies/**`, which now holds
  scripts only. Tests sit under `packages/studies`, so they get the package's treatment: the diff
  review and the mutation pass.
- **Alternative considered: a `tests/` folder inside each study folder.** It needs a `conftest.py`
  and a `pythonpath` entry per folder, makes `studies/` hold tested code that `studies/README.md`
  says it does not hold, and splits `uv run pytest packages/studies` into several locations.
  Rejected.
- **Cost accepted:** with every study folder on pytest's path, a script that imports another
  folder's module passes its tests but fails under `uv run python studies/x/y.py`. The boundary test
  and the smoke run of every script cover that.
- **New tests and what each asserts that fails on `main`:**

  | Test | Fails on `main` because |
  |---|---|
  | `test_study_boundaries` (one AST test, last commit) | `main` has crossing imports, `sys.path` mutations, a `spec_from_file_location` load, and (after the layer-1 moves begin) would catch a bare script import in `packages/studies/src` |
  | `test_production_does_not_import_studies` | passes today; it is a guard, listed as one |
  | characterisation tests listed under "What moves" | the moved functions are untested by name today |
  | a test that `solar_hourly_power` and `wind_hourly_power` give different timestamps for one input | a swap of the two aliased imports would pass every other test |

  `test_study_boundaries` asserts these over every script under `studies/` except
  `era_fold_design/scripts`: no import of a module that lives in a different `studies/` folder;
  no `sys.path` mutation of any form and no `site.addsitedir`; no use of
  `importlib.util.spec_from_file_location` (which `ens_hres_past_wind.py` uses to load
  `weather_downloads/paths.py`; the trial-area box it reads moves into a layer-1 module); and no
  basename shared by two scripts. It also scans every module under `packages/studies/src` and
  fails on any import whose module name is the basename of a script under `studies/`.

## Data

**Every file a study downloads or builds moves under `data/studies/`, in folders named for what the
data is, and each study keeps one folder of its own outputs.** The maintainer decided the rules and
the placement. The code that names the folders is changed first, in a code-only change that leaves
the data where it is (step D1 below), and the files then move in waves, one pull request per wave.

**Decisions taken by the maintainer, and what each settles:**

1. All study downloads live in `data/studies/`, apart from Dagster-managed data (`data/NGED`,
   `data/NWP` and the other pipeline tables), which no study writes.
2. Data used, or plausibly usable, by several studies is filed by what it is: `NWP/`,
   `reanalysis/` and `observations/`. A folder is never named after the study that first wrote it.
   This covers per-site download files too: they stay with their product.
3. Each study has one folder with `inputs/` (cached intermediate frames), `results/` (losses,
   predictions, stamps, intervals), `reports/` and `superseded/`. The 22 `nwp_forecast_comparison_*`
   folders and the flat `nwp_forecast_comparison/` become one folder with one subfolder per batch,
   and the three-way split applies only where "Where the three-way split applies" says so.
4. The nwp-archivist store (`/mnt/data/nwp-archive*`) is not a study and not Dagster-managed. It
   stays where it is, as do its caches and the MOGREPS copy on `/mnt/wd_18tb`, and no study script
   reads them.
5. The ECMWF ENS backfill ultimately belongs in the `data/NWP` Delta table, ingested by Dagster.
   That ingest is not part of this refactor. Issue #959 ("Extend ECMWF ENS training history") carries
   the fetch, and the ingest needs an issue of its own once #959 settles the fetch scope. Until
   then, fetched files are staged in `data/studies/NWP/ENS_BACKFILL_STAGING/`.
6. The three `UKV-CEDA` stores are not merged. They become three sibling folders under `NWP/`.
7. `data.old` is deleted only on the maintainer's explicit go-ahead, after the gate in step G.
8. The three `era_fold_design` Python scripts and its nine shell scripts use `~/` paths (above).

**The target layout.**

```text
data/studies/
  NWP/                       forecasts: runs, previous runs, and the extracts made from them
    ECMWF-AIFS/  ECMWF-AIFS-ENS/  ECMWF-IFS-SINGLE-RUNS/  GEFS/  GFS/
    WeatherNext3/            from WeatherNext3_trial_area
    UKV-CEDA/  UKV-CEDA-part2/  UKV-CEDA-part3/  UKV-CEDA-T120/
    OPEN-METEO-ENSEMBLE-MEANS/
    OPEN-METEO-PREVIOUS-RUNS/<model>/   the eleven Previous Runs products, each with site_points/
    ENS_SITE_EXTRACT/        ens_members, solar and wind inputs, member summaries (shared)
      site_points/           per-site frames built from data/NWP
    ENS_BACKFILL_STAGING/    the #959 fetch, until Dagster ingests it
    windows/                 the eight window or trial copies
    superseded/
  reanalysis/                ERA5  CERRA  NORA3  NORA3_10m  ICON-DREAM-EU  CAMS
  observations/              MIDAS-OPEN  SARAH-3  NGED-ANM
  _scratch/                  transient downloads (today data/_scratch)
  _private/                  trial_area_box.json (never published or committed)
  <study>/                   one folder per study
    beam_diffuse_split  ens_forecast_horizons  nwp_forecast_comparison
    ukv_ceda_blends  cerra_wind  open_meteo_ensemble_means  open_meteo_ens_gap
    icon_eu_compare  era5_wind_compare  ens_backfill_pilot
```

**Where the three-way split applies.** A study folder gets `inputs/`, `results/`, `reports/` and
`superseded/` only where the writing code already keeps the three apart or one path constant can
change it. Files are sorted by role, not by the script that writes them: a frame that a fit reads
but did not fit is an input; losses, predictions, `*_losses.json` stamps and intervals are results;
`.md` files and figures are reports. A file whose role is undecided stays where it is and is listed
in the wave log, and `verification/` and `superseded/` folders move whole. Splitting
`nwp_forecast_comparison/` would rewrite about 120 `output_dir /` uses across its scripts, so it
keeps one folder per batch with every file unchanged.

**Old path to new path, for every entry of `data/studies` and `data/studies/weather`.** Paths are
relative to `data/studies/`. The wave column names the step in "Data migration" that moves the row.

| Old path | New path | Wave |
|---|---|---|
| `data/_scratch` (outside `data/studies`) | `_scratch/` | D2 |
| `superseded/ECMWF-AIFS-3x3`, `superseded/ECMWF-AIFS-ENS-3x3`, `superseded/beam_diffuse_ens_2026-09-26.parquet`, `superseded/beam_diffuse_ens_wind_2026-09-26.parquet` | `NWP/superseded/`, same names (deletion is proposed under "Tidy-up while moving") | D2 |
| `weather/WeatherNext3_icechunk_test` | `_scratch/WeatherNext3_icechunk_test` | D2 |
| `cerra_wind_direction`, `cerra_wind_levels`, `cerra_wind_levels_post_hoc`, `cerra_wind_levels_shear` | `cerra_wind/{direction,levels,levels_post_hoc,shear}`, files and their `superseded*` folders unchanged | D2 |
| `weather/{AROME-FRANCE,ARPEGE-EUROPE,DMI-HARMONIE-AROME,ECMWF-IFS-025,ECMWF-IFS-HRES,GFS-SEAMLESS,ICON-D2,ICON-EU,ICON-GLOBAL,KNMI-HARMONIE-AROME,UKV}` (the eleven Previous Runs products) | `NWP/OPEN-METEO-PREVIOUS-RUNS/<model>/`. The product's `beam_diffuse_<model>.parquet`, `wind_<model>.parquet` and `temperature_2m_site_b.parquet` go to `<model>/site_points/`, so the four `temperature_2m_site_b.parquet` files no longer collide. The download parquet, `previous_runs/`, `README.md` and `lineage.json` stay at the top of `<model>/` | D3 |
| `weather/ECMWF-AIFS-ENS_window_2025-08-01_2025-08-31`, `ECMWF-AIFS_window_2025-02-20_2025-03-05`, `GEFS_window_2024-11-01_None`, `GEFS_window_2025-07-01_2025-07-03`, `GEFS_window_2026-09-22_2026-09-24`, `GFS_window_2025-07-01_2025-07-02`, `WeatherNext3_window_2026-09-20_2026-09-20` | `NWP/windows/<same name>/` | D3 |
| `weather/CERRA` (the wind-level, direction, surface and grid files are all downloads) | `reanalysis/CERRA/`, files unchanged | D4 |
| `weather/CAMS` | `reanalysis/CAMS/`. `beam_diffuse_cams*.parquet` go to `site_points/`; the `cams_site_*.csv` downloads stay | D4 |
| `weather/ERA5` | `reanalysis/ERA5/`. `beam_diffuse_open_meteo*.parquet` and `wind_era5*.parquet` go to `site_points/`; the `beam_diffuse/*.zip` downloads, `wind_native_cds.parquet`, `wind_native_chunks` and `_cds_smoke_test.nc` stay | D4 |
| `weather/ICON-DREAM-EU` | `reanalysis/ICON-DREAM-EU/`. `beam_diffuse_icon-dream-eu.parquet` goes to `site_points/` | D4 |
| `weather/NORA3`, `weather/NORA3_10m` | `reanalysis/NORA3/`, `reanalysis/NORA3_10m/`, files unchanged | D4 |
| `weather/MIDAS-OPEN` | `observations/MIDAS-OPEN/`, files unchanged | D4 |
| `weather/SARAH-3` | `observations/SARAH-3/`. `beam_diffuse_sarah-3.parquet` goes to `site_points/` | D4 |
| `anm/` (the CSV export, `export_cap_23.parquet`, `README.md`, `superseded/`) | `observations/NGED-ANM/`, with the export-cap parquet beside the CSV | D4 |
| `weather/ECMWF-AIFS`, `ECMWF-AIFS-ENS`, `ECMWF-IFS-SINGLE-RUNS`, `GEFS`, `GFS` | `NWP/<same name>/`, files unchanged | D5 |
| `weather/WeatherNext3_trial_area` | `NWP/WeatherNext3/` (`trial_area.zarr`, `_grid_cells.parquet`) | D5 |
| `weather/OPEN-METEO-ENSEMBLE-MEANS` | `NWP/OPEN-METEO-ENSEMBLE-MEANS/`, files unchanged | D5 |
| `weather/ENS` (`beam_diffuse_ens.parquet`, `beam_diffuse_ens_wind.parquet`, `README.md`) | `NWP/ENS_SITE_EXTRACT/site_points/` for the two parquet files, `NWP/ENS_SITE_EXTRACT/README.md` for the README | D5 |
| the 22 folders `nwp_forecast_comparison_{aifs,aifs_blends,aifs_extra_days,day4_shared,day5_aifs_wn3,leaderboard_by_day,leaderboard_by_day_fig3,leads,leads_day10,leads_day10b,leads_day10c,leads_day10d,p4_seeds,product_blends,product_blends_report,vs_ens_dots,vs_ens_dots_all_days,vs_ens_dots_blends,vs_ens_dots_blends_final,vs_ens_dots_final,wn3,wn3_extra_days}` | `nwp_forecast_comparison/<batch>/`, the suffix being the batch name, with every file, `verification/` and `superseded/` unchanged | D6 |
| the flat files of `nwp_forecast_comparison/` (`report.md`, `solar_*`, `wind_*`, `verification/`) | `nwp_forecast_comparison/original/`, files unchanged | D6 |
| `ens_forecast_horizons/{ens_members,solar_inputs,wind_inputs,solar_member_summary,wind_member_summary}.parquet` (top level only) | `NWP/ENS_SITE_EXTRACT/` | D7 |
| `ens_forecast_horizons_day4/` (`ens_members_day4.parquet`, `README.md`) | `NWP/ENS_SITE_EXTRACT/` | D7 |
| `ens_forecast_horizons/era_covered/` (its own `solar_inputs`, `wind_inputs` and member summaries differ from the top-level files) | `ens_forecast_horizons/era_covered/`, unchanged, so nothing collides | D7 |
| the rest of `ens_forecast_horizons/` (losses, predictions, rows, weights, `intervals`, `leaderboard`, `report.md`, `superseded/`) | `ens_forecast_horizons/`, in place | D7 |
| `beam_diffuse_split/beam_diffuse_dataset_*.parquet` (13 frames, written through `dataset_path_for`) | `beam_diffuse_split/inputs/` | D7 |
| `beam_diffuse_split/{beam_diffuse_ens_horizons,beam_diffuse_figures,beam_diffuse_multi_nwp,beam_diffuse_weather_products,beam_diffuse_wind_products,blend_products,past_weather_v2,satellite_blend,superseded}` and `era5_source_agreement.json` | `beam_diffuse_split/`, in place | D7 |
| `open_meteo_ensemble_means`, `open_meteo_ens_gap`, `icon_eu_compare`, `era5_wind_compare`, `ens_backfill_pilot` | same names, files unchanged. `frame_*.parquet` moves to `inputs/` only where one path constant writes it | D7 |
| `weather/UKV-CEDA`, `UKV-CEDA-part2`, `UKV-CEDA-part3` | `NWP/UKV-CEDA/`, `NWP/UKV-CEDA-part2/`, `NWP/UKV-CEDA-part3/`, three stores, never merged | D8 |
| `weather/UKV-CEDA-T120` | `NWP/UKV-CEDA-T120/` | D8 |
| `weather/_t120_trial` | `NWP/windows/UKV-CEDA-T120_trial/` | D8 |
| `ukv_ceda_blends`, `ukv_ceda_blends_run15` | `ukv_ceda_blends/` and `ukv_ceda_blends/run15/` | D8 |
| `weather/_trial_area_box.json` | `_private/trial_area_box.json` | D8, last |
| `weather/` itself | a read-only tombstone (below) | after D8 |

**The per-site `beam_diffuse_*` and `wind_*` files are downloads, so they stay with their
product.** `point_output_path_for` in `sources.py` documents them as "where one per-site download
is written": the Open-Meteo point fetches and the CERRA wind-level files are downloads, not
extracts of a raw folder. Under maintainer rule 2 they therefore go to a `site_points/` subfolder
of their product, not to a study's `inputs/`, which `past_weather/` scripts also read. Only
`ENS/beam_diffuse_ens*.parquet` is derived (from `data/NWP`), and it sits in `ENS_SITE_EXTRACT/`
with the other ENS extracts. `point_output_path_for` and `temperature_site_b_path_for` take the
`site_points/` subfolder in the wave that moves the product.

**Why renaming is safe for the stamps.** A grep over every `*_losses.json`, `losses.fingerprint`,
`build.json` and `lineage.json` found no absolute path, because the stamps hold content hashes and
Icechunk snapshot IDs. A rename therefore leaves every stamp valid, and what breaks is code that
names a folder. The audit lists roughly 25 hard-coded folder names in `nwp_forecast_comparison/`
(several guarded by `output_dir.name == <constant>`, so the constant and the guard change
together), eight copies of `_repo_data_dir()`, and the constants in `weather_downloads/paths.py` and
`beam_diffuse_split/sources.py`.

### Data migration

**The data moves are not part of the git diff, so they run as a separate sequence after the code PR
has merged, one wave per step, each verified before the next starts.** Every move is a rename on the
one device that holds `data/` (`/mnt/data`), so a wave takes seconds. Nothing is deleted by a move,
and nothing a running process holds open is touched. Each wave is one small pull request that flips
the wave's constants, updates the docs and READMEs that name its folders, and lists the `mv`
commands. The maintainer merges it (no agent merges) and runs the renames at the moment it merges.
About 17 other worktrees run old code while a wave is in flight, which the tombstone below turns from
a silent failure into a loud one.

- **Step D0, baseline manifest (no change to data).** From `data/studies`, run `find . -type f
  -not -path '*/UKV-CEDA-T120/*' -not -path '*/ukv_ceda_blends/*' -print0 | xargs -0 -P8
  sha256sum`, writing to a file outside `data/` (about 58 GB read, 5 to 15 minutes). Record the file
  count per top-level folder as well. Links are never followed. The two excluded folders are
  re-baselined once their writers finish, and so is `ukv_ceda_blends_run15` if #1016 writes to it again.
- **Step D1, code-only change (lands with step 2 of the code order).** Every folder name comes from
  one constants module (`studies.sources`, plus the existing `weather_downloads/paths.py` constant
  that now imports from it), the eight `_repo_data_dir()` copies are replaced by that module, and
  `STAMP_GLOB` and the ~25 folder names in `nwp_forecast_comparison/` become constants. **The
  constants keep their old values**, so data stays where it is and the tests and the reproduction
  check cover a change of structure only. The temporary directory in `build_dataset.py:315` and the
  scratch constants in `fetch_cerra.py` and `fetch_cerra_grid.py` become one `SCRATCH_DIR` constant.
  D1 also gives `check_arm_columns_unchanged.py` a required `--expected-stamps` argument, passed as 72
  (today's `ls nwp_forecast_comparison_*/*_losses.json | wc -l`). The script exits non-zero when the
  number of stamps it finds differs from the argument, and the argument has no default, so a glob
  that matches too few stamps cannot pass.
- **Step D2, folders no process reads:** `data/_scratch`, the top-level `superseded/`,
  `WeatherNext3_icechunk_test`, and the four `cerra_wind_*` folders. `fetch_cerra*.py` writes
  `_scratch`, so no CERRA fetch may be running.
- **Step D3, the Previous Runs products and the window copies** (all but `_t120_trial`), with the
  `site_points/` split.
- **Step D4, reanalysis, observations and `anm/`.** No fetch is running on these folders.
- **Step D5, raw forecast products and the ENS per-site extract.**
- **Step D6, `nwp_forecast_comparison` consolidation.** The ~25 constants, the guards and
  `STAMP_GLOB` change in one PR, and the glob becomes `nwp_forecast_comparison/*/*_losses.json`.
  The glob skips `superseded/` folders and `ukv_ceda_blends/`, which are outside the 72, and the check
  still passes `--expected-stamps 72`. The wave does not start while any `fit_*` run is writing to a
  batch folder.
- **Step D7, the shared ENS extract and the remaining study folders** (`ens_forecast_horizons`,
  `ens_forecast_horizons_day4`, the `beam_diffuse_split` dataset frames, and the five small
  studies). `fetch_ens_day4_supplement.py` and `build_forecast_inputs.ens_members` take the new
  constant, the day-4 README's "never edit `ens_forecast_horizons/`" sentence is corrected, and
  `two_shares.py`'s `FILES` entries follow.
- **Step D8, work in flight, last.** `UKV-CEDA-T120`, `_t120_trial`, `ukv_ceda_blends` and
  `ukv_ceda_blends_run15` move only when no process writes to them. The `ukv-t120` unit is a
  transient `systemd-run` unit that has back-filled newest-first towards 2019-09 since 2026-10-02. It
  is stopped (`systemctl --user stop ukv-t120`), the store is moved, and the unit is restarted from a
  checkout on the new code with `--store-dir` pointing at `NWP/UKV-CEDA-T120`, resuming from the
  store's own state. If the back-fill still has weeks to run, only this wave waits: D2 to D7 do not
  touch the T120 store, so the old `weather/` keeps the T120 store, `_t120_trial` and
  `_trial_area_box.json` until D8. `_trial_area_box.json` then moves last, and its old path keeps a
  symlink until the restarted unit runs, because `fetch_ukv_ceda.py` in the `ukv-ceda` worktree reads
  that path. `UKV-CEDA`, `-part2` and `-part3` move as three separate stores, only while no fetch is
  running on them. **Never moved:** the MOGREPS copy on `/mnt/wd_18tb`, and the `nwp-archive*` stores
  and caches.

**A symlink is left at each old path while a wave runs, and the `STAMP_GLOB` double match is handled
explicitly.** A symlink to a directory is followed by `pathlib`, `pl.scan_parquet`, `pl.scan_delta`
and `icechunk.local_filesystem_storage`, and `Path.exists()` is true for it, so the write-once
refusals still fire. Four cases need care:

- **`STAMP_GLOB` and every glob of `nwp_forecast_comparison_*`** match both an old symlink and a new
  real folder, so a stamp can be read twice. In step D6 no symlink is left at any of the 22 old
  batch paths, because the same PR changes every script that read them. The check script keeps one
  entry per `Path.resolve()` result, and `--expected-stamps 72` makes a partial match fail.
- **`output_dir.resolve() == published_dir.resolve()` comparisons** give the same answer through a
  symlink, but a guard that tests `output_dir.name` sees the symlink's name. Those guards are
  converted in step D1.
- **`du`, `find` and `rsync -a`** double-count or copy links. The manifest commands use `find -type f`
  and never follow links. `era_fold_design/scripts/saved_cov.py` uses `rglob`, which double-counts
  through any symlink left at an old path, so symlinks are removed before it is re-run.
- **A symlink cannot replace a folder a writer holds open**, which is why step D8 waits.

**A symlink is removed only when no referrer remains anywhere, and the old `weather/` then becomes a
tombstone.** A writer running stale code calls `mkdir(parents=True, exist_ok=True)`, which silently
recreates a removed path and writes a fresh copy there, and the write-once refusals do not fire
because the path no longer exists. Before removing a symlink, `grep` for its old path in every
checkout listed by `git worktree list`, in `~/.config/systemd/user`, in the transient units shown by
`systemctl --user list-units`, and in `/mnt/data/*.sh`. When `weather/` is empty, replace it with a
read-only tombstone (an empty directory with mode 555, or a regular file), so a stale writer fails
loudly. The tombstone stays until no worktree holds code that names the old paths.

**The `era_fold_design` files follow the data.** `two_shares.py`, `saved_cov.py` and `common.py` name
`data/studies/beam_diffuse_split` and `data/studies`. Each wave that moves a folder one of them names
updates the literal in the same PR, so the two scripts that still run keep resolving their inputs.

**Step G, deleting `data.old` (separate and gated, not part of any move).** `data.old` (143 GB, on
the root device) holds a stale copy of the Dagster tables (`NGED`, `NWP`, `power_forecasts`,
`production_model`, `forecast_metrics`, `effective_capacity`, `eligible_time_series`), so this is the
one step that touches anything Dagster-managed, and it deletes only the stale copy. The gate does not
wait for the last study wave, because the study waves touch no Dagster path. It needs:

- at least one `SUCCESS` run since the `data -> /mnt/data` symlink was made (2026-09-23 09:55) of each
  of `power_time_series_and_metadata_job`, `ecmwf_ens_job` and `live_forecasts_job`, read from the
  run database (`dagster_history/history/runs.db`, opened read-only) or from the Runs page at
  `http://localhost:3000`; and
- `rsync -rn --ignore-existing data.old/ data/` listing no file.

Both conditions held when this plan was revised (2026-10-05, read-only): the run database showed 264,
11 and 44 successes since the symlink, the latest at 13:55, 10:30 and 12:00, and the dry run listed
nothing. The implementer re-checks both immediately before asking. The deletion itself still needs the
maintainer's explicit go-ahead in chat.

**Duplicated caches** (the issue's CAMS example): the audit found no byte-identical CAMS files, so
nothing is merged. The byte-identical files are listed under "Tidy-up while moving".

**`studies/README.md` gains a "Where data lives" table**, one row per kind folder and per study
folder, giving what the folder holds and the script that writes it. The existing table of studies
gains one row per script folder.

### Tidy-up while moving

**Every item here is proposed for the maintainer's approval, and nothing is deleted without an
explicit go-ahead in chat.** The verdicts come from a read-only check on 2026-10-05 of what reads
each folder (a grep of `studies/`, `packages/`, `docs/`, `plans/`, `scripts/` and the skills on
`main`, and a hash of every file above 1 MB that shares its size with another file). Other branches
were not read.

- **WeatherNext 3 test folders:** delete `WeatherNext3_icechunk_test` (67 MB, the output of the fetch
  script's own test mode, read by nothing) and `WeatherNext3_window_2026-09-20_2026-09-20` (0.9 MB, the
  output of an earlier version of the fetch script, read by nothing). Keep `WeatherNext3_trial_area`:
  `build_wn3_inputs.py --build` reads it, and rebuilding it needs Google Cloud credentials.
- **Trial windows:** delete `GEFS_window_2025-07-01_2025-07-03`, `GEFS_window_2026-09-22_2026-09-24`,
  `ECMWF-AIFS_window_2025-02-20_2025-03-05` and `ECMWF-AIFS-ENS_window_2025-08-01_2025-08-31`
  (about 74 MB; low value). Delete `_t120_trial` only after confirming that it is the pre-run trial of
  `fetch_ukv_ceda.py`, which its README says and which the check did not verify by opening its
  `store/`. Keep `GFS_window_2025-07-01_2025-07-02` (read by `verify_previous_runs_leads.py`) and
  `GEFS_window_2024-11-01_None` (hard-coded by three scripts and a README; 647 MB).
- **Duplicates:** delete the seven duplicate parquet files (0.91 GB) in
  `ens_forecast_horizons/superseded/2026-09-23/`, which are byte-identical to the current top-level
  files, and keep that folder's reports, intervals, leaderboards and losses. Do not delete the GEFS
  month caches (0.34 GB). Either keep both copies, or replace the window folder's caches with hard
  links to `GEFS/_month_cache` (same filesystem, so every path stays valid).
- **The 11 `superseded/` folders (8.9 GB):**
    - Keep `beam_diffuse_split/superseded/*_piecewise`, because `docs/studies/beam-diffuse-split.md`
      says the quoted results are filed there and draws Figure 1 from them.
    - Keep `anm/superseded` (raw operator data that cannot be regenerated),
      `cerra_wind_levels/superseded*`, `cerra_wind_levels_shear/superseded*` and
      `nwp_forecast_comparison_leads/superseded` (each under 1 MB).
    - Propose deleting `superseded/ECMWF-AIFS-3x3` (31 MB), `superseded/ECMWF-AIFS-ENS-3x3` (986 MB),
      the two `superseded/beam_diffuse_ens*_2026-09-26.parquet` files (73 MB, rebuilt in minutes from
      `data/NWP`), `nwp_forecast_comparison_aifs_extra_days/superseded` (20 MB), and the large files
      of `ens_forecast_horizons/superseded` (about 2.4 GB; keep the four `report.md` files).
    - Propose deleting the non-`_piecewise` entries of `beam_diffuse_split/superseded` (about 3.4 GB:
      the `as-labelled` and `shifted` convention variants, the `cds_*` frames, `blend_products_first_run`,
      and the `*_before_pages_review` folders). A grep of `docs/`, `studies/`, `packages/*/src` and
      `plans/` on 2026-10-05 found no mention of those names, and `beam-diffuse-split.md` names only
      `_piecewise`. The grep checks names, not each quoted number, so a reviewer confirms that the
      page's figures come from the `_piecewise` entries before the deletion.
- **Stale READMEs under `data/` (73 `README*.md` files):** one `sed` pass after the last wave rewrites
  `data/studies/weather/<dir>` to the new path, and manual fixes cover the rest. Two READMEs name
  `ECMWF-AIFS-WIDE` and `ECMWF-AIFS-ENS-WIDE`, which no longer exist, and the `ERA5` and `CAMS`
  READMEs omit the `_2026-08-20_2026-09-21` extension files. READMEs are not regenerated, because
  regeneration needs the network and rewrites `lineage.json` with a new `retrieved_at_utc`. Stamps do
  not hash READMEs (a grep of every `*.json`, `*.fingerprint` and `*.txt` under `data/studies` found
  no mention), so editing them changes no hash the manifest comparison checks.

## Order of mechanical steps

**Every commit passes `ruff check`, `ruff format --check`, `pydoclint`, `ty check`, `pytest`,
`pymarkdown`, `mkdocs build --strict` and `check_docs_links`**, per the run-every-CI-step-locally
rule. A `git mv` commit changes no content beyond path strings, so `git diff -M` reads as renames.

0. **Baseline (no commit).** On the merged `main`: re-run the import scan (save it as `edges.py` in
   the session scratch), record the real layer-2 count, save the golden outputs (see
   "Reproduction"), and check the CPU load.
1. **Guard.** Add `test_production_does_not_import_studies`.
2. **Layer 1, one commit per module** in topological order, because `export_cap.py:49` imports
   `build_dataset._pv_sites`: `sources`, `era5_grid`, `commissioning`, `physics_model`,
   `pv_dataset`, `export_cap`, `arm_runner`, `open_meteo_point`, `figure_numbers`. Each commit moves
   the code, deletes the script's copy, switches every caller from a bare import to `from
   studies.<module> import`, and adds or moves its tests. Scripts that were only a library
   (`era5_grid`, `commissioning`, `physics_model`, `export_cap`, `figure_numbers`, `sources`) are
   deleted from `studies/`. The implementer checks that the number of `def` and `class`
   statements removed from the scripts equals the number added to the package.
   **Step 2a (code only, data unmoved):** step D1 under "Data migration", the folder-name
   constants and `STAMP_GLOB`, lands here, in or straight after the `sources` commit.
3. **Layer 2, one commit per module,** after the maintainer has seen the step 0 closure report.
   Characterisation tests land in a commit before the move.
4. **The folder move, one commit per destination folder** (`past_weather/`, the four ENS
   scripts into `nwp_forecast_comparison/`, then `ukv_ceda_blends/` into
   `nwp_forecast_comparison/`). `git mv`; replace every `sys.path.insert` into a
   sibling folder (none should remain); update `pyproject.toml` `pythonpath` and `extra-paths`; fix
   each script's own `parents[...]` (depth stays at `studies/<folder>/`, so `parents[2]` is still the
   repository root: the 38 `Path(__file__)` uses are checked, not assumed); move the tests to
   `packages/studies/tests/<study>/`; split `beam_diffuse_split/README.md` into one README per
   folder, each with a table mapping every script to its published page; update every path in
   `docs/`, scripts' docstrings and comments, `studies/README.md`, and the skills that name a path
   (found by `grep`; `data-download` is one). Each folder's `pythonpath` and
   `extra-paths` entries are added in its own commit. **The last of these commits adds
   `test_study_boundaries`.**
5. **Skills and `CLAUDE.md`.** In `.claude/skills/study/SKILL.md`, the "Where a study's pieces live"
   table (new folder rule, the import rule, the test location, the data convention) and the sentence
   about `wind_products.py` importing private helpers, which is obsolete. In `CLAUDE.md`, the
   Packages table row for `studies` and the skills table if any summary changed (sweep the summaries,
   per the project memory). In `docs/architecture/testing.md`, a "Study tests" paragraph recording the
   convention, because the issue requires it there.
6. **Verification and reviews** (below).
7. **Data migration (after the merge).** Steps D0 and D2 to D8, then the gated step G, under
   "Data migration". Each wave repeats the manifest comparison and the grep gate.

**Which parts a Sonnet implementer does mechanically:** steps 0, 1, 4 and 5, and the moves in steps
2 and 3 once the closure report is approved. Judgement stays with the maintainer or Opus in two
places: the public names chosen when an underscore-private symbol becomes a package API, and the
layer-2 module names, which the step 0 scan may change.

## Reproduction

**Every report-producing script must print the same published numbers after the move.** Eighteen
scripts take part. Seventeen have `--report-only`: `blend_products`, `blend_satellites`,
`cerra_past_solar`, `cerra_wind_direction`, `cerra_wind_levels`, `ens_forecast_horizons`,
`ens_hres_past_wind`, `ens_past_solar`, `reanalysis_past_wind`, `station_past_solar`,
`station_wind_arms`, `weather_products`, `wind_icon_dream`, `fit_extra_leads`,
`nwp_forecast_comparison`, `ensemble_means_mae` and `local_ens_gap`. `wind_products` has no
`--report-only` flag: the implementer runs it as far as its report step against the scratch root if
its fits are cached, and otherwise records it as covered only by the `blend_products` gate and the
unit tests. The leaderboard scripts (`past_solar_leaderboard`, `past_wind_leaderboard`) and
`fractions_skill_score` read saved losses with no flag. Step 0 re-counts the scripts after the
`ukv_ceda_blends/` fold.

**What a `--report-only` run proves is limited.** The frame fingerprint casts every float column to
Float32 before hashing (`station_past_solar.py:517-545`), so a match proves that the input frames
agree to Float32 precision, not bit for bit. Five of the scripts (`ens_forecast_horizons`,
`weather_products`, `ensemble_means_mae`, `local_ens_gap`, `blend_products`) carry no frame
fingerprint, so their `--report-only` runs only re-read losses saved on disk. Where feasible the
implementer adds a per-row bit-for-bit check: write the row frame the script builds to Parquet on
`main` and on the branch in the scratch root, and compare with Polars `frame_equal`.

- **Scratch data root, laid out before any run.** `DATA_PATH_INTERNAL` is a directory on the home
  partition (not `/tmp`; hard links are not possible across `/mnt/data`, so symlinks and copies are
  the only options). `REPO_DATA_DIR` (`sources.py:328`) is the root of every input, and each
  fingerprinted `--report-only` run rebuilds its whole row frame from the power Delta table, the
  weather directories, the CAMS extract and the ENS members before it compares the fingerprint.
  The layout is therefore: every input directory is a symlink to the shared `data/` (the study
  inputs, and also `data/NGED`, `data/NWP` and `data/effective_capacity`, which the row builders
  read), and every
  directory a script writes to is a real copy (or an empty directory the script fills). Before any
  run, the implementer lists every write path of every script (`write_parquet`, `write_text`,
  `mkdir`, the cache writers) and puts each under a real directory. A write through a symlinked
  input would land in the shared `data/`, which this plan promises never to touch, so the
  reproduction fails loudly: it runs the scripts as a user without write permission on the shared
  `data/` if the machine allows that, and otherwise takes a `find data -newer <marker>` listing
  before and after, and any file the listing shows is a failure. The implementer measures with `du`
  first (`ens_forecast_horizons/` alone is 4.8 GB).
- **Compare outputs.** Run the scripts on the branch into the scratch root and compare each
  `report.md`, `intervals.parquet` and leaderboard output against the file already on disk in the
  shared `data/studies/`, using `cmp` for reports and a Polars `frame_equal` for parquet files.
  Exclude the generated output READMEs that `ens_hres_past_wind.py:3015` and
  `station_wind_arms.py:2497` write: they name their own path, which changes by design. A
  fingerprint `ValueError` from a `--report-only` run counts as "differs". For any script whose
  output differs, run `main` into the scratch root too: a difference present on `main` is a stale
  file on disk, not a regression. `blend_products --report-only` needs `--power-version N`, read
  from the report it replaces.
- **Per-row outputs are the bit-for-bit level**, as the study skill requires. Because `--report-only`
  recomputes no per-row loss, add three checks that do. First, rebuild one
  `beam_diffuse_dataset_<source>.parquet` with `build_dataset.py` into the scratch root and compare
  its hash with the one on disk. Second, re-run the ENS input builders that consume the moved
  `ens_members` code (`build_forecast_inputs`, `build_wn3_inputs`, and the `verify_*` scripts) into
  scratch: at minimum one ENS-derived input, hashed against the copy on disk. Third, run
  `blend_products.py` once into scratch, which refits every published single-product arm and
  **stops unless each reproduces the published per-row losses bit for bit** (`reproduction.md`).
  The full run goes on to fit every blend and takes hours, so the implementer stops it once
  `reproduction.md` is written and records the result. The machine must be idle first, and the
  runtime is quoted from the last run rather than guessed.
- **Chart scripts.** Re-run every moved chart script inside the worktree and require `git status
  docs/` to show no change. A change means the chart or its output path differs (the
  `ASSETS_DIR` rule above is the likely cause).
- **Smoke every script with only its own folder on `sys.path`.** For each script with a command
  line, `cd studies/<folder> && uv run python <script>.py --help` (exit 0), and for the rest `cd
  studies/<folder> && uv run python -c 'import <script>'`. A pytest or `PYTHONPATH` setup that
  lists every folder hides cross-folder imports, which is the cost this plan names under "Tests".
  The study skill's rule applies too: no linter evaluates a `sys.path` string.
- **Hash manifests and the grep gate, for the data migration.** After each wave, a second SHA-256
  manifest is compared with the D0 baseline through the old-to-new map: every hash reappears
  exactly once, and the file count per folder matches. Active folders are skipped and re-baselined
  after they finish. Stamps are re-checked by recomputing `inputs_sha256` and `published_sha256`
  against the files at their new paths. At the end, this grep returns nothing:

  ```bash
  grep -rnE "studies/weather|nwp_forecast_comparison_[a-z]|data/_scratch|data/studies/anm" \
    studies packages docs .claude CLAUDE.md
  ```

  The same grep runs in every checkout listed by `git worktree list` before a symlink is removed.
  `check_arm_columns_unchanged.py` must report the baseline stamp count, and its
  `--expected-stamps 72` argument fails the run if it finds fewer.
- **`era_fold_design` reproduction check.** The audit and this plan found that
  `.claude/worktrees/era-fold-design` no longer exists, and the frozen copy at
  `.claude/worktrees/scratch/era-fold/code/` holds only `ens_forecast_horizons.py`, so
  `import common` cannot succeed on this machine today. Before the edit, the implementer records the
  result of `cd studies/era_fold_design/scripts && uv run python -c "import common"`. After it, the
  check asserts that each of the six `Path(...).expanduser()` values (and the `str` at `common.py:9`)
  equals the literal it replaced (true when `$HOME` is `/home/jack`), that each `cd ~` line in the
  nine `run*.sh` files resolves to the old directory, that `import common` gives the same result as
  before, and that `ens_forecast_horizons` still imports from the frozen directory. `saved_cov.py`
  and `two_shares.py` are re-run only if they ran before the edit, and their output is compared.
- **Run order respects the review rule.** The branch runs only after the Opus diff review is
  triaged, because the moved scripts are changed scripts.
- **Reproduction log.** The implementer writes one line per script (the 18 above plus the
  `blend_products` gate): the comparison result, and any fingerprint refusal.

## Docs to update

- **81 lines in 11 files under `docs/`** name a script path, and 58 of them name a script that
  moves (70 name a `beam_diffuse_split` script). A `git mv` map file drives one scripted replacement
  of `studies/<old>/<name>.py` with `studies/<new>/<name>.py`. Pages that name only a folder (`wind.md`
  line 1640: "in `studies/beam_diffuse_split/`") are edited by hand.
- **Path check:** `mkdocs build --strict` and `check_docs_links.py` check neither of the two ways
  the docs name a script (81 lines in a code span, and README and skill paths), because
  `check_docs_links.py` checks only URLs on the published site. A grep loop therefore extracts
  every `studies/<folder>/<name>.py` from `docs/`, `studies/`, `packages/`, `.claude/skills/`,
  `CLAUDE.md` and `README.md`, and fails if any file does not exist. The implementer saves it as
  `scripts/lint/check_study_script_paths.py` only if the maintainer wants it kept; otherwise it
  stays in the session scratch and runs before each folder-move commit. Paths in the `data/`
  READMEs are not checked (see Risks).
- **Outside `docs/`:** a `grep` finds 177 lines in scripts and READMEs under `studies/` and
  `packages/` that name a script path, plus the skills that name a path, `pyproject.toml` comments,
  `studies/README.md`, and `packages/studies/README.md`.
- **Data paths:** the files that name `data/studies/weather`, `anm/`, `ens_forecast_horizons/`
  parquet names or `nwp_forecast_comparison_<batch>` are updated in the wave that moves them. The
  audit counts hits in `docs/studies/forecasts/matched-lead.md` (19), `docs/studies/past-weather/wind.md`
  (11), `solar.md` (6), `ensemble-means.md` (4), `cerra-wind-levels.md` (3), `beam-diffuse-split.md`
  (3), and a few more; `docs/live_service/setup.md`, `aws.md`, `docs/getting-started.md`,
  `docs/architecture/ensemble-archive.md` and `packages/dashboard/README.md` probably name the Dagster
  data path, which does not move, so each is checked before editing. Also updated: the study
  READMEs under `studies/` (`nwp_forecast_comparison` 14 hits, `beam_diffuse_split` 12,
  `weather_downloads` 3, and three more with 2 each or fewer),
  `.claude/skills/study/SKILL.md` (the "Where a study's pieces live" table, lines 50 to 52, and
  line 270), `.claude/skills/data-download/SKILL.md` (lines 23 and 306 to 307), and
  `studies/README.md`. `CLAUDE.md` names no `data/studies` path, so it changes only if the
  `studies` Packages row or the skills table does.
- **Prose:** written about the present ("each family has a folder"), with no history of the old
  layout, per the repository's prose rules.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict --site-dir /tmp/ci-site
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files --show-diff-on-failure
test "$(uv export --no-dev --format requirements-txt | grep -ciE '^(pvlib|cdsapi)')" -eq 0
```

The last line is the image check from the Testing page: the library modules gain no new
third-party dependency (`pv_dataset` already needs only `pvlib`, `xarray` and `polars`, all in
`packages/studies/pyproject.toml`), so it should keep passing.

## Reviews

- **Plan reviews (caller runs them):** the simplicity review first, then correctness.
- **Diff review 1 (Opus), once, over the whole branch:** correctness, and cutting the diff to what
  the change needs. It also reviews the public names and module grouping of layers 1 and 2. The
  brief tells it to check that:
    - every moved definition is deleted from its original script, not copied (compare the `def` and
      `class` counts before and after);
    - no module under `packages/studies/src` imports a bare script name;
    - each caller of `solar_hourly_power` and `wind_hourly_power` still calls the function it called
      before the move;
    - no moved code uses `__file__`;
    - each private name made public keeps its signature and default arguments;
    - `logging.getLogger("<old name>")` calls are updated;
    - the `era_fold_design` scripts differ from `main` only in the six `expanduser` path literals
      and the nine `cd ~` lines;
    - the path-check loop passes;
    - the reproduction log shows a result for each script plus the `blend_products` gate, and records
      every fingerprint refusal.
- **Diff review 2 (Opus mutation pass), limited to the tests this work adds** (the characterisation
  tests and the guards), as the study skill requires whenever `packages/studies` changes.
- **No Sonnet review of its own output**: Sonnet implements, Opus reviews, per the project memory.

## Suggestions from the simplicity review, and what became of them

- **Adopted:** two folders for the past-weather family and the ENS-horizons scripts; `sources.py`
  moved wholesale; no `CACHES` registry, role functions or registry test; one boundary test with no
  allowlist, `FOLDER_MAP` or per-script folder test; no `study_script` fixture; corrected counts and
  the two missed crossings; no `studies.nwp_comparison` for the unmerged branch; the on-disk-report
  saving in "Reproduction".
- **Adopted in a different form:** cutting `data_sources_figures/` and `tools/`. The four orphan
  scripts stay in `beam_diffuse_split/` instead of moving to a shared folder, because leaving them
  needs no move. The `python -m studies.page_numbers` entry point is not added.
- **Rejected:** dropping the dataset hash, the `blend_products` per-row refit, or the end-to-end
  runs. The study skill requires per-row proof after a refactor, and `--report-only` alone recomputes
  no per-row loss.
- **Not changed:** the commit count. Layer 2 shrinks to a few modules, so the commit count falls
  without a rule.

## Risks and open questions

- **The 44-versus-79 and 30-versus-81 counts mean the issue under-sized itself.** The plan covers
  the true counts. If the maintainer wants a smaller first PR, the natural cut is steps 1 and 2 (the
  guard and layer 1), with the folder move following.
- **Layer 2 is about 1,040 plus 235 lines, and about 59 more symbols arrive with #1016.** The
  decision rule under "What moves" says which closures stay as documented exceptions. The
  maintainer sees the step 0 closure report before any layer-2 commit.
- **Placement judgement calls:** `stamp_alignment` and the three other orphans stay in
  `beam_diffuse_split/`; `ens_horizons` and `multi_nwp` (early ENS and second-NWP studies, the first
  superseded by the ENS-horizons page) stay there too. Whether they still earn a place under
  `studies/README.md`'s "not an attic" rule is a question for the maintainer; this plan does not
  delete them. `verify_ukv_lineage` sits in `past_weather/` because it imports
  `fetch_open_meteo_point`.
- **The `past_weather/` folder holds 38 scripts.** The folder README's per-page table is the way to
  find a page's scripts, and a reader must use it instead of the folder name.
- **The tests move from `tests/` to `packages/studies/tests/`.** The package is a leaf dependency
  absent from the production image, so the move cannot reach production.
- **The reproduction check runs before any data moves**, against data at the old paths, because the
  code change (step 2a) merges first. The data waves have their own checks (manifests, stamps, grep
  gate).
- **Reproduction needs a scratch data root of several gigabytes.** The home partition has room
  (486 GB free). Nothing is written under the shared `data/`, and the write-path audit and the
  before-and-after listing make a stray write fail the run.
- **Generated READMEs under `data/` go stale.** About 70 READMEs embed the old paths as text. Nothing
  reads them, and stamps do not hash them. "Tidy-up while moving" proposes one `sed` pass after the
  last wave, plus manual fixes, with no regeneration.
- **Other branches and services:** no other open branch touches `studies/` except #1016, and no
  `.github` workflow calls a study script by path. The transient `ukv-t120` unit does run
  `weather_downloads/fetch_ukv_ceda.py` from the `ukv-ceda` worktree, and #1016 report passes write to
  `ukv_ceda_blends/`, so step D8 waits for both. About 17 other worktrees hold old code, which the
  tombstone and the worktree grep in "Data migration" cover. A fresh fit
  after the move records the move commit in `_script_commit()` (`ens_hres_past_wind.py:3048`,
  `station_wind_arms.py:397`); saved `script_commit.txt` files are unaffected.
- **The `_repo_data_dir` copies are in scope.** Step D1 replaces all eight (including those in
  `build_forecast_inputs`, `check_input_steps` and `verify_previous_runs_leads`) with
  `studies.sources.REPO_DATA_DIR`.
- **No reviewer or implementer should publish a generator's name, ID or coordinates** in a commit,
  README or PR body while splitting the READMEs; the existing anonymisation rules still apply to
  every moved chart script.
