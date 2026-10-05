# Plan: give each study its own folder and move shared study code into `packages/studies` (#861)

**The problem is that `studies/beam_diffuse_split/` holds 79 scripts for about ten different study
pages, and the scripts reach each other, and other folders, by bare imports.** The issue counted 44
scripts; `main` now holds 79 in that folder, 128 under `studies/` in all (excluding
`era_fold_design/scripts`), and 77,000 lines. A scan of the imports found about 60 pairs of (importing
folder, imported module) that cross a folder boundary once the scripts are sorted by study page, and
about 130 distinct imported symbols, 34 of them underscore-private. Ten scripts hide a crossing
behind `sys.path.insert`, which `ruff` and `ty` cannot check. Published pages cite 125 script
paths across 12 files in `docs/`, not the 30 the issue estimated.

**The plan is to sort the scripts into one folder per study page, and to make the crossings
impossible rather than merely rare.** A script may import only from its own folder and from
`studies.*` (`packages/studies`). Everything a second folder needs moves into the package, in two
layers, each guarded by a test that lists the crossings still allowed and can only shrink. Study
outputs stay where they are, with one path module and one registry to describe them. All tests of
study scripts move to one tree. The general Fractions Skill Score and paired block bootstrap are not
extracted here (issues #805 and #808): they wait for their production callers.

## Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.** The preconditions (#858, #859)
have merged. This plan is written against `main` plus branch `study-1016-ukv-blends`, which is
running a study now and **merges before this refactor starts**. That branch edits
`nwp_forecast_comparison/fit_aifs.py`, `nwp_forecast_comparison.py` and
`packages/studies/src/studies/ifs_single_runs.py`, and adds `studies/ukv_ceda_blends/` (five
scripts that import from `nwp_forecast_comparison/`, `beam_diffuse_split/` and
`weather_downloads/` through `sys.path` inserts) and `packages/studies/tests/test_ukv_ceda_*.py`.
The implementer re-runs the import scan in step 0 on the merged `main`, because the symbol counts
here predate that branch.

**Size: complex.** The five triggers:

- **What gets stored:** no. No Patito model, Delta table or asset changes, and every output keeps
  its path.
- **Production serving path:** no. Nothing in `src/` or `packages/` imports `studies` today, and a
  new test pins that.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The folder map, the number of extraction layers, the
  test location, and the #805/#808 boundary each have a defensible alternative, listed below.
- **Callers not nameable without searching:** yes. About 130 symbols, 128 scripts, 29 test files and
  125 docs lines.

Either of those two triggers alone makes the issue complex: it gets this plan and both
plan reviews (run by the caller, not by this planner), and both diff reviews. The maintainer asked
for one Opus diff review; this plan keeps that review and adds the mutation pass the study skill
already requires whenever `packages/studies` changes, limited to the tests this work adds.

**Departures from the issue body:**

- **Five folders become eight, and the issue's `ens_forecast_horizons/` and `blending/` stay.**
  `cerra_wind_levels` and `cerra_wind_direction` share one folder, `cerra_wind/`, because the second
  imports 13 symbols from the first. A `data_sources_figures/` folder takes the two charts drawn
  for `docs/roadmap/data-sources.md`, and a `tools/` folder takes two scripts no single page owns.
- **"Move the shared plumbing" is done in two layers, not one.** Layer 1 is code every study uses
  (paths, the PV dataset, the XGBoost arm runner). Layer 2 is code the past-weather pages share
  with each other (report-line builders, chart helpers, row builders). Layer 2 is the larger risk
  and ships in its own commits.
- **Tests go to `packages/studies/tests/<study>/`, not into each study folder.** See "Tests".
- **#805 and #808's extraction is deferred**, with a guard test now. See "Boundary".

## The folder map

**Each script goes to the folder of the study page that first needs it; a script two pages use goes
to the earlier page.** Folder names follow the issue where it named one. `era_fold_design/`,
`weather_downloads/`, `open_meteo_ensemble_means/`, `ens_backfill_pilot/` and
`nwp_forecast_comparison/` (17 scripts, three pages) keep their folders, and `ukv_ceda_blends/` keeps
the folder branch 1016 gives it.

| Folder | Scripts |
|---|---|
| `beam_diffuse_split/` (32) | `era5_grid`\*, `fetch_era5`, `fetch_era5_open_meteo`, `verify_era5_sources`, `fetch_cams`, `build_dataset`\*, `run_experiment`\*, `physics_model`\*, `run_physics_experiment`, `run_hybrid_experiment`, `report_results`, `compare_sources`, `fractions_skill_score`, `elevation_breakdown`, `make_chart`, `make_figures`, `sky_conditions`, `inverter_clipping`, `restart_basins`, `shared_geometry`, `capacity_denominator`, `oracle_capacity`, `multi_nwp`, `ens_horizons`, `fetch_ens_point`, `fetch_ens_point_wind`, `anm_curtailment`, `anm_setpoints`, `export_cap`\*, `commissioning`\*, `sources`\*, `site_e_commissioning` |
| `past_weather_solar/` (15) | `weather_products`\*\*, `cerra_past_solar`\*\*, `ens_past_solar`\*\*, `ens_past_solar_charts`, `station_past_solar`, `station_past_solar_charts`, `past_solar_leaderboard`\*\*, `past_solar_leaderboard_charts`\*\*, `weather_product_charts`\*\*, `check_new_products`, `extract_site_series`\*\*, `verify_icon_lineage`, `verify_ukv_lineage`, `check_station_page_numbers`, `fetch_open_meteo_point`\* |
| `past_weather_wind/` (14) | `wind_products`\*\*, `fetch_wind_point`, `wind_product_charts`, `wind_icon_dream`, `wind_icon_dream_charts`, `station_wind_arms`, `station_wind_arms_charts`, `ens_hres_past_wind`, `ens_hres_past_wind_charts`, `reanalysis_past_wind`, `reanalysis_past_wind_charts`, `past_wind_leaderboard`, `past_wind_leaderboard_charts`, `check_served_wind` |
| `cerra_wind/` (5) | `cerra_wind_levels`\*\*, `cerra_wind_levels_charts`, `cerra_wind_levels_shear`, `cerra_wind_direction`, `cerra_wind_direction_figures` |
| `blending/` (4) | `blend_products`\*\*, `blend_product_charts`, `blend_satellites`, `blend_satellites_charts` |
| `ens_forecast_horizons/` (4) | `fetch_ens_forecast_horizons`\*\*, `fetch_ens_day4_supplement`\*\*, `ens_forecast_horizons`\*\*, `ens_forecast_charts` |
| `data_sources_figures/` (2) | `weather_product_domains` (with the `domains/` outlines), `nwp_horizons` |
| `tools/` (2) | `check_page_numbers` (the command line over `studies.page_numbers`), `stamp_alignment` |
| the package, no script left | `figure_numbers` |

A single asterisk marks a module that moves wholesale to the package in layer 1 and is deleted from
`studies/`. A double asterisk marks a script from which named helpers move to the package in layer 2 (the script stays, shorter).
The `weather_downloads/fetch_era5_wind.py` script keeps its folder and loses its
`sys.path.insert` into `beam_diffuse_split/`, because the helpers it reads come from layer 1.

**Alternative considered: fewer folders.** One `past_weather/` folder for solar, wind, `cerra_wind/`,
blending and ENS horizons would need no layer 2, because the 60-odd layer-2 symbols are all
past-weather-to-past-weather. It would also hold about 50 scripts and recreate the problem this
issue names. Rejected, with the consequence accepted that layer 2 is real work.

## What moves to `packages/studies`

**Layer 1 moves code every study reads; each module is one commit, with tests.** Underscore-private
names lose their underscore on the way, because a name imported from a package is public.

| New module | Comes from | What it holds |
|---|---|---|
| `studies.data_paths` | `sources.py` | The path constants (`REPO_DATA_DIR`, `STUDIES_DATA_DIR`, `WEATHER_DATA_DIR`, `ANM_DATA_DIR`, the study and leaderboard directories) and the registry in "Data". |
| `studies.sources` | `sources.py` | `SourceType`, `SOURCE_CHOICES`, `PER_SITE_SOURCES`, `OpenMeteoModel` and its registry, `point_output_path_for`. |
| `studies.pv_dataset` | `build_dataset.py` | The site rosters (`pv_sites`, `wind_sites`), `hourly_power`, `read_era5`, `nearest_era5_cell`, `read_cams`, the outage and false-zero filters, the separation-model columns. The build's command line stays in `beam_diffuse_split/build_dataset.py`. |
| `studies.arm_runner` | `run_experiment.py` | `Job`, `run_all`, `MAX_CONCURRENT_FITS`, `SHARED_FEATURES`, `add_time_features`, `dataset_path_for`. |
| `studies.commissioning`, `studies.export_cap`, `studies.physics_model`, `studies.era5_grid`, `studies.figure_numbers` | the scripts of the same name | Whole modules; `figure_numbers` is read by the solar and wind chart scripts and one test. |
| `studies.open_meteo_point` | `fetch_open_meteo_point.py` | `fetch_point_frame`, `HOURLY_VARIABLES`, the timestamp-convention checks. The `--model` command line stays. |

**Layer 2 moves the helpers the past-weather pages share with each other.** The scan names them:

| New module | Symbols it takes | Used by |
|---|---|---|
| `studies.report_tables` | `METRIC`, `PERCENTAGE_POINTS`, `mae`, `contrast_line`, `CONTRAST_HEADER`, `geometry_lines`, `with_eras`, the ERA5-by-year builders (from `weather_products.py`); `fingerprint`, `arm_columns_lines`, `absolute_table_lines` (`ens_past_solar.py`); `check_column_counts`, `uncovered_share`, `with_covering_folds` (`cerra_past_solar.py`); `check_same_rows`, `check_settings` (`cerra_wind_levels.py`) | wind, `cerra_wind/`, blending, ENS horizons |
| `studies.past_weather_charts` | `ASSETS_DIR`, `NAMES`, `FAMILIES`, `CAPACITY`, the axis titles and chart-frame builders (from `weather_product_charts.py` and `past_solar_leaderboard_charts.py`); `ABSOLUTE_SECTION`, `contrast_section` (`past_solar_leaderboard.py`) | wind, blending, ENS horizons |
| `studies.product_frames` | `solar_frame`, `wind_frame`, `SOLAR`, `WIND`, `Domain`, `PERMUTATION_GROUPS` (`blend_products.py`); `hourly_power` (`wind_products.py`) | ENS horizons, `ens_past_solar`, ensemble means |
| `studies.ens_members` | `ENSEMBLE_SIZE`, `H3_RESOLUTION`, `SUPPLEMENT_PATH`, `Steps`, `reduce_members`, `shared_features`, `ens_columns` | past-weather solar, `nwp_forecast_comparison` |
| `studies.gridded_site_series` | `ICON_DREAM_CELL_CENTRES` and the cell-centre helpers (`extract_site_series.py`) | wind |
| `studies.nwp_comparison` | The symbols `ukv_ceda_blends/` imports from `nwp_forecast_comparison/`, `fit_extra_leads`, `nwp_forecast_charts` and `weather_downloads/paths.py`; the implementer lists them from the scan | `ukv_ceda_blends/`, `nwp_forecast_comparison/` |

**Departure from "every function carries tests" for layer 2, with its reason.** A moved
report-line builder is covered by a stronger check than a unit test: the saved-loss reports must
come out byte for byte identical (see "Reproduction"). Row builders, fold cutters and anything that
changes a number get a characterisation test written *before* the move, against the unmoved code,
and moved with it. That is `solar_frame`, `wind_frame`, `with_covering_folds`, `add_time_features`,
`fingerprint`, the site rosters, and `with_export_cap`. Existing tests of moved code
(`test_weather_products_served_lead`, `test_cerra_past_solar`, `test_reanalysis_past_wind` and
others) move with it and change only their import line.

**Not moved: study-only code.** Each `main()`, each study's own arm lists and `PLANNED_CONTRASTS`, and
each page's own tables stay in the script. `wind_products.py` stops being imported by anything
outside `past_weather_wind/`.

## Boundary with #805 and #808: defer the extraction

**Recommendation: do not create `packages/evaluation/` in this work.** The Fractions Skill Score
(`studies/fractions_skill_score.py`, 115 lines) is built on an hourly grid and calendar-month
components, and the bootstrap (`studies/bootstrap.py`, 642 lines) is mostly study-shaped intervals
(`YearInterval`, `YearChangeInterval`, `BlendVerdict`). #805 needs half-hourly cadence and an
explicit grouping; #808 needs resampling of whole time blocks across every series at once. Neither
production caller exists, so an API designed now would guess both. The extraction is not mechanical,
it would hold up the folder move, and moving it twice (into an interim package, then into the
production one) doubles the reproduction risk.

**What this work does for the boundary:**

- **A guard test, `test_production_does_not_import_studies`,** scans `src/` and `packages/*/src/`
  (excluding `packages/studies`) for `studies` imports and asserts there are none. It passes on
  `main` today and fails the day #805 or #808 imports `studies.bootstrap` for convenience.
- **`studies.bootstrap` and `studies.fractions_skill_score` stay put and import only `studies.*`
  and third-party code.** When #805 or #808 starts, it creates `packages/evaluation` (depending on
  `polars` and `numpy` only), writes the production-quality functions with its own tests, and then
  switches the study callers. That change runs the same reproduction check as this one.
- **Open question:** whether the maintainer prefers the extraction now. If so, it is a separate
  Opus-reviewed PR after the folder move, not part of it, and #805 and #808 are updated to
  reference it.

## Tests

**One convention: every test of study code lives under `packages/studies/tests/`, with the tests of
a study's scripts in `packages/studies/tests/<study>/`.** Library tests stay at the top of that
directory. Today 18 files in `tests/` and 12 in `packages/studies/tests/` test scripts, using two
styles: `spec_from_file_location` with a copied `SCRIPT_DIR`, and `sys.path.insert` at module top
with `ty` `extra-paths` to match.

- **Discovery needs no configuration change.** Pytest already collects `packages/*/tests/`, and
  importlib mode (`addopts`) lets two test files share a basename. The one rule is that no script
  under `studies/` is named `test_*.py`.
- **Imports resolve through one fixture.** `packages/studies/tests/conftest.py` gains
  `study_script(study, module)`, which puts `studies/<study>/` on `sys.path`, imports the module
  under a unique name, and removes both afterwards. It replaces the two loader styles now in use. Because a script then imports only its
  folder and `studies.*`, one folder on the path is always enough. `ty` `extra-paths` keeps one entry per folder that a test still imports
  statically; entries for folders no test imports are deleted.
- **The study skill's per-script review rule is unchanged.** It covers `studies/**`, which now holds
  scripts only. Tests sit under `packages/studies`, so they get the package's treatment: the diff
  review and the mutation pass.
- **Alternative considered: a `tests/` folder inside each study folder.** It keeps tests beside
  scripts, but it needs a `conftest.py` and a `pythonpath` entry per folder, makes `studies/` hold
  tested code that `studies/README.md` says it does not hold, and splits `uv run pytest
  packages/studies` into 12 locations. Rejected.
- **New tests and what each asserts that fails on `main`:**

  | Test | Fails on `main` because |
  |---|---|
  | `test_study_boundaries` (allowlist form, then empty) | `main` has crossing imports; the final commit asserts the allowlist is empty |
  | `test_production_does_not_import_studies` | passes today; it is a guard, listed as one |
  | `test_every_script_is_in_a_study_folder` | 79 scripts sit in one folder that the folder-map test rejects |
  | `test_data_registry_producers_exist` | no registry exists |
  | characterisation tests listed under "What moves" | the moved functions are untested by name today |

## Data

**Leave every output where it is, and add a path module and a registry.** Moving 45 directories
under `data/studies/` would change the path in saved fingerprints, in 50 `docs/` references and in
every script's constants, and the study skill forbids overwriting an output a merged page quotes.
The issue allows this ("leave them in place and point the new scripts at them").

- **`studies.data_paths` holds every path constant and one function per role:** `weather_dir(product)`,
  `anm_dir()`, `study_dir(name)` for a study's losses, fitted parameters and reports, and
  `cache_dir(kind, name)` for cached intermediate frames.
- **The convention for new work:** a cached intermediate frame goes under `data/studies/cache/<kind>/`,
  and `data/studies/<study>/` holds only final losses, intervals and reports. Existing caches are not
  moved.
- **`data_paths.CACHES` is the registry of existing reusable caches.** One entry per cache: the
  path, the producing script, and the scripts that read it. Candidates are `weather/<PRODUCT>/`,
  `beam_diffuse_dataset_<source>.parquet`, `ens_forecast_horizons/ens_members.parquet`, and the
  per-site frames `extract_site_series.py` and `fetch_open_meteo_point.py` write.
  `test_data_registry_producers_exist` asserts that each entry's producing script exists in the
  folder map.
- **Duplicated caches:** the issue reports two studies rebuilding the same CAMS extract. The
  implementer lists every writer of CAMS-derived frames (`grep` for the CAMS paths, `du` the
  directories; read-only). Where two caches hold byte-identical content, the later script reads the
  earlier path and the registry says so. Where they differ, both stay and the registry says why.
  Nothing under `data/` is deleted or rewritten.
- **`studies/README.md` gains a "Where data lives" section** stating the convention, and its table
  of studies (which lists three of the ten today) gains one row per folder.

## Order of mechanical steps

**Every commit passes `ruff check`, `ruff format --check`, `pydoclint`, `ty check`, `pytest`,
`pymarkdown`, `mkdocs build --strict` and `check_docs_links`**, per the run-every-CI-step-locally
rule. A `git mv` commit changes no content beyond path strings, so `git diff -M` reads as renames.

0. **Baseline (no commit).** On the merged `main`: save the golden outputs (see "Reproduction"), save
   `edges.py` (the import scan used for this plan) to the session scratch, and check the CPU load.
1. **Guards.** Add `test_study_boundaries` with a `FOLDER_MAP` literal (the table above) and an
   allowlist of every current crossing; add `test_production_does_not_import_studies`.
2. **Layer 1, one commit per module** in the table order: `data_paths`, `sources`, `era5_grid`,
   `commissioning`, `physics_model`, `export_cap`, `pv_dataset`, `arm_runner`, `open_meteo_point`,
   `figure_numbers`. Each commit moves the code, deletes the script's copy, switches every caller
   from a bare import to `from studies.<module> import`, shrinks the allowlist, and adds or moves its
   tests. Scripts that were only a library (`era5_grid`, `commissioning`, `physics_model`,
   `export_cap`, `figure_numbers`, `sources`) are deleted from `studies/`.
3. **Layer 2, one commit per module**, in the table order. Characterisation tests land in a commit
   before the move.
4. **The folder move, one commit per destination folder.** `git mv`; replace every
   `sys.path.insert` into a sibling folder (none should remain); update `pyproject.toml`
   `extra-paths`; fix each script's own `parents[...]` (depth stays at `studies/<folder>/`, so
   `parents[2]` is still the repository root: the 39 `Path(__file__)` uses are checked, not assumed);
   move the tests to `packages/studies/tests/<study>/` and swap in `study_script`; split
   `beam_diffuse_split/README.md` into one README per folder (the script table is the bulk); update
   every path in `docs/`, scripts' docstrings and comments, `studies/README.md`, and the three skills
   that name a path (`data-download`, `dataviz`, `data-validation`). Delete `FOLDER_MAP` and the
   allowlist in the last of these commits, leaving the test to read the filesystem.
5. **Data module and registry** (`CACHES`, `README` section), if not already in step 2.
6. **Skills and `CLAUDE.md`.** In `.claude/skills/study/SKILL.md`, the "Where a study's pieces live"
   table (new folder rule, the import rule, the test location, the data convention) and the sentence
   about `wind_products.py` importing private helpers, which is obsolete. In `CLAUDE.md`, the
   Packages table row for `studies` and the skills table if any summary changed (sweep the summaries,
   per the project memory). In `docs/architecture/testing.md`, a "Study tests" paragraph recording the
   convention, because the issue requires it there.
7. **Verification and reviews** (below).

**Which parts a Sonnet implementer does mechanically:** steps 0, 1, 4, 5 and 6, and the moves in
steps 2 and 3. Judgement stays with the maintainer or Opus in two places: the public names chosen
when an underscore-private symbol becomes a package API, and the layer-2 grouping into modules,
which this plan proposes but the scan on the merged `main` may change.

## Reproduction

**Every report-producing script must print the same published numbers after the move.** The 17
scripts with `--report-only` are `blend_products`, `blend_satellites`, `cerra_past_solar`,
`cerra_wind_direction`, `cerra_wind_levels`, `ens_forecast_horizons`, `ens_hres_past_wind`,
`ens_past_solar`, `reanalysis_past_wind`, `station_past_solar`, `station_wind_arms`,
`weather_products`, `wind_icon_dream`, `fit_extra_leads`, `nwp_forecast_comparison`,
`ensemble_means_mae` and `local_ens_gap`. The leaderboard scripts (`past_solar_leaderboard`,
`past_wind_leaderboard`) and `fractions_skill_score` read saved losses with no flag.

- **Run against a scratch data root, never the shared one.** `--report-only` rewrites `report.md`,
  and every worktree shares the main checkout's `data/studies/`. Set `DATA_PATH_INTERNAL` to a
  scratch directory on the home partition (not `/tmp`), holding copies of only the output
  directories those scripts read (small compared with the 4.8 GB `ens_forecast_horizons/`, but the
  implementer measures with `du` first). Run the 19 scripts on `main`, save every `report.md`,
  `intervals.parquet` and the leaderboard outputs, repeat on the branch, and compare with `cmp`
  (reports byte for byte) and a Polars `frame_equal` on parquet files.
- **Per-row outputs are the bit-for-bit level**, as the study skill requires. Because `--report-only`
  recomputes no per-row loss, add two checks that do: rebuild one `beam_diffuse_dataset_<source>.parquet`
  with `build_dataset.py` into the scratch root and compare its hash with `main`'s, and run
  `blend_products.py` once, which refits every published single-product arm and **stops unless each
  reproduces the published per-row losses bit for bit** (`reproduction.md`). That is the one
  expensive run; the implementer confirms the machine is idle first and states the runtime from the
  last run rather than guessing it.
- **Smoke every script:** `uv run python studies/<folder>/<script>.py --help` for each script with a
  command line (exit 0), and an import of each of the rest. The study skill's rule applies: no linter
  evaluates a `sys.path` string.
- **Run order respects the review rule.** The `main` baseline runs first (reviewed code). The branch
  runs only after the Opus diff review is triaged, because the moved scripts are changed scripts.

## Docs to update

- **125 lines in 12 files under `docs/`** name a script path (`past-weather/wind.md` 34,
  `past-weather/solar.md` 25, `beam-diffuse-split.md` 12, `blending.md` 9, and eight pages with 1
  to 5 each, including `roadmap/data-sources.md`). A `git mv` map file drives one scripted
  replacement of `studies/<old>/<name>.py` with `studies/<new>/<name>.py`; pages that name only a
  folder (`wind.md` line 1640: "in `studies/beam_diffuse_split/`") are edited by hand.
- **Rendered check:** `uv run mkdocs build --strict` and `check_docs_links.py` pass, and the
  implementer opens the wind and solar pages' reproduction-command blocks in the built HTML and
  confirms each command's file exists.
- **Outside `docs/`:** 103 lines in the scripts' docstrings and READMEs, 29 test files, three
  skills, `pyproject.toml` comments, `studies/README.md`, `packages/studies/README.md`.
- **Prose:** written about the present ("each study has a folder"), with no history of the old
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
  the change needs. It also reviews the public names and module grouping of layers 1 and 2, which is
  the extraction API review for this work.
- **Diff review 2 (Opus mutation pass), limited to the tests this work adds** (the characterisation
  tests and the guards), as the study skill requires whenever `packages/studies` changes.
- **No Sonnet review of its own output**: Sonnet implements, Opus reviews, per the project memory.

## Risks and open questions

- **The 44-versus-79 and 30-versus-125 counts mean the issue under-sized itself.** The plan covers
  the true counts. If the maintainer wants a smaller first PR, the natural cut is steps 1 to 2 plus
  step 5 (layer 1 and the data module), with the folder move following.
- **Layer 2 is the one place the plan may be wrong.** I grouped the symbols by the scan on today's
  `main`; branch 1016 adds imports from `nwp_forecast_comparison/`, `fit_extra_leads` and
  `nwp_forecast_charts`. The implementer re-runs the scan, and the module grouping above is a
  recommendation, not a contract.
- **Three placements are judgement calls:** `stamp_alignment` (measures NGED's timestamp lag; put in
  `tools/`), `ens_horizons` and `multi_nwp` (early ENS and second-NWP studies kept in
  `beam_diffuse_split/`, though the ENS-horizons page has superseded the first), and
  `verify_ukv_lineage` (in `past_weather_solar/` because it imports `fetch_open_meteo_point`).
  Whether the two early scripts still earn a place under `studies/README.md`'s "not an attic" rule
  is a question for the maintainer; this plan does not delete them.
- **Splitting `nwp_forecast_comparison/` (three pages, 17 scripts) is not planned.** Its scripts
  already import each other tightly. A split is a follow-up if wanted.
- **The tests move from `tests/` to `packages/studies/tests/`.** The package is a leaf dependency
  absent from the production image, so the move cannot reach production; the cost is that
  `packages/studies/tests/` grows to about 80 files.
- **Reproduction needs a scratch data root of several gigabytes.** The home partition has room
  (486 GB free). Nothing is written under the shared `data/`.
- **No reviewer or implementer should publish a generator's name, ID or coordinates** in a commit,
  README or PR body while splitting the READMEs; the existing anonymisation rules still apply to
  every moved chart script.
