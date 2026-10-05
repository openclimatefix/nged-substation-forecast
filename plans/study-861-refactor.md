# Plan: give each study family its own folder and move shared study code into `packages/studies` (#861)

**The problem is that `studies/beam_diffuse_split/` holds 79 scripts for about ten different study
pages, and the scripts reach each other, and other folders, by bare imports.** The issue counted 44
scripts; `main` now holds 79 in that folder, 128 under `studies/` in all (excluding
`era_fold_design/scripts`), and 77,000 lines. An AST scan of the imports found 56 pairs of
(importing folder, imported module) that cross a folder boundary once the scripts are sorted by
study family. Ten scripts hide a crossing behind `sys.path.insert`, which `ruff` and `ty` cannot
check. Published pages cite script paths on 81 lines in 11 files in `docs/`, 58 of them naming a
script that moves, not the 30 the issue estimated.

**The plan is to sort the scripts into folders that mirror the docs page families, and to make the
crossings impossible rather than merely rare.** A script may import only from its own folder and
from `studies.*` (`packages/studies`). Everything a second folder needs moves into the package, in
two layers. One AST test, added in the last commit, enforces the rule. Study outputs stay where they
are. Tests of study scripts move to one tree. The general Fractions Skill Score and paired block
bootstrap are not extracted here (issues #805 and #808): they wait for their production callers.

## Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.** The preconditions (#858, #859)
have merged. This plan is written against `main`. Branch `study-1016-ukv-blends` is running a study
and is unmerged. It edits `nwp_forecast_comparison/fit_aifs.py`, `nwp_forecast_comparison.py` and
`packages/studies/src/studies/ifs_single_runs.py`, and adds `studies/ukv_ceda_blends/` (five
scripts that reach `nwp_forecast_comparison/`, `beam_diffuse_split/` and `weather_downloads/`
through `sys.path` inserts). This plan names no module for that branch. The implementer re-runs the
import scan in step 0 on whatever `main` holds then. If the branch has merged, the scan lists the
crossings it adds, and they join layer 1 or layer 2 like any other.

**Size: complex.** The five triggers:

- **What gets stored:** no. No Patito model, Delta table or asset changes, and every output keeps
  its path.
- **Production serving path:** no. Nothing in `src/` or `packages/` imports `studies` today, and a
  new test pins that.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The folder grouping, the number of extraction layers,
  the test location, and the #805/#808 boundary each have a defensible alternative.
- **Callers not nameable without searching:** yes. 128 scripts, 49 files in `tests/` and 41 in
  `packages/studies/tests/`, and 81 docs lines.

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
`beam_diffuse_split/`, `era_fold_design/`, `weather_downloads/`, `open_meteo_ensemble_means/`,
`ens_backfill_pilot/` and `ukv_ceda_blends/` (where it exists).

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

**`era_fold_design/scripts/partC_*.py` import `ens_forecast_horizons` and `blend_products`, which
both move.** Those scripts are a record the `ruff` and `ty` configurations exclude, and the boundary
test excludes them too. The implementer repoints their imports at the layer-2 `studies.*` modules
and confirms each still imports. A symbol that did not move is left alone and reported.

## What moves to `packages/studies`

**Layer 1 moves code every study reads; each module is one commit, with tests.** Underscore-private
names lose their underscore on the way, because a name imported from a package is public.

| New module | Comes from | What it holds |
|---|---|---|
| `studies.sources` | `sources.py`, wholesale | `SourceType`, `SOURCE_CHOICES`, `PER_SITE_SOURCES`, `OpenMeteoModel` and its registry, `point_output_path_for`, and the path constants (`REPO_DATA_DIR`, `STUDIES_DATA_DIR`, `WEATHER_DATA_DIR`, `ANM_DATA_DIR`). `weather_downloads/paths.py` imports `REPO_DATA_DIR` from here instead of defining it a second time. |
| `studies.pv_dataset` | `build_dataset.py` | The site rosters (`pv_sites`, `wind_sites`), `hourly_power`, `read_era5`, `nearest_era5_cell`, `read_cams`, the outage and false-zero filters, the separation-model columns. The build's command line stays in `beam_diffuse_split/build_dataset.py`. |
| `studies.arm_runner` | `run_experiment.py` | `Job`, `run_all`, `MAX_CONCURRENT_FITS`, `SHARED_FEATURES`, `add_time_features`, `dataset_path_for`. |
| `studies.commissioning`, `studies.export_cap`, `studies.physics_model`, `studies.era5_grid`, `studies.figure_numbers` | the scripts of the same name | Whole modules. |
| `studies.open_meteo_point` | `fetch_open_meteo_point.py` | `fetch_point_frame`, `HOURLY_VARIABLES`, the timestamp-convention checks. The `--model` command line stays. It moves because `weather_downloads/` and `beam_diffuse_split/` scripts import it as well as `past_weather/`. |

**Layer 2 is expected to be about 28 symbols in 8 (importing folder, imported module) pairs.** The
scan counts 114 symbols in 24 pairs under the one-folder-per-page grouping this plan replaced, and
the `past_weather/` and `nwp_forecast_comparison/` grouping turns most of them into same-folder
imports. The 28 are an estimate from the measured scan, not a contract: the implementer re-derives
them in step 0 and records the real number. The estimate divides as follows.

| Candidate module | Symbols | Used by |
|---|---|---|
| `studies.product_frames` | About 9: `solar_frame`, `wind_frame`, `SOLAR`, `WIND`, `Domain`, `PERMUTATION_GROUPS` and neighbours, from `blend_products.py` | `nwp_forecast_comparison/`, `open_meteo_ensemble_means/`, `era_fold_design/scripts/partC_*.py` |
| `studies.ens_members` | About 7: the symbols other folders take from `ens_forecast_horizons.py` (`ENSEMBLE_SIZE`, `H3_RESOLUTION`, `reduce_members`, `ens_columns` and neighbours) | `past_weather/`, `era_fold_design/scripts/partC_*.py` |
| `studies.wn3_fetch` | About 8 attributes of `weather_downloads/fetch_weathernext3.py` | `nwp_forecast_comparison/build_wn3_inputs.py` |
| singletons | About 4 | the scan lists them |

**Departure from "every function carries tests" for layer 2, with its reason.** A moved
report-line builder is covered by a stronger check than a unit test: the saved-loss reports must
come out byte for byte identical (see "Reproduction"). Row builders, fold cutters and anything that
changes a number get a characterisation test written *before* the move, against the unmoved code,
and moved with it. That covers `solar_frame`, `wind_frame`, `add_time_features`, the site rosters
and `with_export_cap`. Existing tests of moved code (`test_weather_products_served_lead`,
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
directory. Today 18 files in `tests/` and 12 in `packages/studies/tests/` test scripts, using two
styles: `spec_from_file_location` with a copied `SCRIPT_DIR`, and `sys.path.insert` at module top
with `ty` `extra-paths` to match.

- **Each study folder is listed once in pytest `pythonpath` and once in `ty` `extra-paths`.** Tests
  then keep static lines such as `from wind_products import ...`, which `ty` checks. Both settings
  already list some folders, and the entries for folders that hold no test-imported script are
  fine to keep, because the boundary test guarantees basenames are unique across `studies/`.
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
  | `test_study_boundaries` (one AST test, last commit) | `main` has crossing imports, `sys.path.insert` calls and duplicate basenames |
  | `test_production_does_not_import_studies` | passes today; it is a guard, listed as one |
  | characterisation tests listed under "What moves" | the moved functions are untested by name today |

  `test_study_boundaries` asserts three things over every script under `studies/` except
  `era_fold_design/scripts`: no import of a module that lives in a different `studies/` folder, no
  `sys.path.insert`, and no basename shared by two scripts.

## Data

**Leave every output where it is.** Moving 45 directories under `data/studies/` would change the
path in saved fingerprints, in 50 `docs/` references and in every script's constants, and the study
skill forbids overwriting an output a merged page quotes. The issue allows this ("leave them in
place and point the new scripts at them"). The path constants move with `studies.sources` and no
registry or role functions are added, because no caller needs them.

- **`studies/README.md` gains a "Where data lives" table.** It states the convention that a cached
  intermediate frame goes under `data/studies/cache/<kind>/` and that `data/studies/<study>/` holds
  only final losses, intervals and reports. Existing caches are not moved. One row per reusable
  cache gives the path, the producing script and the scripts that read it. Candidates are
  `weather/<PRODUCT>/`, `beam_diffuse_dataset_<source>.parquet`,
  `ens_forecast_horizons/ens_members.parquet`, and the per-site frames `extract_site_series.py` and
  `fetch_open_meteo_point.py` write. A table can go stale without a test noticing; the issue asks
  only for a convention a later study can find.
- **Duplicated caches:** the issue reports two studies rebuilding the same CAMS extract. The
  implementer lists every writer of CAMS-derived frames (`grep` for the CAMS paths, `du` the
  directories; read-only). Where two caches hold byte-identical content, the later script reads the
  earlier path and the table says so. Where they differ, both stay and the table says why. Nothing
  under `data/` is deleted or rewritten.
- **The existing table of studies in `studies/README.md` gains one row per folder.** It
  lists three of the ten today. Each folder README maps every script to its published page.

## Order of mechanical steps

**Every commit passes `ruff check`, `ruff format --check`, `pydoclint`, `ty check`, `pytest`,
`pymarkdown`, `mkdocs build --strict` and `check_docs_links`**, per the run-every-CI-step-locally
rule. A `git mv` commit changes no content beyond path strings, so `git diff -M` reads as renames.

0. **Baseline (no commit).** On the merged `main`: re-run the import scan (save it as `edges.py` in
   the session scratch), record the real layer-2 count, save the golden outputs (see
   "Reproduction"), and check the CPU load.
1. **Guard.** Add `test_production_does_not_import_studies`.
2. **Layer 1, one commit per module** in the table order: `sources`, `era5_grid`, `commissioning`,
   `physics_model`, `export_cap`, `pv_dataset`, `arm_runner`, `open_meteo_point`, `figure_numbers`.
   Each commit moves the code, deletes the script's copy, switches every caller from a bare import to
   `from studies.<module> import`, and adds or moves its tests. Scripts that were only a library
   (`era5_grid`, `commissioning`, `physics_model`, `export_cap`, `figure_numbers`, `sources`) are
   deleted from `studies/`.
3. **Layer 2, one commit per module.** Characterisation tests land in a commit before the move.
4. **The folder move, one commit per destination folder** (`past_weather/`, then the four ENS
   scripts into `nwp_forecast_comparison/`). `git mv`; replace every `sys.path.insert` into a
   sibling folder (none should remain); update `pyproject.toml` `pythonpath` and `extra-paths`; fix
   each script's own `parents[...]` (depth stays at `studies/<folder>/`, so `parents[2]` is still the
   repository root: the 38 `Path(__file__)` uses are checked, not assumed); move the tests to
   `packages/studies/tests/<study>/`; split `beam_diffuse_split/README.md` into one README per
   folder, each with a table mapping every script to its published page; update every path in
   `docs/`, scripts' docstrings and comments, `studies/README.md`, and the skills that name a path
   (found by `grep`; `data-download` is one). **The last of these commits adds
   `test_study_boundaries`.**
5. **Skills and `CLAUDE.md`.** In `.claude/skills/study/SKILL.md`, the "Where a study's pieces live"
   table (new folder rule, the import rule, the test location, the data convention) and the sentence
   about `wind_products.py` importing private helpers, which is obsolete. In `CLAUDE.md`, the
   Packages table row for `studies` and the skills table if any summary changed (sweep the summaries,
   per the project memory). In `docs/architecture/testing.md`, a "Study tests" paragraph recording the
   convention, because the issue requires it there.
6. **Verification and reviews** (below).

**Which parts a Sonnet implementer does mechanically:** steps 0, 1, 4 and 5, and the moves in steps 2
and 3. Judgement stays with the maintainer or Opus in two places: the public names chosen when an
underscore-private symbol becomes a package API, and the layer-2 module names, which the step 0 scan
may change.

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
  implementer measures with `du` first). Run the 19 scripts on the branch into the scratch root and
  compare each `report.md`, `intervals.parquet` and leaderboard output against the file already on
  disk in the shared `data/studies/`, using `cmp` for reports and a Polars `frame_equal` for parquet
  files. For a script whose output differs, run `main` into the scratch root too: a difference
  present on `main` is a stale file on disk, not a regression. This saves the full `main` baseline
  and is sound because an equal result needs no baseline and a different one gets one.
- **Per-row outputs are the bit-for-bit level**, as the study skill requires. Because `--report-only`
  recomputes no per-row loss, add two checks that do: rebuild one
  `beam_diffuse_dataset_<source>.parquet` with `build_dataset.py` into the scratch root and compare
  its hash with the one on disk, and run `blend_products.py` once, which refits every published
  single-product arm and **stops unless each reproduces the published per-row losses bit for bit**
  (`reproduction.md`). That is the one expensive run; the implementer confirms the machine is idle
  first and states the runtime from the last run rather than guessing it.
- **Run a moved script end to end, and smoke every script.** `uv run python
  studies/<folder>/<script>.py --help` for each script with a command line (exit 0), and an import
  of each of the rest. The study skill's rule applies: no linter evaluates a `sys.path` string.
- **Run order respects the review rule.** The branch runs only after the Opus diff review is
  triaged, because the moved scripts are changed scripts.

## Docs to update

- **81 lines in 11 files under `docs/`** name a script path, and 58 of them name a script that
  moves (70 name a `beam_diffuse_split` script). A `git mv` map file drives one scripted replacement
  of `studies/<old>/<name>.py` with `studies/<new>/<name>.py`. Pages that name only a folder (`wind.md`
  line 1640: "in `studies/beam_diffuse_split/`") are edited by hand.
- **Rendered check:** `uv run mkdocs build --strict` and `check_docs_links.py` pass, and the
  implementer opens the wind and solar pages' reproduction-command blocks in the built HTML and
  confirms each command's file exists.
- **Outside `docs/`:** a `grep` finds 177 lines in scripts and READMEs under `studies/` and
  `packages/` that name a script path, plus the skills that name a path, `pyproject.toml` comments,
  `studies/README.md`, and `packages/studies/README.md`.
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
  the change needs. It also reviews the public names and module grouping of layers 1 and 2.
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
- **The layer-2 estimate of 28 symbols is from a scan of `main` without branch 1016.** The
  implementer re-runs the scan and records the real number. If it is far above 28, the grouping is
  wrong and the maintainer is told before the moves start.
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
- **Reproduction needs a scratch data root of several gigabytes.** The home partition has room
  (486 GB free). Nothing is written under the shared `data/`.
- **No reviewer or implementer should publish a generator's name, ID or coordinates** in a commit,
  README or PR body while splitting the READMEs; the existing anonymisation rules still apply to
  every moved chart script.
