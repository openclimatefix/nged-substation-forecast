# Plan: one tested fit loop, bootstrap and grid sampler for the studies (#822)

**The problem.** The beam/diffuse study fits its models in eight scripts that each re-derive part of
one fold loop: cut folds, train on the other folds minus curtailed hours, fit per seed, clamp to the
export cap, write per-row losses, bootstrap a paired difference. Five `xgb.train` calls and three
Powell physics fits sit inside those copies, and the copies have already drifted. `era_comparison.py`
and `multi_nwp.py` report their arm tables as a mean of per-row error over capacity and their
contrast tables as a megawatt difference over mean capacity, so the published UKV − ERA5 contrast
(−0.3355) does not match the arm tables it sits beside (9.572 − 9.895 = −0.323).
`run_hybrid_experiment.py` reaches the loop by monkeypatching two `run_experiment` module attributes.
#809 and #810 are the next two studies, and both need this loop.

**The solution.** Move the loop into `packages/studies/` as plain functions with tests, parameterised
by a fit-and-predict callable, so the XGBoost and physics fits share one loop. Give it one
normalisation: every error is divided by its own row's capacity before any mean or difference. Give
it one mean-difference bootstrap, and keep the Fractions Skill Score bootstrap separate because it
bootstraps a different statistic. Promote fold assignment with an optional within-era cut for #809,
and promote the nearest-cell grid sampler for #806. The refactor must reproduce the headline run
**bit for bit**. That property is checkable, so the plan checks it by running the old code and the
new code on the same dataset and comparing the outputs exactly. Only the two scripts carrying the
normalisation trap, and `ens_horizons.py`'s interval digits, change their published output.

## Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.** #809 names this refactor as its
precondition, and the survey found the drift the issue predicted.

**Size: complex.** The five triggers:

- **What gets stored:** yes. The study's result files are rewritten, and two published tables
  change.
- **Production serving path:** no. Nothing under `studies/` or `packages/studies/` is imported by
  `src/`.
- **Degradation rule:** no. This is R&D code, which fails fast.
- **More than one defensible design:** yes. The choices are whether the loop is generic over the
  fitter, whether to unify the bootstraps, and how to cut folds within eras.
- **Callers I could not name without searching:** yes. Twelve scripts call `_bootstrap_difference`,
  and eight fit models.

That buys both plan reviews and both diff reviews, run on Opus 5.5 as the maintainer asked.

**Departures from the issue body:**

1. **`shared_geometry._fit` is not an XGBoost fit.** The issue lists it among the XGBoost fit
   sites. It is a Powell fit of the physical model. Three physics scripts repeat the fold loop
   around that fitter: `run_physics_experiment._run_site_arm`, `shared_geometry._losses_for_site`
   and `run_hybrid_experiment._add_physics_predictions`. The promoted loop therefore takes a fitter
   as an argument, rather than being an XGBoost loop.
2. **The Fractions Skill Score bootstrap stays a separate function.** It bootstraps a difference
   of two ratios of sums, which cannot be written as a mean of per-row differences. It moves into
   `studies.fractions_skill_score`, beside the statistic it resamples, and gets a test.
   `ens_horizons._paired_month_bootstrap` is dropped. For one seed and equal monthly row counts it
   is the same statistic as `_bootstrap_difference`, so `ens_horizons.py` calls the shared function
   instead.
3. **`ens_horizons.py` keeps its own training choices.** Those are the squared-error objective,
   400 rounds, one seed, and no `~constrained` filter. It calls the shared loop with those choices
   passed explicitly, so every deviation from the other scripts is visible at one call site.
   Whether the deviations are right belongs to #810, which owns the ENS study and will re-run it.
4. **"Done when the study is re-run" narrows to what the refactor can change.** Four kinds of run
   are in scope:
   - the headline `run_experiment.py --source cams` run, compared bit for bit;
   - the physics and hybrid runs on `cams`, compared bit for bit;
   - `era_comparison.py` and `multi_nwp.py`, whose contrast tables change by design;
   - `ens_horizons.py`, whose interval digits change.

   Every other script only reads the saved losses, and its output is unchanged if the losses are.

## What changes, file by file

### New and changed modules in `packages/studies/src/studies/`

- **`folds.py`** (new) exports one function, `assign_folds(*, dataset, n_folds, era_starts=())`.
  It moves `run_experiment._assign_folds` (a dense month rank per site, scaled to `n_folds` and
  clipped) and adds one optional argument. `era_starts` is a tuple of `%Y-%m` month labels. Each
  label begins a new era, and the rank is taken per `(site, era)`, so every fold holds a slice of
  every era. With `era_starts=()` the function reproduces today's folds exactly. #809 is the first
  caller to need the within-era cut: it is the fix for the UKV upgrade confound the issue describes.
  `era_comparison.py` keeps its current scheme of separate fits per era, because its question is
  "is the ordering the same in each era", which needs disjoint fits.
- **`boosted_trees.py`** (new) holds the XGBoost settings and primitives moved from
  `run_experiment.py`:
  - the `HyperParameters` TypedDict, `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS`,
    with the `colsample_bytree` docstring moved with them;
  - `QUANTILE_LEVELS` and `QUANTILE_LEVEL_SPACING`;
  - `booster_parameters(*, hyper_parameters, seed, threads)`;
  - `crps(*, actual, quantiles)`;
  - `fit_and_predict(*, train, test, features, target, hyper_parameters, seed, objective, threads,
    rounds=None)`, which returns the point prediction for the named objective. `rounds` overrides
    `num_boost_round`; `ens_horizons.py` passes 400;
  - `fit_and_predict_quantiles(...)`, which returns the `(n_rows, n_levels)` quantile matrix.

  Two differences from today, both intended to change nothing: `threads` stops being a module
  constant, and the objective becomes an argument rather than being added by each caller. The five
  `xgb.train` calls all route through these two functions.
- **`cross_validation.py`** (new) holds the loop.
  `out_of_fold_predictions(*, site_rows, fit_and_predict, seeds, n_folds, train_rows)`:
  - loops fold, then seed, in today's order;
  - calls `fit_and_predict(train=..., test=..., fold=..., seed=...)`, which returns a
    `Predictions` named tuple of a point array and an optional quantile matrix;
  - skips an empty train or test fold, as today;
  - returns the test rows with `fold`, `seed`, `prediction_mw`, and `quantiles` when present.

  `train_rows` is the training-row filter as a Polars expression. Every caller passes
  `~pl.col("constrained")` except `ens_horizons.py`, which passes `pl.lit(True)`.

  `losses_from_predictions(*, predictions, target)` builds the six loss columns `run_experiment`
  writes today, in the same dtypes and the same arithmetic order. The order matters because
  `absolute_error_mw` is computed in float32 before its cast, and changing that moves the seventh
  significant figure. It also adds the two `*_fraction_of_capacity` columns, which today are added
  in `run_experiment.main`.

  `clamp_to_cap` moves here from `export_cap.py`, because the loss builder needs it and it is pure.
  `export_cap.py` keeps `with_export_cap`, which reads the data store.
- **`bootstrap.py`** (new) holds `paired_differences`, `bootstrap_difference` and
  `per_fold_differences`, moved from `run_experiment.py`, with two generalisations:
  - the seed axis is read from the data (`sorted(unique seeds)`) rather than from the `SEEDS`
    constant, which is what lets `ens_horizons.py` call it with one seed;
  - the column naming the variant is an argument that defaults to `"arm"`.

  The random stream is preserved exactly: a fresh `default_rng(seed)` per call, then per resample
  the seed draw followed by the month draw. `n_resamples` and `rng_seed` become arguments with
  today's defaults.
- **`fractions_skill_score.py`** gains `bootstrap_fss_difference`, moved from the study script
  unchanged, with a docstring saying why it is not `bootstrap_difference`.
- **`grid_sampling.py`** (new) holds
  `sample_nearest_cell(*, field, sites, crs=None, x_coordinate, y_coordinate)`, moved from
  `verify_ukv_lineage._sample_grid`:
  - it takes the 2-D `xarray.DataArray` rather than finding it, so the "exactly one 2-D variable"
    rule stays in the UKV script;
  - it takes the coordinate names rather than hard-coding UKV's;
  - with `crs=None` it samples a latitude/longitude grid directly, which is SARAH-3's case in #806.
- **`packages/studies/pyproject.toml`** adds `xgboost`, `xarray` and `pyproj`. All three are already
  in the lock, so the lock changes only in the dependency edges. The `--no-dev` check has to stay
  green: none of the three may reach the production image through `studies`, which is a dev
  dependency.

### Study scripts in `studies/beam_diffuse_split/`

- **`run_experiment.py`** keeps its arms, contrasts, jobs, thread pool and `main`, and loses
  `_assign_folds`, `_booster_parameters`, `_fit_one_fold`, `_crps`, `_paired_differences`,
  `_bootstrap_difference`, `_per_fold_differences` and the hyperparameter constants.
  `_run_site_arm` becomes a call to `out_of_fold_predictions` with a closure over the arm's
  features, then `losses_from_predictions`. `_direct_fraction_predictability` and
  `_add_learned_split` call `fit_and_predict` with `objective="reg:squarederror"`, seed 0, and
  their own all-site training rows.
- **`_run_all` takes `arm_features: Mapping[str, tuple[str, ...]]` and
  `shared_features: tuple[str, ...]` as arguments.** That removes `run_hybrid_experiment.py`'s
  monkeypatch of `run_experiment.ARM_FEATURES` and `_features_for`, which would silently stop
  working once the loop moved. `H_physics_only` passes an empty `shared_features`.
- **`run_physics_experiment.py`, `shared_geometry.py` and `run_hybrid_experiment.py`** call
  `out_of_fold_predictions` with their Powell fitter wrapped as the `fit_and_predict` callable. The
  physics `_run_all` and `_intervals_for` duplicates go, in favour of `run_experiment`'s and the
  package's.
- **`era_comparison.py` and `multi_nwp.py`** call the loop, and bootstrap
  `absolute_error_capped_fraction_of_capacity` rather than the megawatt metric divided by mean
  capacity afterwards. Their arm and contrast tables then subtract exactly. This is the one
  intended change to published numbers.
- **`ens_horizons.py`** calls the loop and `bootstrap_difference`, and drops
  `_paired_month_bootstrap`.
- **`verify_ukv_lineage.py`** finds its one 2-D variable and passes it, with the UKV coordinate
  names and CRS, to `sample_nearest_cell`.
- **The twelve `_bootstrap_difference` importers** import `studies.bootstrap.bootstrap_difference`
  instead: `anm_curtailment`, `capacity_denominator`, `compare_sources`, `inverter_clipping`,
  `oracle_capacity`, `sky_conditions` and the fitting scripts. `restart_basins` and `sky_conditions`
  import `assign_folds`.
- **`studies/beam_diffuse_split/fractions_skill_score.py`** imports `bootstrap_fss_difference`.

## Design-philosophy check

This is R&D code, which fails fast
(<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/>).
The loop raises on a malformed frame rather than degrading. It adds no asset and no asset check.

Of the design principles, the change buys "one copy of a rule" for the fold loop and the
normalisation, at the price of one callable-typed parameter. No principle is traded away.

## Tests (`packages/studies/tests/`)

Each test names what it asserts that fails on `main`. On `main` none of these functions exists in
the package, so each test first fails on import. The assertion that matters is listed.

- **`test_folds.py`:**
  - folds are contiguous whole months per site, with no month split across folds;
  - a site with a shorter span still fills all `n_folds`;
  - with `era_starts=("2026-02",)`, every fold of every site holds months from both eras;
  - with `era_starts=()` the folds equal the old `_assign_folds` on a fixed frame, pinned as a
    literal.
- **`test_boosted_trees.py`:**
  - `crps` on a hand-computed two-row case;
  - crossing quantiles are sorted before scoring;
  - `booster_parameters` carries no `colsample_bytree`, so the docstring's rule is pinned;
  - `fit_and_predict` with the same seed twice gives identical predictions, and different seeds
    give different predictions.
- **`test_cross_validation.py`:**
  - no training row ever comes from the test fold;
  - constrained rows are absent from training but present in scoring;
  - an empty fold is skipped;
  - every `(test row, seed)` appears exactly once.

  These use a fitter stub that records its train and test frames, so the loop is tested without
  XGBoost.
- **`losses_from_predictions`:**
  - the clamp applies to the capped columns only;
  - the fraction-of-capacity column equals the megawatt error divided by that row's capacity;
  - two sites with different capacities give a pooled mean of per-row ratios, not a ratio of means.
    That is the #809 normalisation trap, pinned.
- **`test_bootstrap.py`:**
  - the point estimate is the mean paired difference;
  - a constant difference gives a zero-width interval at that value;
  - one seed works, where `SEEDS` was hard-coded before;
  - the random stream equals the old implementation's on a fixed frame, pinned as literal interval
    bounds.
- **`bootstrap_fss_difference`:** identical arms give an interval of exactly zero, and swapping the
  arms negates it.
- **`test_grid_sampling.py`:**
  - an in-memory Lambert azimuthal equal-area grid, with each cell holding its own index, returns
    the right cell for points placed inside known cells;
  - a latitude/longitude grid with `crs=None` does the same;
  - no netCDF library, network or data store is involved.

## The reproduction check (the evidence that the refactor changed nothing it should not)

1. On `main`, rebuild the CAMS dataset with `build_dataset.py --source cams` into a scratch copy of
   the study data directory. Run `run_experiment.py --source cams`, `run_physics_experiment.py
   --source cams` and `run_hybrid_experiment.py --source cams` into that copy. `DATA_PATH_INTERNAL`
   points each run at the copy, which holds symlinks to the NGED tables and the weather downloads.
2. On the branch, run the same three scripts on the same dataset into a second copy.
3. Compare `per_row_losses.parquet` sorted by `(setting, arm, site, time, seed)`, and
   `bootstrap_intervals.parquet` and `per_site_summary.parquet`, with `polars.testing` exact
   equality. Any difference is a bug in the refactor, not a finding.
4. Run `era_comparison.py`, `multi_nwp.py` and `ens_horizons.py` on the branch, and record the new
   contrast tables in the PR body beside the old ones.

The rebuilt dataset should also equal the `_piecewise` build already on disk, which the first review
of #818 confirmed for `power_mw` row by row. Rebuilding is still necessary, because the committed
code reads unsuffixed names.

## Docs to update

- **`docs/studies/beam-diffuse-split.md`:** nothing in the headline, because it is reproduced. The
  page does not publish the era or multi-NWP contrasts, so nothing else changes there.
- **`docs/roadmap/data-sources.md` 475–535:** replace the UKV − ERA5 "0.34 points, 0.10 to 0.56"
  with the re-normalised contrast.
- **The #809 and #810 issue bodies:** one comment each, giving the re-normalised figures, because
  their tables quote these scripts.
- **`packages/studies/README.md`:** one line per new module.
- **`studies/beam_diffuse_split/README.md`:** the script table where a script's role changed.

## Verification commands

The implement-issue set:

- `uv run ruff check .`, `uv run ruff format .`, and `uv run --all-packages ty check`;
- `uv run pytest`;
- pydoclint and pymarkdown, as in `ci.yml`, and `scripts/lint/check_docs_links.py`;
- `uv run mkdocs build --strict`.

Two more for this change:

- **The `--no-dev` check:**
  `test "$(uv export --no-dev --format requirements-txt | grep -ciE '^(pvlib|cdsapi|xgboost)')" -eq 0`.
  xgboost is already in the image through `xgboost_forecaster`, so that pattern is wrong for
  xgboost. The check to run is that the `--no-dev` export is unchanged from `main`.
- **The reproduction check** above.

## Risks and open questions

1. **Should the loop be generic over the fitter, or XGBoost-only with the physics loops left
   alone?** *Recommendation: generic.* The price is one `Callable` parameter. The alternative
   leaves three copies of the fold loop, the ones the physics scripts carry, and #809 will fit a
   physical model per product too.
2. **Should the within-era fold cut ship now, with no caller until #809?** *Recommendation: yes.* It
   is a four-line extension of a function being moved anyway, #822's body names it, and #809 starts
   next. The alternative is adding it inside #809's PR, which is also defensible.
3. **Should `ens_horizons.py`'s deviations be harmonised here?** *Recommendation: no, leave them to
   #810.* These are the missing `~constrained` filter, squared error, and 400 rounds. The
   `~constrained` omission is probably a defect, because the model trains on hours the network
   operator turned down, and #810's plan should settle it.
4. **Row order.** With `subsample=0.8`, XGBoost samples rows by position, so a refactor that
   reorders training rows changes every prediction. The loop must filter from the same site-sorted
   frame in the same order. The reproduction check is what would catch a slip.
5. **The data on disk carries `_piecewise` and similar suffixes that the committed code never
   writes.** The re-run writes unsuffixed names beside them. *Recommendation:* once the reproduction
   check passes, move the suffixed outputs into `data/studies/beam_diffuse_split/superseded/` rather
   than deleting them.
