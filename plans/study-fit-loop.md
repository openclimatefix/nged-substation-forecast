# Plan: one tested fit loop, bootstrap and grid sampler for the studies (#822)

**The problem.** The beam/diffuse study fits its XGBoost models in four scripts, and three of them
re-derive the same per-site, out-of-fold loop:

- `run_experiment._run_site_arm`;
- `era_comparison._losses`;
- `multi_nwp._losses_for_site`.

Each loop cuts folds, trains on the other folds minus curtailed hours, fits once per seed, clamps to
the export cap, and writes per-row losses. The copies have already drifted. `era_comparison.py` and
`multi_nwp.py` report their arm tables as a mean of per-row error over capacity. Their contrast
tables are a megawatt difference over mean capacity instead. So the published UKV − ERA5 contrast
(−0.3355) does not match the arm tables beside it (9.572 − 9.895 = −0.323).

`run_hybrid_experiment.py` reaches the loop by monkeypatching two `run_experiment` module attributes.
That patch would silently stop working once the loop moved. Twelve scripts import the paired
bootstrap from `run_experiment.py`, and the bootstrap hard-codes the three seeds that script uses.
#809 is the next study, and #810 the one after; both need a tested loop to build on.

**The solution.** Move the XGBoost loop into `packages/studies/` almost verbatim, with tests, and
make the two sibling scripts call it. Move the paired bootstrap beside it, reading its seeds from the
data. Move fold assignment, with a grouping argument that lets #809 cut folds within each era of a
product. Move the nearest-cell sampler that #806 needs.

Every error gets one normalisation: divided by its own row's capacity before any mean or difference.
The refactor must reproduce the headline run **bit for bit**. The plan checks that by running the
old code and the new code on the same dataset and comparing the outputs exactly. The two scripts
that carry the normalisation trap are the only ones whose published output changes. Their arm
tables must still reproduce, and only their contrast columns may move.

## Verdict, size and departures

**Verdict: worth doing, with five departures from the issue body.** #809 names this refactor as its
precondition, and a survey of the scripts found the drift the issue predicted.

**Size: complex.** The five triggers:

- **What gets stored:** yes. The study's result files are rewritten, and two published contrast
  tables change.
- **Production serving path:** no. Nothing under `studies/` or `packages/studies/` is imported by
  `src/`.
- **Degradation rule:** no. This is R&D code, which fails fast.
- **More than one defensible design:** yes. The choices are how much of the loop to generalise,
  whether to unify the bootstraps, and how to cut folds within eras.
- **Callers I could not name without searching:** yes. Twelve scripts call the bootstrap.

That buys both plan reviews and both diff reviews, run on Opus 5.5 as the maintainer asked.

**Departures from the issue body:**

1. **The physics fold loops stay in their scripts.** The issue's list includes `shared_geometry._fit`,
   but it is a Powell fit of the physical model, not XGBoost. Three physics scripts repeat a fold
   loop, and only one has the per-arm, per-fold shape the XGBoost loop has:
   `run_physics_experiment._run_site_arm`. The other two do not fit it:
   - `shared_geometry._losses_for_site` runs one reference fit per `(fold, seed)` and feeds its
     geometry to 7 arms under 2 schemes;
   - `run_hybrid_experiment._add_physics_predictions` fits per `(site, {scored fold, own fold})`
     pair to build a feature.

   A loop generic over the fitter would absorb one copy, at the cost of a callable parameter and a
   result type. The scripts' results are published and final, and neither #809 nor #810 names a
   physical model. These scripts change only their imports.
2. **`ens_horizons.py` is left alone.** It trains once per fold and predicts 51 ensemble members
   from that fit. It uses squared error, 400 rounds, one seed, and no `~constrained` filter. Its
   numbers appear only in the #810 issue body, and #810 lists those deviations as traps it will
   redo. `_paired_month_bootstrap`'s docstring already states why it is not `run_experiment`'s
   bootstrap, which is the issue's "three with the reason each differs stated" option.
3. **The Fractions Skill Score bootstrap stays in its study script.** It bootstraps a difference of
   two ratios of sums, which cannot be written as a mean of per-row differences, and no interval
   from it is published. One docstring line is added, saying why it is not `bootstrap_difference`.
4. **The two squared-error fits in `run_experiment.py` keep their own `xgb.train` calls.**
   `_direct_fraction_predictability` and `_add_learned_split` train on all sites with calendar
   months withheld. That is a different scheme from the per-site loop. They import
   `booster_parameters` from the package and change nothing else.
5. **"Done when the study is re-run" narrows to the runs the refactor can change.** Those are the
   headline tree run and the hybrid run, both compared bit for bit, plus `era_comparison.py` and
   `multi_nwp.py`. Every other script either keeps its own loop or only reads saved losses.

## What changes, file by file

### `packages/studies/src/studies/` — three new modules

- **`cross_validation.py`** holds the per-site out-of-fold XGBoost machinery, moved from
  `run_experiment.py` with its docstrings:
  - `N_FOLDS`, `SEEDS` and `THREADS_PER_FIT`;
  - the `HyperParameters` TypedDict, with `PRIMARY_HYPER_PARAMETERS` and
    `SENSITIVITY_HYPER_PARAMETERS`, including the load-bearing `colsample_bytree` docstring;
  - `QUANTILE_LEVELS` and `QUANTILE_LEVEL_SPACING`;
  - `assign_folds(*, dataset, n_folds=N_FOLDS, by=("site",))`, which is today's `_assign_folds` with
    `.over("site")` generalised to `.over(by)`. #809 adds an `era` column and passes
    `("site", "era")`, so every fold holds a slice of every era.
  - `booster_parameters`, `crps`, and `fit_one_fold`, which is today's absolute-error point fit plus
    optional quantiles;
  - `clamp_to_cap`, moved from `export_cap.py` because the loop needs it and it is pure.
    `export_cap.py` keeps `with_export_cap`, which reads the data store.
  - `out_of_fold_losses(*, site_rows, features, target, hyper_parameters, with_quantiles)`, which
    is today's `_run_site_arm` body minus the `arm` and `setting` labels.

  In `out_of_fold_losses`, `features` is a list whose entries may carry a `{fold}` placeholder,
  resolved per fold, which is how arm `B_learned` and the hybrid arms keep their own fold out of
  training. It returns today's six loss columns in today's dtypes and arithmetic order: the float32
  `absolute_error_mw` is computed before its cast, and changing that moves the seventh significant
  figure. It also returns the two `*_fraction_of_capacity` columns, which today are added in
  `run_experiment.main`. The caller adds the labels.
- **`bootstrap.py`** holds `paired_differences`, `bootstrap_difference` and
  `per_fold_differences`, moved from `run_experiment.py`. Two things change:
  - the seed axis is read from the data (`sorted(unique seeds)`) rather than from the `SEEDS`
    constant;
  - `n_resamples` and `rng_seed` become keyword arguments with today's defaults.

  The random stream is preserved exactly: a fresh `default_rng(20260920)` per call, then per
  resample the seed draw followed by the month draw.
- **`grid_sampling.py`** holds `sample_nearest_cell(*, field, sites, crs)`, moved from
  `verify_ukv_lineage._sample_grid`. It takes the 2-D `xarray.DataArray` and the grid's `pyproj`
  CRS, and samples on UKV's `projection_x_coordinate` and `projection_y_coordinate`. The UKV script
  keeps two things: finding the one 2-D variable (`_flux_variable_name`), and reading the CRS from
  its `lambert_azimuthal_equal_area` attributes. #806 adds a latitude/longitude path once it knows
  where SARAH-3 is served and what its coordinates are named.
- **`packages/studies/pyproject.toml`** adds `xgboost`, `xarray` and `pyproj`. All three are
  already in the lock.

### `studies/beam_diffuse_split/`

- **`run_experiment.py`** keeps its arms, contrasts, jobs, thread pool, the two squared-error fits,
  and `main`. It loses everything moved above.
  - `_run_all` takes jobs that each carry a resolved feature list, rather than an arm name looked up
    in the module-level `ARM_FEATURES`. That removes the monkeypatch.
  - `_run_site_arm` becomes a call to `out_of_fold_losses` plus the labels.
  - Row order into XGBoost is unchanged: the same site-sorted frame, filtered the same way. With
    `subsample=0.8`, XGBoost samples rows by position, so any reordering would move every
    prediction.
- **`run_hybrid_experiment.py`** builds its own jobs with the hybrid feature lists. It deletes the
  writes to `run_experiment.ARM_FEATURES` and the replacement `_features_for`. Its physics half is
  unchanged.
- **`era_comparison.py` and `multi_nwp.py`** replace their loops with `out_of_fold_losses`. They
  bootstrap `absolute_error_capped_fraction_of_capacity` instead of the megawatt metric divided by
  mean capacity afterwards, so their arm and contrast tables subtract exactly. This is the intended
  change to published numbers.
- **The twelve `_bootstrap_difference` importers**, and the importers of the moved constants and
  folds, import from the package. These are `anm_curtailment`, `capacity_denominator`,
  `compare_sources`, `inverter_clipping`, `oracle_capacity`, `sky_conditions`, `restart_basins`,
  `shared_geometry`, `run_physics_experiment`, `ens_horizons` (folds and hyperparameters only), and
  the three fitting scripts.
- **`verify_ukv_lineage.py`** passes its variable and CRS to `sample_nearest_cell`.
- **`fractions_skill_score.py`** (the study script) gains one docstring line on
  `_bootstrap_fss_difference`.

## Design-philosophy check

This is R&D code, which fails fast
(<https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/>).
It adds no asset and no asset check. It serves the "one copy of a rule" principle for the fold loop
and the normalisation, and trades none away.

## Tests (`packages/studies/tests/`)

On `main` none of these functions exists in the package, so every test fails there on import. The
assertion that matters is listed for each test.

- **Folds, in `test_cross_validation.py`:**
  - folds are contiguous whole months per site, with no month split across two folds;
  - a site with a shorter span still fills all five folds;
  - with `by=("site", "era")` every fold of every site holds months from both eras;
  - the default folds equal a literal pinned from today's `_assign_folds` on a fixed frame.
- **Booster settings and losses, also in `test_cross_validation.py`:**
  - `crps` on a hand-computed two-row case, with crossing quantiles sorted before scoring;
  - `booster_parameters` carries no `colsample_bytree`;
  - `clamp_to_cap` clamps where a cap exists and passes through where it is null.
- **The loop, on a small synthetic frame with a real XGBoost fit of a few rounds:**
  - no row of the test fold is ever a training row, which is checked by the loop's own output
    covering each row exactly once per seed;
  - constrained rows are scored but not trained on;
  - an empty fold is skipped;
  - a `{fold}` placeholder resolves to that fold's column;
  - the capped columns are clamped and the uncapped columns are not;
  - two sites with different capacities give a pooled mean of per-row ratios, not a ratio of means.
    That is the #809 normalisation trap, pinned.
- **`test_bootstrap.py`:**
  - the point estimate is the mean paired difference;
  - a constant difference gives a zero-width interval at that value;
  - a frame with one seed works, where the hard-coded `SEEDS` failed before;
  - the interval equals a literal pinned from today's implementation on a fixed frame, which pins
    the random stream.
- **`test_grid_sampling.py`:** an in-memory Lambert azimuthal equal-area grid, with each cell
  holding its own index, returns the right cell for points placed inside known cells. It needs no
  netCDF library, network or data store.

## The reproduction check

1. **Baseline, already running.** On `main` (a detached worktree at `276db291`), rebuild the CAMS
   dataset into a scratch data directory. Then run `run_experiment.py`, `run_physics_experiment.py`
   and `run_hybrid_experiment.py`, each with `--source cams`. `DATA_PATH_INTERNAL` points the runs
   at the scratch directory, which symlinks the NGED tables, the weather downloads and the ANM
   exports.
2. **On the branch,** run `run_experiment.py --source cams` against the same dataset. Then run
   `run_hybrid_experiment.py --source cams` against the baseline's physics losses. The physics code
   is unchanged, so its run is not repeated.
3. **Compare** `per_row_losses.parquet`, sorted by `(setting, arm, site, time, seed)`, and
   `bootstrap_intervals.parquet` and `per_site_summary.parquet`, with exact `polars.testing`
   equality. Any difference is a bug in the refactor.
4. **Build the open-meteo, UKV and ICON-D2 datasets on the branch,** then run `era_comparison.py`
   and `multi_nwp.py`. Their arm tables must reproduce the published 9.572 and 9.895, and 6.922 and
   7.224, exactly, because the arm tables already use per-row ratios. Only the contrast columns may
   move. Record the old and new contrasts in the PR body.

## Docs to update

- **`docs/roadmap/data-sources.md` 475–535:** the UKV − ERA5 contrast of "0.34 points, 0.10 to
  0.56" becomes the re-normalised figure.
- **The #809 and #810 issue bodies:** a comment on each giving the re-normalised figures, because
  they quote these scripts.
- **`packages/studies/README.md`:** one line per new module.
- **`studies/beam_diffuse_split/README.md`:** the script table, where a script's role changed.

## Verification commands

- The implement-issue set: `ruff check`, `ruff format`, `uv run --all-packages ty check`,
  `uv run pytest`, pydoclint and pymarkdown as in `ci.yml`, `scripts/lint/check_docs_links.py`, and
  `uv run mkdocs build --strict`.
- **The `--no-dev` export is unchanged from `main`:** diff the output of
  `uv export --no-dev --format requirements-txt` on `main` against the same command on the branch.
  The three new dependencies reach only the dev group, through `studies`.
- The reproduction check above.

## Risks and open questions

1. **Is it right to leave the physics loops and `ens_horizons.py` alone?** *Recommendation: yes.*
   Neither shape fits a shared loop without a callable and a result type. The physics results are
   final, and #810 rewrites `ens_horizons.py`. If #809 or a later study needs a physics fold loop,
   generalise then, with a real second caller in hand.
2. **Should the within-era grouping ship before #809 uses it?** *Recommendation: yes.* It is a
   one-argument change to a function being moved anyway, and #822's body asks for it.
3. **`ens_horizons.py` omits the `~constrained` training filter**, so its model trains on hours the
   network operator turned down. This is probably a defect. It is recorded here for #810 to settle,
   not fixed.
4. **The row-order risk.** The reproduction check is what would catch a slip.

## Rejected from the simplicity review

- Deferring the within-era fold grouping to #809 entirely: the issue asks for it, and it is one
  argument.
