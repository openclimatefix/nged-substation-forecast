# Plan: does blending weather products beat the best single product? (#836)

**Problem.** The two weather-product studies score one product at a time. On the solar page CAMS has
a mean absolute error of 5.05% of capacity and the best weather model, ICON-D2, 7.71%. On the wind
page ICON-D2 scores 6.69% and UKV 6.83%. Neither study measures whether a model shown several
products at once beats the best of them. If it does, the consumers of past weather should read a
blend instead.

**Solution.** Refit on the two studies' own common rows, folds, and seeds, adding arms that combine
products four ways:

- XGBoost shown every combined product's columns.
- A statistical blend: the mean of the products' values, shown as one column set, so the arm has
  exactly as many columns as a single-product arm.
- A linear stack of the single-product models' out-of-fold predictions, with non-negative weights
  that sum to 1, fitted per generator on the folds other than the one scored.
- The equal-weight mean of the single-product predictions.

Every XGBoost blend is tested against a control arm whose extra columns carry the same products'
climatology in place of their weather. Every model's out-of-fold predictions are saved to disk. The
result is a new page, "Does blending weather products help?", linked from both existing pages.

## Verdict, size and departures

**Verdict: worth doing.** Feeding several weather products to one model is common practice in
power forecasting. The existing `multi_nwp.py` measured one pair, ICON-D2 with UKV, for issue #800,
on its own row set. This study measures blends across all six solar products and all five wind
products on the rows both published pages already use, so its numbers sit directly beside theirs.

**Size: complex.** The five trigger answers:

- **What gets stored:** yes. A published page, and new result files under `data/studies/`.
- **Production serving path:** no.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The combinations, the blending methods, and the
  controls are all choices.
- **Code whose callers cannot be named without searching:** no.

**Reviews:** both plan reviews, then two Opus scientific-validity reviews of the results, as the
`study` skill requires, then a diff review. A mutation pass on the new `packages/studies` code.

**Departures from the issue:** none.

## The arms

**Every arm sees the shared features the published studies use** (time of day, season, the UKV era;
for solar, sun position and ERA5's air temperature) plus the columns below. The single-product arms
are refitted in this run, not read from the published losses, so every contrast is paired on
identical rows, folds, and seeds. The refitted single-product errors must reproduce the published
reports to the printed digit, which checks that the rows and folds are unchanged.

### Solar: each product's global horizontal irradiance

| Set | Products | Consumer it serves |
|---|---|---|
| Singles | each of the six | the reference |
| Satellite + best model | CAMS, ICON-D2 | history, where ICON-D2 covers |
| Satellite + GB-wide model | CAMS, ICON-EU | history anywhere in Great Britain |
| Satellite + reanalysis | CAMS, ERA5 | history back to 2004 (scored only from December 2022) |
| Live, GB-wide | UKV, ICON-EU | historical features in the live service |
| Live, all (weather models only) | UKV, ICON-D2, ICON-EU, ICON global | the same, where ICON-D2 covers |
| Everything | all six | the upper bound |

Global irradiance only, not each product's beam split: the solar page found the published split
adds 0.03 to 0.11 points, and carrying three columns per product would triple the column count.

### Wind: each product's hub-height speed, its direction, and its 10 m speed

| Set | Products | Consumer it serves |
|---|---|---|
| Singles | each of the five | the reference |
| Best pair | ICON-D2, UKV | the two leaders |
| Live, GB-wide | UKV, ICON-EU | historical features in the live service |
| Live, all | UKV, ICON-D2, ICON-EU, ICON global | the same, where ICON-D2 covers |
| Everything | all five | the upper bound, inside ICON-D2's domain |

**Every blend uses ICON global as served, without the wind study's step indicator
(`step_period`).** The indicator is worth about 0.06 points even to ICON-EU, which has no steps
(`icon_eu_step` 6.894 against `icon_eu_wind` 6.958), so it works as a date-regime feature. The
single-product arms and the stack go without it, and a blend carrying it would be tilted against
them. One exploratory arm, `everything_xgb_step`, adds it back.

**Each set's best single product is fixed from the published tables:**

| Set | Best single |
|---|---|
| Solar sets containing CAMS | CAMS |
| Solar live, GB-wide | ICON-EU (8.30 against UKV's 8.79) |
| Solar live, all | ICON-D2 |
| Wind sets containing ICON-D2 | ICON-D2 |
| Wind live, GB-wide | UKV (6.83 against ICON-EU's 6.96, close) |

### Four blending methods per set

1. **XGBoost on every column** (`xgb`). Column subsampling stays at 1.
2. **The mean of the inputs** (`mean`). Solar: the mean of the products' global irradiance, one
   column. Wind: the mean hub-height speed, the mean 10 m speed, and the means of the products'
   direction sines and cosines, which together give the circular mean direction: four columns, as
   in a single-product wind arm. The mean hub-height speed mixes ICON's 80 m wind with the others'
   100 m wind, which a per-generator tree absorbs, and the page says so.
3. **A linear stack** (`stack`). For each generator and each fold k, non-negative weights that sum to
   1 are fitted by least squares on the generator's other folds and applied to fold k. With weights
   summing to 1, the stacked error is the weighted sum of the single-product models' capped signed
   errors, so the stack is fitted and scored from `signed_error_capped_mw` with no refit. The weights
   are per generator, so no neighbour's target reaches them. Each seed is stacked separately, from
   that seed's single-product models, so `bootstrap_difference`'s seed draw stays valid. The fit
   drops `constrained` hours, as the base models' training does, and scoring keeps them. The fit is
   least squares with `scipy.optimize.nnls` and one heavily weighted sum-to-one row, and the weights
   are renormalised to sum to exactly 1 afterwards. A least-absolute-deviation fit, closer to the
   metric, moved the preview by 0.01 points, and the page says so.

   **Blending capped predictions needs no re-clamp.** A weighted mean of values at or below the
   cap is at or below the cap, so the stack blends the capped predictions exactly. Clamping a blend
   of uncapped predictions would give a result no lower than this; wind has no cap.
4. **The equal-weight mean of the single-product predictions** (`equal`), the standard benchmark for
   combining forecasts. It needs no fitting.

**The stack's weights are cross-fitted, and the report measures how optimistic fitting them is.**
Weights fitted on every fold, fold k included, beat the cross-fitted weights by 0.005 to 0.016
points in the preview, and the report prints that gap for every set. A second-order leak remains
and is not bounded here: an out-of-fold prediction for fold j comes from a model trained on every
other fold, including the scored fold k, and the weights for fold k are fitted on those predictions.
This is standard in cross-validated stacking
([Breiman (1996)](https://doi.org/10.1007/BF00117832)), and the page says so.

**A stack control measures what ensembling alone buys.** The stack of the best single product's
three seeds measures pure ensembling, about 0.01 points in the preview. The bootstrap treats the weights
as fixed; across folds each weight's standard deviation is 0.05 or less, so the intervals are
barely affected, and the page says so.

**A zero-fit preview of the stack exists.** The simplicity review stacked the published
single-product errors before this run. The named contrasts below were written before that preview
and are unchanged, and the page records that the preview existed.

### Controls

**Every XGBoost blend has a climatology control, whether or not it is behind a named contrast.**
The control arm holds the set's best single product's real columns, plus every other product's
columns permuted among the rows sharing a site, a month (the year-month the folds use), and an hour
of day. Each product's columns move together, one row permutation per product, so direction sines
and cosines stay on the unit circle and each product's hub-height and 10 m speeds stay paired. That
keeps each month's mean at each hour of day and removes the hour-to-hour weather. The month's own
mean explains about 3% of UKV's hub-height speed variance beyond the calendar-month mean, which
makes the control slightly harder to beat, so a blend's gain against its control is conservative.
Folds are whole months, so a permuted value never crosses a fold. The permutation helper moves from
`multi_nwp.py` into `packages/studies` with tests.

### Named contrasts, fixed before the run

**Each named XGBoost blend gets two deciding contrasts: blend against its control, and control
against the single product.** The first measures the other products' information. The second
measures what the wider column set alone gives, about 0.4% of mean absolute error at column
subsampling 1 on synthetic data. The page reports blend against single beside them.

Solar:

1. `everything_xgb`, against CAMS: does anything beat the satellite retrieval?
2. `cams_icon_eu_xgb`, against CAMS: the maintainer's own example.
3. `live_all_xgb`, against ICON-D2: the best blend available in the live service against the best
   live single product.
4. `everything_stack − everything_xgb`: a linear stack against XGBoost.

Wind:

1. `everything_xgb`, against ICON-D2: does anything beat the best single product?
2. `live_gb_xgb`, against UKV: a GB-wide live blend against GB-wide UKV.
3. `everything_stack − everything_xgb`.

**The positive control is `cams_icon_d2_xgb − icon_d2_global`**, where a large gain must appear. A
null result elsewhere is read as "no effect" only if the positive control shows one.

Every other contrast is exploratory, including the `mean` and `equal` arms,
`everything_xgb_step`, and the per-site, per-season and per-era splits. The page states how many
exploratory intervals it shows, and that about 1 in 20 would exclude zero by chance. The deciding
contrasts also run under `SENSITIVITY_HYPER_PARAMETERS`.

## What changes, file by file

- **`packages/studies/src/studies/blending.py` (new, tested).**
    - `climatology_permutation(*, frame, column_groups, by, seed)`: permutes each group of columns
      jointly within groups of rows, moved from `multi_nwp.py`, which then imports it.
    - `simplex_weights(*, errors)`: non-negative weights summing to 1 that minimise the squared
      weighted-sum error.
    - `stacked_errors(*, errors, sites, folds, seeds, fit_rows)`: per generator, seed and fold,
      fits the weights on the other folds' `fit_rows` (the unconstrained hours) and applies them to
      every row of the scored fold. Returns the stacked errors and the weights.
- **`packages/studies/src/studies/cross_validation.py` is unchanged.** The blend script writes each
  arm's predictions as measured power plus the capped signed error.
- **`studies/beam_diffuse_split/blend_products.py` (new).** Builds the solar and wind common rows
  with the published studies' own functions (solar through `_joined`, `_common_rows`,
  `_with_eras` and `with_export_cap`, in that order, because `_joined`'s inner joins shape the
  published rows), records the power Delta table's version, runs every arm, stacks, bootstraps
  every contrast with `studies.bootstrap`, and writes the results below.
- **Results, in `data/studies/beam_diffuse_split/blend_products/`:** `predictions.parquet` (every
  arm's out-of-fold prediction with site label, time, month, fold, seed, arm, setting, and measured
  power), `losses.parquet`, `stack_weights.parquet`, `intervals.parquet` (every interval as a
  table), and `report.md`.
- **`docs/studies/blending-weather-products.md` (new),** with charts, in `mkdocs.yml`'s navigation
  and `docs/studies/index.md`. One sentence and a link from each existing weather-product page.
- **`studies/beam_diffuse_split/README.md`:** the new script and its outputs.

## Charts

The page follows the `study` skill's chart rules, and uses `studies.charts` if PR #831 has merged.
Otherwise it uses the form that PR defines.

1. **Headline:** for solar and for wind, each named set's `xgb` blend and its control against the
   set's best single product, and the `stack`, with intervals. Only arms fixed before the run
   appear, never the best of several methods.
2. **Method comparison:** for each set, `xgb`, `mean`, `stack`, and `equal` against the set's best
   single product.
3. **Stack weights:** each product's mean weight per set, across sites and folds, with its range.
4. **Where a blend helps:** the named blend contrasts by season and by generator.

## Tests

`packages/studies/tests/test_blending.py`:

- **The permutation keeps each group's values and changes their order**, and leaves other groups
  untouched: a column whose group holds known values comes back as the same multiset, and at least
  one row changes.
- **A product's columns move together:** after permutation, each row's sine and cosine still
  satisfy sin² + cos² = 1.
- **The stack recovers inverse-variance weights.** Independent errors with variances σ²_A and σ²_B
  give weights σ²_B/(σ²_A+σ²_B) and σ²_A/(σ²_A+σ²_B), within 0.01.
- **The stack never sees the fold it scores.** Corrupting fold k's errors must leave fold k's weights
  unchanged, and must change the other folds' weights.
- **One generator's errors do not change another generator's weights**, and each seed is stacked
  from its own errors only.
- **Constrained rows do not reach the fit**: corrupting them leaves the weights unchanged, and they
  are still scored.
- **The weights are non-negative and sum to exactly 1**, and each returned stacked error equals the
  weighted sum of the fold's errors.

The mutation pass then checks each test fails on the bug it targets.

## Verification

- The green-before-push set, as in `ci.yml`, plus `uv run mkdocs build --strict` and reading the
  rendered page.
- **The refitted single-product arms reproduce the published `losses.parquet` row for row:** equal
  (site, time, fold, seed) keys and bit-identical `signed_error_capped_mw`, which also reproduces
  both reports' error tables.
- Every chart looked at as a PNG render.

## Risks and open questions

- **Run time.** Solar fits 6 singles, 6 XGBoost blends, 6 controls, and 6 mean arms, 24 arms; wind
  fits 5 singles and 4 of each, 17 arms; plus the second hyperparameter setting on the deciding
  arms. At about 2 minutes per arm, the run takes a couple of hours on this 32-core machine.
- **A blend containing CAMS cannot serve the live service,** because CAMS arrives a day late. The
  page states each set's latency beside its result: a blend waits for its slowest input, about
  4 hours for the UKV and ICON sets, about 1 day with CAMS, and about 5 days with ERA5.
- **A live blend needs every input at run time.** Under inherent stability each missing product
  needs a fallback, which a single product avoids; the page says so.
- **Wind's "Everything" set covers only ICON-D2's domain, and UKV's hub-height wind starts on
  12 August 2024,** so the set says nothing about training history before that.
- **The consumer recommendations on the two existing pages may change.** If a blend wins, the pages'
  "Which product each consumer should read" sections point to the new page rather than restating it.

## Findings from the simplicity review

- **Accepted:** bound the stack's leak from the in-sample against cross-fitted gap, not a nested run;
  simplex weights fitted from the saved signed errors, with no change to `cross_validation.py`;
  per-generator weights on the other folds, without the every-site month rule; the wind mean
  direction from the published sine and cosine columns; an exact key check against the published
  losses; the equal-weight benchmark; the "weather models only" label.
- **Rejected: run controls and mean arms only for the sets behind named contrasts.** An arm takes
  about 2 minutes, and a blend reported without its control invites exactly the column-count
  misreading the control exists to rule out.

## Findings from the correctness review

- **Accepted:** ICON global without `step_period` in every blend, with one exploratory arm adding it
  back; blend against control and control against single as the deciding contrasts, with each set's
  best single fixed in advance; a headline with pre-specified arms only, and a count of exploratory
  intervals; per-seed stacking, a fit on unconstrained rows, and exact renormalisation; the leak
  wording (the gap measures the optimism of fitting the weights, not the second-order leak), and
  dropping "the nested stack"; the capping argument stated; the control described as keeping each
  month's mean, with each product's columns permuted jointly; the reworked tests; per-row
  reproduction, the Delta version recorded, the solar build order, the positive control, and the
  seed-ensemble stack control; the latency, coverage and fallback statements; four methods, and
  `equal` on the method chart.
- **Accepted with a different fix:** least squares against least absolute deviation. The plan keeps
  least squares, which `nnls` solves directly, and the page reports the 0.01-point difference.
