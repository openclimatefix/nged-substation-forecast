# Plan: quick wins for the production XGBoost forecast (exploratory study)

Draft plan, branch to be renamed when the issue exists. No issue or PR yet (awaiting approval).

## Context: the auto-research infrastructure does not exist yet

The v0.3 plan for an LLM agent that runs experiments (trusted submit command, protected scorer,
hypothesis store, leakage test; issues #958, #1031, #1038) is not built, and rolling-origin folds
(#960) are not designed. This study is therefore a manual precursor. It does not build, call, or
depend on any of that infrastructure, and its scores are never leaderboard submissions. Two
consequences: the study's fold B is an ad hoc robustness check, not a stand-in for the #960 design;
and the page ends with a list of which ideas were cheap and which looked worth sending to the agent,
as evidence for the screening decision in #1038. The study reads the validation year directly, which
is allowed for a study but is exactly the access issue #958 plans to bar from agent sessions.

## Question

Which cheap changes to the XGBoost forecast, using only data the production system already holds
(ECMWF ENS NWP, NGED power), lower CRPS and MAE relative to the forecast in production today, and
do any of them close the gap to `manual_heuristic`?

On the leaderboard fold `mid_2025_to_mid_2026` (31 series, all horizons) the stored rows give:
`manual_heuristic` CRPS 4.05 MW / NMAE 0.110, `climatology` 4.43 / 0.127, `xgboost_baseline`
6.24 / 0.1235. These are unreviewed readings of `data/forecast_metrics`; Phase 0 re-derives them.

## Style of this study

**Exploratory and iterative, closer to ML R&D than to a pre-registered study.** Round 1 tries ideas
one at a time against the production arm. Round 2 combines the better ones by judgement. Round 3
checks the finalists on a second fold and a second seed. Every number is exploratory except three
planned contrasts (below). The page says so, and says which arms were chosen after seeing results.

**No production code changes.** The diff touches only `studies/quick_wins/`, `packages/studies/`
(if shared code is needed, with tests), and `docs/studies/`. Nothing in `src/`, `conf/`, or any
other package changes. Extra features are computed in the study's own code on top of the frame the
production feature engineer returns.

## Size (five triggers)

- Changes what gets stored: no (study outputs under `data/studies/` only).
- Touches the production serving path: no.
- Touches a degradation rule: no.
- Admits more than one defensible design: yes (calibration method, combination rule, fold design).
- Spans code whose callers cannot be named without searching: no.

Sized complex because of the fourth trigger and because a study page publishes numbers. Review
effort is deliberately reduced for credits (see "Reviews").

## Data and rows

- **Series:** the 31 leaderboard series (`data/eligible_time_series`): demand (including
  disaggregated demand and raw flow), PV, wind, and the 3 remaining generation types. Results are
  reported by `time_series_type`, never per generator. PV and wind are NGED metered generators, so
  no per-generator MW or label reaches any chart, report, or page; errors are normalised by
  effective capacity and aggregated by type.
- **Inputs:** ECMWF ENS only (`data/NWP`, 2024-04-01 onward, one 00:00 run per day, control plus
  about 50 members), NGED cleaned power, series metadata. Nothing else.
- **Production arm:** the `conf/model/xgboost.yaml` forecaster refitted in the study: 24 features,
  500 trees, `reg:squarederror`, trained on the control member, predicted for every member.
- **Rows:** the stored `xgboost_baseline` keys `(time_series_id, power_fcst_init_time, valid_time,
  ensemble_member)` restricted to a fixed, pre-declared sample of init times: every 7th day, about
  52 runs. A full-window 51-member frame is several hundred million rows per arm, so the sample is
  what keeps about ten arms, two seeds, and two folds affordable. Every arm is asserted to hold the
  same keys (`studies.guards`). The full set is scored once, for production only, in Phase 0.
- **Fold A (main):** the leaderboard fold. Train 2024-04-01 to 2025-06-30, validate 2025-07-01 to
  2026-06-30. Nothing after `final_test_start` (2026-07-01) is read.
- **Fold B (robustness check, round 3 only):** train 2025-07-01 to 2026-06-30, validate 2024-04-01
  to 2025-06-30. Never used to choose an arm. It tests a different training period, not the
  production direction of time, so the page calls it a robustness check, not the honest estimate. It
  has caveats stated on the page: 12 months of training against 15; early validation months have no
  power history for the lags; series eligibility (six months of history before validation starts)
  differs from fold A; effective capacity grows over time, so training on the future mis-scales it;
  and ECMWF's 12 November 2024 model upgrade falls inside the validation period with only one side
  of it in training.

## Phase 0 — reproduce production (gate for everything else)

1. Build the production arm from a script, with `load_engineering_inputs`, the forecaster's
   `feature_engineer.engineer`, and `XGBoostForecaster.train` and `predict`, not through Dagster.
   Fit on fold A's training period at the stored configuration (`colsample_bytree` 0.8).
2. Score it with `ml_core.metrics`, not study code. **Gate:** on the full set of stored keys, CRPS
   and MAE agree with the stored `xgboost_baseline` rows to within the spread of the stored retrains
   (`_retrain_638`, `_retrain_790`: CRPS 6.23 to 6.24). If not, stop and find out why before running
   any idea. Then restrict both to the 7th-day sample and print the sampled production score, which
   is the reference for everything after.
3. Score the stored `manual_heuristic` and `climatology` rows with the same code on the same rows.
   Print all three, per type and for the horizon slices 0 to 2 days, 3 to 10 days, and all, into
   `report.md`.
4. Decision for the user, not made here: `score_study.py` needs the reference experiment
   `xgboost_no_power_lags`, which has no stored rows. The study's numbers are therefore "scored with
   `ml_core.metrics` in the study", never called leaderboard numbers. Running the reference and
   submitting the finalist to `score_study.py` is an optional follow-up.

## Metrics

`compute_metrics` returns aggregates per series and horizon slice, not per-row losses, and its
per-run frame is private. The study therefore adds a small tested wrapper in `packages/studies`
that calls the reviewed per-run expressions (`_fair_crps_expr` and the ensemble-mean absolute
error) and returns one row per `(series, init, valid)` group, in batches by init month so a
51-member frame never sits in memory whole. A test checks that the wrapper's aggregates equal
`compute_metrics` on a small frame. The month bootstrap and the saved per-row losses come from that
wrapper.

- **CRPS:** `ml_core.metrics` fair CRPS over the forecast's ensemble members (Ferro 2014), per
  `(series, init, valid)` group, normalised by effective capacity for aggregation, as the
  leaderboard does. Also reported in MW per type.
- **MAE:** as `ml_core.metrics` defines it: the absolute error of the ensemble mean, per
  `(series, init, valid)` group (and NMAE). CRPS is computed per group over the members. An arm that
  only rescales members about their mean (B1) cannot change MAE by construction, so B1 is reported
  on CRPS alone.
- **Ensemble-mean reference arm:** the ensemble mean as a single member, whose CRPS equals its MAE.
  The stored CRPS (6.24 MW) against `manual_heuristic` (4.05 MW) hints that the members may hurt;
  if the single-member arm beats the member spread on CRPS, shrinking toward the mean is the
  cheapest win.
- **Slices:** all horizons, 0 to 2 days, 3 to 10 days; by `time_series_type` (demand, PV, wind;
  "other" folded into a fourth group).
- **Extra diagnostics for calibration arms:** spread-skill ratio and the 10% to 90% interval
  coverage by lead day, from `ml_core.metrics` where it exists. Fair CRPS is the only
  ensemble-size-fair score, so calibration arms that change the member count are compared at the
  same count (51).

## Intervals

Paired differences against the production arm, resampling whole validation calendar months
(12 months; 2,000 resamples, `studies.bootstrap.bootstrap_difference`). Add a t-interval over the
12 monthly differences, because a percentile bootstrap over so few clusters is too narrow. Also
print, per arm, how many of the 12 months improved. Every arm's absolute CRPS and MAE are reported
beside its contrast. No multiple-comparison correction on exploratory rows; the page says that
about 1 in 20 rows with no real effect will look significant and that those rows cluster, because
they share months. The final round-2 winner is therefore confirmed on fold B and a second seed.

## Planned contrasts (written before any result)

1. **P1:** the round-2 combination (fixed in the report before it is scored on fold B) against the
   production arm, CRPS and MAE. Fold B is the deciding number; fold A is exploratory, because the
   combination was chosen on it.
2. **P2:** A2 (type-specific lists, PV rung A1 included) against the production arm for PV series
   only, CRPS and MAE, fold A. (Question: does PV forecast better without power lags and weekday
   features?)
3. **P3:** B1 against the same forecast unrescaled, CRPS only, all types, fold A. B1 is named here
   in advance.

Every other number is exploratory. Anything added after a first look is labelled post hoc.

## Round 0 — training-data cleaning (added at the maintainer's request)

**Question.** Production trains on the cleaned power table (`cleaned_power_time_series`, the
unflagged rows of `flag_nged_power`), which does minimal cleaning. Does stronger cleaning of the
*training target* lower the error? The maintainer's first idea is to drop each series' first few
months after its first non-zero reading. The maintainer also notes that NGED's data shows no
stuck values, so stuck-value rules are not tried.

**Design.** Only the training rows change. Every cleaning arm is scored against the identical
validation actuals and rows as production (the existing cleaned table), so the arms stay
comparable and cleaning cannot flatter itself by removing hard validation hours. Rules are applied
to the engineered training frame by dropping `(series, valid_time)` rows; power lags still read the
raw cleaned table, as in production. Each rule is a pure function of the series' own history before
`train_end`, so no validation information reaches the training set.

**Arms (tinkering allowed, listed here so the page can say what was tried):**

| ID | Rule | Applies to |
|---|---|---|
| R0 | Production: no extra cleaning | all |
| R1 | Drop the first N months after the first non-zero reading, for N in 1, 2, 3, 6 | all |
| R2 | Drop isolated exact-zero half-hours (a zero with non-zero neighbours) | all |
| R3 | Drop gross outliers: readings beyond k times the series' effective capacity, or beyond a robust band around the series' own hour-of-week median, for k and band width read from the training data only | all |
| R4 | Drop commissioning ramps and level shifts found by comparing each series' monthly mean output with its own long-run level (generators) | PV, wind |
| R5 | The best-looking combination of the above, chosen on fold A | all |

Unscored extras for the page's data-cleaning section: how many training rows each rule removes per
series type, plots of what was removed for one example per type (anonymised, normalised by
capacity, no generator names), and a check that the validation rows are untouched.

**Selection.** The cleaning rules chosen on fold A become "the cleaning" for the rest of the study,
so round 1 to 3 arms run on cleaned training data. The leaderboard keeps both references: the
original production forecast (R0) and production with the chosen cleaning (R5 or best single rule),
and reports every later arm against both. Fold B repeats the cleaning comparison in round 3.

**Planned contrast P0:** the chosen cleaning (R5, fixed before fold B is scored) against R0, CRPS
and MAE, fold B deciding. The ladder of R1 to R4 is exploratory.

**Page section.** "What the data cleaning showed": the table of rules with rows removed and
paired differences against R0 with intervals, the plots, and a recommendation on which rule to
move into the production cleaning step, with the limits (six PV and three wind series; one
validation year).

## Round 1 — ideas one at a time, each against production and against production with the chosen cleaning

**Column rule.** Phase 0 reproduces production at its stored `colsample_bytree` of 0.8. Every
round 1 contrast then runs against a second control, "production at `colsample_bytree` 1.0", and
every arm also runs at 1.0, because at 0.8 a wider arm wins from column subsampling alone. An arm
that adds columns is compared with a control padded to the same width with shuffled copies of
real columns. An arm that drops columns cannot win from subsampling, so it needs no padding. The
page states the residual bias at 1.0 (about 0.4% of error on synthetic data). The production control is fitted with 3 seeds in
round 1, giving a noise floor; arms use seed 0, and finalists get more seeds in round 3. GPU fits (`device="cuda"`) if `nvidia-smi` shows a free GPU and CPU load is low; one device inside
each contrast.

| ID | Idea | Applies to | Source on the roadmap page |
|---|---|---|---|
| A1 | Drop the power lags (24, 48, 168, 336 h) and the calendar features that mean nothing for generators | PV and wind | Per-`time_series_type` feature lists |
| A2 | A1 plus type-specific physics columns: PV adds solar elevation and clear-sky index; wind adds `u`, `v`, speed, and a farm-level power-curve proxy; demand adds effective temperature and degree-days, built from the engineer's own rolling-feature names so `nwp_init_time` grouping holds | by type | Per-type lists; physics features; effective temperature |
| B1 | Post hoc rescaling of each member's deviation from the member mean by a factor fitted per (type, lead day), allowing factors below 1 (shrink toward the mean) | all, CRPS only | Probabilistic evaluation / calibration |
| C1 | Lead time as a feature (`nwp_lead_time_hours`) | all | Feed the model the forecast lead time |
| C2 | Early stopping on the last weeks of the training window | all | Early stopping |
| C3 | Recency sample weights (exponential decay with sample age) | all | Early stopping item |
| C4 | `reg:absoluteerror` (MAE objective) instead of squared error | all | Objective |
| C7 | Holiday and raw ordinal time features (`is_bank_holiday`, days to nearest holiday, `half_hour_of_day`, `day_of_year`) | demand | Holidays; raw ordinals |

A1 is the first rung of A2's ladder and is reported as its own row. Controls: production at 0.8,
production at 1.0 with 3 seeds, a shuffled-column-padded control for each width, and the
ensemble-mean single-member arm.

B1 is applied to the production arm's predictions and needs no refit. Calibration rows for a scored
run must have `valid_time` at or before that run's `power_fcst_init_time`, so only outcomes already
known at issue time are used; the first two validation months have too little history and are
dropped from B1 and from its control, and both are scored on the same rows. A residual-quantile band
rebuilt from recent errors made CRPS worse in an earlier solar study; B1 keeps the model's own
members and only rescales them, and the page compares against that result. A wider-than-members
variant that adds sampled residuals (an EMOS-style design) goes on the future-research list.

Not tried here, because they cost more than a quick win: training on more ensemble members, ensemble
statistics as features, monotone constraints, a residual-draw calibration, neighbouring-cell weather
(all listed as future research). Also not tried, because they need data the production system lacks
or code outside the study:
estimates of past weather and longer history, the clear-sky-index NWP resample as a pipeline change
(measured in the ENS horizons study, so cited and not repeated), neighbouring-cell weather, residual
and weather-delta lag features, BMRS price, and anything needing new ingest.

## Round 2 — combinations, by judgement

From round 1, take the arms whose paired CRPS and MAE both improved on fold A and improved in more
than half of the 12 months. Combine them by type (the best feature list per type, plus the best
training tweaks, plus the best calibration on top), in 3 to 5 named combinations, building up one
idea at a time (a ladder) so each step's contribution is visible. Record the choice and its reason
in the report before scoring on fold B. C2 and C3 are tried together explicitly, because both address drift.

## Round 3 — confirmation

The finalist combinations are refitted with 3 seeds on fold A and with seed 0 on fold B. A finalist
counts as a quick win only if it improves on production in both folds and by more than the 3-seed
noise floor of the production control.

## Charts and page

Figure-led page `docs/studies/forecasts/quick-wins.md`, following the `study` skill's structure,
loosened for exploration. The bottom line and Figure 1 (the leaderboard) sit on the first screen.

1. **Figure 1, the visual leaderboard:** every arm sorted best first, two panels (CRPS, MAE), each
   arm's value with its 95% interval, with vertical reference lines for production, `manual_heuristic`,
   and `climatology`. Colour by round and idea family. A second panel pair, built only if the draft needs it, shows the
   paired difference to production.
2. **Figure 2:** the same by type (demand, PV, wind), since the ideas differ by type.
3. **Figures 3 onward:** production reproduces the stored result (Phase 0); error by lead day for
   production, `manual_heuristic`, and the finalist; calibration plots (coverage by lead
   day, spread-skill) before and after B1; the round-2 ladder as a chain of paired steps; fold A
   against fold B. Example forecast weeks per type, chosen by a stated rule, are added only if the
   page needs them.

Charts follow the `dataviz` skill and `plotting.ocf_theme`; text of every figure is reviewed before
rendering. The page also holds: recommendations for the production forecast, ordered by size of win
and effort to land, each with its roadmap item; ideas for future research (what the study could not
try, and what the results suggest); limitations; scope; reproduction commands.

## Code layout

- `studies/quick_wins/`: `build_frames.py` (rows, features, folds), `fit_arms.py` (round 1 to 3,
  checkpointed per arm so a crash costs one arm), `calibrate.py`, `report_quick_wins.py` (prints
  every table the page quotes into `report.md` and saves per-row losses and predictions as
  parquet), `quick_wins_charts.py`, `README.md` (what each file holds).
- Reuse: `studies.guards`, `studies.bootstrap`, `studies.checkpointed_fits`, `studies.power`,
  `studies.anonymise`, `studies.charts`, `ml_core.metrics`, and the production feature engineer
  (imported, not edited). Shared code goes into `packages/studies/src/studies/` only if a second
  folder needs it, with tests.
- Outputs under `data/studies/per_study/quick_wins/`; nothing overwritten.

## Reviews (reduced for credits; stated in the PR body)

- One Sonnet adversarial review of this plan (simplicity and correctness together).
- Sonnet review of every script before its first run (the `study` skill's rule), with triage.
- **One** Opus scientific-validity review after round 3 and the draft page (the `study` skill asks
  for two; the PR says why it ran one). It attacks leakage in the calibration windows, the fold
  design, selection effects, and every page number.
- Sonnet first-stumble and `prose-review` passes on the page and the figure text.
- No mutation pass unless `packages/studies/` changes.

## Risks

- **Selection on fold A.** About eight ideas are screened on one validation year. Mitigated by
  fold B, the 3-seed noise floor, and the months-improved count, not eliminated.
- **Phase 0 might not reproduce.** Then fixing the harness comes before any idea.
- **Validation-set size.** The 7th-day sample and batching by init month keep memory bounded;
  Phase 0 checks the stored experiment's row count and init-time sampling first.
- **Few generators per type** (7 PV, 3 wind): type-level results are wide and are labelled so.
- **Calibration leakage.** B1 uses only outcomes with `valid_time` at or before the scored run's
  `power_fcst_init_time`; the Opus review attacks this rule.
- **Era change.** The Met Office and ECMWF upgrades inside the periods are named on the page; the
  ECMWF upgrade of 12 November 2024 is not cut from the folds.
