# Study plan: forecast fixes and lean feature sets

**This study compares the production XGBoost recipe and the best quick-wins recipe after the
wind and radiation corrections, then investigates Part 1b of issue 1156.** It uses the repository's
`study` skill. The user authorised autonomous execution on 12 October 2026, including retraining
the quick-wins winner with PRs 1167 and 1166. No new paid resources or data licences are authorised.

## Scope and process

**This is a complex study, with independent plan, script, scientific, and prose reviews.** The
five sizing triggers are: stored results and leaderboard submissions (yes); production serving
changes (no); degradation policies for incomplete inputs (yes); multiple defensible experimental
designs (yes); and callers requiring investigation (yes). The study skill permits implementation
after plan reviews without human approval. Scripts are reviewed before their first execution.
Two fresh scientific-validity reviewers inspect the first results and revised page separately.
Available Codex reviewers substitute for unavailable Opus/Sonnet agents; record that substitution.
Run mutation testing only if shared `packages/studies/` machinery changes.

**Production code and data remain inputs to the experiment.** Pin the latest reviewed heads of
PRs 1167 and 1166, resolve their combination in an isolated experimental checkout, and record
every conflict resolution. Refresh heads before building features; a subsequent substantive
change invalidates affected caches. Study deliverables stay under `studies/`, `packages/studies/`,
and `docs/studies/`. The final study PR must not carry the implementation PRs' production changes.

## Input and readiness gates

**The wind archive must be complete and validated before a comparison can use it.** Monitor
the chat titled "NWP u and v". Direct cross-chat messaging is unavailable; the user agreed to
relay the continuation and readiness request. Require the destination path, source snapshot,
destination version, full inventory validation, non-wind preservation checks, and wind
round-trip precision checks. Pin both archives. Do not silently use a partially migrated table.
The radiation implementation must include the user's latest PCHIP long-wave revision and the
short-wave interval reconstruction. Rebuild features through that code, rather than patching
cached engineered columns with the older audit's approximations.

**Each treatment pairs compatible code and archive schemas.** The execution matrix is old main
code with the original polar archive; radiation-only package changes with the original archive;
wind-only package changes with the migrated component archive; and combined package changes
with the migrated archive. Export the same engineered-frame contract from each environment.
Never reconstruct the old archive from component values. Assert equal raw keys and non-wind
values across pinned archives, using migration validation plus independent spot checks.
Read each already-issued run's native temporal context before engineering, then clip output to
the fold window. Verify full-run-then-clip equivalence at boundaries; native future forecast
values are available information, future observed power is not.

**New scripts have explicit contexts and never redirect existing module globals.** Implement
uniquely named `fixed_pipeline_*.py` scripts within the existing `studies/quick_wins/` family,
reusing pure same-folder helpers. Use a separate `quick_wins_fixed_pipeline_1156` output root.
Every cache is bound to input snapshots, pipeline revision, fold, columns, settings, seed, and
device through immutable manifests. A mismatching cache raises. For `scan_power`, verify its
source Delta version before and after materialisation, then freeze the bounded frame as parquet.
Other Delta scans pass explicit snapshot versions. Keep all published quick-wins files read only.

**Pin the production recipe and training cutoff before scoring its comparison.** The locally
promoted model is recorded in `production_model/promotion.json` and `meta.json`; verify its
provenance against MLflow and, where possible, the actual serving deployment. Call a matched
historical refit a refit. Do not evaluate a deployed model on rows that precede its training
cutoff, or present the local promoted artifact as verified live deployment without evidence.
Record the source code, configuration, dependencies, feature order, row order, cleaned-power
version, capacity version, eligible population, forecast archive versions, and frame hashes.

## Rows, fitting, and scores

**Every contrast uses identical forecast keys, targets, capacities, fitting seeds, and device.**
Start from the quick-wins study's production-system inputs, folds, population, sampled-run design,
cleaning rules, and training-control-member/predict-all-51-members protocol. Fold A trains on
April 2024 to June 2025 and scores July 2025 to June 2026. Fold B reverses those windows and is a
retrospective robustness diagnostic, not an honest future holdout. Guard all observed power with
`studies.power.scan_power`, explicitly filtered to times before 1 July 2026: current main has
removed the former global date guard. Assert this study boundary and each fold's training and
scoring bounds in manifests. Freeze the existing score-capacity table; do not rebuild it from
later outcomes. Retain the power lags and their
initialisation-time masking. Never train or select on observations after the relevant cutoff.
Use common target-supported rows, report excluded rows and nulls by type, era, and lead, and
assert one-to-one keys and complete expected member labels. No outcome-based selection of
validation rows is permitted.

**Point-only contrasts use capacity-normalised fair CRPS, with ensemble-mean MAE secondary.**
For comparisons involving the quantile winner, use twice mean pinball loss at the 19 midpoint
levels `(i - 0.5) / 19`, i from 1 to 19, for every recipe. To preserve the original delivered
sample without inventing a new calibration, treat both the 51-member and 459-value outputs as
their equally weighted empirical distributions. Sort finite sample values per forecast key and
evaluate their empirical inverse CDF at each level using rank `ceil(n * level) - 1`. This fixes
tails and crossing treatment without extrapolation: sorting rearranges crossed samples only for
scoring and does not change their empirical distribution. Print all 19 pinball losses, their
twice-mean aggregate, and nominal 80%/90% interval coverage. Historical fair CRPS of 459 values
is a reproduction diagnostic; never make it the deciding cross-recipe metric. Report
absolute errors and paired differences, overall and by series type and the quick-wins lead bands.
Retain equal-series weighting and document the different equal-month estimand for the issue's
month-level t-interval. Use paired whole-calendar-month resampling with a paired fitting seed,
2,000 resamples, seeds 0, 1, and 2. Use the skill's second hyperparameter setting for planned and
deciding contrasts, and one CPU/GPU refit as a numerical noise check. One GPU fitting job runs
at a time; cap CPU preparation at eight cores.

**Separate exact production reproduction from feature-information comparisons.** Reproduce
production's column subsampling of 0.8 in its own before/after comparison. The main feature
comparisons use column subsampling 1.0 and equal column counts, with deterministic redundant
columns built only from each narrower model's existing inputs. Print every actual feature list
and count; separately check unpadded versions to expose padding artefacts. Keep padding results
distinct from the exact quick-wins configuration. No arm changes row subsampling merely to
obtain a better score.
The fixed width is the largest real feature count among the preregistered arms for that series
type. Append duplicates by cycling through the lexicographically sorted real feature names;
use unique padding names, and never borrow an omitted feature.
Production reproduction calls the public `XGBoostForecaster.train` and `predict`, preserving
`QuantileDMatrix` construction and public feature preparation. Quick-wins recipe refits preserve
their historical `DMatrix` fitting path. Unrecoverable dirty-source production provenance is
reported explicitly rather than treated as an exact reproduction.

**Leaderboard scores come only from `scripts/forecasting/score_study.py`.** Sampled-run diagnostic
scores are labelled study scores. Before official submission, predict exactly the reference
experiment's series, initialisation times, valid times, and ensemble members and validate keys
offline. Submit the reference and reported compatible finalists once under distinct
`study/1156-fixed-pipeline-<arm>` names. A quantile forecast's 459 samples do not match a
51-member reference: never relabel them as weather members to pass the scorer. Report that
quantile comparison diagnostically unless an already supported compliant scoring path exists.
Do not modify the scorer, CV configuration, or forecast contract. State selection bias on fold A.

## Stage 1: production and quick-wins benchmarks

**Retrain both recipes on matched old and corrected inputs.** Planned contrasts, before results:

1. Corrected production recipe versus the old production recipe, with identical hyperparameters.
2. Corrected production recipe versus radiation-only and wind-only corrections, to separate the
   two changes and their interaction.
3. Corrected quick-wins winner versus the same winner on old inputs.
4. Corrected quick-wins winner versus corrected production, as a recipe comparison rather than
   attributing that whole difference to the pipeline fixes.

**The quick-wins reference is `r2_e1_c1_a3w_d3`, the three-seed headline winner.** Its recipe is
500 rounds, depth 6, learning rate 0.05, minimum child weight 1, row subsampling 0.8, column
subsampling 1, quantile levels 0.1 through 0.9, `new_meter_settling` cleaning, and no early
stopping, calibration, recency weighting, or analogue features. The page identifies quantile
training at nine levels, forecast lead, lean wind
columns, and pooling by series type with a site identifier, with commissioning cleaning already
applied. Preserve its exact named arm and parameters from the committed scripts, and
preserve quantile levels, cleaning, pooling, scaling, and sample semantics. The page's one-seed
weekly-analogue variant is exploratory and is not silently substituted. Also retain the best
point configuration `r2_c4_c2_c1_a3w_d3` as a secondary reference where needed for a compatible
51-member score. It uses absolute-error loss and a last-six-training-weeks early-stopping probe
with patience 50, then refits with at least ten rounds, preserving the same pooling and cleaning.
The issue's concerns about treating fixed quantiles as exchangeable members must accompany fair
CRPS results; use a consistent quantile/pinball diagnostic and interval coverage for all
quantile comparisons, without claiming that diagnostic is the official leaderboard score.

**Correctness controls precede interpreting small forecast differences.** Check synthetic wind
crossing north and cancelling vectors; short-wave interval timing, daylight and energy; and
long-wave PCHIP against an independent oracle. Check bulk/single-run feature agreement and
derived wind features. Use a deliberately injected one-hour radiation shift as a positive timing
control and removal of UTC offset/weekday ordinal as negative controls. A null result requires
a detectable positive control or an explicit bound on the unresolved effect.

## Stage 2: Part 1b on the corrected pipeline

**Rebuild the lean sets from the quick-wins definitions and investigate additions in issue order.**
The primary Part 1b control is `production_cleaned`: 24 production features, 500 squared-error
trees, depth 6, learning rate 0.05, minimum child weight 1, row subsampling 0.8, column
subsampling 1, per-series fits, and `new_meter_settling` cleaning. Preserve sorted real feature
order and equal-width padding. Re-evaluate full versus lean solar and full versus lean wind as
planned contrasts before add-backs. All these use the combined PR feature pipeline;
the quick-wins winner remains a benchmark. Do not switch to the one-row ensemble-statistic
screening model because gate G9 has not been established. Do not add Part 2 calibration work.
Resolve the issue's "11" and "8" removed-column counts against the actual committed lists and
print all differences before running; descriptive prose is not a feature specification.

**Solar additions form a cumulative ladder with a predefined stopping rule.** Start with the
lean solar columns. Add 10 m wind speed; then both pressure columns; then 500 hPa geopotential
height; then raw downward long-wave and long-wave minus sigma times absolute temperature to
the fourth power; then replace time-of-day sine/cosine with zenith and azimuth and remove UTC
offset. Each gain is against the immediately preceding retained rung. Derive the long-wave
residual using temperature in kelvin and the physical constant's
correct units. At the first step improving solar CRPS by less than 0.15 percentage points, stop
the planned ladder and do not adopt the lean solar set. The continuation decision uses the
three-seed mean, equal-series-weighted fair CRPS reduction on fold A. Above the 0.15 threshold
the ladder continues even if uncertainty is unresolved; continuation is not a claim of benefit.
A decisive gain must also meet the
issue's t-interval criterion, improve at least 9 of 12 fold-A months, and have the same sign in
a refitted fold B. Report seed, bootstrap, t-interval, and sensitivity disagreements explicitly.

**Wind additions are separate tests from the lean wind baseline.** Test short-wave radiation,
dew point, precipitation type, and each time-of-year sine/cosine separately. Test each of the
three weekday columns separately, documenting overlap with the already-run controls. Keep
other series' predictions fixed within type-specific contrasts. Retain lags unless the actual
lean definition removes them. Total cloud cover is deferred: twelve post-21-November-2025
months have not accrued, and a shorter-window comparison would confound the result.
For precipitation, print scores and missingness separately before and after its November 2024
availability boundary. Fold B's earlier, wholly missing era measures availability skew, not
precipitation's information content. The preserved folds cannot identify an era-balanced benefit;
make that limitation explicit instead of asserting causality from an era sign reversal.

**The two PRs do not repair every Part 1 diagnostic.** Describe their combined output as the
wind/radiation-corrected pipeline, never as eliminating all audited faults. Check the windchill
domain with a separately named study-only guarded version, using temperature at or below 10 C
and speed at or above 4.8 km/h and otherwise air temperature, before interpreting cuts that
remove windchill. Count affected rows. Keep that additional change distinct from the PR-only
stage-1 contrasts, and record whether the feature-cut verdict depends on it.

## Plan-review triage

**The first independent plan review found two must-fix and three should-fix issues, all accepted.**
The removed global date guard is replaced with explicit study-window assertions. A common
19-midpoint empirical-quantile pinball score is defined before cross-recipe comparisons. The
quick-wins and Part 1b controls, full-versus-lean contrasts, padding, ladder continuation rule,
and precipitation-era interpretation are now explicit. No finding was rejected.

**The second independent plan review found two must-fix and two should-fix issues, all accepted.**
The runner now has an explicit isolated context and cache namespace, a compatible four-treatment
code/archive matrix, the public production training path, and native temporal context before
clipping engineered rows. No finding was rejected. Script reviews must check these contracts
before execution.

## Overnight execution and reporting

**Finish the required comparisons before adding exploratory work.** No unplanned model families,
new archives, paid compute, calibration search, or data licence acceptance. Up to two post hoc
diagnostic contrasts may clarify an unexpected result, using existing saved predictions where
possible and labelled post hoc. Checkpoint each completed fit and its manifest. Retry a failed
fit only after diagnosing its cause; never retry a leaderboard submission to select a score.
If the archive, reviewer, compute, or time becomes limiting, preserve completed outputs and
give an explicit partial report rather than inventing results or calling unreviewed work final.

**The page and report must distinguish measured findings from pending work.** Every quoted table
comes from a committed report script. Plan figures around input validity and known-answer
controls, absolute errors, paired fixes, the quick-wins comparison, the stopped solar ladder,
wind additions, coverage, and descriptive feature importance. Review figure text before rendering,
inspect rendered charts, and review the page for evidence, first-reader understanding, prose,
and user/statistician perspectives. Anonymise private generators before writing public artifacts.
Publish a study-only reviewed PR and a concise report of what ran, what moved, the stopping
decisions, limitations, and one proposed follow-up with rough cost. Never claim human review
before it happens.
