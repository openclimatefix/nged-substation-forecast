# Study plan: calibration diagnostics after the wind and radiation corrections

**This is a separate Part 2 study of issue 1156, starting with saved predictions.** The maintainer
authorised a start on Part 2 if time remains after the fixed-pipeline and Part 1b work. This first
study asks whether the corrected production recipe, best point recipe, and best quantile recipe
show different marginal calibration, and how pooling the quantile forecasts over weather members
changes their coverage. It fits no calibrators and makes no production or forecast-contract changes.

## Process and scope

**The study follows the repository's study skill, including reviews before execution.** The five
sizing triggers are stored study results (yes), production serving changes (no), incomplete-input
policy (yes), multiple defensible experimental designs (yes), and investigation of callers and
sample semantics (yes). Independent plan and script reviews precede execution; two fresh
scientific-validity reviews precede publication. Available Codex agents substitute for unavailable
Opus/Sonnet agents, with the substitution recorded in the PR. Mutation testing applies only if
shared `packages/studies/` code changes.

**Use a separate output namespace and published page.** Scripts belong to the separate
`studies/forecast_calibration/` folder, with unique names. Private outputs
belong under `data/studies/per_study/calibration_diagnostics_1156/`, separate from the Part 1 study.
The proposed page is `docs/studies/forecasts/calibration-diagnostics-1156.md`. Existing forecasts,
scores and reports remain read only; each new output has a manifest and refuses overwrite.

**The main study takes priority.** This diagnostic stage requires CPU scoring only, capped at eight
cores. It uses existing production-system data and incurs no new paid compute, data purchase,
licence acceptance, archive download, or external publication outside the study workflow.

## Inputs and readiness gates

**Use only completed, verified corrected-pipeline fits.** Require the combined PR 1167/1166
pipeline, its pinned archive versions and migration validation, and the refitted recipes
`production`, `r2_c4_c2_c1_a3w_d3`, and `r2_e1_c1_a3w_d3`. Record exact parameters, feature lists,
cleaning, pooling/scaling, source-member count, delivered-sample construction, device, seeds,
frame hashes, dependency hashes, and code revisions. Verify completion manifests and forecast
hashes before scoring. The wind archive remains reconstructed from rounded polar values;
do not describe the reconstruction as upstream lossless components.

**Retain the parent study's folds and boundaries.** Fold A trains April 2024 through June 2025 and
scores July 2025 through June 2026. Fold B reverses those windows and is retrospective robustness
evidence. All observed power remains before 1 July 2026. Keep seeds 0, 1 and 2 separate until
aggregation. A missing fit is reported as pending; do not substitute an old-pipeline model or a
partial migration to complete a table.

**Each comparison needs identical targets, capacities and forecast keys.** Validate unique series,
initialisation and valid-time keys, finite observations and predictions, positive frozen
capacities, expected sample counts, and timestamp-derived leads. Require exact target, capacity,
type and key agreement across arms and seeds. Decide common target-supported rows without
consulting forecast values, and print excluded-row counts. Report both absolute scores and paired
differences with equal weight per series.

## Planned contrasts and scoring

**Name three deciding contrasts before producing diagnostics.** Each uses the corrected pipeline:

1. Best quantile recipe versus the production configuration refit, using the common 19-level
   pinball score.
2. Best quantile recipe versus the best point recipe, using the same score.
3. The quantile recipe's pooled weather-member sample versus its control-member quantile sample,
   using a separate nine-level pinball diagnostic and descriptive nominal 80% coverage. Run this
   contrast only if the saved
   sample construction recovers the control member and its nine levels unambiguously.

**The common marginal score is twice the mean pinball loss at 19 midpoint probabilities.** Use
`(i - 0.5) / 19` for `i = 1,...,19`, normalise each row by its own capacity before aggregation,
and print each level separately. For the delivered empirical samples, sort finite values and use
inverse-CDF rank `ceil(n*q)-1`; record `n` and the exact rule. Rearrangement for marginal scoring
does not establish coherent joint scenarios. Nine fitted quantiles are quantile knots, not nine
exchangeable draws. Historical fair CRPS is exploratory for cross-recipe comparisons and cannot
decide the quantile comparison. Any leaderboard number must come from the official scorer with
its required exact reference keys; this saved-prediction study promises no leaderboard submission.

**The control-member comparison preserves its nine supported nominal levels.** Its knots are
trained at 0.1, 0.2, ..., 0.9. Count crossings and their extent before rearrangement. Score the
native nine knots at their nominal levels, against the pooled
sample's empirical inverse-CDF quantiles at the same nine levels. Report twice their mean pinball
as a separate nine-level diagnostic, not the primary 19-level score. Also report a separately
labelled rearranged variant after sorting knot values against the same nominal levels. Use the
native 0.1 and 0.9 knots for nominal 80% control-member coverage: require both endpoint inequalities,
so crossed endpoints fail coverage; report their invalid-interval rate and signed endpoint width.
Report the rearranged 80% interval separately. Control-member 90% coverage is pending because
0.05 and 0.95 knots were not fitted; do not extrapolate or treat knots as nine exchangeable atoms.

**Report calibration diagnostics overall, by series type, and by the existing lead bands.** The
bands are less than 72 hours, 72 to less than 264 hours, and 264 hours onward. Print 80% and 90%
coverage, interval width, mean signed error, ensemble-mean MAE, all 19 pinball losses, and the
common aggregate. Coverage is percent of cases; interval width and losses are percentages of
capacity. Show nominal coverage beside observed coverage, without calling wider intervals better
from coverage alone.

**Use a descriptive empirical-CDF diagnostic for delivered samples only.** At observed `y`, compute
`(#values < y + 0.5 * #values == y) / n`, with no randomisation. Outside sample support it is zero
or one. For continuous independent identically distributed samples and an independent outcome,
the no-tie reference has equal mass at `k/n`, `k=0,...,n`, rather than a continuous uniform density.
That exchangeability reference does not apply to the pooled fitted quantile knots. Control-member
knot PIT remains pending; no CDF tails are invented. Synthetic calibrated finite iid ensembles,
underdispersed, biased and tied samples must expose the intended behaviour. CDF shape is
descriptive and cannot identify the cause of miscalibration.

**Use paired whole-month inference without treating generators as independent weather episodes.**
Resample whole valid-time calendar months and one paired fitting seed, 2,000 times. Keep
equal-series means as the primary estimand. Print the separate equal-month mean and t-interval,
months favouring each loss contrast, seed spread, and row/series/month counts. Mark fewer than
three seeds or six months as provisional; do not issue a full-year verdict from them. Coverage
differences remain descriptive. Confirm any recommendation on the refitted fold B and the
parent study's second hyperparameter setting; otherwise label it unresolved or pending.

## Diagnostic branches and deferred experiments

**Check the source-upgrade boundary descriptively before later calibration.** Report spread,
absolute error, coverage and missingness before and after 12 November 2024, assigned by
`nwp_init_time` rather than valid time, where the saved fold supports them. Verify this archive
boundary against ingestion provenance. This question is restricted to the November boundary;
inventory the 12 May 2026 cycle boundary but leave its separate comparison pending. These era summaries
have different seasons and training exposure, so they do not identify a causal upgrade effect.
Do not fit a calibrator across the boundary in this study.

**Training coverage remains pending unless suitable predictions already exist.** Validation
predictions cannot substitute for training predictions. If exact matching training samples are
available with preserved provenance, report training versus scored-row coverage as exploratory,
including the control-member interval. A near-nominal training interval and undercoverage on scored
rows are consistent with overfitting; they do not prove it or exclude other sources of distribution
shift. Otherwise record this test as pending and specify the required later prediction pass.

**No coherence ranking is planned.** Dependence across lead times and generators requires a stated
use case and defensible scenario semantics. Marginal nine-quantile outputs do not supply that
automatically. Energy score, variogram score, weather-cell sums, hierarchy sums, ECC and Schaake
shuffle comparisons remain outside this first diagnostic study.

**No calibration gate is passed by this report alone.** A later, separately preregistered G5 screen
would leave one scored month out, fit calibration on the other eleven, and compare against a
matched `manual_heuristic` baseline without using the held-out month for parameter selection.
Only after that screen meets the issue's gate would the cross-fitted training forecasts and full
calibrator experiments proceed. Preserve the 14-day purge and upgrade checks for that later
cross-fit. G9 remains closed: this study does not authorise an ensemble-statistic screening model,
change `ensemble_member` semantics, or score July 2026 onward.

## Reporting and review

**Publish only numbers printed by committed reporting code.** Keep raw identifiers and fit
manifests private. Public tables and figures use type aggregates or the approved anonymised labels,
never generator names, coordinates or individual MW output. Record input and output hashes,
specification, bootstrap seed and metric definitions in the new report's provenance.

**The first deliverable may honestly be a partial diagnostic study.** List completed recipes and
folds, pending training coverage, missing sensitivity fits, and any unrecoverable sample lineage.
Review figure text before rendering and inspect the rendered figures. Complete evidence,
scientific-validity, first-stumble, prose and persona reviews before a publication recommendation.
The proposed follow-up is the preregistered G5 calibration screen, requiring existing saved
predictions and CPU work, with no new data purchase; its cost must be estimated from measured
scoring throughput before execution is proposed.

## Plan-review triage

Both independent reviewers found the nine-knot distribution policy ambiguous. Accepted:
control-member diagnostics now use its nine supported nominal levels and 80% interval,
with 90% coverage pending. The main 19-level empirical score remains for delivered samples.
Also accepted source-cycle era assignment, native and rearranged diagnostics, crossing counts,
and an explicit deterministic empirical-CDF definition with its limited reference interpretation.
