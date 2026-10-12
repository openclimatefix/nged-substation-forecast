# How well calibrated are the corrected XGBoost forecasts?

## Summary

**Calibration findings remain pending: this page is a methods draft for a separate start on
Part 2 of issue 1156.** It will compare how often observations fall inside the corrected models'
forecast intervals, alongside forecast error and interval width. Completed results, scientific
reviews and prose reviews are required before drawing conclusions or making recommendations.

<!-- HEADLINE_RESULTS -->

> **How this page was made.** The research question came from a human. This methods draft was
> written by OpenAI Codex. Results and figures have not yet been incorporated into this page;
> scientific, prose and human review of the completed page remain pending.

## Introduction

**The question is whether the corrected models describe uncertainty as well as they predict
power.** An interval labelled 80% should contain observations at its stated rate under an
appropriate interpretation of the forecast distribution. Coverage alone is insufficient: an
interval can cover more observations simply by becoming wider.

This study uses saved forecasts from the
[wind/radiation and lean-input study](fixed-pipeline-lean-features.md). It compares refits of the
production configuration and the quick-wins quantile winner. The best point recipe is included
when the parent study's material-effect gate calls for that refit. The intended reader is a
forecast maintainer deciding which calibration questions deserve a subsequent experiment.

## Data and methods

### Inputs and comparisons

**Only completed fits using both wind and radiation corrections are eligible.** The parent study
documents the exact features, parameters, preprocessing and frozen inputs. Wind components in the
migrated archive are reconstructed from rounded polar values, rather than lossless upstream
components. Forecast hashes, sample counts, targets, capacities and keys must agree with their
recorded provenance before comparisons proceed. Missing recipes or fitting seeds remain pending.

Fold A trains on April 2024–June 2025 and scores July 2025–June 2026. Fold B reverses those windows
and provides retrospective robustness evidence. The intended fitting seeds are 0, 1 and 2;
observed power from July 2026 onward is excluded.

A *planned* contrast was specified in the study plan before results existed; other comparisons
are *exploratory*. The three planned contrasts compare the quantile winner with the production
configuration, with the best point recipe, and with its own control weather member. The last
comparison requires unambiguous recovery of the control member's quantile levels. Subgroup,
source-era and rearrangement diagnostics are exploratory. Intervals describe individual paired
contrasts; they do not adjust for multiple comparisons or establish a family-wide significance
claim. Coverage differences remain descriptive.

### Samples, quantile knots and scores

**Delivered samples and fitted quantile knots have different meanings.** The point recipe delivers
51 predictions, one per weather member. The quantile winner fits nine levels, 0.1 through 0.9,
for each member and pools the resulting 459 values. These are structured predictions, without an
established interpretation as independent scenarios. The control member's nine fitted values
retain their nominal probability labels.

For delivered samples, the common score is twice mean pinball loss at the 19 midpoint probabilities
`(i - 0.5) / 19`, for `i = 1, …, 19`. Pinball loss measures error at a specified quantile, with
different penalties above and below it. After sorting the sample, its empirical inverse CDF uses
zero-based rank `ceil(n * q) - 1`. Each row's loss is divided by its frozen capacity before series
receive equal weight. Frozen effective capacity is the full-history 99th percentile of absolute
observed power; it can use readings after the study cutoff. It enters scoring only, and makes
normalised loss and width retrospective and conditional on that common snapshot. Losses and
widths are percentages of frozen effective capacity; coverage is a percentage of cases. Negative
bias means underprediction. These are study diagnostics, not official leaderboard scores.

The control comparison uses a separate nine-level pinball score at its supported nominal levels.
Native 80% coverage uses the 0.1 and 0.9 knots: both endpoint inequalities must hold, so inverted
endpoints fail coverage and retain signed width. Crossings and their extent are reported before
rearrangement. Sorting the knots and reassigning the same levels is a separately labelled
exploratory variant. Neither variant supplies the 0.05 and 0.95 knots needed for native 90%
coverage or a justified full CDF; those diagnostics remain pending.

### Marginal diagnostics and uncertainty

Delivered 80% and 90% coverage, widths, signed error, ensemble-mean absolute error, spread and
level-specific pinball losses are examined overall, by series type and by lead bands below
72 hours, 72 to below 264 hours, and 264 hours onward. Coverage is interpreted alongside width
and loss.

The empirical-CDF diagnostic at observation `y` is
`(count(values < y) + 0.5 * count(values == y)) / n`, without randomisation. For continuous
independent identically distributed samples and an independent outcome from the same distribution,
its no-tie reference is equal mass at `k/n`, for `k = 0, …, n`, rather than a continuous uniform
histogram. Under those assumptions, an interval with one-based endpoint ranks `L` and `U` has
reference coverage `(U - L) / (n + 1)`. These finite-sample references are no guarantee for either
delivered construction, particularly the fixed quantile pool. No independent-sample reference is
applied to the nine control knots. Histogram shape cannot identify the cause of miscalibration.

Paired uncertainty uses 2,000 resamples of whole valid-time calendar months and one shared fitting
seed per draw, preserving the primary equal-series mean. A separate equal-month mean and
t-interval, months favouring each loss contrast, seed spread and support counts aid interpretation.
Fewer than three seeds or six months makes a comparison provisional. If a draw contains no
supported months for a series, that series is omitted from that draw and the remaining series
receive equal weight. Reports expose month support and the frequency and extent of these
omissions; an interval does not remove the limitation of sparse support.

## Results

**Completed aggregate results and their provenance remain to be inserted.** The final page must
identify delivered recipes, folds, seeds and settings, then present sample semantics before
interpreting coverage, loss or empirical-CDF shape. Missing comparisons must remain labelled
pending, including the second hyperparameter setting where unavailable.

<!-- SAMPLE_SEMANTICS -->

<!-- DIAGNOSTIC_RESULTS -->

## Discussion

**No calibration or production recommendation follows from this draft.** Once results are
available, the useful question is whether changes in coverage also improve loss without relying
on excessive width. Any recommendation needs the retrospective Fold B check and the parent
study's second hyperparameter setting. Marginal performance cannot establish coherent scenarios
across sites or forecast times.

## Proposed follow-up study

A separately planned calibration screen could test whether held-out-month calibration improves
on the matched `manual_heuristic` baseline. It would fit on eleven scored months and evaluate the
twelfth without using it to select parameters. Its value depends on the diagnostic findings;
little reproducible improvement or inadequate support would argue against proceeding.

For planning only, assume about one day of agent work, one hour of maintainer review and up to
one CPU-day on the existing workstation, with no new downloads or spend. These are rough
assumptions, not measured costs or an execution commitment. Measured scoring throughput is needed
before proposing execution. This draft passes neither the issue's G5 calibration gate nor its
G9 gate for later ensemble-statistic experiments.

## Limitations

The quick-wins winner was selected using Fold A, so this comparison is not an independent repeat
of that selection. Fold B is retrospective and does not simulate a deployment trained only on
earlier data. Shared weather episodes, frozen capacities, missing targets and sparse month
support limit interpretation beyond the observed period and population.

Source-era summaries use recorded weather initialisation times before or from
13 November 2024 at 00:00 UTC, the archive's precipitation-availability boundary. This is distinct
from the operational cycle change. Different seasons and training exposure prevent a causal
upgrade interpretation. The separate comparison around 12 May 2026 at 06:00 UTC remains pending;
inventorying that boundary does not verify a cycle effect.

## Scope

This stage examines saved-prediction marginal distributions. Training-versus-validation coverage,
cross-fitted training forecasts, spread rescaling, EMOS, residual-quantile and isotonic calibration
remain pending. Validation forecasts cannot substitute for matching training predictions, and
coverage alone neither proves nor rules out overfitting.

Joint energy and variogram scores, scenario reordering, weather-cell or hierarchy sums and the
issue's later model experiments remain outside this stage. There is no coherence claim, forecast
contract change, production code change or evaluation of observed power from July 2026 onward.

## Data and code availability

The study-local scorer is `studies/forecast_calibration/calibration_diagnostics.py`. Forecasts,
power observations, fit manifests and detailed provenance remain private under the study's data
namespace. Public evidence is restricted to aggregate tables and figures without site identifiers,
names or coordinates. Exact input/output hashes, code revision, dependency versions, XGBoost
settings and completed-run inventory must be supplied with the results; none is inferred from
this draft.

<!-- PROVENANCE -->

## Reproducing the figures

**No figures have been incorporated into this draft.** The following is a future scoring command
template, not a record of a completed run. Variables must point to verified, matching completed
fits for one fold and setting, and a new output directory; omit a missing point fit and report it
as pending rather than substituting an older pipeline.

```bash
.venv/bin/python studies/forecast_calibration/calibration_diagnostics.py \
  --frames "$COMBINED_FRAMES" \
  --fit "$PRODUCTION_FIT" "$QUANTILE_FIT" "$POINT_FIT" \
  --output "$NEW_DIAGNOSTIC_OUTPUT"
```

The scorer writes aggregate diagnostic tables, sample semantics and a manifest. Exact completed
commands, figure-generation code and rendering instructions remain pending, along with scientific,
first-reader and prose review of the completed page.
