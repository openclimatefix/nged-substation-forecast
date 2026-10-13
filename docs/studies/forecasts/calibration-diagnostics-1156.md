# How well calibrated are the corrected XGBoost forecasts?

## Summary

This study compares a refit of the production configuration with a recipe that predicts power
at specified probability levels, called quantiles. It evaluates the forecast distributions
they deliver in two training/test splits and two sets of model hyperparameters.
**The quantile recipe has lower distribution loss than the production refit in all four comparisons.**
The marginal diagnostics leave calibration questions open: coverage changes accompany substantial
width changes, and pooling's small nine-level loss effect differs between folds. This separate
Part 2 draft of issue 1156 supports no calibration or production recommendation. Scientific
and first-reader reviews are complete; final prose and publication-inventory checks remain pending.

**Part 2 implementation is on hold while the maintainer revises its plan.** The write-up records
only the completed diagnostics: two folds, two hyperparameter settings, and three fitting
seeds, presented in twenty-six diagnostic tables and one coverage figure. No calibration correction
or temporal or spatial coherence method has been implemented or evaluated. Further Part 2 code and
experiments are deferred pending the revised plan.

![Coverage on Fold A, primary training settings.](../assets/calibration-diagnostics-1156/coverage-fold-a.svg)

> **How this page was made.** The research question came from a human. OpenAI Codex wrote the new
> study code, analysis and text, reusing recipes and fitting helpers from the Claude-authored
> [quick-wins study](quick-wins.md). Two independent AI scientific reviews checked the results
> and draft page; final prose and publication checks remain pending. Human review is pending.

## Key findings

The quantile recipe combines nine fitted quantiles from each of 51 weather members into a pool
of 459 values. We also compare that pool with the nine labelled quantiles from the control
weather member, using a separate nine-level loss score. Here, control denotes a weather member,
not the production recipe.

- Pooling's nine-level loss effect is unresolved in both Fold A settings and favours pooling in
  both retrospective Fold B settings. Pooled 80% intervals are narrower and cover fewer cases
  than native control intervals in every completed comparison.
- Quantile-pool solar coverage exceeds the 80% label but falls below the 90% label in all four
  fold/settings; wind falls below both labels. Coverage must be read alongside width and loss.
- [Supplementary tables](calibration-diagnostics-1156-tables.md) expose empirical-CDF bins,
  lead bands, descriptive November eras and support counts without a causal or coherence claim.

## Introduction

**The question is whether the corrected models describe uncertainty as well as they predict
power.** An interval labelled 80% should contain observations at its stated rate under an
appropriate interpretation of the forecast distribution. Coverage alone is insufficient: an
interval can cover more observations simply by becoming wider.

This study uses saved forecasts from the
[wind/radiation and lean-input study](fixed-pipeline-lean-features.md). It compares refits of the
production configuration and the quick-wins quantile winner. The best point refit was not run
because the parent study's preregistered material-effect gate was not reached. The intended reader
is a forecast maintainer deciding which calibration questions deserve a subsequent experiment.

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

Each fold uses two sets of model hyperparameters: the primary settings, labelled main in the
tables, and alternative settings used for the sensitivity comparison. The parent study records
the exact parameter values.

A *planned* contrast was specified in the study plan before results existed; other comparisons
are *exploratory*. The three planned contrasts compare the quantile winner with the production
configuration, with the best point recipe, and with the quantiles it fits for the weather
ensemble's control member. The last
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

The reported difference averages three fitting seeds, with equal weight per series. Each of 2,000
uncertainty draws resamples whole valid-time calendar months and selects one shared fitting seed
for the compared forecasts. Its central 95% range describes month and individual-fit variation;
it is not a confidence interval for the mean of a three-fit ensemble. The separate equal-month
t-interval first averages the seeds within each month and then estimates variation between months.
Within each series, the central loss difference weights supported forecast cases equally;
months with more cases therefore contribute more. The equal-month t-interval gives calendar
months equal weight and targets a different mean. Mean coverage is descriptive and has no
reported confidence interval. Reports also show months favouring each loss contrast, seed spread
and support counts. Forecast cases from different issuances can share the same valid hour.
The parent study uses the same bootstrap method with a different deterministic resample sequence;
small differences between its interval endpoints and these reflect Monte Carlo variation.
Fewer than three seeds or six months makes a comparison provisional. If a draw contains no
supported months for a series, that series is omitted from that draw and the remaining series
receive equal weight. Reports expose month support and the frequency and extent of these
omissions; an interval does not remove the limitation of sparse support.

## Results

**Both folds and both training settings are complete for production and the quantile recipe.**
Each comparison uses fitting seeds 0, 1 and 2 for both corrected-pipeline recipes. Fold B
provides retrospective robustness evidence. The
conditional best point comparison was not run because the preregistered gate was not reached.

**Production delivers 51 weather-member point predictions; the quantile recipe delivers a pool
of 459 fitted values.** The quantile pool combines nine fitted levels from each of 51 weather
members. The control member retains nine labelled quantile knots, which are not nine independent
draws. No independent-sample reference is applied to those control knots.

**The rank table gives conditional analytic references, not measured coverage or calibration
guarantees.** For lower and upper one-based endpoint ranks `L` and `U`, reference coverage is
`(U - L) / (n + 1)` only for continuous independent identically distributed samples and an
independent outcome from the same distribution. Neither delivered recipe is asserted to satisfy
those assumptions; the structured quantile pool in particular has no such guarantee. Observed
coverage must therefore be read with the delivered construction and interval width.

| Sample count | Nominal coverage (% of cases) | Lower endpoint rank (one-based) | Upper endpoint rank (one-based) | Analytic continuous iid coverage (% of cases) |
|---|---|---|---|---|
| 51 | 80.000 | 6 | 46 | 76.923 |
| 51 | 90.000 | 3 | 49 | 88.462 |
| 459 | 80.000 | 46 | 414 | 80.000 |
| 459 | 90.000 | 23 | 437 | 90.000 |

### Fold A: primary settings

**The quantile recipe improves the shared nineteen-level pinball score in Fold A main, while
the effect of pooling on the separate nine-level score remains unresolved.** Completed results
compare `production` with the quantile recipe `r2_e1_c1_a3w_d3`, using fitting seeds 0, 1 and 2.
Each fit contributes 1,070,930 supported forecast rows across 31 series and 12 scored months.
Every series has all 12 months, and no uncertainty draw drops a series. The second hyperparameter
setting for Fold A follows below. The conditional comparison with `r2_c4_c2_c1_a3w_d3` was not run because
the preregistered gate was not reached.

**The quantile recipe's lower delivered-sample loss is supported across all 12 months in this
fold (planned).** Its nineteen-level score is 7.756% of frozen effective capacity, against
11.479% for production. The paired difference is -3.723 percentage points of frozen capacity,
with a central 95% month/individual-fit range of [-4.059, -3.382]. The separate equal-month
t-interval also lies below zero. Both recipes use the corrected pipeline; the recipe contrast
does not isolate the effect of the wind or radiation corrections.

**Coverage rises alongside much wider delivered intervals.** In the tables, `coverage80` and
`coverage90` mean the percentage of cases inside intervals labelled 80% and 90%; `width80`,
`width90`, bias and pinball loss are percentages of frozen effective capacity. Negative bias
means underprediction. Production's coverage describes empirical intervals made from weather-member
point predictions. The coverage increase alone does not establish better calibration.

| Recipe | Metric | Value (%) |
|---|---|---|
| production | pinball19 | 11.479 |
| production | bias | -2.041 |
| production | coverage80 | 11.654 |
| production | coverage90 | 15.312 |
| production | width80 | 4.069 |
| production | width90 | 5.341 |
| r2_e1_c1_a3w_d3 | pinball19 | 7.756 |
| r2_e1_c1_a3w_d3 | bias | -0.786 |
| r2_e1_c1_a3w_d3 | coverage80 | 71.617 |
| r2_e1_c1_a3w_d3 | coverage90 | 75.549 |
| r2_e1_c1_a3w_d3 | width80 | 26.698 |
| r2_e1_c1_a3w_d3 | width90 | 29.389 |

**Solar coverage exceeds the 80% label but falls below the 90% label; wind coverage falls below
both labels (exploratory).** For the quantile pool, solar (`PV`) coverage is 82.561% and 86.996%,
with respective widths of 20.550% and 23.044% of frozen capacity. Wind coverage is 72.872% and
81.004%, with widths of 42.060% and 50.097%. The two interval labels therefore need separate
interpretations, and the type-specific widths matter when reading coverage.

| Recipe | Generation type | Metric | Value (%) |
|---|---|---|---|
| production | PV | coverage80 | 18.950 |
| production | PV | coverage90 | 25.101 |
| production | PV | width80 | 4.122 |
| production | PV | width90 | 5.379 |
| production | Wind | coverage80 | 15.355 |
| production | Wind | coverage90 | 19.682 |
| production | Wind | width80 | 9.855 |
| production | Wind | width90 | 12.984 |
| r2_e1_c1_a3w_d3 | PV | coverage80 | 82.561 |
| r2_e1_c1_a3w_d3 | PV | coverage90 | 86.996 |
| r2_e1_c1_a3w_d3 | PV | width80 | 20.550 |
| r2_e1_c1_a3w_d3 | PV | width90 | 23.044 |
| r2_e1_c1_a3w_d3 | Wind | coverage80 | 72.872 |
| r2_e1_c1_a3w_d3 | Wind | coverage90 | 81.004 |
| r2_e1_c1_a3w_d3 | Wind | width80 | 42.060 |
| r2_e1_c1_a3w_d3 | Wind | width90 | 50.097 |

**Pooling has an unresolved nine-level loss effect relative to the native control knots
(planned).** Pooled minus native loss is -0.028 percentage points of frozen capacity, with a
central 95% month/individual-fit range of [-0.072, 0.012]; pooling has lower loss in 6 of 12
months. The separate equal-month t-interval also includes zero. Pooled 80% coverage is lower
than native coverage (71.617% versus 73.922%), and the pooled interval is narrower (26.698%
versus 28.223% of frozen capacity). Neither the loss uncertainty nor the descriptive
coverage/width change establishes that pooling improves calibration. The nine-level scores
must be read separately from the delivered nineteen-level scores.

**Native and rearranged control results agree because no saved control knots cross
(exploratory).** Zero crossings and zero invalid intervals are observed properties of these
saved predictions, without a general guarantee for future forecasts. `crossing_rate` and
`invalid_interval_rate` are percentages of cases. Native and rearranged control 90% coverage
remain pending because the required endpoint knots were not fitted.

| Distribution variant | Metric | Value (%) |
|---|---|---|
| control_native | pinball9 | 8.344 |
| control_native | coverage80 | 73.922 |
| control_native | width80 | 28.223 |
| control_native | crossing_rate | 0.000 |
| control_native | invalid_interval_rate | 0.000 |
| pooled_nine | pinball9 | 8.316 |
| pooled_nine | coverage80 | 71.617 |
| pooled_nine | width80 | 26.698 |
| pooled_nine | crossing_rate | 0.000 |
| pooled_nine | invalid_interval_rate | 0.000 |
| control_rearranged | pinball9 | 8.344 |
| control_rearranged | coverage80 | 73.922 |
| control_rearranged | width80 | 28.223 |
| control_rearranged | crossing_rate | 0.000 |
| control_rearranged | invalid_interval_rate | 0.000 |

**The paired loss table separates the supported recipe improvement from the unresolved pooling
effect.** Treatment is `r2_e1_c1_a3w_d3` throughout; differences are treatment minus reference,
so negative values favour treatment. “Capacity points” means percentage points of frozen
effective capacity. The printed 95% intervals are the month/individual-fit ranges described
above. The delivered and native-control contrasts are planned; the rearranged-control contrast
is exploratory. Fold A sensitivity results follow before the retrospective Fold B results.

| Treatment variant | Reference recipe | Reference variant | Pinball metric | Difference (capacity points) | 95% interval lower (capacity points) | 95% interval upper (capacity points) | Months with lower loss |
|---|---|---|---|---|---|---|---|
| delivered | production | delivered | pinball19 | -3.723 | -4.059 | -3.382 | 12 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_native | pinball9 | -0.028 | -0.072 | 0.012 | 6 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_rearranged | pinball9 | -0.028 | -0.072 | 0.012 | 6 |

### Fold A: sensitivity settings

**The alternative training settings retain Fold A's delivered loss advantage and unresolved
pooling effect.** Both recipes use fitting seeds 0, 1 and 2, with 1,070,930 supported forecast
rows per fit across 31 series and 12 months. Every series has all 12 months, and no uncertainty
draw drops a series. The shared nineteen-level loss difference is -3.497 percentage points of
frozen capacity, with a central 95% month/individual-fit range of [-3.801, -3.163] and lower loss
in all 12 months (planned). The separate equal-month t-interval also lies below zero.

| Recipe | Metric | Value (%) |
|---|---|---|
| production | pinball19 | 11.271 |
| production | bias | -1.933 |
| production | coverage80 | 12.162 |
| production | coverage90 | 15.835 |
| production | width80 | 4.448 |
| production | width90 | 5.797 |
| r2_e1_c1_a3w_d3 | pinball19 | 7.775 |
| r2_e1_c1_a3w_d3 | bias | -0.726 |
| r2_e1_c1_a3w_d3 | coverage80 | 73.400 |
| r2_e1_c1_a3w_d3 | coverage90 | 77.108 |
| r2_e1_c1_a3w_d3 | width80 | 28.450 |
| r2_e1_c1_a3w_d3 | width90 | 31.139 |

**Solar coverage remains above the 80% label and below the 90% label; wind remains below both
(exploratory).** Quantile-pool solar coverage is 84.950% and 89.361%, with respective widths of
21.921% and 24.678% of frozen capacity. Wind coverage is 74.117% and 81.946%, with widths of
42.881% and 50.910%. Coverage is a percentage of cases; widths, bias and losses are percentages
of frozen effective capacity. The coverage changes must be read alongside width and loss, and
do not establish better calibration.

| Recipe | Generation type | Metric | Value (%) |
|---|---|---|---|
| production | PV | coverage80 | 18.540 |
| production | PV | coverage90 | 24.269 |
| production | PV | width80 | 4.692 |
| production | PV | width90 | 6.066 |
| production | Wind | coverage80 | 15.724 |
| production | Wind | coverage90 | 19.971 |
| production | Wind | width80 | 9.744 |
| production | Wind | width90 | 12.740 |
| r2_e1_c1_a3w_d3 | PV | coverage80 | 84.950 |
| r2_e1_c1_a3w_d3 | PV | coverage90 | 89.361 |
| r2_e1_c1_a3w_d3 | PV | width80 | 21.921 |
| r2_e1_c1_a3w_d3 | PV | width90 | 24.678 |
| r2_e1_c1_a3w_d3 | Wind | coverage80 | 74.117 |
| r2_e1_c1_a3w_d3 | Wind | coverage90 | 81.946 |
| r2_e1_c1_a3w_d3 | Wind | width80 | 42.881 |
| r2_e1_c1_a3w_d3 | Wind | width90 | 50.910 |

**Pooling remains unresolved on the separate nine-level score under Fold A sensitivity
settings (planned).** Pooled minus native loss is -0.035 percentage points of frozen capacity,
with a central 95% month/individual-fit range of [-0.079, 0.004] and lower loss in 7 of 12
months. The separate equal-month t-interval also includes zero. Pooled 80% coverage is lower
than native coverage (73.400% versus 75.587%), alongside a narrower interval (28.450% versus
30.019% of frozen capacity). Those descriptive changes do not establish improved calibration.

**The saved control knots also have no crossings or invalid intervals in Fold A sensitivity
results (exploratory).** Native and rearranged diagnostics therefore agree. Observed zero rates
cover both settings in both folds; the observation provides no guarantee for future forecasts.
Control 90% coverage remains unavailable from the fitted endpoint knots.

| Distribution variant | Metric | Value (%) |
|---|---|---|
| control_rearranged | pinball9 | 8.390 |
| control_rearranged | coverage80 | 75.587 |
| control_rearranged | width80 | 30.019 |
| control_rearranged | crossing_rate | 0.000 |
| control_rearranged | invalid_interval_rate | 0.000 |
| control_native | pinball9 | 8.390 |
| control_native | coverage80 | 75.587 |
| control_native | width80 | 30.019 |
| control_native | crossing_rate | 0.000 |
| control_native | invalid_interval_rate | 0.000 |
| pooled_nine | pinball9 | 8.355 |
| pooled_nine | coverage80 | 73.400 |
| pooled_nine | width80 | 28.450 |
| pooled_nine | crossing_rate | 0.000 |
| pooled_nine | invalid_interval_rate | 0.000 |

**The sensitivity table preserves the separate nineteen-level and nine-level comparisons.**
Treatment is `r2_e1_c1_a3w_d3` throughout; negative differences favour treatment. Capacity points
and the printed 95% intervals retain the definitions above. The delivered and native-control
contrasts are planned; the rearranged-control contrast is exploratory.

| Treatment variant | Reference recipe | Reference variant | Pinball metric | Difference (capacity points) | 95% interval lower (capacity points) | 95% interval upper (capacity points) | Months with lower loss |
|---|---|---|---|---|---|---|---|
| delivered | production | delivered | pinball19 | -3.497 | -3.801 | -3.163 | 12 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_native | pinball9 | -0.035 | -0.079 | 0.004 | 7 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_rearranged | pinball9 | -0.035 | -0.079 | 0.004 | 7 |

### Retrospective Fold B: primary settings

**The quantile recipe retains its delivered-sample loss advantage in retrospective Fold B
(planned).** Its nineteen-level score is 8.745% of frozen effective capacity, against 12.179%
for production. The paired difference is -3.435 percentage points of frozen capacity, with a
central 95% month/individual-fit range of [-3.959, -2.981]. Loss is lower in all 15 scored
months, and the separate equal-month t-interval also lies below zero. Fold B trains on the later
window and scores the earlier window, so the agreement provides retrospective evidence rather
than a simulated deployment result.

**Fold B's all-series estimate includes incomplete month support for two demand series.** Each
fit contributes 1,326,430 supported forecast rows across 31 series and 15 scored months; series
support ranges from 5 to 15 months. Two of the 2,000 uncertainty draws omit one series because
none of that series' supported months is drawn. The remaining series receive equal weight in
those draws. This diagnostic's resample sequence differs from the parent study's sequence, so
the omission count need not match the parent's report.

| Recipe | Metric | Value (%) |
|---|---|---|
| r2_e1_c1_a3w_d3 | pinball19 | 8.745 |
| r2_e1_c1_a3w_d3 | bias | 1.988 |
| r2_e1_c1_a3w_d3 | coverage80 | 70.424 |
| r2_e1_c1_a3w_d3 | coverage90 | 74.393 |
| r2_e1_c1_a3w_d3 | width80 | 26.338 |
| r2_e1_c1_a3w_d3 | width90 | 29.170 |
| production | pinball19 | 12.179 |
| production | bias | 2.576 |
| production | coverage80 | 10.992 |
| production | coverage90 | 14.415 |
| production | width80 | 4.051 |
| production | width90 | 5.360 |

**Fold B solar coverage is above the 80% label and below the 90% label; wind remains below
both labels (exploratory).** Quantile-pool solar coverage is 80.442% and 84.946%, with respective
widths of 19.432% and 22.249% of frozen capacity. Wind coverage is 72.069% and 80.378%, with
widths of 42.205% and 50.882%. Coverage is a percentage of cases; widths, bias and losses in
these tables are percentages of frozen effective capacity. The observed coverage and width
changes do not establish better overall calibration.

| Recipe | Generation type | Metric | Value (%) |
|---|---|---|---|
| r2_e1_c1_a3w_d3 | PV | coverage80 | 80.442 |
| r2_e1_c1_a3w_d3 | PV | coverage90 | 84.946 |
| r2_e1_c1_a3w_d3 | PV | width80 | 19.432 |
| r2_e1_c1_a3w_d3 | PV | width90 | 22.249 |
| r2_e1_c1_a3w_d3 | Wind | coverage80 | 72.069 |
| r2_e1_c1_a3w_d3 | Wind | coverage90 | 80.378 |
| r2_e1_c1_a3w_d3 | Wind | width80 | 42.205 |
| r2_e1_c1_a3w_d3 | Wind | width90 | 50.882 |
| production | PV | coverage80 | 17.100 |
| production | PV | coverage90 | 22.489 |
| production | PV | width80 | 3.959 |
| production | PV | width90 | 5.211 |
| production | Wind | coverage80 | 14.840 |
| production | Wind | coverage90 | 19.246 |
| production | Wind | width80 | 9.594 |
| production | Wind | width90 | 12.775 |

**Pooling shows a small nine-level loss improvement in Fold B, while Fold A's effect remains
unresolved (planned).** Pooled minus native loss is -0.047 percentage points of frozen capacity,
with a central 95% month/individual-fit range of [-0.083, -0.016] and lower loss in 10 of 15
months. The separate equal-month t-interval also lies below zero. Pooled 80% coverage is lower
than native coverage (70.424% versus 72.746%), alongside a narrower interval (26.338% versus
27.905% of frozen capacity). The small loss gain does not establish improved calibration from
pooling. The nine-level comparison remains separate from the delivered nineteen-level score.

**Native and rearranged control results again agree because the saved control knots do not
cross (exploratory).** Crossing and invalid-interval rates are percentages of cases. Their
observed zeros provide no guarantee for future forecasts. Control 90% coverage remains pending.

| Distribution variant | Metric | Value (%) |
|---|---|---|
| control_native | pinball9 | 9.351 |
| control_native | coverage80 | 72.746 |
| control_native | width80 | 27.905 |
| control_native | crossing_rate | 0.000 |
| control_native | invalid_interval_rate | 0.000 |
| control_rearranged | pinball9 | 9.351 |
| control_rearranged | coverage80 | 72.746 |
| control_rearranged | width80 | 27.905 |
| control_rearranged | crossing_rate | 0.000 |
| control_rearranged | invalid_interval_rate | 0.000 |
| pooled_nine | pinball9 | 9.304 |
| pooled_nine | coverage80 | 70.424 |
| pooled_nine | width80 | 26.338 |
| pooled_nine | crossing_rate | 0.000 |
| pooled_nine | invalid_interval_rate | 0.000 |

**The Fold B paired table preserves the distinct loss comparisons and their uncertainty.**
Treatment is `r2_e1_c1_a3w_d3` throughout; negative differences favour treatment. Capacity
points and the printed 95% intervals retain the definitions used for Fold A. The delivered and
native-control contrasts are planned; the rearranged-control contrast is exploratory.

| Treatment variant | Reference recipe | Reference variant | Pinball metric | Difference (capacity points) | 95% interval lower (capacity points) | 95% interval upper (capacity points) | Months with lower loss |
|---|---|---|---|---|---|---|---|
| delivered | production | delivered | pinball19 | -3.435 | -3.959 | -2.981 | 15 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_native | pinball9 | -0.047 | -0.083 | -0.016 | 10 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_rearranged | pinball9 | -0.047 | -0.083 | -0.016 | 10 |

### Retrospective Fold B: sensitivity settings

**Fold B's alternative settings retain the delivered loss advantage (planned).** The nineteen-level
score is 8.510% of frozen capacity for the quantile recipe and 12.079% for production. The paired
difference is -3.569 percentage points of frozen capacity, with a central 95% month/individual-fit
range of [-4.134, -2.974] and lower loss in all 15 months. The separate equal-month t-interval
also lies below zero. Support matches Fold B main: 1,326,430 forecast cases per seed across 31
series, with 5 to 15 supported months per series, two series missing months, and two draws each
omitting one series. Fitting seeds are 0, 1 and 2.

#### Fold B (retrospective), sensitivity: delivered overall diagnostics

| Recipe | Metric | Value (%) |
|---|---|---|
| production | pinball19 | 12.079 |
| production | bias | 2.556 |
| production | coverage80 | 11.581 |
| production | coverage90 | 15.032 |
| production | width80 | 4.290 |
| production | width90 | 5.646 |
| r2_e1_c1_a3w_d3 | pinball19 | 8.510 |
| r2_e1_c1_a3w_d3 | bias | 1.822 |
| r2_e1_c1_a3w_d3 | coverage80 | 72.917 |
| r2_e1_c1_a3w_d3 | coverage90 | 76.614 |
| r2_e1_c1_a3w_d3 | width80 | 28.634 |
| r2_e1_c1_a3w_d3 | width90 | 31.384 |

**Solar remains above the 80% label and below the 90% label; wind remains below both
(exploratory).** Quantile-pool solar coverage is 82.781% and 87.652%, with respective widths of
21.008% and 24.072% of frozen capacity. Wind coverage is 73.039% and 81.103%, with widths of
43.632% and 52.086%. Coverage is a percentage of cases; widths, bias and losses are percentages
of frozen effective capacity. No confidence intervals for these mean coverages are reported.

#### Fold B (retrospective), sensitivity: solar and wind interval diagnostics

| Recipe | Generation type | Metric | Value (%) |
|---|---|---|---|
| production | PV | coverage80 | 18.098 |
| production | PV | coverage90 | 23.361 |
| production | PV | width80 | 4.487 |
| production | PV | width90 | 5.843 |
| production | Wind | coverage80 | 13.595 |
| production | Wind | coverage90 | 17.479 |
| production | Wind | width80 | 9.032 |
| production | Wind | width90 | 12.028 |
| r2_e1_c1_a3w_d3 | PV | coverage80 | 82.781 |
| r2_e1_c1_a3w_d3 | PV | coverage90 | 87.652 |
| r2_e1_c1_a3w_d3 | PV | width80 | 21.008 |
| r2_e1_c1_a3w_d3 | PV | width90 | 24.072 |
| r2_e1_c1_a3w_d3 | Wind | coverage80 | 73.039 |
| r2_e1_c1_a3w_d3 | Wind | coverage90 | 81.103 |
| r2_e1_c1_a3w_d3 | Wind | width80 | 43.632 |
| r2_e1_c1_a3w_d3 | Wind | width90 | 52.086 |

**Pooling's small nine-level improvement is supported in both Fold B settings (planned).** Under
sensitivity settings, pooled minus native loss is -0.063 capacity points, with a central 95%
month/individual-fit range of [-0.099, -0.030] and lower loss in 11 of 15 months. The separate
equal-month t-interval also lies below zero. Pooled 80% coverage is lower than native coverage
(72.917% versus 75.067%), alongside a narrower interval (28.634% versus 30.213% of frozen
capacity). The loss gain and descriptive coverage/width changes do not establish better calibration.
Native and rearranged control results agree because no saved control knots cross; zero crossing
and invalid-interval rates are observed in all four completed fold/settings, without a future
guarantee. Control 90% coverage remains unavailable from the fitted knots.

#### Fold B (retrospective), sensitivity: native, rearranged and pooled nine-level diagnostics

| Distribution variant | Metric | Value (%) |
|---|---|---|
| control_native | pinball9 | 9.149 |
| control_native | coverage80 | 75.067 |
| control_native | width80 | 30.213 |
| control_native | crossing_rate | 0.000 |
| control_native | invalid_interval_rate | 0.000 |
| control_rearranged | pinball9 | 9.149 |
| control_rearranged | coverage80 | 75.067 |
| control_rearranged | width80 | 30.213 |
| control_rearranged | crossing_rate | 0.000 |
| control_rearranged | invalid_interval_rate | 0.000 |
| pooled_nine | pinball9 | 9.085 |
| pooled_nine | coverage80 | 72.917 |
| pooled_nine | width80 | 28.634 |
| pooled_nine | crossing_rate | 0.000 |
| pooled_nine | invalid_interval_rate | 0.000 |

**The paired sensitivity results retain the separate loss scales and planned contrasts.** Treatment
is `r2_e1_c1_a3w_d3` throughout. Negative differences favour treatment; capacity points and the
95% ranges retain the definitions above. The rearranged-control contrast is exploratory.

#### Fold B (retrospective), sensitivity: paired nine- and nineteen-level losses

| Treatment variant | Reference recipe | Reference variant | Pinball metric | Difference (capacity points) | 95% interval lower (capacity points) | 95% interval upper (capacity points) | Months with lower loss |
|---|---|---|---|---|---|---|---|
| delivered | production | delivered | pinball19 | -3.569 | -4.134 | -2.974 | 15 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_native | pinball9 | -0.063 | -0.099 | -0.030 | 11 |
| pooled_nine | r2_e1_c1_a3w_d3 | control_rearranged | pinball9 | -0.063 | -0.099 | -0.030 | 11 |

## Discussion

**Lower marginal loss does not settle the calibration or production decision.** The quantile
recipe's delivered nineteen-level loss advantage persists in both folds and both settings, but
the recipe contrast changes objective, inputs, pooling and preprocessing together. Pooling's
small nine-level improvement is supported in both retrospective Fold B settings and unresolved
in both Fold A settings. Narrower pooled 80% intervals also cover fewer observations than the
native control intervals. Those findings motivate a separate calibration question without
establishing that pooling improves calibration.

**The completed diagnostics justify neither a calibrator choice nor a coherence claim.** No
calibrator was fitted, the conditional best-point refit was not triggered, and marginal scores
cannot establish joint scenarios across sites or forecast times. The supplementary CDF, lead
and era tables are descriptive. Final prose review and the public reproduction inventory remain
pending. The completed diagnostics support no calibration or production recommendation.

## Proposed follow-up study

**The earlier calibration study proposal is retained for the plan revision.** The proposal does
not authorise implementation. Further Part 2 code and experiments are on hold.

A separately planned calibration screen could test whether held-out-month calibration improves
on the matched `manual_heuristic` baseline. It would fit on eleven scored months and evaluate the
twelfth without using it to select parameters. Its value depends on the diagnostic findings;
little reproducible improvement or inadequate support would argue against proceeding.

For planning only, assume about one day of agent work, one hour of maintainer review and up to
one CPU-day on the existing workstation, with no new downloads or spend. These are rough
assumptions, not measured costs or an execution commitment. Measured scoring throughput is needed
before proposing execution. These completed diagnostics do not establish the requirements for
proceeding with calibration or later ensemble-statistic experiments under the issue's G5 and
G9 gates.

## Limitations

The quick-wins winner was selected using Fold A, so this comparison is not an independent repeat
of that selection. Fold B is retrospective and does not simulate a deployment trained only on
earlier data. Shared weather episodes, frozen capacities, missing targets and sparse month
support limit interpretation beyond the observed period and population.

Source-era summaries use recorded weather initialisation times before or from
13 November 2024 at 00:00 UTC, the verified archive precipitation-availability boundary. This
grouping differs from the operational upgrade cycle at 12 November 2024 at 06:00 UTC and does
not evaluate that cycle's effect. Different seasons and training exposure prevent a causal
upgrade interpretation. The separate comparison around 12 May 2026 at 06:00 UTC remains pending;
inventorying that boundary does not verify a cycle effect.

## Scope

This stage examines saved-prediction marginal distributions. Training-versus-validation coverage,
cross-fitted training forecasts, spread rescaling, EMOS, residual-quantile and isotonic calibration
were not evaluated and are deferred pending the revised Part 2 plan. Validation forecasts cannot
substitute for matching training predictions, and coverage alone neither proves nor rules out
overfitting.

Joint energy and variogram scores, scenario reordering, weather-cell or hierarchy sums and the
issue's later model experiments remain outside this stage. There is no coherence claim, forecast
contract change, production code change or evaluation of observed power from July 2026 onward.

## Data and code availability

**OpenAI Codex wrote the new study work; the reused code has historical Claude credits.**
Following the imported study helpers and chart theme through their commit histories finds
Claude Fable 5, Opus 5 and 5.5 (including 1M-context credits), and Sonnet 4.6, 5 and 5.5;
some commits credit Claude without a model name. These are authorship records for reused code,
not the identities of this study's reviewers. Available Codex agents performed the independent
reviews in place of the study skill's unavailable named models.

**The public evidence consists of aggregate tables and figures; reproducing the scores requires
the private frozen inputs.** The scorer is `studies/forecast_calibration/calibration_diagnostics.py`;
`studies/quick_wins/fixed_pipeline_tables.py` selects the displayed tables without recalculating
statistics. Forecasts, observations, fit manifests and detailed input inventories remain private.
Completed diagnostic manifests cover Fold A and Fold B, each at main and sensitivity settings,
using CUDA fits without padding and seeds 0, 1 and 2 for production and `r2_e1_c1_a3w_d3`.
The conditional best-point refit was not run because the preregistered gate was not reached.

**The feature implementations and diagnostic environment are pinned.** The parent study's
`studies/quick_wins/fixed_pipeline_runtime/README.md` describes reconstruction from base revision
`4e820d24c0129c8c95c567ff7355abff8d416144`, wind PR revision
`38025f4eab457b9605b8c124006609869dbfca0d` and radiation PR revision
`0f094c356da82e75c4014664ad7fb33fe1df8b7f`, with the recorded combined integration patch.
Completed fit provenance records XGBoost 3.4.1; diagnostic manifests record NumPy 2.5.3,
Polars 2.0.0 and SciPy 1.18.1. The Fold A coverage render records Altair 6.3.0 and
vl-convert-python 1.9.0.post1. Diagnostics use bootstrap seed 20261012 and 2,000 resamples.

| Recorded source | SHA-256 |
|---|---|
| Diagnostic scorer, all four completed reports | `adba37654e98da9f49f8be1dfad906e1318a537b25250821f0a36108253dd5c7` |
| Fitter for the saved corrected-pipeline forecasts | `b073be43f8809290746d952bf1acf988ea1f01cf61a7e2ed11e6f51f2de57ca3` |
| Publication-table selector, Fold A main extract | `6c2bc6a59eb9ad5b8e8f4326c7c1fa850a2b90e11423c2c4e8da36d68d691362` |
| Coverage chart script, Fold A main render | `21cc9fde248b27b45790374dc154e9bc94e955d4f62bf6a343e603882d61dbda` |
| Executed combined integration patch | `23f4f76a30687b1c741b897273aba67fb5319dc43b6e51fdf062dc5083cf47a9` |
| Published equivalent integration patch | `fca0456fde5f6009290908c3c9b895bba285b4af7cdfa98972745a0adcc116b3` |

**The final public reproduction inventory is still being prepared.** Completed artifacts retain
original run configurations, input and output hashes, dependency versions and completion records.
The private preparation inventory includes unexecuted jobs, so that inventory alone is not a
completion record. Publication of the sanitised configurations and complete hash inventory remains
pending final reconciliation. Remaining public files are the four completed scoring configurations,
the four primary publication-table specifications, the supplementary table specification and their
sanitised hash inventory. Their public destinations have not been established, so no configuration
download links are asserted here. Whole private manifests are not public reproduction artifacts.

## Reproducing the figures

**The commands below rescore all four completed fold/settings from their six saved fits.** Run
from the repository root with the recorded environment. Set `STUDY_DATA_ROOT` to the private
`data/studies/per_study` directory and choose a new `STUDY_REPRO_ROOT`; each output directory must
be absent. The recipe and seed loops name the exact completed fit-directory layout. These are
portable reproduction commands, not a claim that new runs have been executed. They omit the
untriggered best-point refit. The [supplementary tables](calibration-diagnostics-1156-tables.md)
copy the completed selector output; reproducing that selection also requires the public
specification whose publication remains pending.

```bash
set -euo pipefail
STUDY_DATA_ROOT="/path/to/data/studies/per_study"
STUDY_REPRO_ROOT="/path/to/new/calibration-reproduction"
STUDY_PARENT_ROOT="$STUDY_DATA_ROOT/quick_wins_fixed_pipeline_1156"
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MLFLOW_DISABLE_AGENT_HINT=1

for STUDY_RUN in fold_a_main fold_a_sensitivity fold_b_main fold_b_sensitivity; do
  STUDY_FOLD="${STUDY_RUN%_*}"
  STUDY_SETTING="${STUDY_RUN##*_}"
  STUDY_FITS=()
  for STUDY_RECIPE in production r2_e1_c1_a3w_d3; do
    for STUDY_SEED in 0 1 2; do
      STUDY_FITS+=("$STUDY_PARENT_ROOT/fits/combined/$STUDY_FOLD/$STUDY_RECIPE/$STUDY_SETTING/cuda_seed${STUDY_SEED}_none")
    done
  done
  .venv/bin/python studies/forecast_calibration/calibration_diagnostics.py \
    --frames "$STUDY_PARENT_ROOT/frames/combined/$STUDY_FOLD" \
    --fit "${STUDY_FITS[@]}" \
    --output "$STUDY_REPRO_ROOT/reports/$STUDY_RUN"
done
```

**Figure text must be reviewed before rendering.** The coverage script verifies a completed
report's output hashes and exports the text with `--text-only`. The first-reader and prose reviews
must clear that exact text before a separate render command uses `--reviewed-text`. Use distinct,
new output directories for text export and rendering, and inspect the rendered labels and layout.
The following commands reproduce the Fold A main coverage figure from the rescored report above.

```bash
.venv/bin/python studies/forecast_calibration/calibration_coverage_charts.py \
  --report "$STUDY_REPRO_ROOT/reports/fold_a_main" \
  --output "$STUDY_REPRO_ROOT/coverage-fold-a-text" \
  --text-only
```

After the exported `figure-text.md` has passed the required text reviews:

```bash
.venv/bin/python studies/forecast_calibration/calibration_coverage_charts.py \
  --report "$STUDY_REPRO_ROOT/reports/fold_a_main" \
  --output "$STUDY_REPRO_ROOT/coverage-fold-a-render" \
  --reviewed-text "$STUDY_REPRO_ROOT/coverage-fold-a-text/figure-text.md"
```

**Scientific reviews are complete; final reading and publication checks remain pending.** The
scorer's immutable report, aggregate tables and manifest preserve their evidence; the reproduction
commands do not replace those checks.
