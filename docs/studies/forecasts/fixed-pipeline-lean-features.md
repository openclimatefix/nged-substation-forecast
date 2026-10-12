# Do wind and radiation fixes improve XGBoost forecasts, and which lean inputs help?

> **Draft study.** Matched refits are running. Corrected-pipeline results and final scientific
> reviews are pending; no forecast-effect recommendation is made yet.

This study compares two changes to the weather-feature pipeline and revisits the lean solar and
wind inputs in Part 1b of [issue 1156](https://github.com/openclimatefix/nged-substation-forecast/issues/1156).
It follows the [feature-fault diagnostic audit](production-feature-faults.md) and
[quick wins](quick-wins.md), using the latter study's winning configuration alongside a
refit of the production XGBoost configuration. Production here means the training recipe fitted
on the study's historical data, rather than a deployed model artifact.

<!-- HEADLINE_RESULTS -->

> **How this page was made.** A human supplied the research question. OpenAI Codex wrote this
> study's new code, analysis and text, using existing repository study machinery. Independent
> AI scientific, evidence and prose reviews are required before publication. Human review of this
> page is pending.

<!-- KEY_FINDINGS -->

## Introduction

[PR 1166](https://github.com/openclimatefix/nged-substation-forecast/pull/1166) reconstructs
half-hourly radiation from the means over the intervals ending at the weather timestamps.
[PR 1167](https://github.com/openclimatefix/nged-substation-forecast/pull/1167) interpolates wind
components before deriving speed and bearing. This avoids interpolating a bearing through south
when it crosses north. Interpolating components also changes intermediate wind speeds, so the
wind comparison measures the joint speed-and-bearing processing change. The four treatments
are the old pipeline, radiation corrected, wind
corrected, and both corrected. Every comparison refits the same recipe on matched training keys
and scores matched forecasts against the same observed power and frozen capacities.

The wind archive was reconstructed from rounded archived speed and bearing. Its validation checks
that reconstruction against the archived values; it does not establish lossless recovery of
upstream components. These two PRs address wind and radiation processing. They do not resolve every
fault listed in issue 1156. Guarding the windchill formula's temperature and speed domain is a
separate study-only comparison.

## Data and methods

### Reference recipes

The production recipe uses 500 trees, depth 6, learning rate 0.05, row subsampling 0.8 and column
subsampling 0.8. It fits one model per series through the public production training and prediction
methods. The quick-wins winner is `r2_e1_c1_a3w_d3`: nine-quantile training, a forecast-lead feature,
lean wind inputs and one pooled model per series type. It retains numeric site identifiers,
training-only per-series scaling, column subsampling 1 and the historical fitting method. It adds
no early stopping, analogues or calibration.

The production reference and the Part 1b control are different recipes. The latter keeps the
historical quick-wins fitting path, settling filter and full columns, with column subsampling
set to one. It provides a common control for the feature comparisons; its score must not be
labelled the public production configuration's score.

| Recipe | Training groups | Objective | Training path | Column subsampling | Scaling |
|---|---|---|---|---|---|
| Production configuration | One per series | Squared error | Public production methods, quantile data matrix | 0.8 | Production preprocessing |
| Part 1b full-input control | One per series | Squared error | Historical quick-wins data matrix | 1 | Historical preprocessing |
| Quick-wins winner | One pooled fit per series type | Nine quantiles, 0.1–0.9 | Historical quick-wins data matrix | 1 | Per-series training-only scaling |

The pooled winner adds a numeric Float32 series identifier after feature preparation. The
identifier is a feature, rather than a categorical encoding. It uses 26 trained columns for
solar and demand and 18 for wind; the per-series full-input controls use 24. Feature comparisons
within Part 1b retain 24 columns by duplicating retained columns, and their unpadded checks are
reported separately.

### Data and target support

The all-series estimates include demand as well as the six solar farms (`PV`) and three wind
farms (`Wind`). Rows below are unique target-supported forecast cases per seed; ensemble members
are not additional cases. Both treatments in a paired comparison use these same observations.
Fold A has complete month support. In Fold B, two demand series lack some months; three of the
2,000 bootstrap draws omit a series when its supported months are not drawn. The all-series
bootstrap then averages the remaining represented series. Solar and wind retain complete support.

#### Target-supported validation rows: Fold A

| Series type | Series | Forecast cases per seed | Fitting seeds | Validation months |
|---|---|---|---|---|
| all | 31 | 1070930 | 3 | 12 |
| PV | 6 | 175795 | 3 | 12 |
| Wind | 3 | 107121 | 3 | 12 |

#### Target-supported validation rows: Fold B

| Series type | Series | Forecast cases per seed | Fitting seeds | Validation months |
|---|---|---|---|---|
| all | 31 | 1326430 | 3 | 15 |
| PV | 6 | 253003 | 3 | 15 |
| Wind | 3 | 135921 | 3 | 15 |

### Scoring and uncertainty

The main comparison between recipes uses twice the mean pinball loss at the same 19 midpoint
probabilities for every issued distribution. Pinball loss measures whether a forecast quantile is
too high or too low, with the penalty set by its probability. Each loss is divided by that series'
frozen effective capacity before aggregation. This scale is the 99th percentile of absolute power
over the available history, shared by all recipes. It is a measured historical scale rather than a
connection rating. The frozen snapshot includes readings after the scored periods, so normalised
aggregate results are retrospective and conditional on that common scale. The scale enters
scoring only; pooled fitting uses a separately calculated training-only scale. Lower is better; a difference of one point means
one percentage point of capacity. This finite-grid score is a study diagnostic, rather than the
official leaderboard score or an exact integral of continuous ranked probability score (CRPS).

The original study's fair CRPS is retained for reproduction and comparisons within the same
sample construction. Its sample-size correction assumes independent draws. The winner instead
crosses nine fixed quantile predictions with 51 weather members to make 459 values, so fair CRPS
cannot decide its comparison with a 51-value point-model ensemble. The main score sorts each
issued sample and uses empirical inverse-CDF rank `ceil(n*q)-1`, without extrapolating tails.
Coverage is the fraction of observations inside the specified interval; width and coverage alone
do not rank forecast quality.

Each estimate gives equal weight to each series and fitting seed. Paired intervals resample whole
validation months and one shared seed 2,000 times. The bootstrap represents variation across
validation months and an individual fitted seed; the central estimate averages the three observed
seeds. The separately reported t-interval gives months equal weight and averages the three fitted
seeds within each month before estimating variation across months. It does not include fitting-seed
variation in the same way as the bootstrap. Three seeds, a second hyperparameter setting and a
retrospective second fold provide robustness checks. An unresolved interval does not establish
that an effect is zero.

Fold A trains from April 2024 through June 2025 and validates from July 2025 through June 2026.
Fold B reverses these periods and is retrospective evidence. The original quick-wins selection
already used Fold A, so refitting its winner on that year is reproduction, rather than a new
independent confirmation. Validation uses weekly weather initialisations and all 51 members;
these scores must not be presented as official leaderboard results. All fitting targets, power-lag observations and scored outcomes precede 1 July 2026. Score
normalisation uses a separately frozen effective-capacity table: each series' full-history
99th percentile of absolute observed power. That table can include observations after the study
cutoff. It is shared across treatments and does not enter fitting or prediction. Lags are checked
to precede forecast issuance.

Planned contrasts were written into the study plan before results existed; other analyses are
exploratory, and additions after results are labelled post hoc. The month-resampled intervals
cover weather-month variation and fitting seed, not generalisation to different generators.
There is no multiple-comparison correction: a null exploratory contrast has a nominal
5% chance of statistical significance at the 5% level. The number of false findings is unknown,
and shared months can make false findings cluster.

### Rebuilding the lean inputs

The feature comparisons use the cleaned 51-member point-model recipe, with 500 trees and column
subsampling 1. Deterministic duplicate columns keep the input width equal at 24 per series type;
with every column available to each tree, this controls random column availability. A separate
unpadded lean check tests sensitivity to that construction. The exact production and quick-wins
reference refits remain unpadded.

For solar, the starting cut removes weather-regime proxies and weekday columns while retaining
power lags. The planned cumulative ladder adds 10 m wind speed, pressure, 500 hPa height, longwave
radiation and its temperature-adjusted residual, then replaces the clock features with sun
position. It stops at the first step whose three-seed mean solar gain is below 0.15 points. A gain
above that threshold only permits the next step; a recommendation also needs the uncertainty,
month consistency, second fold and sensitivity checks.

For wind, each removed radiation, dewpoint, precipitation-type, annual-cycle and weekday column
is added separately. Removing the weekday ordinal and local UTC offset provides negative controls.
The windchill domain guard is tested separately on the full and lean inputs. Total cloud cover
waits for the full post-introduction year required by the issue; this study does not evaluate it.



### Input audit

The input audit compares engineered columns on matched rows, counts changed values and missingness,
and checks domain and timing diagnostics. A deliberately delayed radiation input tests whether the
correlation diagnostic moves by an hour. That synthetic recovery does not establish power to
detect a forecast-loss improvement. Precipitation availability eras have different seasons and
training exposure, so their comparison is descriptive rather than a causal upgrade effect.



## Results

### Reproducing the old-pipeline references

These three-seed refits provide the matched old-pipeline references. `old_production` is the
production configuration, `old_quick_wins` is the named pooled quantile winner, and `old_cleaned`
is the cleaned point-model baseline used for the feature comparisons. The tables below were
selected and formatted directly from the aggregate report by the publication-table script.
The earlier page's headline comparison used commissioning cleaning and column subsampling 1;
that is the cleaned baseline here. The production-configuration row instead preserves column
subsampling 0.8 and the public production fitting path. Their difference is not a pipeline-fix effect.

#### Old-pipeline reproduction: common pinball score (% capacity)

| Recipe | Estimate | Lower 95% | Upper 95% | Seeds | Months |
|---|---|---|---|---|---|
| old_production | 11.59 | 10.93 | 12.22 | 3 | 12 |
| old_quick_wins | 7.87 | 7.43 | 8.28 | 3 | 12 |
| old_cleaned | 11.40 | 10.78 | 11.99 | 3 | 12 |

#### Old-pipeline reproduction: nominal 80% coverage (% cases)

| Recipe | Estimate | Lower 95% | Upper 95% | Seeds | Months |
|---|---|---|---|---|---|
| old_production | 11.00 | 10.19 | 11.77 | 3 | 12 |
| old_quick_wins | 71.27 | 70.17 | 72.40 | 3 | 12 |
| old_cleaned | 10.48 | 9.93 | 11.07 | 3 | 12 |


The earlier study's quoted 57% coverage came from its standalone `e1_quantile` model on seed 0.
The headline winner also pooled series, added lead and used lean wind inputs. Its coverage is
measured separately here. A difference from the old quoted coverage cannot be credited to the
wind or radiation fixes; their effect requires the matched old-versus-corrected winner contrast.
The historical interval convention also differed from the empirical inverse-CDF convention here.

### Radiation correction improves the solar score in both folds (provisional)

![Radiation-only changes in forecast loss on Fold A, with matched forecasts and month-resampled intervals. Negative differences favour corrected radiation.](../assets/fixed-pipeline-lean-features/radiation-fold-a.svg)

With the production configuration refitted, the radiation-only correction lowers solar's common
pinball score in both folds and both hyperparameter settings. The paired month-and-seed intervals
are statistically significant at the 5% level in all four comparisons. Solar's score improves in
every validation month in these comparisons. Wind's bootstrap intervals include both improvement
and deterioration in both settings and folds. In Fold B's main setting, the equal-month t-interval
favours deterioration. The point estimates agree in direction, but the intervals use different
weighting and treatment of fitting-seed variation, so the wind conclusion remains unresolved.

The following planned contrasts subtract the old pipeline's common pinball score from the
radiation-only pipeline's score. Negative differences favour the correction. Loss differences
are percentage points of capacity; the bounds are paired month-and-seed intervals. `PV` means
solar farms. Fold A has 12 validation months and Fold B has 15. Other loss measures are retained
in the aggregate reports; these tables show the common score used for the recipe comparison.

#### Radiation correction: July 2025–June 2026, main setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.122 | -0.175 | -0.075 | 11 |
| PV | -0.337 | -0.473 | -0.236 | 12 |
| Wind | 0.043 | -0.048 | 0.122 | 4 |

#### Radiation correction: July 2025–June 2026, sensitivity setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.112 | -0.189 | -0.002 | 10 |
| PV | -0.397 | -0.511 | -0.292 | 12 |
| Wind | 0.038 | -0.063 | 0.113 | 4 |

#### Radiation correction: April 2024–June 2025, retrospective, main setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.088 | -0.146 | -0.022 | 15 |
| PV | -0.346 | -0.498 | -0.237 | 15 |
| Wind | 0.119 | -0.066 | 0.361 | 4 |

#### Radiation correction: April 2024–June 2025, retrospective, sensitivity setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.106 | -0.156 | -0.060 | 12 |
| PV | -0.329 | -0.432 | -0.237 | 15 |
| Wind | 0.039 | -0.060 | 0.172 | 6 |

<!-- PIPELINE_RESULTS -->


<!-- FEATURE_RESULTS -->

<!-- AUDIT_RESULTS -->

The separate [calibration diagnostics study](calibration-diagnostics-1156.md) starts Part 2 from
saved corrected-pipeline predictions. It does not fit a calibrator or establish joint scenario
coherence. This study changes no production code or forecast contract.

<!-- PROVENANCE_AND_DECISIONS -->


## Discussion: what to use

<!-- DISCUSSION_RESULTS -->

## Proposed follow-up study

<!-- FOLLOWUP_RESULTS -->

## Limitations

Effective capacity is an empirical output scale, rather than a nameplate or connection rating.
Because its frozen full-history estimate can use later observations, the normalised aggregate
results are retrospective and conditional on that capacity snapshot. They do not reproduce a
score normalised using only information available at forecast issuance.

These are weekly-initialisation diagnostics for this fixed eligible population and historical
windows. Fold B is retrospective, and the original quick-wins winner was selected using Fold A.
The component archive reconstructs rounded polar observations. Frozen capacities, commissioning
cleaning and available target months affect the estimand. In Fold B, some demand series have
incomplete month support; the report discloses that support and bootstrap draws omitting a series.
The two PRs leave other audited feature defects outside their scope. No official leaderboard
comparison follows from these weekly scores.

## Scope

This study covers the pinned ECMWF ensemble archive and the named XGBoost recipes. It does not
establish performance at new generators, on later forecast periods or under a live deployment.
Total cloud cover and calibration experiments remain outside Part 1b. The separate Part 2 study
examines marginal distributions and does not establish coherent joint scenarios.

## Data and code availability

ECMWF ensemble weather and NGED observed power are the production-system inputs. The weather
component migration reconstructs archived speed and direction; it does not download new native
components. The NGED observations, generator metadata, fitted boosters and row-level forecasts
remain private. Published artifacts contain aggregate scores, feature names, code and execution
specifications, without a private generator mapping.

The old implementation is pinned to repository commit
`4e820d24c0129c8c95c567ff7355abff8d416144`; the wind and radiation PR heads are
`38025f4eab457b9605b8c124006609869dbfca0d` and
`0f094c356da82e75c4014664ad7fb33fe1df8b7f`. Isolated package source trees preserve those
implementations while the study code evolves. A committed integration patch combines their
feature-processing order, and the runtime inventory hashes every selected Python source file.
The runs use XGBoost 3.4.1 and Polars 2.0.0. Fit manifests record source and dependency hashes,
parameters, data snapshots, seeds and hardware selection; report provenance verifies saved fit
artifacts before aggregation.

<!-- AVAILABILITY_RESULTS -->

## Reproducing the figures

<!-- REPRODUCING_RESULTS -->
