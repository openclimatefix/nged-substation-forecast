# Do wind and radiation fixes improve XGBoost forecasts, and which lean inputs help?

> **Draft study.** Pipeline comparisons are complete. Lean-feature robustness checks and final
> scientific reviews are pending; final input recommendations remain open.

This study compares two changes to the weather-feature pipeline and revisits the lean solar and
wind inputs in Part 1b of [issue 1156](https://github.com/openclimatefix/nged-substation-forecast/issues/1156).
It follows the [feature-fault diagnostic audit](production-feature-faults.md) and
[quick wins](quick-wins.md), using the latter study's winning configuration alongside a
refit of the production XGBoost configuration. Production here means the training recipe fitted
on the study's historical data, rather than a deployed model artifact.

**Correcting radiation processing consistently improves solar forecasts; correcting wind
processing does not establish a general forecast improvement.** Across both historical folds
and both training settings, radiation-only processing lowers solar loss, and the combined
corrections lower overall and solar loss within each reference recipe. Wind-only processing
worsens wind-farm loss in retrospective Fold B. Its Fold A effects remain unresolved.

The quick-wins winner retains a substantially lower common pinball score than the refitted
production configuration on the corrected inputs. This compares whole training recipes and
cannot be credited to the processing fixes. Lower loss also does not establish adequate interval
coverage; the separate [Part 2 study](calibration-diagnostics-1156.md) examines that question.
These are weekly-initialisation diagnostics for a fixed population, with a retrospective capacity
denominator. Final lean-wind recommendations await the remaining Part 1b settings and controls.

![Input fixes and recipe comparison.](../assets/fixed-pipeline-lean-features/headline-fold-a.svg)

> **How this page was made.** A human supplied the research question. OpenAI Codex wrote this
> study's new code, analysis and text, using existing repository study machinery. Independent
> AI scientific, evidence and prose reviews are required before publication. Human review of this
> page is pending.

## Key findings

- **Radiation correction consistently helps solar.** With the production configuration, its
  main-setting common pinball difference is -0.337 capacity points in Fold A (95% paired
  month-and-seed interval -0.473 to -0.236) and -0.346 in retrospective Fold B (-0.498 to -0.237).
  Both sensitivity settings also support lower solar loss. These comparisons cover six solar
  series and three fitting seeds, over 12 validation months in Fold A and 15 in Fold B.
- **Wind processing has no established general loss benefit.** Wind-only effects remain
  unresolved under the primary bootstrap in Fold A. For the three wind series in retrospective
  Fold B, wind-only processing increases common pinball loss by 0.194 capacity points in the
  main setting (0.002 to 0.470) and 0.121 in sensitivity (0.048 to 0.204). The combined corrections
  also worsen wind loss in Fold B sensitivity; their wind effects remain unresolved in the
  other fold/settings.
- **The winner's advantage is a recipe difference.** On corrected inputs, its all-series common
  pinball difference from refitted production is -3.723 capacity points in Fold A's main setting
  (-4.059 to -3.392) and -3.435 in Fold B's main setting (-3.936 to -2.982). Both sensitivity
  settings retain the advantage. The exact winner uses nine quantiles, pooling, forecast lead,
  lean wind inputs and training-only scaling, without early stopping. These contrasts do not
  isolate any one ingredient or establish adequate coverage.
- **The conditional best-point refit was not run.** Neither fold passes G3's preregistered
  0.15-capacity-point magnitude gate on main-setting production fair CRPS. Statistical
  significance alone does not pass that gate. Final lean-feature findings remain pending the
  Part 1b alternate settings and controls.

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
upstream components. The wind comparison includes the component migration and its storage
quantisation as well as component interpolation; native wind values can differ slightly too.
These two PRs address wind and radiation processing. They do not resolve every
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

The all-series estimates cover 31 metered series, including demand, solar, wind, biofuel and battery
storage. Feature comparisons target the six solar farms (`PV`) and three wind farms (`Wind`).
Rows below are unique target-supported forecast cases per seed; ensemble members
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
scoring only; pooled fitting uses a separately calculated training-only scale. Lower is better;
a difference of one point means one percentage point of capacity. This finite-grid score is a
study diagnostic, rather than the
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
these scores must not be presented as official leaderboard results. All fitting targets, power-lag
observations and scored outcomes precede 1 July 2026. Score normalisation uses a separately frozen
effective-capacity table: each series' full-history
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

The [feature inventory](fixed-pipeline-feature-inventory.md) prints every configured recipe's
ordered inputs and duplicate padding. Solar retains 13 real inputs and removes 11 from the
production list; wind retains 16 and removes eight. Those removals include three weekday
columns for each type. The registry also lists conditional recipes that were not necessarily run.

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
is the cleaned point-model baseline used for the feature comparisons.
The [reproduction tables](fixed-pipeline-results-tables.md#old-pipeline-reproduction) were
selected and formatted directly from the aggregate report by the publication-table script.
The earlier page's headline comparison used commissioning cleaning and column subsampling 1;
that is the cleaned baseline here. The production-configuration row instead preserves column
subsampling 0.8 and the public production fitting path. Their difference is not a pipeline-fix effect.

The earlier study's quoted 57% coverage came from its standalone `e1_quantile` model on seed 0.
The headline winner also pooled series, added lead and used lean wind inputs. Its coverage is
measured separately here. A difference from the old quoted coverage cannot be credited to the
wind or radiation fixes; their effect requires the matched old-versus-corrected winner contrast.
The historical interval convention also differed from the empirical inverse-CDF convention here.

### Radiation correction improves the solar score in both folds (provisional)

![Radiation correction on Fold A.](../assets/fixed-pipeline-lean-features/radiation-fold-a.svg)

With the production configuration refitted, the radiation-only correction lowers solar's common
pinball score in both folds and both hyperparameter settings. The paired month-and-seed intervals
are statistically significant at the 5% level in all four comparisons. Solar's score improves in
every validation month in these comparisons. Wind's bootstrap intervals include both improvement
and deterioration in both settings and folds. In Fold B's main setting, the equal-month t-interval
favours deterioration. The point estimates agree in direction, but the intervals use different
weighting and treatment of fitting-seed variation, so the wind conclusion remains unresolved.

These planned contrasts subtract the old pipeline's common pinball score from the
radiation-only pipeline's score. Negative differences favour the correction. Loss differences
are percentage points of capacity; the bounds are paired month-and-seed intervals. `PV` means
solar farms. Fold A has 12 validation months and Fold B has 15. Other loss measures are retained
in the aggregate reports; the [radiation tables](fixed-pipeline-results-tables.md#radiation-comparisons)
show the common score used for the recipe comparison.

### Both corrections lower overall loss in the first fold (provisional)

**Fold A's main setting shows a modest pipeline improvement alongside a much larger recipe
difference.** The combined wind and radiation corrections lower the all-series common pinball loss
by 0.115 capacity points within both the production configuration and the quick-wins winner. The
corrected winner's 3.723-point advantage over corrected production compares different training
recipes. That comparison changes the objective, pooling, features, and preprocessing together; it
cannot isolate the effect of quantile training or be credited to the pipeline corrections. These
combined-pipeline results are provisional pending final reviews. The sensitivity settings and
retrospective Fold B results follow the Fold A main results.

**The absolute scores keep the pipeline and recipe comparisons separate.** `old_production` and
`combined_production` use the production configuration; `old_winner` and `combined_winner` use the
quick-wins recipe. `combined` means both processing corrections. The deciding table below and
[Fold A appendix tables](fixed-pipeline-results-tables.md#fold-a-main) reproduce the report's
printed aggregate selections without recalculation. All loss differences use the same
19-level score and are in percentage points of frozen capacity. Negative differences favour the
treatment named first.

[Absolute scores](fixed-pipeline-results-tables.md#fold-a-main).

**Both corrections improve overall and solar loss within each recipe in Fold A's main setting,
while the wind effect remains unresolved.** For production, the all-series difference is
-0.115 points (95% interval -0.159 to
-0.067), and solar's difference is -0.317 (-0.443 to -0.224). For the winner, the corresponding
differences are -0.115 (-0.155 to -0.082) and -0.287 (-0.414 to -0.184). These paired improvements
are statistically significant at the 5% level in this fold and setting. Wind's production difference
is 0.049 (-0.079 to 0.160), and the winner's is -0.006 (-0.017 to 0.003). Both wind intervals allow
improvement and deterioration.

#### Deciding comparisons on common 19-level pinball loss: July 2025–June 2026, main setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.115 | -0.159 | -0.067 | 11 |
| Both corrections minus old pipeline | PV | -0.317 | -0.443 | -0.224 | 12 |
| Both corrections minus old pipeline | Wind | 0.049 | -0.079 | 0.160 | 2 |
| Both corrections on quick-wins winner | all | -0.115 | -0.155 | -0.082 | 12 |
| Both corrections on quick-wins winner | PV | -0.287 | -0.414 | -0.184 | 12 |
| Both corrections on quick-wins winner | Wind | -0.006 | -0.017 | 0.003 | 8 |
| Corrected quick-wins winner minus corrected production configuration | all | -3.723 | -4.059 | -3.392 | 12 |
| Corrected quick-wins winner minus corrected production configuration | PV | -2.631 | -3.356 | -1.940 | 12 |
| Corrected quick-wins winner minus corrected production configuration | Wind | -5.810 | -6.544 | -5.202 | 12 |

**The wind correction alone has no resolved loss effect in this fold and setting.** With the
production configuration, the wind-only difference is 0.005 points overall (95% interval -0.033 to
0.064) and -0.019 for wind farms (-0.117 to 0.080). Adding the wind correction after radiation is
corrected also leaves overall and wind effects unresolved. The intervals do not establish that the
wind correction has zero effect.

[Pipeline-isolation comparisons](fixed-pipeline-results-tables.md#fold-a-main).

In Fold A's main setting, the primary bootstrap remains unresolved for adding wind processing
after radiation on solar farms and for adding radiation processing after wind on wind farms.
Their equal-month t-intervals instead favour deterioration: 0.028 points (0.004 to 0.053) and
0.067 points (0.027 to 0.107), respectively. These intervals weight months and fitted seeds
differently, so the primary bootstrap conclusions remain unresolved.

**The interaction between the processing corrections remains unresolved.** The interaction subtracts
the radiation-only and wind-only scores from the combined score and adds the old-pipeline score. The
all-series interaction is 0.002 capacity points (95% interval -0.083 to 0.061). Solar and wind
interactions also remain unresolved. An interaction measures departure from additive correction
effects; the count of months with negative interaction is not a count of months with lower forecast
loss.

[Wind–radiation interaction](fixed-pipeline-results-tables.md#fold-a-main).

**The cleaned feature baseline also gains overall and for solar, with wind unresolved.** Both
corrections lower its all-series common pinball score by 0.147 points (95% interval -0.201 to
-0.079) and its solar score by 0.337 points (difference -0.337; -0.447 to -0.251). The wind
difference is 0.045 (-0.054 to 0.177). The cleaned baseline uses the historical point-model fitting
path and column subsampling one, so this comparison supports the feature-study control rather than
replacing the production-configuration comparison.

[Cleaned-control comparison](fixed-pipeline-results-tables.md#fold-a-main).

**Fold A does not pass the G3 material-fault threshold.** G3 uses the fair-CRPS comparison between
the matched 51-member production ensembles and requires an absolute change of at least 0.15 capacity
points. Fold A's three-seed fair-CRPS difference is -0.118 points, below that magnitude. The
separate equal-month mean is -0.124, with a month t-interval of -0.174 to -0.074: statistically
significant at the 5% level, but statistical significance does not satisfy the material-size
requirement. Fold B also falls below the threshold, as shown below; the material-fault gate is
not reached in either fold, so the conditional best-point refit is not triggered.

[G3 material-fault gate](fixed-pipeline-results-tables.md#fold-a-main).

### Both corrections lower overall loss in retrospective Fold B (provisional)

**Fold B supports modest overall and solar improvements within each recipe, alongside a larger
recipe difference.** Both corrected production and the corrected quick-wins winner have lower
common pinball loss than their old-pipeline counterparts overall and for solar. Their paired
95% intervals lie below zero. In this main setting, the combined corrections' wind effects
remain unresolved for both recipes. The winner's advantage over production again compares
whole recipes rather than
isolating the processing corrections. Fold B is retrospective; its sensitivity results follow
below. These pipeline comparisons support no feature recommendation.

[Absolute scores](fixed-pipeline-results-tables.md#fold-b-main-retrospective).

#### Deciding comparisons on common 19-level pinball loss: April 2024–June 2025 (retrospective), main setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.054 | -0.093 | -0.016 | 13 |
| Both corrections minus old pipeline | PV | -0.332 | -0.467 | -0.218 | 15 |
| Both corrections minus old pipeline | Wind | 0.159 | -0.082 | 0.431 | 4 |
| Both corrections on quick-wins winner | all | -0.080 | -0.116 | -0.043 | 13 |
| Both corrections on quick-wins winner | PV | -0.180 | -0.265 | -0.104 | 14 |
| Both corrections on quick-wins winner | Wind | 0.006 | -0.003 | 0.015 | 5 |
| Corrected quick-wins winner minus corrected production configuration | all | -3.435 | -3.936 | -2.982 | 15 |
| Corrected quick-wins winner minus corrected production configuration | PV | -2.916 | -3.496 | -2.332 | 15 |
| Corrected quick-wins winner minus corrected production configuration | Wind | -6.680 | -8.045 | -5.357 | 15 |

**Wind-only processing worsens wind's score in Fold B's main setting, while adding wind
processing after radiation remains unresolved.** The wind-only wind-farm interval lies above
zero; the conditional wind-after-radiation interval includes both improvement and deterioration.
The wind-only finding differs from Fold A's unresolved comparison; Fold B's sensitivity setting
also supports deterioration, as reported below. The primary bootstrap interaction intervals include
zero for every reported type. Negative loss
differences favour the treatment named first; all pinball differences are percentage points of
frozen capacity.

[Pipeline-isolation comparisons](fixed-pipeline-results-tables.md#fold-b-main-retrospective).

Fold B's main setting also has interval-method disagreements. The equal-month t-intervals
favour deterioration for wind-only processing across all series, both corrections on wind farms,
and wind processing after radiation across all series: 0.027 points (0.011 to 0.042),
0.159 (0.038 to 0.279), and 0.038 (0.015 to 0.060). The wind interaction has a negative equal-month
estimate of -0.152 (-0.266 to -0.039), while its primary bootstrap interval includes zero.
This interaction measures departures from additivity of losses; it does not establish a forecast
improvement from adding wind processing. The primary bootstrap remains unresolved for these
comparisons.

[Wind–radiation interaction](fixed-pipeline-results-tables.md#fold-b-main-retrospective).

**The cleaned feature baseline also improves overall and for solar in Fold B's main setting.**
Its wind interval remains unresolved. This pipeline comparison does not decide which feature
cuts or additions help; those require their own matched held-out tests.

[Cleaned-control comparison](fixed-pipeline-results-tables.md#fold-b-main-retrospective).

**Neither fold reaches G3's preregistered material-size threshold, so the conditional best-point
refit is not run.** Fold B's fair-CRPS difference is -0.0572 capacity points, with a separate
equal-month mean of -0.0534 and a month t-interval of [-0.0827, -0.0240]. The supported improvement
is below the required absolute change of 0.15 points, as is Fold A's improvement. Statistical
significance alone does not pass this gate.

[G3 material-fault gate](fixed-pipeline-results-tables.md#fold-b-main-retrospective).

### Fold A's sensitivity setting retains overall and solar gains (provisional)

**The second training setting retains the overall and solar improvements within each recipe.**
For the production configuration, both corrections change the all-series common pinball loss by
-0.102 capacity points (95% interval -0.174 to -0.004). For the quick-wins winner, the difference
is -0.134 (-0.174 to -0.099). The overall and solar gains are statistically significant at the 5%
level, and solar improves in all 12 validation months within both recipes. The corrected winner's
all-series advantage over corrected production is -3.497 points (-3.827 to -3.165). That larger
comparison changes the whole training recipe and does not isolate the processing corrections.

[Absolute scores](fixed-pipeline-results-tables.md#fold-a-sensitivity).

#### Deciding comparisons on common 19-level pinball loss: July 2025–June 2026, sensitivity setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.102 | -0.174 | -0.004 | 10 |
| Both corrections minus old pipeline | PV | -0.396 | -0.503 | -0.300 | 12 |
| Both corrections minus old pipeline | Wind | 0.055 | -0.077 | 0.153 | 3 |
| Both corrections on quick-wins winner | all | -0.134 | -0.174 | -0.099 | 12 |
| Both corrections on quick-wins winner | PV | -0.367 | -0.493 | -0.263 | 12 |
| Both corrections on quick-wins winner | Wind | 0.001 | -0.006 | 0.008 | 6 |
| Corrected quick-wins winner minus corrected production configuration | all | -3.497 | -3.827 | -3.165 | 12 |
| Corrected quick-wins winner minus corrected production configuration | PV | -2.633 | -3.340 | -1.971 | 12 |
| Corrected quick-wins winner minus corrected production configuration | Wind | -5.497 | -6.259 | -4.873 | 12 |

**Wind effects remain unresolved under the primary month-and-seed bootstrap in this setting.**
Both corrections and the wind-only correction have intervals allowing improvement and
deterioration for wind farms. Adding wind processing after radiation is corrected also remains
unresolved. For the wind-only correction across all series, the equal-month t-interval favours
deterioration: the equal-month difference is 0.018 points (0.001 to 0.036), while the primary
bootstrap difference is 0.013 (-0.032 to 0.075). The two intervals use different weighting and
treatment of fitting-seed variation, so the all-series wind-only effect remains unresolved across
those methods. The full sensitivity extract retains the equal-month estimates and support counts.

[Pipeline-isolation comparisons](fixed-pipeline-results-tables.md#fold-a-sensitivity).

**The wind–radiation interaction remains unresolved for all series, solar, and wind.**
The interaction intervals allow both signs, as in Fold A's main setting. These sensitivity checks
cover Fold A; the retrospective Fold B sensitivity results follow below. The cleaned control has
no sensitivity fit in this pipeline-isolation comparison.

[Wind–radiation interaction](fixed-pipeline-results-tables.md#fold-a-sensitivity).

### Fold B sensitivity retains overall and solar gains but worsens wind scores (provisional)

**Both corrections lower overall and solar loss within each recipe in Fold B's sensitivity
setting, but increase wind loss.** For production, the all-series common pinball difference is
-0.090 capacity points (95% paired month-and-seed interval -0.148 to -0.038), and solar's is
-0.322 (-0.418 to -0.235). Wind's difference is 0.172 (0.044 to 0.324), supporting deterioration
at the 5% level in this setting. For the quick-wins winner, the wind increase is much smaller:
0.009 (0.001 to 0.017), also statistically significant at 5%. Both recipes' combined-correction
wind effects remain unresolved in Fold B's main setting and in both Fold A settings. The
supported harm therefore applies to these Fold B sensitivity comparisons.

The corrected winner's all-series advantage over corrected production is -3.569 points
(-4.127 to -2.993). This changes the whole recipe; it does not isolate the processing corrections.
[Absolute scores](fixed-pipeline-results-tables.md#fold-b-sensitivity-retrospective) are retained
in the appendix. Fold B remains retrospective evidence.

#### Deciding comparisons on common 19-level pinball loss: April 2024–June 2025 (retrospective), sensitivity setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.090 | -0.148 | -0.038 | 11 |
| Both corrections minus old pipeline | PV | -0.322 | -0.418 | -0.235 | 15 |
| Both corrections minus old pipeline | Wind | 0.172 | 0.044 | 0.324 | 3 |
| Both corrections on quick-wins winner | all | -0.095 | -0.120 | -0.071 | 15 |
| Both corrections on quick-wins winner | PV | -0.197 | -0.254 | -0.145 | 15 |
| Both corrections on quick-wins winner | Wind | 0.009 | 0.001 | 0.017 | 5 |
| Corrected quick-wins winner minus corrected production configuration | all | -3.569 | -4.127 | -2.993 | 15 |
| Corrected quick-wins winner minus corrected production configuration | PV | -3.121 | -3.760 | -2.477 | 15 |
| Corrected quick-wins winner minus corrected production configuration | Wind | -6.734 | -8.075 | -5.458 | 15 |

**Wind-only processing worsens wind loss in both Fold B settings; adding wind processing after
radiation remains unresolved under the primary bootstrap.** The sensitivity wind-only difference
is 0.121 points (0.048 to 0.204), alongside 0.194 (0.002 to 0.470) in the main setting. In the
sensitivity setting, adding wind processing after radiation changes wind loss by 0.133 points
(-0.018 to 0.310), allowing both improvement and deterioration. The interaction's primary
bootstrap intervals also include zero for every reported type. See the
[pipeline-isolation and interaction tables](fixed-pipeline-results-tables.md#fold-b-sensitivity-retrospective).

**The equal-month t-intervals disagree with the primary bootstrap on several sensitivity
comparisons.** Across all series, wind-only processing has a bootstrap difference of 0.013 points
(-0.011 to 0.044), while its equal-month estimate is 0.013 (0.001 to 0.025). Adding wind processing
after radiation has an all-series bootstrap difference of 0.016 (-0.016 to 0.046), versus an
equal-month estimate of 0.015 (0.001 to 0.028). For wind farms, that conditional comparison's
equal-month estimate is 0.132 (0.051 to 0.213), favouring deterioration, while its bootstrap
interval includes zero. Solar's interaction likewise remains unresolved under the bootstrap
(0.015; -0.023 to 0.062), while the equal-month t-interval is positive: 0.0143
(0.0004 to 0.0282). These methods weight months and fitting-seed variation differently; the
primary bootstrap does not resolve those comparisons.

For the winner's combined-correction wind comparison, both intervals favour deterioration.
The equal-month t-interval's lower bound is 0.0002 points in the report; its printed 0.000 in the
three-decimal full extract is rounding, rather than an interval containing zero. The production
combined-correction wind and wind-only wind intervals also favour deterioration under both methods.
The [fair-CRPS sensitivity table](fixed-pipeline-results-tables.md#fold-b-sensitivity-retrospective)
is supplementary and does not replace the main-setting G3 gate. These pipeline comparisons do
not decide which lean inputs help; the matched feature comparisons remain separate.

### Lean wind lowers loss, while lean solar raises it in both main folds

![Full and lean point-model inputs in Fold A.](../assets/fixed-pipeline-lean-features/lean-fold-a.svg)

**The lean wind inputs lower fair CRPS in both main folds, while the lean solar inputs raise
it.** These comparisons use the combined wind/radiation-corrected pipeline and the historical
cleaned point-model fitting path. All arms have 24 trained columns, using deterministic duplicate
padding where needed, and column subsampling one. Their full-input control is
`production_cleaned`, which differs from the public production configuration. Each forecast has
51 values; fair CRPS compares these matched sample constructions. The results below cover
three fitting seeds, six solar series or three wind series, with 12 validation months in Fold A
and 15 in retrospective Fold B. Loss is a percentage of frozen effective capacity; differences
are capacity percentage points. Lower is better.

**Wind's loss reduction is statistically significant at the 5% level in both folds under both
interval methods.** In Fold A, full-input fair CRPS is 17.151 (95% month-and-seed bootstrap
interval 15.507 to 18.710), versus 13.821 (12.261 to 15.203) for lean wind. The paired
lean-minus-full difference is -3.330 capacity points (-3.989 to -2.739). In retrospective
Fold B, the corresponding absolute scores are 16.620 (14.859 to 18.291) and 12.876
(11.768 to 13.944), with a paired difference of -3.744 (-4.849 to -2.678). Every validation
month has lower loss: 12 of 12 in Fold A and 15 of 15 in Fold B.

The separate equal-month wind differences are -3.309 in Fold A (95% t-interval -4.026 to
-2.592) and -3.725 in Fold B (-4.923 to -2.527). The primary bootstrap resamples whole months
and one paired fitting seed; the equal-month t-interval averages the fitted seeds within each
month and gives months equal weight. Those methods represent different month weights and
treatment of fitting-seed variation. Their agreement here does not make their estimands identical.

**Solar's higher loss is statistically significant at the 5% level under the primary bootstrap
in both folds, while Fold A's equal-month t-interval remains unresolved.** In Fold A, full-input
fair CRPS is 7.889 (95% bootstrap interval 5.802 to 10.144), versus 8.095 (5.953 to 10.420)
for lean solar. The paired increase is 0.206 capacity points (0.044 to 0.393), while the
equal-month difference is 0.107 (t-interval -0.170 to 0.384). In retrospective Fold B, the
absolute scores are 8.612 (6.976 to 10.134) and 8.816 (7.103 to 10.392). Its paired increase
is 0.204 (0.070 to 0.337); the equal-month increase is 0.200 (0.044 to 0.355), also
statistically significant at the 5% level. Lean solar has lower loss in only four of 12 Fold A
months and four of 15 Fold B months.

The [feature-results appendix](fixed-pipeline-feature-results-tables.md) retains the unaltered
publication tables for the feature comparisons.

### The first solar addition stops the registered ladder

![The first solar addition in Fold A.](../assets/fixed-pipeline-lean-features/solar-first-fold-a.svg)

**Adding 10 m wind speed does not reach G4's continuation threshold.** Against lean solar,
Fold A's three-seed mean fair-CRPS reduction is 0.025 capacity points, below the registered
0.15-point threshold. Its absolute fair CRPS is 8.070 (95% bootstrap interval 5.940 to 10.402),
compared with lean solar's 8.095 (5.953 to 10.420). The paired difference is -0.025
(-0.063 to 0.022), so the primary bootstrap allows improvement and deterioration. The separate
equal-month difference is -0.021 (t-interval -0.037 to -0.005), statistically significant at the
5% level, with lower loss in nine of 12 months. The magnitude rule stops the ladder despite
that disagreement.
Later solar rungs are not run, and the registered rule does not adopt the lean solar set.

**Retrospective Fold B does not resolve a benefit from the first addition.** Its paired
difference is 0.005 capacity points (95% bootstrap interval -0.038 to 0.051), with the opposite
sign from Fold A. Its equal-month difference is 0.006 (t-interval -0.032 to 0.043); both
methods allow improvement and deterioration. The added-input arm's absolute fair CRPS is
8.821 (7.120 to 10.385), compared with lean solar's 8.816 (7.103 to 10.392), with lower loss
in seven of 15 months. The first rung still receives its planned alternate-setting check;
that check does not bypass the stopped ladder.

### Restoring individual calendar inputs worsens wind loss in Fold A

![Separate wind input additions in Fold A.](../assets/fixed-pipeline-lean-features/wind-addbacks-fold-a.svg)

**Each tested calendar addition raises wind loss against the lean wind baseline in Fold A's
main setting, statistically significant at the unadjusted 5% level under both interval methods.** The
annual-cycle sine and cosine and the three weekday columns are separate additions, each with
the other lean inputs retained. Precipitation type also raises loss in this fold and setting,
statistically significant at the unadjusted 5% level under both methods.
Dewpoint and downward short-wave radiation remain unresolved under both interval methods;
their intervals allow improvement and deterioration. These comparisons cover the same three
wind series, three seeds and 12 validation months as the lean-wind contrast.

**The isolated additions do not decompose the full-input model's loss.** Each test refits a
different feature set, and the effects need not add together. They do not identify a causal
mechanism for an individual feature's harm or show that a feature is useless under another
recipe. Precipitation also has an archive-availability boundary: the retrospective earlier era
cannot identify its information content from a wholly missing input. Its second-fold and era
comparisons require their own interpretation. The dewpoint and short-wave intervals do not
establish zero effects.

### Calendar additions also worsen wind loss in retrospective Fold B's main setting

**Each tested calendar addition raises wind loss in retrospective Fold B's main setting,
statistically significant at the unadjusted 5% level under both interval methods.** These are
separate additions to lean wind, using the historical cleaned point-model recipe, equal-width
padding to 24 trained columns and the combined wind/radiation-corrected pipeline. They cover
three wind series, three fitting seeds and 15 validation months. Lean wind's absolute fair CRPS
is 12.876 percent of frozen effective capacity (95% month-and-seed bootstrap interval 11.768
to 13.944). The differences below subtract that lean baseline and are capacity percentage points;
positive differences mean higher loss.

The annual-cycle sine increases fair CRPS by 2.741 points (95% paired bootstrap interval 1.960
to 3.491), and the cosine by 2.377 (1.566 to 3.211). Each addition has lower loss in only one
of 15 months. The weekday ordinal increases loss by 0.288 points (0.164 to 0.393), weekday
sine by 0.251 (0.160 to 0.331), and weekday cosine by 0.242 (0.131 to 0.347). Their separate
equal-month t-intervals also support higher loss. This agreement concerns these individual
main-setting refits; it does not establish additive effects or identify why a feature harms
this recipe.

**Precipitation type, dewpoint and short-wave radiation remain unresolved in this fold and
setting.** The precipitation-type difference is 0.225 capacity points (95% paired bootstrap
interval -0.072 to 0.572), dewpoint's is 0.007 (-0.067 to 0.086), and downward short-wave
radiation's is 0.055 (-0.013 to 0.127). Each equal-month t-interval also allows improvement
and deterioration. None of these intervals establishes a zero effect. The precipitation result
does not identify its information content: the archive's earlier wholly missing era and later
availability require separate descriptive comparisons, which cannot establish an era-balanced
causal benefit from this aggregate contrast.

**These main-setting add-backs do not yet decide a final input recommendation.** The alternate
training setting and guarded-windchill, negative-control and unpadded checks remain pending.
Their results must be reviewed before concluding which lean inputs to retain. The tests concern
this historical point-model recipe, rather than the public production configuration or the
pooled quantile winner.

**Splitting Fold B by precipitation availability does not resolve either era's loss effect.**
After the archive boundary, adding precipitation type changes wind fair CRPS by 0.032 capacity
points (95% paired bootstrap interval -0.008 to 0.085). Before it, the difference is 0.409
(-0.161 to 0.953). Both separate equal-month t-intervals also include zero. Each era has eight
supported valid-time months, sharing part of November rather than forming independent samples
of eight whole months. The [absolute and paired era tables](fixed-pipeline-feature-results-tables.md)
retain both arms' scores and support counts.

The split uses weather initialisation before or from 13 November 2024 at 00:00 UTC, the
[documented archive-availability boundary](../../architecture/ecmwf-ens-known-issues.md).
It differs from the [operational weather-model upgrade](../../roadmap/data-sources.md)
at 12 November 2024 at 06:00 UTC. An absent precipitation column in the earlier era cannot
identify that input's information content. Different seasons and training exposure also prevent
these summaries from identifying a causal upgrade effect. Fold A's scoring window is wholly
after the archive boundary.

<!-- PART1B_ALTERNATE_SETTINGS_SLOT: add the reviewed second-setting lean, first-solar-rung
and wind-addition findings, including disagreements with these main-setting results. -->

<!-- PART1B_REQUIRED_CONTROLS_SLOT: add the guarded-windchill, negative-control and unpadded
comparisons after completion and evidence review. -->

**A final lean-wind recommendation remains conditional on the required checks.** The two
main-fold reductions support lean wind within this padded historical point-model recipe, but
the alternate settings and guarded, negative and unpadded controls are pending. Their results
must be reviewed before deciding what to retain. These weekly-initialisation scores do not
establish an official leaderboard result, adequate calibration or joint scenario coherence.

### Descriptive feature importance

**Training gain describes how the point-control models divide split credit; held-out feature
tests decide whether an input helps.** The figure uses squared-error models fitted separately
for each meter on corrected wind and radiation inputs, with all 24 production features. Short-wave
radiation receives the largest mean gain share for solar; time-of-year cosine and sine receive
the largest shares for wind. Correlated features can divide credit between themselves, so those
rankings do not establish feature benefit or justify removing a low-gain input. Quantile models
are excluded from this diagnostic.

![Training gain shares for solar and wind point-control models.](../assets/fixed-pipeline-lean-features/importance.svg)

**The displayed ranges describe variability across six refits per meter, not confidence
intervals.** Each meter's model is fitted with three random seeds in each of the two training
periods. Feature gains are scaled to sum to one within each model, with zero for unused columns.
Meters within each type receive equal weight for each refit, and the six refits receive equal
weight in the final mean. The blue dots show mean gain shares; the blue lines show the minimum
and maximum across refits. These training summaries remain descriptive alongside the separate
held-out loss comparisons.

The logarithmic axes show small shares alongside large ones. Means and range endpoints below
0.01%, including zero, appear at that display floor; a floor mark does not establish positive gain.

**The input audit confirms changed radiation and wind-derived features in both folds; forecast
benefit requires the matched loss comparisons above.** Each treatment exposes the same 32
engineered fields on matched rows. That audit inventory includes fields available for feature
experiments and is distinct from the 24 trained columns in the full-input control. Radiation
changes short-wave and long-wave values and the temperature-adjusted long-wave residual. Wind
processing changes speed, bearing and derived windchill. Counts cover all weather members;
bearing differences use shortest angular distances, and other changes compare paired finite
Float32 values exactly.

**Null counts and domain screens need different interpretations.** Power-lag nulls include the
expected masking of observations unavailable at forecast issuance; missing lags alone do not
establish a pipeline error. The audit reports null and nonfinite counts separately and excludes
both from changed-value shares. Windchill is still evaluated outside its formula's temperature
and speed domain in the PR-only treatments; a separate domain guard remains to be evaluated.
Radiation threshold and midpoint-night flags are screens rather than definitive invalid values:
a half-hour whose midpoint is at night can still be partly sunlit.

**The deliberately delayed radiation passes the timing-detector check in both folds.** The
solar correlation peak moves from offset zero for the original input to offset one hour for
the input delayed by one hour; both reports print `Shift detector passed: True`. Offsets compare
power at time `t` with control-member weather at `t + offset`, after removing each series'
within-day, within-run mean and giving series equal weight. Recovery supports the detector's
response to the injected delay. Autocorrelation, curtailment and cloud regimes still confound
the observed peak, so the check establishes neither causal timing accuracy nor the ability of a
forecast-loss comparison to detect a timing fault.

**Precipitation availability remains an era limitation rather than evidence of an upgrade
benefit.** Fold B's earlier era has missing precipitation-type inputs, while the later era has
available inputs. The audit assigns eras by weather initialisation time, using the November
archive-availability boundary and a separate May cycle marker. Source-provenance verification
remains required before interpreting those markers. Different seasons and training exposure
prevent the era summaries from identifying precipitation's information content or a causal
upgrade effect.

The [input-audit tables](fixed-pipeline-input-audit-tables.md) print the solar and wind domain
counts, precipitation missingness and timing correlations for both folds. Counts include all
weather members and overlapping forecast windows; they count forecast input rows rather than
distinct observed hours. The separate guarded-windchill fits test whether the lean-feature
conclusions depend on inputs outside that formula's domain.

The separate [calibration diagnostics study](calibration-diagnostics-1156.md) starts Part 2 from
saved corrected-pipeline predictions. It does not fit a calibrator or establish joint scenario
coherence. This study changes no production code or forecast contract.

<!-- PROVENANCE_AND_DECISIONS -->

## Discussion: what to use

<!-- DISCUSSION_RESULTS -->

**The strongest processing result is the solar benefit from correcting radiation intervals.**
Its direction is supported in both folds and both training settings. The combined corrections
also lower overall and solar common pinball loss within both production and the quick-wins
winner. The wind evidence needs a narrower interpretation: wind-only processing has unresolved
Fold A effects and worsens wind-farm loss in retrospective Fold B. Adding the wind correction
after radiation remains unresolved under the primary bootstrap, so these results do not establish
a general wind forecast benefit. The combined-correction wind deterioration in Fold B sensitivity
also prevents an unqualified claim that both fixes improve every series type. Processing
correctness and measured forecast benefit are separate questions; these historical loss tests
address the latter within the pinned archives and recipes.

**The corrected winner remains a stronger distributional-loss benchmark than refitted
production, while its coverage requires separate assessment.** The deciding score evaluates
both delivered distributions at the same 19 midpoint probabilities. It supports the winner in
both folds and both settings, but the comparison changes objective, pooling, inputs and
preprocessing together. Its advantage cannot be attributed to the PR fixes, quantile training
alone or early stopping, which the exact winner does not use. The original quick-wins selection
used Fold A, and Fold B is retrospective; neither comparison establishes future deployment
performance. Part 2 reports interval coverage alongside widths and assesses undercoverage
separately by interval and series type. A lower common loss does not establish adequate coverage
or coherent joint scenarios.

**The preregistered gate rules out the conditional best-point experiment in this study.** G3
uses the main-setting production fair-CRPS contrast between matched 51-member ensembles. Its
three-seed differences are -0.1176 capacity points in Fold A and -0.0572 in Fold B, each below
the required absolute change of 0.15. The separate equal-month t-intervals favour improvement,
but do not satisfy the magnitude requirement. The conditional best-point refit was therefore
not run. Fair CRPS remains a diagnostic for matched sample constructions; it does not decide
the point-versus-quantile recipe comparison.

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

**OpenAI Codex wrote the new study work; the reused code has historical Claude credits.**
Following the imported study helpers and chart theme through their commit histories finds
Claude Fable 5, Opus 5 and 5.5 (including 1M-context credits), and Sonnet 4.6, 5 and 5.5;
some commits credit Claude without a model name. These are authorship records for reused code,
not the identities of this study's reviewers. Available Codex agents performed the independent
reviews in place of the study skill's unavailable named models.

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

**Reproduction requires the registered private snapshots.** Old and radiation-only frames use
polar archive version 1830; wind-only and combined frames use component archive version 924.
All four treatments use observed-power version 70, capacity version 1 and eligibility snapshot 1.
Observed power is bounded before 1 July 2026, with each fold's additional training and scoring
bounds recorded separately. The frozen effective-capacity denominator is a full-history 99th
percentile and can include readings after that cutoff. It enters scoring only; normalised scores
therefore remain retrospective and conditional on this common snapshot.

**The public configuration inventory is not yet final.** The intended indexes are
`studies/quick_wins/fixed_pipeline_specs/index.json` and
`studies/quick_wins/fixed_pipeline_runtime_records/index.json`.
TODO: reconcile completed artifacts, publish the reviewed entries and hashes, and verify these
destinations before treating them as available downloads. Exact executed JSON specifications
must be distinguished from records reconstructed from completed provenance: a derived CLI record
is neither an original executed JSON specification nor a transcript of historical shell arguments.
Dependency snapshots document the recorded environment; they do not constitute a complete
environment lock or guarantee numerical equality on different hardware. Prepared queues and the
full recipe registry are not evidence that every listed job completed. Private manifests and
generator mappings are excluded from the public inventory.

## Reproducing the figures

**Reconstruct the four feature treatments before rebuilding frames.** Run from a repository
clone containing the three pinned commits, using the recorded dependencies and authorised access
to the frozen inputs. Follow the commands in
`studies/quick_wins/fixed_pipeline_runtime/README.md`
to create the wind, radiation and combined source trees and compare their source hashes with
`all-source-hashes.json`. In the same shell, add the unchanged treatment:

```bash
git worktree add --detach "$STUDY_RUNTIME_ROOT/old" \
  4e820d24c0129c8c95c567ff7355abff8d416144
```

The component migration must have passed its full readiness checks before either component
treatment is built. Source reconstruction supplies no weather or power data. Use new output
directories throughout; these commands describe reproduction and have not themselves been run
as a new experiment. Set the following paths to your authorised input copies and an absent
output root. The data roots must expose the registered power, capacity and eligibility snapshots;
the frame builder checks power version 70 and records the eligibility snapshot it reads, which
must match snapshot 1. It has no eligibility-version command-line override.

```bash
set -euo pipefail
STUDY_REPRO_ROOT="/path/to/new/fixed-pipeline-reproduction"
STUDY_POLAR_ARCHIVE="/path/to/frozen/polar-archive"
STUDY_COMPONENT_ARCHIVE="/path/to/frozen/component-archive"
export DATA_PATH_INTERNAL="/path/to/authorised/internal-data"
export DATA_PATH_DELIVERY="/path/to/authorised/delivery-data"
export REPO_DATA_DIR="$DATA_PATH_INTERNAL"
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MLFLOW_DISABLE_AGENT_HINT=1

for STUDY_PIPELINE in old radiation wind combined; do
  export PYTHONPATH="$STUDY_RUNTIME_ROOT/$STUDY_PIPELINE/packages/contracts/src:$STUDY_RUNTIME_ROOT/$STUDY_PIPELINE/packages/ml_core/src"
  case "$STUDY_PIPELINE" in
    old)
      STUDY_ARCHIVE="$STUDY_POLAR_ARCHIVE"; STUDY_ARCHIVE_VERSION=1830
      STUDY_REVISION="4e820d24c0129c8c95c567ff7355abff8d416144" ;;
    radiation)
      STUDY_ARCHIVE="$STUDY_POLAR_ARCHIVE"; STUDY_ARCHIVE_VERSION=1830
      STUDY_REVISION="0f094c35" ;;
    wind)
      STUDY_ARCHIVE="$STUDY_COMPONENT_ARCHIVE"; STUDY_ARCHIVE_VERSION=924
      STUDY_REVISION="PR1167:38025f4eab457b9605b8c124006609869dbfca0d+base:4e820d24+source1830+destination924" ;;
    combined)
      STUDY_ARCHIVE="$STUDY_COMPONENT_ARCHIVE"; STUDY_ARCHIVE_VERSION=924
      STUDY_REVISION="PR1167:38025f4eab457b9605b8c124006609869dbfca0d+PR1166:0f094c356da82e75c4014664ad7fb33fe1df8b7f+base:4e820d24+source1830+destination924" ;;
  esac
  for STUDY_FOLD in fold_a fold_b; do
    case "$STUDY_FOLD" in
      fold_a) STUDY_WINDOW=mid_2025_to_mid_2026 ;;
      fold_b) STUDY_WINDOW=backwards_2025_07_to_2026_06 ;;
    esac
    .venv/bin/python studies/quick_wins/fixed_pipeline_frames.py \
      --output "$STUDY_REPRO_ROOT/frames/$STUDY_PIPELINE/$STUDY_FOLD" \
      --fold "$STUDY_WINDOW" --pipeline-tag "$STUDY_PIPELINE" \
      --pipeline-revision "$STUDY_REVISION" --archive "$STUDY_ARCHIVE" \
      --archive-version "$STUDY_ARCHIVE_VERSION" --power-version 70 --capacity-version 1
  done
done
```

**Refit the report-listed recipes on their matching frames.** The `production` recipe calls the public
production training and prediction path; `r2_e1_c1_a3w_d3` preserves the exact quick-wins winner's
historical fitting path. Both use no padding. The Part 1b `production_cleaned`, `lean_pv` and
`lean_wind` controls, the registered wind add-backs, guarded-windchill arms, negative controls and
first solar rung use their separately recorded padding and setting. The unpadded controls are
distinct main-setting checks. Select the completed fits from the exact executed report
specifications in the forthcoming public spec index. Each report lists its runs, including fit
path, frames, label and seed, together with fold, setting and padding. The canonical fit path
`fits/<treatment>/<fold>/<recipe>/<setting>/<device>_seed<seed>_<padding>` identifies the treatment,
recipe and device; check these against the fit manifest. The derived runtime index does not
contain per-fit records. Do not infer a Cartesian product from the recipe registry. For each
selected run, set these variables from its report specification and checked fit path, then run
the following command, one CUDA fit at a time:

```bash
: "${STUDY_PIPELINE:?Select the treatment from the checked fit path}"
: "${STUDY_FOLD:?Select the report fold}"
: "${STUDY_RECIPE:?Select the recipe from the checked fit path}"
: "${STUDY_SETTING:?Select the report setting}"
: "${STUDY_SEED:?Select the report run seed}"
: "${STUDY_DEVICE:?Select the device from the checked fit path}"
: "${STUDY_PADDING:?Select the report padding}"
export PYTHONPATH="$STUDY_RUNTIME_ROOT/$STUDY_PIPELINE/packages/contracts/src:$STUDY_RUNTIME_ROOT/$STUDY_PIPELINE/packages/ml_core/src"
.venv/bin/python studies/quick_wins/fixed_pipeline_fit.py \
  --frames "$STUDY_REPRO_ROOT/frames/$STUDY_PIPELINE/$STUDY_FOLD" \
  --output "$STUDY_REPRO_ROOT/fits/$STUDY_PIPELINE/$STUDY_FOLD" \
  --recipe "$STUDY_RECIPE" --seed "$STUDY_SEED" --device "$STUDY_DEVICE" \
  --setting "$STUDY_SETTING" --padding "$STUDY_PADDING"
```

The combined production and exact-winner refits use seeds 0, 1 and 2 at both main and sensitivity
settings in both folds. Do not add the conditional G3 best-point refit: its preregistered gate
failed in both folds, so it was not run. The solar ladder starts with `solar_add_wind10`; report
that rung and apply the registered stopping rule before considering another. Listing later
rungs in the registry does not establish their execution or authorise bypassing that rule.

**Regenerate reports and publication artifacts from reviewed specifications.** Select the exact
completed report, table, chart and importance specifications from the forthcoming public spec
index. Retain the original executed bytes and their hashes. If input or output paths need to
change, make a separately labelled reproduction copy bound to the rebuilt artifacts; it is a
derived configuration, not the original executed specification. TODO: finalise the public
inventory and its path-rebinding instructions. Until then, the following variables are explicit
placeholders for selected, reviewed copies, rather than asserted public filenames:

```bash
: "${STUDY_REPORT_SPEC:?Select and rebind the reviewed report specification}"
: "${STUDY_TABLE_SPEC:?Select and rebind the reviewed publication-table specification}"
: "${STUDY_CHART_SPEC:?Select and rebind the reviewed figure specification}"
: "${STUDY_ARTIFACT:?Choose the indexed artifact name for fresh outputs}"
.venv/bin/python studies/quick_wins/fixed_pipeline_report.py \
  --spec "$STUDY_REPORT_SPEC" --output "$STUDY_REPRO_ROOT/reports/$STUDY_ARTIFACT"
.venv/bin/python studies/quick_wins/fixed_pipeline_tables.py \
  --spec "$STUDY_TABLE_SPEC" --output "$STUDY_REPRO_ROOT/publication/$STUDY_ARTIFACT"
.venv/bin/python studies/quick_wins/fixed_pipeline_charts.py \
  --spec "$STUDY_CHART_SPEC" --output "$STUDY_REPRO_ROOT/figure-text/$STUDY_ARTIFACT" \
  --text-only
```

Review the exported figure text before rendering the corresponding chart into another new
directory:

```bash
.venv/bin/python studies/quick_wins/fixed_pipeline_charts.py \
  --spec "$STUDY_CHART_SPEC" --output "$STUDY_REPRO_ROOT/figures/$STUDY_ARTIFACT" \
  --reviewed-text "$STUDY_REPRO_ROOT/figure-text/$STUDY_ARTIFACT/figure-text.md"
```

The input-audit report has a direct CLI rather than an executed JSON spec. Repeat for each fold,
with `STUDY_FOLD` set to `fold_a` or `fold_b`:

```bash
.venv/bin/python studies/quick_wins/fixed_pipeline_input_audit.py \
  --old "$STUDY_REPRO_ROOT/frames/old/$STUDY_FOLD" \
  --radiation "$STUDY_REPRO_ROOT/frames/radiation/$STUDY_FOLD" \
  --wind "$STUDY_REPRO_ROOT/frames/wind/$STUDY_FOLD" \
  --combined "$STUDY_REPRO_ROOT/frames/combined/$STUDY_FOLD" \
  --output "$STUDY_REPRO_ROOT/input_audit/$STUDY_FOLD"
```

**Verify that each feature experiment preserves forecasts for unaffected series types.** After
the required combined-pipeline fits complete, the verifier compares every saved sample key and
forecast value exactly for those types. It checks the two lean arms against their full-input
control, and the first solar addition against lean solar. The unpadded check covers only the
two registered lean comparisons. It checks input inventories and hashes before and after the
comparison and writes an aggregate proof without private row keys or predictions. Use new
proof filenames:

```bash
for STUDY_FOLD in fold_a fold_b; do
  for STUDY_SETTING in main sensitivity; do
    .venv/bin/python studies/quick_wins/fixed_pipeline_unaffected.py \
      --data-root "$STUDY_REPRO_ROOT" --fold "$STUDY_FOLD" \
      --setting "$STUDY_SETTING" --padding equal \
      --output "$STUDY_REPRO_ROOT/unaffected-$STUDY_FOLD-$STUDY_SETTING-equal.json"
  done
  .venv/bin/python studies/quick_wins/fixed_pipeline_unaffected.py \
    --data-root "$STUDY_REPRO_ROOT" --fold "$STUDY_FOLD" \
    --setting main --padding none \
    --output "$STUDY_REPRO_ROOT/unaffected-$STUDY_FOLD-main-none.json"
done
```

This proof concerns unaffected types within each padding policy. It does not test equality
between the padded and unpadded forecasts for the affected type.

The importance figure uses its own completed specification and report. After selecting and
rebinding `STUDY_IMPORTANCE_SPEC` from the public index, regenerate its aggregate report and text:

```bash
: "${STUDY_IMPORTANCE_SPEC:?Select and rebind the reviewed importance specification}"
.venv/bin/python studies/quick_wins/fixed_pipeline_importance.py \
  --spec "$STUDY_IMPORTANCE_SPEC" --output "$STUDY_REPRO_ROOT/importance/report"
.venv/bin/python studies/quick_wins/fixed_pipeline_importance_charts.py \
  --report "$STUDY_REPRO_ROOT/importance/report" \
  --output "$STUDY_REPRO_ROOT/importance/text" --number 6 --text-only
```

After reviewing that text, render with `fixed_pipeline_importance_charts.py` using the same
`--report` and `--number 6`, a new `--output` directory and
`--reviewed-text "$STUDY_REPRO_ROOT/importance/text/figure-text.md"`.
