# Lagged power did not improve day-ahead solar power forecasts at six plants in Lincolnshire

**At six solar plants over 18 months, giving an XGBoost model the plant's own power from an earlier
day did not change its day-ahead error by a statistically significant amount in any of the five
planned comparisons, and the planned intervals rule out a gain as large as the 2% decision threshold
in four of the five.** For a single lag, the 99% interval of the change in mean absolute error runs
from −0.075 to +0.073 percentage points of the plant's capacity, against a baseline error of 8.93
points and a threshold of 0.18 points. For Flexpectation, a lag of one day is therefore not worth
building into the day-ahead service for utility-scale solar on this evidence, as long as plants run
normally, because the cleaning removed outage hours and the study cannot say whether a lag helps
when a plant is partly offline. Inputs that average a plant's recent history or use satellite
irradiance did better in post hoc analyses, but most of their gain on the months they were not
selected on comes from one plant in one month, so they are hypotheses to test on more plants rather
than a recommendation. Eighteen months at six plants in one 25 km box cannot show whether a longer
history, more plants, or a faster data feed would change this.

![Figure 1: No planned contrast found lagged power changing day-ahead error significantly. Five
rows, each a planned contrast with a dot and a 99% interval at two hyperparameter settings; every
interval includes zero.](../assets/lag_features_figure_1.svg)

**Can lagged power improve a day-ahead point forecast?** No change was detectable. A single lag
changed the error by −0.001 points [−0.075, +0.073] at the primary hyperparameter setting and by
+0.012 points [−0.043, +0.058] at a second, more regularised setting (planned contrast P1, 99%
intervals, Figure 1).

**Can lagged power improve the probabilistic forecast?** No. The continuous ranked probability score
changed by +0.009 points [−0.020, +0.036] (P4). The 10% to 90% interval held 68.9% of hours without
the lag and 68.5% with it, against a nominal 80% ([Figure
16](#the-10-to-90-interval-holds-about-69-of-hours-with-or-without-the-lag)).

**Does a lag help one XGBoost model shared by all plants?** No change was detectable: −0.059 points
[−0.151, +0.029] (P5).

**Do the other lag ideas help?** None of the planned comparisons found a gain. The analogue
ensemble, the arm with the lowest error in the first 10 months, was no better than a single lag on
the 8 months it was not chosen on (+0.012 points [−0.243, +0.278], P2, which cannot rule out a gain
of the threshold's size), and a two-step forecast was no better than a lag with its weather (+0.039
points [−0.059, +0.143], P3).

**Can this study detect a planted power loss, and which inputs could?** A single lag cannot: in a
post hoc test that cuts every plant's power by 10% in the same 9 months, one lag recovered 0.09
[−0.04, 0.27] of what an oracle given the true loss recovers, the analogue ensemble 0.23 [0.05,
0.46], weekly statistics 0.39 [0.21, 0.73], and the ratio of power to satellite irradiance 0.74
[0.57, 1.21] (95% intervals, [Figure
8](#pc-recovers-74-of-a-planted-10-power-loss-in-the-month-level-control-one-days-lag-recovers-9)).
In a control that cuts each plant separately for weeks at a time, no input recovered a share
distinguishable from zero. A null for a single lag is therefore not evidence that no input could
track a plant's level.

**Do the level-tracking inputs help in the real data?** Not clearly. The satellite ratio, weekly
statistics, and the ratio of power to forecast irradiance gained 0.14 to 0.23 points over the
baseline on the 8 months not used for screening (95% intervals that exclude zero, post hoc), but
dropping one plant in one month cuts those gains to 0.05 to 0.12 points, with intervals that include
zero ([Figure
10](#on-the-8-months-not-used-for-screening-pc-w7-and-tf-gained-most-over-b0-but-an-did-not)).

**Do lags help at the longest leads?** Weekly statistics lowered the error by 0.45 points [−0.74,
−0.16] at day 14 (post hoc), but a 50/50 blend of the baseline with climatology built only from
earlier years lowered it by 1.16 points [−1.50, −0.80], so no lag arm beat that simpler fix ([Figure
13](#blending-the-baseline-with-climatology-lowers-long-lead-error-more-than-any-lag-arm-shown-does)).

**Can lagged inputs describe each plant to a model shared by many plants?** Partly (post hoc).
Adding each plant's ratio of power to forecast irradiance and its daily shape lowered a shared
model's error by 0.31 points [−0.49, −0.14], and by 0.32 points [−0.51, −0.14] for a plant left out
of training; adding the plant's clipping ceiling to those inputs raised the error for an unseen
plant by 0.15 points [+0.02, +0.29] ([Figure
15](#plant-information-lowers-a-shared-models-error-the-clipping-ceiling-does-not-help-unseen-plants)).

**What can these data support?** They support the absence of a gain as large as the 2% threshold
from a single lag at these six plants in normal operation, and hypotheses about level-tracking
inputs. They do not support a claim about domestic solar, other regions, more than 18 months, plants
with faults, or hundreds of plants.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5, Claude Opus 5.5, and Claude Sonnet 5). Several
> independent Claude reviewers have checked the method, the evidence, and the prose adversarially.

## Key findings

- **A single lag is a weak guide to a plant's level, because one day's weather swamps a level
  change.** In a planted 10% power loss the single lag recovers 0.09 of the oracle's gain, against
  0.39 for weekly statistics and 0.74 for the satellite ratio (post hoc, [Figure
  7](#a-single-days-lag-neither-detects-a-planted-power-loss-nor-beats-a-random-day-lag), [Figure
  8](#pc-recovers-74-of-a-planted-10-power-loss-in-the-month-level-control-one-days-lag-recovers-9)).
- **The planned comparisons rule out a gain as large as the 2% threshold for P1, P3, P4, and P5 at
  both settings.** P2, which rests on 8 months, cannot.
- **The screening winner lost most of its gain outside the months it was chosen on.** The analogue
  ensemble beat the baseline by 0.231 points [−0.392, −0.102] in the 10 screening months, and by
  0.046 points [−0.243, +0.169] on the 8 months it was not chosen on ([Figure
  9](#only-an-clearly-beat-b0-in-the-screening-months-by-023-points-6-arms-clearly-lost-to-it),
  [Figure
  10](#on-the-8-months-not-used-for-screening-pc-w7-and-tf-gained-most-over-b0-but-an-did-not)).
- **The gains of the arms that did best outside the screening months rest on one plant in one
  month.** There the baseline under-forecast the plant's output by 14.6 points of capacity on
  average, and removing that plant-month leaves the satellite ratio at −0.04 points over 18 months
  and −0.08 over the 8 [post hoc, [Figure
  10](#on-the-8-months-not-used-for-screening-pc-w7-and-tf-gained-most-over-b0-but-an-did-not)].
- **At day 14 the ENS mean's error is above climatology's, and a random-day lag gains too.** The
  mean of the ensemble forecast scored 15.35 points against 14.18 for a climatology built from
  earlier years, and a random-day lag lowered the error by 0.07 to 0.12 points depending on the draw
  ([Figure 12](#at-lead-day-14-b0-is-worse-than-climatology-and-a-random-day-lag-gains-too), [Figure
  13](#blending-the-baseline-with-climatology-lowers-long-lead-error-more-than-any-lag-arm-shown-does)).
- **The quantile forecasts are over-confident with or without a lag** ([Figure
  16](#the-10-to-90-interval-holds-about-69-of-hours-with-or-without-the-lag), [Figure
  17](#b0-and-l1-are-too-narrow-too-many-hours-fall-outside-their-10-and-90-quantiles)).

## Introduction

**Flexpectation needs to know whether a plant's own recent power helps its forecast.** A lag could
correct for soiling, panel degradation, a failed string, part of a plant offline, and changes in the
weather forecast product. In a model shared by many plants, a lag could also tell the model about
each plant, such as its ratio of direct-current to alternating-current capacity, without a physical
model of the plant. The question came from [issue #1138 (Lag features for solar power
forecasting)](https://github.com/openclimatefix/nged-substation-forecast/issues/1138).

**The study compares an XGBoost model with no lag (B0) against XGBoost models given one of 17 other
inputs, at six NGED solar plants.** The forecast is for the day after the day it is issued (lead-day
1), driven by the mean of the 51-member ECMWF ensemble forecast (the ENS mean). A lag is the plant's
power at the same clock hour on the latest whole day before the forecast was issued, which is two
days before the target day for a day-ahead forecast.

| Input | Grid | Lead read | Plant data read | Delay assumed |
|---|---|---|---|---|
| ENS mean of ECMWF's 51 members | 0.25° | the 00 UTC run, lead-days 0 to 14 | power up to the issue time (09:00 UTC on the run's day) | none: the live service currently receives NGED power at least 6 hours late, which the study ignores |
| CAMS satellite irradiance (one input, PC) | point service | whole days before the issue day | the previous day is available at 09:00 UTC | 1 day (the service's latency since February 2026) |

## Data and methods

**The rows, the arms, and the models.** The study uses the matched-lead study's rows: 35,263 hourly
solar rows at 6 plants from December 2024 to September 2026. The power record the lags are read from
ends on 2026-07-01 (the final-test date, which the study does not open), so 29,445 rows over 18
calendar months remain at lead-day 1. January 2026 is absent from the rows. Every error is divided
by its own plant's capacity (its 99th percentile of metered output) before any mean or difference,
and is reported in percentage points of capacity. Each XGBoost model uses the absolute-error
objective, column subsampling off, three fitting seeds, and five folds of whole months. The primary
hyperparameter setting is depth 6, learning rate 0.05, 500 rounds, and row subsampling 0.8. The
baseline B0 has seven columns: hour of day, day of year, an era code, the sun's elevation and
azimuth, and the ENS mean's irradiance and temperature at the same lead. The era code and the folds
follow the matched-lead study's cuts at 2025-10 and 2026-02; the 2026-02 cut follows the Met
Office's UKV upgrade, which does not affect the ENS mean. The ENS mean changed once inside the
record, at ECMWF's Cycle 50r1 on 2026-05-12, which is not cut. A training row whose input is a
test-month label does not put that label in the model's output, so the study does not purge rows
near fold edges.

**The arms.** Every arm other than B0 adds columns to those seven, and no arm is padded to equal
width, because a duplicate column cannot win a split and so cannot remove the width bias. The bias
of extra columns is measured with shuffled-lag controls instead: one extra random-day lag raised
B0's error by 0.011 points [−0.025, +0.051], two by 0.038 [−0.005, +0.085], and three by 0.033
[−0.024, +0.092] (lead-day 1, 95% intervals). The 18 sweep arms are L1 (the lag), IM (the morning's
power on the forecast day), L2 (L1 plus that hour's weather forecast), CTX7 (seven separate daily
lags and their weather), W7 (the minimum, maximum, and mean of the same hour over the last 7 whole
days), Q30 (the 30-day 90th percentile and median of the same hour), CK (the clipping ceiling), TF
(the 30-day ratio of power to forecast irradiance), AN (the analogue ensemble: the mean, the
forecast clear-sky index, and the spread of the power, rescaled by clear-sky irradiance, on the 5 of
the last 30 days whose forecast clear-sky index was closest to the target's), PC (the 7-day ratio of
power to CAMS satellite irradiance, and the 30-day ratio of CAMS to forecast irradiance), S2 and S3
(two-step forecasts built on a first-stage model), RP (the plant's recent energy relative to the
other plants), T1 (days since the record began, an interpolation bound), KS (a combination), N1 and
N2 (random-day lag controls), and B0. Post-model corrections need no fit. Every window is anchored
at the issue day, and the build asserts and probes that none reads data after the issue time. PC's
second column contains no power, so part of PC's and TF's gain may correct the weather forecast's
bias rather than track the plant.

**Planned and exploratory results.** A planned contrast was written into the study plan before any
result existed; every other number is exploratory, and results added after the first run are post
hoc. The five planned contrasts are P1 (L1 minus B0, per plant), P2 (the best sweep arm minus L1 on
the 8 months the sweep never screened), P3 (S2 minus L2), P4 (the continuous ranked probability
score of L1 minus that of B0), and P5 (L1 minus B0 in one model shared by all plants). A lag "helps"
if both hyperparameter settings give a 99% interval wholly below zero and a point estimate of at
least 2% of the baseline's own error, which is about 0.18 points. The sweep scores each arm on the
first 10 months, and the arm with the lowest error among 11 eligible arms (the analogue ensemble,
AN) is chosen before any planned contrast is run.

**What the significance test covers.** Each interval resamples whole calendar months, paired across
arms, and one of the three fitting seeds, 2,000 times. The test covers month-to-month weather and
the fitting seed, not differences between plants. An exploratory row with no real effect has a
nominal 5% chance of a 95% interval that excludes zero, the number of such rows is unknown, shared
months make them cluster, and the page does not correct exploratory intervals for multiple
comparisons. Of 83 exploratory intervals in the first run, 14 lay wholly below zero and 20 wholly
above; of 87 in the first follow-ups, 27 and 7. The planned contrasts use 99% intervals (a
Bonferroni correction over five) and need both settings to agree. All other intervals on this page
are 95%.

**Controls.** N2 (a random day from months not being scored) and N1 (a random day 8 to 28 days
earlier, keeping the plant's slow level) show what an extra column and a level-only lag do. A
positive control multiplies power by 1 minus a planted loss in 9 of the 18 scored months and in a
random half of the other months, applied before the lags are built, at 5% and 10% (and 2% in the
first run). An oracle arm given the true shift factor shows what knowing the shift exactly gains a
tree model, and a second control gives each plant persistent steps of 4 to 12 weeks. An arm's
recovery share is its gain over B0 on the shifted data, minus its gain on the unshifted data,
divided by the oracle's gain.

## Results

### Power follows forecast irradiance more closely than it follows the lag

![Figure 2: Measured power against forecast irradiance, and against the lag. Two scatter
panels.](../assets/lag_features_figure_2.svg)

### How each lag column is built, and what the first-stage forecast never sees

![Figure 3: How each lag column is read for one target day.](../assets/lag_features_figure_3.svg)

![Figure 4: The first-stage forecasts never train on the months they are scored
on.](../assets/lag_features_figure_4.svg)

### The XGBoost models follow measured power

![Figure 5: Day-ahead forecasts against measured power in three weeks chosen by
rule.](../assets/lag_features_figure_5.svg)

![Figure 6: Each arm's error at each plant, and its gain over B0
there.](../assets/lag_features_figure_6.svg)

### A single day's lag neither detects a planted power loss nor beats a random-day lag

![Figure 7: A single day's lag neither detects a planted power loss nor beats a random-day
lag.](../assets/lag_features_figure_7.svg)

**A random-day lag changes the error as much as the real lag does.** The change in error from the
shuffled-lag control N2 against B0 is +0.011 points [−0.025, +0.051], and L1 minus N2 is −0.013
points [−0.066, +0.037] (exploratory). The planted-loss control recovers none of 2%, 5%, or 10% for
the single lag: L1 minus B0 is −0.036 points [−0.126, +0.049] at 10%, where the oracle gains 0.392
points [−0.708, −0.052] (post hoc).

### PC recovers 74% of a planted 10% power loss in the month-level control, one day's lag recovers 9%

![Figure 8: PC recovers 74% of a planted 10% power loss in the month-level control; one day's lag
recovers 9% (post hoc).](../assets/lag_features_figure_8.svg)

**In a control that cuts every plant's power in the same 9 months, the satellite ratio recovers 0.74
of the oracle's gain [0.57, 1.21], weekly statistics 0.39 [0.21, 0.73], the analogue ensemble 0.23
[0.05, 0.46], the 30-day tracker 0.16 [−0.04, 0.36], the transfer ratio 0.17 [−0.13, 0.48], and one
lag 0.09 [−0.04, 0.27] (post hoc, 10% loss).** A 5% loss is too small for even the oracle to be
statistically detectable (−0.110 points [−0.214, +0.013]), so no share is computed for it, and the
2% rule cannot be the pass criterion for a control. In the control that cuts each plant separately
for 4 to 12 weeks, the oracle gains only 0.057 and 0.096 points at 5% and 10%, and every arm's share
interval includes zero, including the satellite ratio (−0.21 [−0.67, 0.41] at 10%). The month-level
control shifts every plant in the same months, which suits an input that tracks the fleet's forecast
bias as much as a plant's own level.

### Only AN clearly beat B0 in the screening months, by 0.23 points; 6 arms clearly lost to it

![Figure 9: Only AN clearly beat B0 in the screening months, by 0.23 points; 6 arms clearly lost to
it.](../assets/lag_features_figure_9.svg)

**The analogue ensemble had the lowest error in the first 10 months (8.97 points against 9.20 for
B0), and beat B0 by 0.231 points [−0.392, −0.102] in the paired comparison.** It was the best of 11
eligible arms, so some of that gain is a winner's curse. Six arms with many columns or two-step
inputs were worse than B0: L2 by 0.105, RP by 0.130, S2 by 0.132, CK by 0.156, CTX7 by 0.246, and S3
by 0.286 points (exploratory 95% intervals; about 1 of the 17 would reach statistical significance
by chance).

### On the 8 months not used for screening, PC, W7, and TF gained most over B0, but AN did not

![Figure 10: On the 8 months not used for screening, PC, W7, and TF gained most over B0, but AN did
not (post hoc).](../assets/lag_features_figure_10.svg)

**Outside the screening months the analogue ensemble's gain over the baseline fell from 0.231 points
to 0.046 points [−0.243, +0.169].** The satellite ratio (−0.234 points [−0.507, −0.031]), weekly
statistics (−0.165 [−0.353, −0.005]), and transfer ratio (−0.141 [−0.281, −0.015]) gained most on
those 8 months, but the intervals include zero at 99%, and selecting them now is a second selection,
labelled post hoc. **Most of the satellite ratio's and weekly statistics' gain on those months comes
from one plant in one month, where the baseline under-forecast the plant's output by 14.6 points of
capacity on average.** Without that plant-month the 8-month gains are −0.075 [−0.196, +0.025] for
the satellite ratio, −0.046 [−0.140, +0.052] for weekly statistics, and −0.120 [−0.267, +0.014] for
the transfer ratio, and over all 18 months −0.041 [−0.108, +0.027], −0.038 [−0.114, +0.044], and
−0.048 [−0.130, +0.040]. The 30-day tracker's gain does not depend on that plant-month (−0.066
[−0.113, −0.030] on the 8 months). The same plant is over-forecast by 15.6 points of capacity on
average in the same calendar month of the other year, and the baseline carries day of year, so the
lag arms may be repairing a cross-validation artefact that a live service would not have. Refitting
without day of year left the satellite ratio's and weekly statistics' gains unchanged within their
intervals, and only the transfer ratio's gain disappeared (a change of −0.110 points [−0.214,
−0.017], post hoc). Alternatively that plant-month is the issue's own use case, a plant partly
offline that the cleaning did not remove. Rescaling each plant-month's baseline forecast in
hindsight gains 0.35 to 0.44 points on these 8 months, the gain from a perfect monthly scale and not
a ceiling for finer corrections. A version of the satellite ratio that waits two days for CAMS gains
0.240 points [−0.554, −0.012] on the same 8 months, as 0.234 did with a one-day wait.

### Month by month, the differences from the baseline are small and change sign

![Figure 11: Month by month, the paired difference from B0 at the day-ahead
lead.](../assets/lag_features_figure_11.svg)

### At lead-day 14, B0 is worse than climatology, and a random-day lag gains too

![Figure 12: At lead-day 14, B0 is worse than climatology and a random-day lag gains
too.](../assets/lag_features_figure_12.svg)

**At lead-day 14 the ENS-mean baseline's error (15.35 points) is above that of a climatology built
from earlier years (14.18), a difference of −1.17 points [−1.82, −0.50] on the rows where that
climatology exists.** Weekly statistics gain 0.45 points [−0.74, −0.16] at day 14 on all rows and
0.20 points [−0.37, −0.004] at day 10, and the 30-day tracker 0.22 points [−0.34, −0.10] at day 10
and 0.22 points [−0.35, −0.07] at day 14 (exploratory). The weekly-statistics gain at day 10 is not
distinguishable from that of a random-day control of the same width (W7 minus N2-3 is −0.058 points
[−0.173, +0.048]); at day 14 it is (−0.275 [−0.508, −0.048]). At lead-days 0 and 1, no lag arm beat
B0 by more than 0.1 points except the analogue ensemble (−0.164 points [−0.281, −0.044] at day 1);
four arms with many columns or two-step inputs (CK, CTX7, RP, S3) were worse by 0.12 to 0.19 points.

### Blending the baseline with climatology lowers long-lead error more than any lag arm shown does

![Figure 13: Blending B0 with climatology lowers long-lead error more than any lag arm shown does
(post hoc).](../assets/lag_features_figure_13.svg)

**A 50/50 blend of the baseline with a climatology built from earlier years lowers the error by 0.54
points [−0.90, −0.16] at day 7, 0.90 [−1.20, −0.59] at day 10, and 1.16 [−1.50, −0.80] at day 14,
and a blend of the weekly-statistics arm with it by 1.31 points [−1.74, −0.85] at day 14.** The same
blends at lead-day 1 raise the error by about 1.6 points. Adding the climatology as one input column
gains little (−0.060 points [−0.234, +0.109] at day 14). The climatology in the figure's arms uses
months after the target day, which a live service would not have, so the figure's blends are 0.2
points weaker than those built from earlier years only; the past-only climatology reads up to 6
earlier years from 2019 whereas the saved one reads the 18-month record, so the gap mixes look-ahead
with history depth. Random-day lag controls gain too, and the shuffled-lag control N2 does not carry
no information at long leads, because it samples the plant's own power from other months, which acts
as a weak climatology: the refit gained 0.069 points [−0.147, +0.018] at day 14, the first run's
draw 0.120 points [−0.213, −0.032], and a control as wide as W7 0.174 points [−0.334, −0.006]. The
first run's draw cannot be regenerated, because its build did not fix the draw order.

### A shared XGBoost model scored worse than one XGBoost model per plant

![Figure 14: A shared XGBoost model scored worse than one XGBoost model per plant
(descriptive).](../assets/lag_features_figure_14.svg)

**On the same folds, one shared model given no plant information scored 0.255 points [+0.078,
+0.450] worse than one model per plant, and a plant code closed the gap (+0.002 points [−0.113,
+0.137]).**

### Plant information lowers a shared model's error; the clipping ceiling does not help unseen plants

![Figure 15: Plant information lowers a shared model's error; the clipping ceiling does not help
unseen plants (post hoc).](../assets/lag_features_figure_15.svg)

**The transfer ratio and daily shape together lower a shared model's error by about as much as a
plant code does.** Compared with the shared baseline G-B0, a plant code lowers the error by 0.252
points [−0.368, −0.137], a plant code with the transfer ratio by 0.435 points [−0.567, −0.308], the
transfer ratio and daily shape together by 0.311 points [−0.485, −0.144], and the full fingerprint
(those plus the clipping ceiling) by 0.402 points [−0.596, −0.209]. For a plant left out of training
the full fingerprint gains 0.172 points [−0.381, +0.055] and the version without the ceiling 0.321
points [−0.510, −0.136]. Adding the ceiling changes the error by −0.091 points [−0.188, +0.017] for
seen plants and +0.149 points [+0.018, +0.290] for unseen plants. The ceiling is nearly constant
within a plant and differs between plants, so in training it acts as a plant code, which cannot
transfer to an unseen plant. Six plants say nothing about whether a fingerprint scales to hundreds
of systems, because the trees have only five training plants to interpolate between.

### The 10% to 90% interval holds about 69% of hours, with or without the lag

![Figure 16: The 10% to 90% interval holds about 69% of hours rather than 80%, with and without the
lag.](../assets/lag_features_figure_16.svg)

### B0 and L1 are too narrow: too many hours fall outside their 10% and 90% quantiles

![Figure 17: B0 and L1 are too narrow: too many hours fall outside their 10% and 90% quantiles (post
hoc).](../assets/lag_features_figure_17.svg)

**The share of hours below the 10% quantile is 14.0%, and above the 90% quantile 17.0%, for B0, and
the lag changes neither.** Adding the quantiles of recent residuals to the baseline's point forecast
(R5) made the continuous ranked probability score 0.562 points [+0.415, +0.693] worse.

### Where each arm's trees split (descriptive)

![Figure 18: Where each arm's trees split, by feature group
(descriptive).](../assets/lag_features_figure_18.svg)

**Importance is descriptive, not evidence.** The lagged power of L1 carries 1.3% of the total split
gain [0.8%, 2.5% across refits].

## Discussion: what to use

**For the day-ahead utility-scale solar forecast, do not add a one-day lagged power input on this
evidence.** The 99% interval of its effect on mean absolute error is within ±0.075 points of the
baseline's 8.93, which rules out a gain as large as the 0.18-point threshold at both settings, in
data from which outage hours were removed.

**If the project wants a level-tracking input, test the satellite ratio, weekly power statistics,
and the transfer ratio in a planned study on more plants and a longer record.** Each gained 0.14 to
0.23 points on the 8 months outside the screening months, but those results were chosen after the
first run and rest mostly on one plant in one month, and only the satellite ratio clearly recovered
a planted fleet-wide loss. The satellite ratio needs CAMS ingested into the live system, with a
delay of one day.

**At days 10 to 14, test a blend with a climatology built from earlier years before testing lag
inputs.** The blend beat every lag arm at those leads, and it needs no extra data feed. It raises
the error at day 1, so it belongs only at the longest leads.

**For a model shared by many plants, treat a plant fingerprint as level tracking and test it on more
plants.** The gain for an unseen plant came from the transfer ratio and daily shape, not from the
clipping ceiling.

## Limitations

- **Sample:** 18 scored months at 6 plants in one 25 km box. P2 rests on 8 months (2025-10 to
  2026-06, without 2026-01), spanning the matched-lead study's second and third fold eras and with
  no July to September.
- **Intervals:** a percentile bootstrap over 18 month clusters tends to under-cover; plants are
  pooled and weighted by row count; the test does not cover differences between plants. A 99% bound
  within 0.01 points of zero changes with the random stream, so such results are called borderline
  at 99%, neither significant nor not significant.
- **Selection:** the analogue ensemble was chosen on the first 10 months, and the arms highlighted
  in the follow-ups were chosen after seeing the first run, so those results are hypotheses.
- **Cleaning removes the case the issue names:** outage, commissioning-ramp, and export-cap hours
  are dropped from targets and lags, so the study cannot show whether a lag helps when a plant is
  partly offline.
- **Controls:** the month-level control shifts every plant in the same months, which suits inputs
  that track fleet-wide forecast bias, and the plant-specific control has a small oracle gain, so
  its shares are imprecise.
- **Climatology and the climatology column in Figure 13** read months after the target day, which a
  live service would not have; the past-only climatology figures above do not.
- **The first run's N2** came from a build whose draw order was not fixed and cannot be regenerated;
  the follow-ups refit N2 on a fixed seeded draw and both are shown.
- **Latency:** the study assumes observed power up to the issue time, and CAMS with a one-day delay.
  The live system currently receives power at least 6 hours late, which the study ignores. Before
  February 2026 CAMS's latency was longer, so the backtest reads CAMS slightly fresher than the
  service then offered.
- **Capacity table:** errors are divided by the effective-capacity table built for the matched-lead
  study.
- **IFS HRES replicate:** with IFS HRES as the weather input, L1 minus B0 was +0.014 points [−0.030,
  +0.053] at lead-day 1 (B0's error 9.86 against 8.95 with the ENS mean on the same rows) and −0.027
  [−0.101, +0.047] at lead-day 2, so the single-lag null does not depend on the weather product
  (exploratory).

## Scope

The study does not cover domestic solar, wind, other regions, plants with known faults, lead times
shorter than the 00 UTC run, a power feed that is minutes old, longer histories than 18 months, or
hundreds of plants.

## Data and code availability

The ENS-mean forecasts, CAMS irradiance, and the matched-lead study's rows are inputs the project
holds; NGED's power data are private, and no plant's name, identifier, or coordinates appear on this
page. The scripts are in
[`studies/lag_features/`](https://github.com/openclimatefix/nged-substation-forecast/tree/main/studies/lag_features),
at the commit that merged this page. XGBoost 3.4.1 fitted every model on a GPU with 4 workers, at
the primary setting (maximum depth 6, learning rate 0.05, 500 rounds, row subsampling 0.8) and the
sensitivity setting (maximum depth 4, learning rate 0.03, 1,200 rounds). The reproduction check
refitted the ENS-mean baseline at lead-day 1 and matched the matched-lead study's published 8.771%
exactly, and a refit of one plant's baseline on the GPU reproduced the first run's per-row losses
with a maximum difference of zero.

## Reproducing the figures

```bash
uv run python studies/lag_features/build_lag_frame.py
uv run python studies/lag_features/fit_lag_arms.py --reproduction-check --device cuda --max-workers 4
uv run python studies/lag_features/fit_lag_arms.py --device cuda --max-workers 4
uv run python studies/lag_features/report_lag_features.py
uv run python studies/lag_features/followup_frames.py
uv run python studies/lag_features/fit_followups.py --device cuda --max-workers 4
uv run python studies/lag_features/report_followups.py
uv run python studies/lag_features/followup2_frames.py
uv run python studies/lag_features/fit_followups2.py --device cuda --max-workers 4
uv run python studies/lag_features/report_followups2.py
uv run python studies/lag_features/build_lag_frame.py --weather-product ifs_single
uv run python studies/lag_features/lag_features_charts.py
uv run python studies/lag_features/lag_features_charts.py --followups
```
