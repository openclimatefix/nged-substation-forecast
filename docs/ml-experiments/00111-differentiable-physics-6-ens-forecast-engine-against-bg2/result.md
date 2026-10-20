---
id: "00111"
slug: differentiable-physics-6-ens-forecast-engine-against-bg2
title: "Differentiable physics, part 6: the DP as a forecast engine on ENS, against BG2"
part: 6
theme: differentiable-physics-forecast-engine
also_relevant_to: [production-reference-and-best-guess, lagged-power-and-error-correction, weather-products-and-blending]
status: done
size: >-
  complex for the module changes (the ensemble weather layer, the calibration, the DP run per
  ensemble member, the filter on ensemble residuals, and forward-chained margins, all in
  packages/studies, reviewed and merged in PR 1388); medium for the experiment scripts. Stored:
  private files under data/ml-experiments/00111-differentiable-physics-6-ens-forecast-engine-against-bg2/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (the calibration, the filter's residuals, the margin), and the plan pre-registered one
  of each. Callers: the experiment's scripts only.
hypothesis: >-
  Driven by ECMWF ENS forecasts, the convex DP demand model forecasts Primary demand up to 7 days
  ahead better than lagless XGBoost given the same recent-level correction, and, used as the base
  margin of the BG2 recipe's lagged XGBoost, makes the production recipe better like for like.
prediction: see "Predictions against results"
decision_rule: see "Decision rule and verdict"
benchmark: >-
  Lagless XGBoost on ENS (SIMP1's no-lag monthly refits, with and without the same Kalman filter,
  with a zero start and with experiment 64's solar offset); the lagged XGBoost member with its
  default start, a zero start, and the offset start; BG2 as stored (experiment 87); climatology;
  manual_heuristic
row_set_id: >-
  fold_a_daily_gap_v1 (the 292 daily ENS runs of experiment 87) at the five decision substations
  D5, D10, D11, D14, and D16 (D9 scored beside them); replicate on the even months of 2024-07 to
  2025-06
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, over
  leads 0.5 to 168 hours from the 09:00 UTC power_fcst_init_time
verdict: inconclusive
best_arm: >-
  GAM-K (convex model plus Kalman filter), 2.800 points at the five decision substations against
  2.780 for BG2 on the same rows
headline: >-
  The convex model with a Kalman filter scored level with BG2 (+0.020 points, 95% interval -0.109
  to +0.157) and 0.134 points better than lagless XGBoost given the same filter (98.75% interval
  -0.305 to +0.014), so neither contrast is decided; the filter lowered the score by 0.151 points
  (-0.295 to -0.008) and its gain came in the first two days; the DP's forecast as XGBoost's
  starting point changed nothing measurable (-0.023, -0.217 to +0.183).
lessons:
  - >-
    The filter on recent errors helps the convex model and lagless XGBoost by the same amount
    (about 0.15 points, nearly all within two days), so a model comparison needs the filter on both
    sides.
  - >-
    The convex model's edge over XGBoost is small and not stable: it reversed sign in February,
    April, and June 2025, where XGBoost has more than 10 months of ensemble history.
  - >-
    The plan's GPU estimate for the XGBoost arms was about 20 times too low, so the margin arms ran
    monthly, not weekly, and the confirmation run against BG2 was not run.
production_status: not-for-production
adopted_in: none
date: 2026-10-20
pr: 1389
take_home_message: >-
  With real ensemble weather forecasts as input, the physics-based demand model with a filter on
  its recent errors forecast seven days ahead about as well as the production recipe at five
  Primary substations (2.80 against 2.78 points), and 0.13 points better than lagless XGBoost with
  the same filter, though neither gap is statistically decided. The filter is what helps, mainly in
  the first two days, and it helps XGBoost just as much; using the physics model's forecast as
  XGBoost's starting point did nothing measurable.
advice:
  - >-
    Put the Kalman recent-level filter on every model before comparing model families on ensemble
    weather: it lowered the score by 0.151 points for the convex model and by 0.154 for lagless
    XGBoost (C4 and X-nl+K minus X-nl).
  - >-
    Feed the filter ensemble-driven residuals in a live service: ERA5-driven residuals scored no
    better (-0.008 points, GAM-K-era5 minus GAM-K).
  - >-
    Do not give lagged XGBoost the convex model's forecast as a base margin on monthly refits: the
    scored year showed no gain (C3) and February, April, and June 2025 showed a loss of 0.334.
  - >-
    Train the convex model on past weather, not on ensemble rows: training on ensemble control rows
    scored 0.248 points worse (GAM-ens minus GAM-0).
  - >-
    Budget 250 s of GPU time per 500-round XGBoost refit on the whole history before planning any
    arm that refits weekly; this experiment's estimate was 20 times too low.
  - >-
    Compare a convex model fitted on years of past weather with XGBoost only in months where XGBoost
    has more than 10 months of ensemble history: in all replicate months the contrast was -0.679
    points and in February, April, and June 2025 it was +0.221 (C1).
---

**With ECMWF ensemble weather forecasts as input, the convex differentiable-physics demand model plus
a Kalman filter on its recent errors (GAM-K) scored level with the production recipe (BG2) and
about 0.13 points better than lagless XGBoost given the same filter, but neither gap is decided by
this year of data.** Scores are quantile scores in points of the substation's effective capacity
(lower is better), at the five decision Primary substations D5, D10, D11, D14, and D16, over leads
of 0.5 to 168 hours, on every half-hour outside the hand-filtered switching events. The scored year
is 1 July 2025 to 30 June 2026. The filter lowered the score by 0.151 points (98.75% interval
-0.295 to -0.008), by 0.392 points at leads under 24 hours and by 0.017 beyond 144 hours, and lowered
lagless XGBoost's score by 0.154. The DP's own forecast, given to lagged XGBoost as a starting
point, changed nothing measurable.

We asked whether the physical model, driven by real weather forecasts as production would be,
forecasts demand better than XGBoost when both see the same weather forecast and the same recent
observations, and whether giving the production recipe the physical model's forecast as a starting
point improves it. Parts 1 to 5 of the series drove the physical model with estimates of past
weather (ERA5 temperature and Copernicus Atmosphere Monitoring Service (CAMS) irradiance), which a
live service would not have. Here the model runs once per member of the 51-member ensemble, with
the ensemble's temperature and irradiance calibrated onto ERA5 and CAMS, and the filter reads the
model's recent errors made with the ensemble's own 9.5-to-33-hour forecasts. XGBoost trains on
ensemble control-member rows. We took the branches of the plan from Parts 3 and 4 (see "Arms, rows,
and settings"). BG2 is compared with GAM-K only as an exploratory contrast, because BG2 differs in
three lags, a five-member blend, and weekly refits. The confirmation run that would compare the
production recipe with and without the DP's margin like for like was not run, because the margin
contrast (C3) showed no gain.

![The planned and exploratory contrasts, with intervals and site counts](figure_02_headline.svg)

![Observed power and four forecasts in a winter week at each decision substation](figure_01a_winter_week.svg)

![Observed power and four forecasts in a summer week at each decision substation](figure_01b_summer_week.svg)

![Every arm scores worse as the lead grows; the filtered convex model and BG2 are level](figure_03_by_lead.svg)

![The ranking of the arms differs from substation to substation](figure_04_by_site.svg)

![The filter's correction sits in its persistent level](figure_05_correction.svg)

![The ensemble runs close to the weather the convex model was trained on](figure_06_calibration.svg)

![Tail calibration of every arm](figure_07_tails.svg)

![The known-answer gate passed every reading](figure_08_gate.svg)

## Arms, rows, and settings

- **Branches taken (Parts 3 and 4).** GAM-K has no drift (Part 4's C4 passed non-inferiority in
  both years) and a 24-month training window (Part 4's C1 was not supported); GAM-0 keeps drift.
  The capacity is pinned to experiment 45's real-time vintage in every arm. The pooled arms of
  Part 3 were not run, because Part 3's onboarding and short-history contrasts were not supported.
  Part 4's drift step of 30 sat at the edge of its grid, so GAM-0's step was tuned afresh on the
  tuning months over 30 to 100,000 and the step of 30,000 won, inside the grid (mean tuning scores
  3.180 at 30, 3.021 at 10,000, 3.018 at 30,000, 3.040 at 100,000).
- **Periods.** Tuning is the odd months of 2024-07 to 2025-06, replicate is the even months, and the
  scored year is the 292 runs of `fold_a_daily_gap_v1`. The solar shrink chosen on the tuning months
  was 1 in every lead band (no shrink); the spread factors were 0.9 to 1.15, with 0.9 at the lower
  edge of the plan's grid for GAM-K in three bands, where the score differs by 0.003 to 0.013
  points from a factor of 1.
- **Calibration.** Fitted on the tuning months per site, season, lead band, and hour band: the
  temperature shift ranged from -1.15 to +2.05 K (mean +0.20) and the daylight irradiance ratio from
  0.80 to 1.11 (mean 0.95) over the decision substations (Figure 6).
- **DP arms.** GAM-0 (drift, no correction), GAM-K (the main arm), GAM-E (14-day weighted mean in
  place of the filter), GAM-K-era5 (the filter on ERA5 and CAMS residuals, which a live service would
  not have), and GAM-ens (trained on ensemble control rows, fitted once at 2025-07-01).
- **XGBoost arms.** X-nl is SIMP1's no-lag monthly recipe (500 rounds, eta 0.05, half-life 0.5
  years), reused for the scored year from SIMP1's stored refits after one refit at D10 reproduced
  within -0.003 points of quantile score, and refitted here for the 12 tuning and replicate months.
  X-nl+K adds the same Kalman filter, fitted on X-nl's 24-hour block residuals. Xlag-d is the lagged
  member with three origin lags, GAM-margin is Xlag-d with the DP's forward-chained forecast as the
  base margin, both refitted monthly at 150 rounds and eta 0.15 on leads up to 168 hours. X-nl-z and
  X-nl-o (zero start; median plus experiment 64's solar offset) ran for the scored year only.
- **Rows.** 464,358 half-hours shared by the arms of C1 at the five decision substations in the
  scored year; D9 is scored beside them in the 6-substation rows; D18 is descriptive and enters no
  contrast.

## Decision rule and verdict

- **The verdict is `inconclusive`.** C5, the only contrast that can say "beats BG2", was not run.
  C3 is neither `supported` nor `small gain` (its interval spans zero), and C2 is not `supported` by
  the series' rule (its interval spans zero: -0.134, -0.305 to +0.014, with 4 of 5 substations
  better). Neither C2 nor C3 is wholly above zero, so the verdict is not `refuted`.
- **Two readings the verdict does not capture.** GAM-K scored level with BG2 (+0.020, -0.109 to
  +0.157), and GAM-K's gain from the filter was significant and consistent in sign in the replicate
  months.

## Headline numbers

Scored year, five decision substations, quantile score in points (negative favours the first arm);
the interval of C1 to C4 is 98.75% (Bonferroni across four) and every other interval is 95%, from
resampling whole calendar months of run initialisations, paired across arms. The numbers come from
`e00111_report.py`, saved as `report.md`.

- **C1, GAM-0 minus X-nl (X-bar): -0.137 (-0.513 to +0.185), 3 of 5 substations better.** Absolute
  scores 2.951 and 3.088. The median-forecast error contrast is -0.218. It is -0.670 from November
  to March and +0.215 from April to October. Without D14 it is -0.031.
- **C2, GAM-K minus X-nl+K: -0.134 (-0.305 to +0.014), 4 of 5 better.** Absolute scores 2.800 and
  2.934; median-forecast error -0.159. At leads under 24 hours it is -0.162 (-0.315 to -0.029).
  Without D5 it is -0.210 (-0.337 to -0.098).
- **C3, GAM-margin minus Xlag-d: -0.023 (-0.217 to +0.183), 3 of 5 better.** Absolute scores 2.853
  and 2.877; median-forecast error +0.001. Without D5 it is +0.014.
- **C4, GAM-K minus GAM-0: -0.151 (-0.295 to -0.008), 3 of 5 better.** By lead: -0.392 (-0.498 to
  -0.288) under 24 hours, -0.244 from 24 to 48 hours, -0.100 from 48 to 144 hours, and -0.017
  (-0.165 to +0.159) from 144 to 168 hours. It is -0.026 from November to March and -0.234 from April
  to October.
- **Replicate months.** All even months of 2024-07 to 2025-06, where XGBoost trained on 3 to 14
  months of ensemble rows: C1 -0.679 (-2.193 to +0.257), 4 of 5 better; C2 -0.316 (-0.721 to
  +0.035), 5 of 5; C3 -0.103 (-0.691 to +0.386); C4 -0.215 (-0.396 to -0.057), 5 of 5. In February,
  April, and June 2025 only (10 or more months of XGBoost rows): C1 +0.221 (-0.092 to +0.446), 1 of 5
  better; C2 -0.098 (-0.363 to +0.241), 4 of 5; C3 +0.334 (+0.092 to +0.510), 1 of 5; C4 -0.377
  (-0.494 to -0.312), 5 of 5. So the sign of C1 and C3 is not stable, and C2's and C4's is.
- **Exploratory, scored year.** GAM-K minus BG2 +0.020 (-0.109 to +0.157), better at 0 to 24 hours
  (-0.060) and worse beyond 144 hours (+0.063). X-nl minus BG2 +0.308 (+0.196 to +0.450). GAM-K
  minus Xlag-d -0.077 (-0.218 to +0.067). X-nl+K minus X-nl -0.154 (-0.308 to -0.023). GAM-K-era5
  minus GAM-K -0.008 (-0.024 to +0.008). GAM-E minus GAM-K +0.034 (-0.011 to +0.082), worse at 0 to
  24 hours (+0.174). GAM-ens minus GAM-0 +0.248 (+0.007 to +0.513). X-nl-o minus X-nl-z -0.023
  (-0.077 to +0.023) and X-nl-o minus X-nl -0.006 (-0.039 to +0.037).

## Absolute scores

| Arm (scored year, 5 decision substations) | Quantile score |
|---|---|
| BG2 | 2.780 |
| GAM-K-era5 | 2.792 |
| GAM-K | 2.800 |
| GAM-E | 2.834 |
| GAM-margin | 2.853 |
| Xlag-d | 2.877 |
| X-nl+K | 2.934 |
| GAM-0 | 2.951 |
| X-nl | 3.088 |
| GAM-ens | 3.200 |

Each row is on the rows it shares with its contrast partner, so scores quoted from two contrasts
differ in the third decimal. Climatology and the manual heuristic are in Figure 3 and 7.

## What the filter does

- **The filter's gain sits in the first two days and is the same for both model families.** GAM-K
  minus GAM-0 is -0.392 under 24 hours and -0.017 beyond 144 hours, and X-nl+K minus X-nl is -0.382
  and -0.028 (Figure 3 and the by-lead rows above). Part 4 found -0.61 on past weather at the
  day-ahead lead; the ensemble gain at 0 to 24 hours is about two thirds of that.
- **The persistent level carries the correction (Figure 5).** The transient part is about 0 at every
  lead.
- **ERA5 residuals add nothing.** GAM-K-era5 minus GAM-K is -0.008 (-0.024 to +0.008), so a live
  service gains nothing from waiting for ERA5.
- **The 14-day weighted mean is worse at short range.** GAM-E minus GAM-K is +0.174 at 0 to 24 hours
  and -0.043 beyond 144 hours.

## What did not help

- **The DP's forecast as a base margin.** C3 -0.023 in the scored year and +0.334 in February, April,
  and June 2025. D5 contributes the gain: without D5 C3 is +0.014.
- **Training the convex model on ensemble rows.** GAM-ens minus GAM-0 +0.248 (+0.007 to +0.513),
  +0.591 from November to March, which is the opposite of what the train-each-weather-on-its-own-
  history rule would predict for this model, and agrees with the plan's note that GAM-ens tests the
  departure.
- **Experiment 64's solar offset on lagless XGBoost.** X-nl-o minus X-nl-z -0.023 and minus X-nl
  -0.006, against the plan's -0.05 to -0.20 on April-to-September daylight rows. The offset
  rows by month were not drawn.

## Predictions against results

| Prediction | Result | Within? |
|---|---|---|
| C1 -0.05 to -0.40, gain concentrated in 0-48 h | -0.137; the gain is spread across leads (-0.152 at 0-24 h, -0.093 at 144-168 h) | estimate yes, shape no |
| C2 -0.05 to -0.25, same sign in replicate months | -0.134 scored; -0.098 in February, April, and June | yes |
| C3 -0.02 to -0.15 | -0.023 scored; +0.334 in the replicate months | barely, and the replicate disagrees |
| C4 -0.3 to -0.6 at 0-24 h, under -0.1 beyond 144 h | -0.392 and -0.017 | yes |
| C5 -0.01 to -0.08 | not run | no result |
| GAM-K 0.0 to 0.4 worse than BG2 | +0.020 | edge |
| X-nl 0.5 to 1.4 worse than BG2 | +0.308 | below the range |
| GAM-K-era5 beats GAM-K by 0.05 to 0.20 at 0-24 h | -0.044 | edge |
| X-nl-o minus X-nl-z -0.05 to -0.20 on April-to-September daylight rows | -0.023 over all rows | not tested on those rows |
| Tails: q05 and q95 within 3 points of nominal, q01 and q99 within 1.5 | GAM-K's coverage is within 1.84 points of nominal at every level and band (largest: q05 at 24-48 h, +1.84; q01 at 24-48 h, +1.40) | yes |
| Each arm under 10 minutes | seconds for the DP, about 1 minute per XGBoost refit when the GPU is free | yes |

The SIMP1 audit explains the X-nl gap: the lag penalty is 0.36 points over the 16 Primaries and
0.125 without the one substation whose level changed, not 0.9 (briefs of 2026-10-19). Our 0.308 over
five substations sits between those two figures.

## Checks

- **The median forecast minus the mean forecast is under 0.17 points at every site and DP arm** (for
  GAM-K -0.117 at D5, -0.043 at D11, -0.037 at D10, +0.015 at D14, +0.002 at D16), because the
  ensemble mixture is not symmetric. XGBoost arms hold only the median.
- **The median-forecast error agrees in sign with the quantile score for C1, C2, and C4, and
  disagrees for C3** (-0.023 against +0.001, both about 0) and for X-nl-o minus X-nl-z (-0.023
  against +0.032).
- **Five of the seven triggers fired.**
    - **Trigger 1, a control fails:** the first run of the gate failed two readings, both from a
      sign error in the gate's own fake irradiance (the fake scaled CAMS up by 1.25 where the plan
      scaled it down by 0.8, and the test expected the wrong ratio); the second run passed 22 of
      22. X-nl's reproduction of SIMP1 held at the one refit checked.
    - **Trigger 2, a result contradicts a prediction:** fired for X-nl minus BG2 (+0.308 against
      +0.5 to +1.4), explained by the SIMP1 audit, and for the offset rows.
    - **Trigger 3, the quantile score and the median-forecast error disagree:** fired for C3 and for
      X-nl-o minus X-nl-z, both within noise.
    - **Trigger 4, leaving a site out flips a contrast:** fired for C3 (without D5 +0.014, without
      D16 +0.048) and changes C2's label (without D5 -0.210, -0.337 to -0.098, wholly below zero).
    - **Trigger 5, a nested arm beats its container, or a null control moves:** did not fire (GAM-K-era5
      minus GAM-K -0.008 is within its own interval).
    - **Trigger 6, an interval far wider or narrower than a sister's:** fired for C3, whose half-width
      of 0.20 is about four times experiment 87's 0.05 for nested steps, and for C1, whose width of
      0.70 is under half of Part 4's C6 (1.69).
    - **Trigger 7, a later design depends on an unreplicated result:** did not fire, because no arm
      became a production candidate.
- **An audit is called for.** Triggers 1, 2, 3, 4, and 6 fired, so a fresh Opus audit runs before the
  prose review, as the experiment skill requires.

## Known-answer gate

The gate ran on the tuning months before any replicate month or scored run and passed 22 of 22
readings: the calibration recovered a temperature bias of +1.5 to +3 K and an irradiance factor of
0.8 within 15% (and the calibrated fake scored within 0.05 points of ERA5 and CAMS); the calibrated
ensemble's irradiance peaked within one half-hour of CAMS's at every site, with a midday ratio of
0.88 to 0.90; the filter carried a median 1.07 of an injected transfer by the end of day 2, 0.93
from day 2 to the end, and 0.000 three days after the end (16 transfers at D10, D11, D14, and D16);
the margin's error on earlier runs was 0.98 of its error on the tuning months' runs; and the module's
tests passed (Figure 8). The recovery readings are exact (0.000) because the fake's member noise has
zero mean across the 51 members, so they check the indexing and the arithmetic and not the
calibration's response to noise, and the calibrated fake's error differed from ERA5's by 0.049
against a limit of 0.05.

## Departures from the plan

- **Compute.** The plan estimated 1.5 to 2 GPU-hours for the XGBoost arms; one 500-round refit on
  the whole history took 247 s while the GPU was shared, matching SIMP1's log, so the estimate was
  about 20 times too low. The margin family therefore refitted monthly, not weekly, on leads to 168
  hours at 150 rounds and eta 0.15; X-nl for the scored year reused SIMP1's refits; X-nl-z and X-nl-o
  ran for the scored year only; the placebo margin, the zero-start lagged arm, GAM-K-w, GAM-K-fc,
  and the tuning-month choice among start rules were not run, so X-bar is X-nl with its default
  start; and C5, the like-for-like comparison with BG2, was not run because C3 showed no gain. The
  XGBoost jobs logged 7.88 job-hours, plus the early refits, which were not logged.
- **Implementation.** No `WeatherInputLayer` was filled (the pooled arms did not run). The PV shape
  is not derated by the member's temperature in the weather layer, because the convex model derates
  it itself. A member with a missing shortwave value is dropped from its run at that site (member 36
  of the 2025-06-04 run at D9 and D11). The last scored run (2026-06-30) is left out of the DP and
  margin arms because the grid ends before the held-out period. The offset arms' training rows
  omit the last two weeks of June 2025, which experiment 87's anomaly table lacks. The filter's
  hyperparameters were fitted in one round, with Part 4's fitted prior, not by the plan's
  empirical-Bayes rounds. The "all rows" robustness row was not run.
- **D18.** SIMP1's refit with D18's commissioning months cut is the source of D18's X-nl rows;
  D18 enters no contrast.

## Caveats

- **XGBoost had 3 to 27 months of ensemble history, and the convex model years of past weather.**
  The replicate year flatters the DP for that reason (C1 -0.679 in all replicate months, +0.221 in
  the three months with 10 or more months of rows).
- **The tuning months are in-sample for the spread**, and the replicate months are interleaved with
  them.
- **The 73 stride runs of the scored year chose BG2's blend and offset**, and BG2's spread factor
  of 1.15 is experiment 18's.
- **Two mild shared leaks:** the 2024-25 margins use a calibration fitted on the same odd months,
  and the convex model's last training rows use ERA5 that a live refit would not yet have.
- **The analysis proxy is the control member of the latest run**, not an analysis.
- **Hindsight switching masks** apply to every scored row.
- **Five substations, one year.** The intervals resample 12 months.

## Tail calibration

GAM-K's q01, q05, q95, and q99 coverage sits within 1.84 points of nominal in every lead band, and
lagless XGBoost's q95 coverage is 4.4 to 5.5 points below nominal at every lead band (Figure 7).
Curtailment cost against `manual_heuristic` was not computed: at the MW Primaries experiment 87
found no event at the scaled limit that the metric could rank forecasts on.

## Substation cards

![Provisional substation cards](figure_09_cards.svg)

**Provisional: fitted to past weather, forecast on ensemble weather, screening run.** The cards
show the weekday daily profile of GAM-K's fit at the 2025-07-01 cut, the pinned solar capacity
(an input from experiment 45's real-time estimates, not a fitted value), and the filter's weekly
mean level over the scored year. No heating slope or fitted capacity is shown. Heating slopes and
capacities are weakly identified at real noise levels (experiment 112); compare across substations
with care, and interpret no threshold per site. Identification improves with more weather
anomalies, which is a reason a longer history could help interpretability.

## Open questions after this experiment

- Does a longer ensemble history (the 2021 to 2024 backfill on Source Cooperative) settle C1 and C2?
  The maintainer decides whether to fetch it.
- Does the filter lower the production XGBoost blend's score as it lowers X-nl's? The result is
  a candidate for the next release review.
- Does the margin help at weekly cadence? The plan's cadence needs about 60 GPU-hours at the speed
  measured here.

## Reproducing the figures

```bash
uv run python ml-experiments/00111-*/e00111_build_frames.py
uv run python ml-experiments/00111-*/e00111_store.py
uv run python ml-experiments/00111-*/e00111_fits.py
uv run python ml-experiments/00111-*/e00111_tune.py
uv run python ml-experiments/00111-*/e00111_drift.py
uv run python ml-experiments/00111-*/e00111_gate.py
uv run python ml-experiments/00111-*/e00111_dp_run.py
uv run python ml-experiments/00111-*/e00111_gam_ens.py
uv run python ml-experiments/00111-*/e00111_xgb.py --arm xnl --before 2025-07-01
uv run python ml-experiments/00111-*/e00111_xnl_k.py
uv run python ml-experiments/00111-*/e00111_report.py
uv run python ml-experiments/00111-*/e00111_charts.py
```
