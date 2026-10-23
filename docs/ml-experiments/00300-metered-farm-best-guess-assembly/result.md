---
id: '00300'
slug: metered-farm-best-guess-assembly
title: >-
  How good is the assembled best-guess day-ahead forecast for the six metered solar farms, and what
  does a new farm with no history get?
theme: metered-solar-farms
also_relevant_to:
- production-reference-and-best-guess
- calibration-and-uncertainty
status: finished
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00300-metered-farm-best-guess-assembly/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: one design (the candidate that experiments 270, 287, and
  260 picked), so no design choice is open. Callers: this folder's scripts, copied from experiments
  270, 287, and 260, with no new reviewed module. Asset graph: not touched.
hypothesis: >-
  The assembled candidate (spread-binned plant P2, a generalised additive model (GAM) band on the
  plant's residuals, a mixture-density head (MDN), blended per quantile with XGBoost) scores better
  than XGBoost alone on the quantile score and at the 0.99 quantile, and its advantage is larger
  for a farm with no history because the plant carries what the missing history cannot.
prediction: >-
  Written before any score is read. On own history, the candidate beats XGBoost on the quantile
  score by 1% to 2% (98.75% interval wholly below zero), beats it by at least 20% at the 0.99
  pinball loss, and is within 1% of XGBoost on the absolute error of the median. With zero history
  (leave-one-farm-out) the candidate beats XGBoost by at least 3%. The 3% and the 20% are the
  predictions that could fail.
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months at 98.75% (Bonferroni across four contrasts: candidate minus XGBoost on the quantile
  score in own history and in zero history, candidate minus XGBoost at the 0.99 pinball loss in own
  history, and the candidate minus XGBoost on the absolute error of the median in own history).
  Supported when the own-history score interval is wholly below zero, the 0.99 gain is at least
  20%, and the zero-history gain is at least 3%. Partly supported when one or two of the three
  hold. The 1% score threshold is set by the cost of the extra stages (a plant, a band model, a
  blend), which a gain under 1% would not repay. Stop rule: if the candidate's score interval
  against XGBoost includes zero in both settings, report that the plant-plus-band and XGBoost are
  equivalent for scoring and that the choice is explainability; add no arms.
benchmark: >-
  XGBoost production-style (experiment 275's x2 in own history, experiment 270's l6 in zero
  history, both read from stored predictions when the refit schedule below matches, otherwise refit
  here), the manual_heuristic and climatology floors of experiment 224, and capacity_forecast
row_set_id: >-
  Experiment 287's rows: experiment 224's scored-year daylight half-hours at 12 to 36 hour leads
  that every arm holds (farms A to F, 2025-09 to 2026-08)
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), mean absolute
  error (MAE) of the median, pinball loss and coverage at 0.01, 0.05, 0.95, and 0.99, per farm, arm,
  lead, clear-sky band, hour, and season
code: ml-experiments/00300-metered-farm-best-guess-assembly/
verdict: partly supported
best_arm: >-
  The blend of GAM, MDN, and XGBoost scores 5.27 points of capacity in own history and 5.45 in zero
  history (12 to 36 h leads, 2025-09 to 2026-08, monthly refits).
headline: >-
  The assembled candidate beats XGBoost by 3.3% on the quantile score (98.75% interval -4.3% to
  -2.0%) and by 43% at the 0.99 pinball loss in own history, but by only 2.0% (-2.9% to -1.0%) in
  zero history, short of the 3% bar.
lessons: >-
  Most of the candidate's edge over XGBoost sits in the upper tail, where the mixture head carries
  the 0.95 and 0.99 quantiles. The lagged-power inputs and the spread add under 0.4% to the
  additive model, so the plant carries most of the skill.
take_home_message: >-
  The assembled solar-farm forecast (a physical power model, an additive and a mixture-density band
  on top, blended with XGBoost) is 3.3% better than XGBoost alone on the whole forecast band and
  43% better at the 0.99 quantile that matters for curtailment, when each farm trains on its own
  history. A farm with no history still gets a forecast 2.0% better than XGBoost's, which is less
  than the 3% we predicted, so the physical model helps less than expected when history is missing.
advice:
- >-
  Take the blend of the additive model, the mixture-density head, and XGBoost as the metered-farm
  candidate: it scores 3.3% below XGBoost in own history and 2.0% below in zero history, and 43%
  below at the 0.99 quantile (C1, C2, C3, Figure 3).
- >-
  Do not spend effort on lagged power or the availability ratio in the metered-farm band: dropping
  either changes the additive model's score by under 0.3% (Figure 6).
- >-
  Keep the ensemble spread in the plant's bins: dropping it raises the additive model's score by
  0.3% in own history and 0.6% in zero history, with intervals that include zero (Figure 6).
- >-
  Judge a farm's day-ahead forecast on its 0.01 and 0.05 quantiles before use: actual output falls
  below the 0.01 quantile 3.5 times too often for every arm (Figure 4).
production_status: candidate
adopted_in: none
---

**The assembled forecast beats XGBoost by 3.3% on the whole forecast band and by 43% at the 0.99
quantile when a farm trains on its own history, and by 2.0% when it has no history.** National Grid
Electricity Distribution (NGED) meters six solar farms, labelled A to F here. We asked how the
forecast that experiments [270](../00270-metered-farm-xgboost-vs-plant-diagnostic/result.md),
[287](../00287-metered-farm-spread-binned-plant-heads/result.md), and
[260](../00260-metered-farm-quantile-combination/result.md) picked scores against the references a
reader will ask for. The forecast is a physical power model of the farm ("P2"), a generalised
additive model (GAM) and a mixture-density network (MDN) that correct P2's 11 quantiles, and a
per-quantile blend of those two with [XGBoost (Chen and Guestrin, 2016)](https://doi.org/10.1145/2939672.2939785).
We refitted every arm each month for the 12 scored months. "Own history" means a model trains on
the farm's earlier months, and "zero history" means it trains on the other five farms only.

**The verdict is partly supported.** The own-history score gain (-3.3%, 98.75% interval -4.3% to
-2.0%) and the 0.99 pinball-loss gain (-42.8%, -49.2% to -36.0%) met the plan. The zero-history gain
(-2.0%, -2.9% to -1.0%) did not reach the 3% we predicted, although its interval excludes zero. The
own-history score gain is larger than the 1% to 2% we predicted, and the next section says why
that matters.

![Figure 1: absolute quantile score of every arm](figure_1_scores.svg)

![Figure 2: score and error of the median against lead time](figure_2_by_lead.svg)

![Figure 3: the planned and supporting contrasts](figure_3_contrasts.svg)

![Figure 4: tail calibration](figure_4_coverage.svg)

![Figure 5: score by clear-sky band, hour, and season](figure_5_slices.svg)

![Figure 6: ablations](figure_6_ablations.svg)

## Results

- **Quantile score (points of capacity; own history, then zero history).** Candidate blend 5.27 and
  5.45, blend of GAM and XGBoost 5.32 and 5.47, MDN (five seeds averaged) 5.34 and 5.55, XGBoost
  5.45 and 5.56, GAM 5.43 and 5.59, P2 5.54 and 5.71, `climatology` 8.64, `manual_heuristic` 8.69.
  The candidate is 39% below `manual_heuristic` in own history (-42.3% to -36.3%) and 37% below in
  zero history.
- **Planned contrasts, candidate minus XGBoost (98.75% interval).** Own history: score -3.27%
  (-4.30 to -2.04); 0.99 pinball loss -42.8% (-49.2 to -36.0); error of the median -1.74%
  (-2.92 to -0.15). Zero history: score -2.03% (-2.90 to -0.99).
- **What each stage adds (own history).** Candidate minus P2 -4.75% (-7.84 to -2.96). GAM minus P2
  -1.9% (the ablations' reference). MDN minus GAM -1.57% (-2.91 to -0.44). Candidate minus the
  blend without the MDN -0.87% (-1.32 to -0.31). In zero history MDN minus GAM is -0.80% (-2.61 to
  +0.70) and the candidate minus the blend without the MDN is -0.42% (-1.05 to +0.23).
- **Ablations on the GAM (score change against the full GAM; own history, zero history).** No
  spread +0.31% (-0.83 to +1.09) and +0.58% (-0.38 to +1.24). No lagged power +0.01% (-0.06 to
  +0.07) and -0.01%. No availability ratio +0.17% (-0.01 to +0.35) and +0.27% (+0.13 to +0.42).
  Lagged power and the availability ratio add almost nothing.
- **Tail calibration (own history; nominal 0.01, 0.05, 0.95, 0.99).** Candidate coverage 0.036,
  0.070, 0.957, 0.990; XGBoost 0.035, 0.083, 0.946, 0.991. Pinball loss at 0.99: candidate 0.31,
  XGBoost 0.54, `manual_heuristic` 0.58, `climatology` 0.43. At the 0.01 quantile every
  forecast arm leaves the actual output below it 3.4 to 3.6 times too often. At the 0.05 quantile
  the arms do so 1.3 to 1.5 times too often, and XGBoost 1.7 times.
- **Where the gain sits (own history).** By season the candidate beats XGBoost by 4.1% in June to
  August (-5.5 to -2.9), 3.2% in March to May, and 2.3% in September to November and in December to
  February, where the December to February interval includes zero. By clear-sky band the candidate
  is 8% below XGBoost under cloud (index below 0.3), 5% below under clear sky (0.9 and above), and
  2% below in between. The candidate is worse than
  XGBoost in one month (December 2025 in own history, April 2026 in zero history).
- **Lead axis.** The stored ensemble tables cover leads of 12 to 36 hours from the 00 UTC run, and
  daylight leaves 3-hour bins starting at 12, 15, 18, 27, 30, and 33 hours. The candidate scores
  6.7 at the lowest bin and 6.5 at the highest, and about 1.1 to 1.2 at the 18 and 27 hour bins.
  The lead range of 0.5 to 54 hours that the brief asked for needs other ensemble runs and was not
  scored.
- **Farm cards (own history, August 2026 refit; fractions of capacity).** The plant scale is 1.00
  to 1.16 and the output ceiling 1.05 to 1.11. The mean width of the 5% to 95% band is 16 to 23
  points of capacity under cloud (clear-sky index below 0.3), 48 to 55 under partly clear sky (0.6
  to 0.9), and 29 to 39 under clear sky (0.9 and above). Across farms and bands, the band holds 85%
  to 95% of actual output against a nominal 90%. The per-farm tables are in the private data folder.
- **What the candidate cannot do.** It cannot forecast outages or curtailment by the network
  operator, and a 25% loss of capacity is detected too slowly to act on. A drift under about
  0.5% per year is invisible to it ([experiment 286](../00286-metered-farm-drift-soiling-injection/result.md)).
- **Curtailment value.** Not scored. The node-plus-generators metric needs a substation node with
  its generators, and these six farms are single series, so the comparison against
  `capacity_forecast` could not be run. Experiments 255, 268, and 277 give the metric's limited
  power.
- **Reference rows, quoted and not refitted.** The plant given observed weather explains a sunny
  fraction of 0.973, and the plant given the ensemble mean 0.91 ([experiment 218](../00218-metered-solar-farm-plant-fit/result.md)).
- **Schedule and weather.** Monthly refits on all earlier months, each month's training rows ending
  one day before the month starts. All forecast weather is the ECMWF ensemble 00 UTC run, and the
  farm's own power one or two days earlier; both exist at forecast time. Weekly refits were not run,
  so the absolute scores could shift a little. The MDN has five seeds, and the ablations use the
  GAM only.

## Checks

- **Median minus mean forecast, own history (points of capacity, farms pooled).** Candidate +0.01,
  XGBoost -0.92, GAM +0.29, MDN +0.37, P2 +0.40. XGBoost's median sits below its mean forecast; no
  other arm's does by more than 0.4.
- **Error of the mean forecast beside each score contrast.** Candidate minus XGBoost in own
  history: mean-forecast error -3.78% (-5.82 to -2.43), against a score change of -3.27%. The score
  and the mean-forecast error point the same way, so trigger 3 did not fire.
- **Audit triggers.** Trigger 2 (a result contradicting its prediction) fired in size only: the
  own-history gain is 3.3% against the predicted 1% to 2%. The XGBoost arm here has no past-weather
  rows, because those arrive about 5 days late. [Experiment 287](../00287-metered-farm-spread-binned-plant-heads/result.md)'s
  XGBoost scored 5.40 with past weather in a single fit, and the candidate is 2.4% below that
  figure, which brings the result into the predicted range. The two schedules differ, so the
  comparison is approximate. Trigger 4 did not fire: dropping each farm in turn leaves the
  own-history score change between -2.9% and -3.5%, and the zero-history change between -1.9% and
  -2.2%. Trigger 5 did not fire: the ablated GAMs score within 0.6% of the full GAM, and none
  scores better by more than the interval's half-width. Triggers 1, 3, 6, and 7 did not fire. No
  audit ran.
- **Row set.** Every arm holds the same 41,676 rows in each setting (the code checks equal counts).

## Caveats and next steps

- **Post hoc.** The blend without the MDN is an added arm and not a planned contrast. The 98.75%
  level covers the four planned contrasts only.
- **Not covered.** Leads under 12 hours and over 36 hours, weekly refits, and the curtailment value.
- **Next.** Run the same candidate on the ensemble's other runs to cover the 0.5 to 54 hour lead
  range, and score it with the node metric once a farm is paired with its substation.
