---
id: '00257'
slug: metered-farm-fair-plant-arm
title: >-
  Is the neural head needed once the solar-farm power model sees the same inputs?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: finished
size: >-
  medium. Stored: predictions and tables under data/ml-experiments/00257-metered-farm-fair-plant-arm/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  way to give the plant the lagged power (a ratio or a difference), so the ratio is fixed in this
  plan before any fit. Callers: this folder's scripts, copied from experiments 224 and 243, and no
  new reviewed module. Asset graph: not touched.
hypothesis: >-
  The 4.4% gain of H5 (the plant plus a neural network that outputs all 11 quantile offsets) over
  the plant alone in experiment 243 came from the inputs that H5 saw and the plant did not (the
  lagged power and the ensemble spread), so a convex linear quantile regression of the 11 offsets
  on those same inputs (arm P2) recovers most of it, and the neural network is not needed.
prediction: >-
  H5 beats P2 by at least 2% in quantile score, with a 95% month-block interval wholly below zero,
  and by at least 10% in pinball loss at the 0.99 quantile (the prediction that could fail).
  Experiment 243 found H5 4.4% better than the plant alone, and experiment 224's two-number plant
  (P1) already recovers about 1.7% of that, so P2 within 1% of H5 is plausible.
decision_rule: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), equal weight
  per farm, on the 41,654 daylight half-hours of experiment 243, 12 to 36 hours ahead, 2025-09 to
  2026-08, with intervals from resampling the 12 scored months 20,000 times, at the 99% level
  (Bonferroni over five planned contrasts). C1: H5 minus P2, quantile score. C2: H5 minus P2,
  pinball loss at 0.99. C3: P2 minus P1. C4: P3 minus H5 with zero history. C5: P2 minus P2 without
  the availability ratio. Supported when C1 is at least 2% below zero with an interval wholly below
  zero and C2 is at least 10% below zero. The 2% threshold is the cost of serving and retraining a
  neural network beside a convex fit. Stop rule: if P2 is within 1% of H5 in quantile score,
  report that the gain came from the inputs and not from the network, and that P2 replaces H5 as
  the metered-farm candidate. If P2 is worse than P1, report a fitting fault before any verdict.
benchmark: >-
  H5 and XGBoost from experiment 243 (stored, own and zero history), the plant alone (P0), the
  two-number plant (P1), and experiment 224's manual_heuristic and climatology floors
row_set_id: >-
  Experiment 243's rows: daylight half-hours (sun more than 5 degrees above the horizon) at 12 to
  36 hour leads, 2025-09 to 2026-08, farms A to F
metric: >-
  Quantile score, MAE of the median, MAE of the mean forecast, and pinball loss and coverage at
  0.01, 0.05, 0.95, and 0.99, per farm and arm
code: ml-experiments/00257-metered-farm-fair-plant-arm/
verdict: supported
verdict_note: >-
  The quantile regression P2 scores 5.500 against H5's 5.475 with own history (0.45% worse, 99%
  interval -2.2% to +0.6%), so the network's 2% advantage in the prediction fails and the stop rule
  fires. P2 recovers 90% of H5's 4.4% gain over the plant alone. H5 is 3.7% worse than P2 at the
  0.99 quantile, with an interval that includes zero.
best_arm: >-
  H5 has the lowest quantile score with own history, 5.475 points of capacity (95% interval 4.95 to
  5.93), and P2 follows at 5.500 (5.02 to 5.94)
headline: >-
  A linear quantile regression on the lagged power and ensemble spread scores within 0.5% of the
  neural network (99% interval -2.2% to +0.6%) and 4.0% better than the plant alone.
take_home_message: >-
  The neural network was not needed. Giving the solar-farm power model the lagged power and the
  spread of the ensemble members, through a plain linear fit, recovers about 90% of the neural
  network's gain and scores within half a percent of it. The linear fit is also better at the
  0.99 quantile, so it replaces the neural network as the metered-farm candidate.
production_status: candidate
adopted_in: none
advice:
  - >-
    Fit the metered-farm quantiles with a linear quantile regression on the plant forecast, the
    ensemble spread, the availability ratio, and lead, clear-sky, and elevation bands before adding
    a neural network (C1, Figure 2).
  - >-
    Give the plant forecast the ensemble spread and the lagged-power availability ratio, because
    the two-number plant scores 2.5% worse than the linear fit that has them (C3, Figure 2).
  - >-
    Do not credit the lagged power alone for the gain: the availability ratio is worth 0.3% with
    an interval that includes zero (C5, Figure 2).
  - >-
    Judge any new network for metered solar farms against the linear fit and not against the plant
    alone, because the plant alone overstates the network's gain by about 10% of it (Figure 1).
---
**A linear quantile regression on the lagged power and the ensemble spread scores within 0.5% of
the neural network of experiment 243, so the network was not needed.** National Grid Electricity
Distribution (NGED) meters six solar farms, labelled A to F. [Experiment
243](../00243-metered-farm-architecture-comparison/result.md) found that H5, the plant (a solar-farm
power model) plus a neural network that outputs 11 quantile offsets, scores 4.4% better than the
plant alone, but H5 saw the lagged power and the ensemble spread and the plant did not. We gave a
linear quantile regression (P2) the same inputs, with no network. P2 scores 5.500 against H5's
5.475 points of capacity with own history, so P2 is 0.45% worse (95% interval -1.7% to +0.4%; 99%
interval -2.2% to +0.6%). P2 is 3.95% better than the plant alone, which is 90% of H5's gain.
**The prediction fails and the stop rule fires.** The prediction was that H5 beats P2 by at least
2% in quantile score and by at least 10% in pinball loss at the 0.99 quantile. The first gap is
0.45%, and at the 0.99 quantile P2 is better, with H5 3.7% worse (99% interval -1.8% to +10.9%).
**The ensemble spread and the availability ratio matter, and the ratio matters least.** P2 beats the
two-number plant (P1) of experiment 224 by 2.5% (99% interval -3.8% to -1.2%). Removing the
availability ratio costs 0.33% (99% interval -0.67% to +0.02%). With zero history, P2 is 0.63%
worse than H5 (99% interval -0.25% to +1.96%).

![Figure 1: quantile score per arm](figure_1_headline.svg)

![Figure 2: planned contrasts](figure_2_contrasts.svg)

![Figure 3: pinball loss at the outer quantiles](figure_3_tails.svg)

## Numbers

- **Quantile score** (points of capacity, mean over six farms, own history): H5 5.475, XGBoost
  5.490, P2 5.500, P2 without the ratio 5.518, P1 5.642, plant 5.726, climatology 8.638, manual
  heuristic 8.687. Zero history: XGBoost 5.610, H5 5.610, P2 5.645, plant 5.871.
- **Per farm, own history:** P2 is within 0.1 of H5 at five farms (A, B, C, D, F) and 0.09 better
  at farm E (6.26 against 6.35).
- **Median error** (MAE of the median, points of capacity, own history): H5 9.85, XGBoost 9.89, P2
  9.89, plant 9.95.
- **Outer quantiles, own history** (pinball loss at 0.01 and 0.99): H5 0.334 and 0.344, XGBoost
  0.369 and 0.485, P2 0.349 and 0.331, plant 0.372 and 0.418. Coverage of the 0.01 and 0.99
  quantiles: H5 0.033 and 0.985, P2 0.035 and 0.985, plant 0.034 and 0.984.
- **Cloudy and clear half-hours** (clear-sky index under 0.5 and 0.5 or more): H5 minus P2 is -2.7%
  when cloudy (99% interval -6.4% to +0.3%) and -0.15% when clear. P2 with the ratio is 0.48% worse
  than P2 without it when cloudy (99% interval -0.8% to +2.3%) and 0.42% better when clear (99%
  interval -0.8% to -0.1%).

## Checks

- **Fitting fault.** P2 beats P1, which the plan named as the test of a fitting fault, so the fit
  is sound.
- **Member alignment.** The mean of the 51 members' responses differs from experiment 224's
  `unit_mean` by under 0.02 on average, as expected from a non-linear plant, and the correlation
  exceeds 0.99.
- **Triggers.** Trigger 2 fires: the prediction that H5 beats P2 by 2% fails. No other trigger
  fires. The stored H5, XGBoost, and plant predictions are unchanged from experiment 243, and the
  rows are experiment 243's, so the contrasts pair on the same half-hours.
- **Departures from the plan.** The MAE of the mean forecast is not reported, because P2 outputs
  quantiles only.

## What the next experiment should answer

**Whether a small neural network beats P2 once both see more than the inputs here, such as the
neighbouring farms' power.** P2 leaves 0.45% unexplained against H5 with the window's uncertainty
of about 1.4% either side, so a network with extra inputs is the next test, scored against P2 and
not against the plant alone.
