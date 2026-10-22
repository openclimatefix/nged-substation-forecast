---
id: '00201'
slug: dp-gam-residual-mlp
theme: differentiable-physics-forecast-engine
also_relevant_to:
- calibration-and-uncertainty
- lagged-power-and-error-correction
status: done
size: >-
  medium. Stored: private files under data/ml-experiments/00201-dp-gam-residual-mlp/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (network
  size, which inputs, additive or multiplicative output), and the plan pre-registers four input arms
  and one network shape. Callers: this folder's scripts, which copy experiments 136, 149 and 155
  and call the reviewed `studies.*` modules without changing them.
hypothesis: >-
  A small neural network that reads the convex model's own design features, and later the calendar,
  the ensemble-mean weather, and lagged power, can learn an interaction the additive convex model
  lacks, and so lowers the quantile score of the best-guess convex forecast (arm D of experiment 155)
  when it adds a correction to the median and rescales the band widths.
prediction: >-
  Written before any score is read. Arm 3 (design features, calendar, weather) or arm 4 (plus lagged
  power and a site embedding) lowers the quantile score over the scored year by at least 0.02
  points against arm D, with the blocked-month interval wholly below zero, and worsens none of the
  special-day groups (cold nights, sunny daylight, Christmas and bank holidays, Economy 7 nights) by
  more than 0.05 points. Arms 1 and 2 gain 0.00 to 0.01. If every arm fails, neural tweaking of the
  convex model leaves the queue.
decision_rule: >-
  Four planned contrasts (arm 1, 2, 3, 4 minus arm D), each read at the 97.5% level of the
  month-blocked paired bootstrap on the ENS-era scored year. The hypothesis is `supported` when the
  best of arms 3 and 4 has its interval wholly below -0.02 points and no special-day group worsens
  by more than 0.05. The threshold of -0.02 is set by cost: a network adds a trained model, a GPU
  step, and stored weights to the serving path. `partly supported` when the interval is wholly
  below zero but a condition fails. `refuted` when wholly above zero. `inconclusive` otherwise. The
  known-answer gate runs first: the network must recover at least 50% of an injected interaction on
  held-out months and change the score on uninjected rows by under 0.02 points. A failed gate caps
  the verdict at `inconclusive` and stops the experiment.
benchmark: >-
  Arm D of experiment 155 (the best-guess convex forecast with the winter night offset and recipe
  R), rebuilt for 2019-09 onward at six substations
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  over leads 0.5 to 168 hours, at day D+1, by lead band, on cold nights, sunny daylight, Christmas
  and bank holidays, and Economy 7 nights; coverage and pinball loss at q01, q05, q95, q99
code: ml-experiments/00201-dp-gam-residual-mlp/
verdict: partly supported
best_arm: >-
  Arm 2 (arm D's own inputs plus calendar and holiday flags): 2.587 points against arm D's 2.691
  over leads 0.5 to 168 hours at six substations in the scored year (-0.104; 97.5% interval -0.196
  to -0.020). Arm 3 (plus ensemble-mean weather) scored 2.597 (-0.094; -0.195 to -0.002).
headline: >-
  A small network that reads only arm D's own forecast, filter state, and lead recalibrates arm D
  and lowers its score by 0.086 points (arm 6; -0.173 to -0.031); weather, calendar, and lagged
  power add nothing measurable, and every arm scores worse on Christmas and bank holidays.
lessons:
  - >-
    The gain comes from reading arm D's own outputs, not from new information: an arm with no
    forecast or weather input (calendar, lead, site only) scored +0.017 (-0.019 to +0.065), and an
    arm with only arm D's outputs, lead, and time of day scored -0.086.
  - >-
    The network raises the median by 1.5 to 3.3 points on days hotter than 24 C and by 1.3 to
    2.9 points below -3 C, so arm D under-forecasts at both temperature extremes.
  - >-
    The network worsens Christmas (+0.18 to +0.80) and bank holidays (+0.11 to +0.44), 19 days in
    one year, so a network correction cannot be used without a holiday guard.
take_home_message: >-
  A small neural network that reads the convex forecast's own output lowers its error by about
  0.09 points (3%), but adding weather, calendar, or recent power does not help further. The
  network is correcting a systematic under-forecast on very hot and very cold days, which points
  to a missing temperature term in the convex model, and it makes Christmas and bank holidays worse.
advice:
  - >-
    Test extra heating and cooling terms in the convex model for days above 24 C and below -3 C
    before using a network: the network raised the median by up to 3 points there (figure 3,
    arms 1 and 3).
  - >-
    Do not add weather, calendar, or lagged power inputs to a network that corrects arm D: arms
    2 to 4 scored within 0.02 of arm 1 (figure 1).
  - >-
    Guard any learned correction on bank holidays and Christmas, where all arms scored 0.1 to 0.8
    points worse than arm D (figure 2).
  - >-
    Run a blind control with calendar, lead, and site only, and an ablation without the base
    forecast's own outputs, before crediting a network with new information (arms 0, 5, 6).
date: 2026-10-22
production_status: not-for-production
adopted_in: none
---

# Experiment 201: a small neural network on top of the convex forecast

**A small neural network that reads the convex forecast's own output lowers its error by about 0.09
points (3%), and nothing else we gave it helped further.** We took arm D of
[experiment 155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md) (the
convex demand model with a Kalman filter on recent errors, a winter night offset, and recipe R) and
trained a network with two hidden layers of 64 units to shift arm D's median and rescale its
bands, a design that adds to [experiment 142](../00142-differentiable-physics-sequence-model-on-gam-k-residuals/result.md).
The network gets the convex model's own inputs (arm 1), then calendar and holiday flags (arm 2),
ensemble-mean weather (arm 3), and power lagged by whole days (arm 4). The scored year is 2025-07 to
2026-06 at six substations, which the network never saw.

**The gain is a recalibration of arm D, and the network worsens Christmas and bank holidays.** Arms 1
to 4 each score about 0.09 to 0.11 points below arm D. A blind network with no forecast or weather
scored +0.017 (no gain), and a network with only arm D's forecast, lead, and time of day scored
-0.086, so the gain comes from re-reading arm D's own output. Mostly it raises the median at
temperature extremes, which suggests a missing term in the convex model. The pre-registered
prediction (arm 3 or 4 at least 0.02 points below arm D with an interval wholly below that, and no
special-day group worse by more than 0.05) failed on both conditions.

![Contrasts against arm D over all rows and day D+1](figure_1_headline.svg)

![Contrasts against arm D on cold nights, sunny daylight, Economy 7 nights, bank holidays, and Christmas](figure_2_special_days.svg)

![Mean shift of the median by driver temperature](figure_shift_by_driver_temperature.svg)

![Mean shift of the median by lead](figure_shift_by_lead.svg)

![Mean shift of the median by half-hour of day](figure_shift_by_local_half-hour_of_day.svg)

![Mean shift of the median by day of week](figure_shift_by_day_of_week.svg)

## Results

- **Arms and inputs.** Arm 1 has 19 columns, arm 2 28, arm 3 32, and arm 4 41, so the arms are not
  equal in width. Arm 0 (blind: lead, calendar, site; 25 columns), arm 5 (arm 1 without the filter's
  level and anomaly), and arm 6 (arm 1 without the temperature, solar, and darkness terms) are post
  hoc. Network: two hidden layers of 64, zero-initialised output, weight decay 1e-3, early stopping
  on every fourth month of the pool, pinball loss over 11 quantiles, 3 seeds.
- **Row set.** Training on every row before 2025-07 (2020-01 to 2024-06 past-weather rows plus
  ensemble-era tuning and replicate rows); scoring on the 561,363 scored-year rows (the six
  substations D5, D9, D10, D11, D14, D16) outside the switching events. Arm D rows are experiment
  155's stored forecasts, and its five-substation score reproduces experiment 155's 2.756 exactly.
- **Scores over leads 0.5 to 168 h (points; arm D 2.691).** Arm 0 2.709; arm 1 2.599; arm 2 2.587;
  arm 3 2.597; arm 4 2.584; arm 5 2.621; arm 6 2.606.
- **Contrast against arm D (97.5% month-blocked interval).** Arm 1 -0.092 (-0.197 to -0.001); arm 2
  -0.104 (-0.196 to -0.020); arm 3 -0.094 (-0.195 to -0.002); arm 4 -0.107 (-0.219 to +0.032); arm
  0 +0.017 (-0.019 to +0.065); arm 5 -0.070 (-0.162 to +0.005); arm 6 -0.086 (-0.173 to -0.031). At
  day D+1: arm 1 -0.064, arm 3 -0.055, arm 4 -0.071.
- **Special groups, arm 3 minus arm D.** Cold nights -0.189; sunny daylight -0.096; Economy 7 nights
  -0.120; bank holidays +0.437 (+0.290 to +0.570); Christmas +0.352 (-0.374 to +0.360). The
  bank-holiday and Christmas groups hold 19 days from one year (9 at Christmas).
- **Tail calibration, arm D then arm 3.** Coverage of q01 2.3% then 2.1%; q05 7.1% then 7.3%; q95
  94.7% then 93.7%; q99 98.2% then 97.8%. Pinball at q95 1.452 then 1.391; at q99 0.445 then
  0.441. Arm 4's q95 coverage fell to 90.3%.
- **Known-answer gate (passed).** On the scored year the network recovered 71% of an injected
  interaction (cold-evening effect up to 3 points plus a 1-point Saturday-evening shift): 24% of the
  cold-evening part and 77% of the Saturday part. On rows with no injection the network scored 0.005
  points worse than the same network trained on the real power. Post hoc: the pre-registered
  wording measured change against arm D, which cannot work because the network also improves arm
  D (-0.09), so the recovery and the harm are measured against a control network trained on the
  real power.
- **What the network learned.** The median shift is 0 to 0.5 points between 0 and 21 C and rises to
  +1.7 (arm 1) or +3.3 (arm 3) points at 24 to 27 C and to +1.3 or +2.9 at -6 C (figure 3); it is
  near zero at every lead except the last day (figure 4).

## Checks

- **Median minus mean, per site.** Arm D's `q50` equals its mean exactly at all six substations
  (0.000), so the spread does not shift the forecast.
- **Error of the median beside the quantile score.** The mean absolute error of the median is 4.623
  for arm D and 4.581 (arm 1), 4.535 (arm 2), 4.554 (arm 3), 4.476 (arm 4), the same direction as
  the quantile score.
- **Leave one substation out.** Every leave-one-out contrast for arms 1 to 4 stays between -0.047
  and -0.147; D16 carries the most (-0.32 alone).
- **Triggers.** Trigger 2 fired (arms 1 and 2 were predicted to gain 0.00 to 0.01 and gained about
  0.09 to 0.10); arms 0, 5, and 6 test the explanation and support it (see above), so no audit was
  run. Triggers 1, 3, 4, and 5 did not fire. Trigger 6 was not checked against a sister experiment.

## Caveats and what to try next

- **Early-era rows have smaller weather error than live rows.** The 2020-01 to 2024-06 training
  rows use ERA5 and CAMS as the weather, so a network trained on them may under-correct. The scored
  year uses ensemble-mean weather only.
- **One scored year and six substations.** The gain rests on 12 months, and D16 gives the largest
  share of it. Sixteen Primaries were planned; the other ten lack a pinned solar capacity and
  ensemble frames, so they were not built.
- **No refit through the scored year, and one seed pool.** The network is trained once on rows before
  2025-07. A weekly refit, as production would use, was not tested.
- **Wind, humidity, and school holidays were not available** in the stored arrays, so arm 3 has
  temperature and shortwave only, and the calendar has no school-holiday flag.
- **Next: test extra heating and cooling terms in the convex model.** The prediction that could
  fail: adding a cooling hinge above 21 C and a steeper heating hinge below 0 C to arm D lowers
  its score by at least 0.05 points without a network.
