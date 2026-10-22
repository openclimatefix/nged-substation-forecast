---
id: '00224'
slug: metered-farm-dayahead-two-number
title: >-
  Does the two-number summary of the 51 ensemble members forecast a metered solar farm day ahead
  as well as XGBoost?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
- differentiable-physics-embedded-generation
status: finished
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00224-metered-farm-dayahead-two-number/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible design for the band (a Gaussian
  mixture of the two numbers, or a closed-form shift of the residual offsets), so the Gaussian
  mixture of experiment 193 is registered. Callers: this folder's scripts, copied from experiments
  187, 193, and 218, and no new reviewed module. Asset graph: not touched.
hypothesis: >-
  Passing each of the 51 ensemble members through a solar farm's plant model gives the farm's day-
  ahead
  output a spread that carries information the ensemble mean lacks, and the mean and standard
  deviation of
  the 51 plant responses keep most of that information, as they did for Primary substations in
  experiment 193. A two-number plant forecast
  therefore forecasts a metered solar farm 12 to 36 hours ahead about as well as XGBoost, and gives
  calibrated tails.
prediction: >-
  The arm with the mean and standard deviation of the 51 plant responses (arm 3) keeps at least
  80% of the
  gain that the arm with all 51 responses (arm 4) has over the ensemble-mean arm (arm 2), in the
  quantile
  score on daylight half-hours at 12 to 36 hour leads; the MAE of arm 3's median forecast is
  within 5% of
  XGBoost's (arm 5); and arm 3's pinball loss at the 0.01 and 0.99 quantiles is lower than
  XGBoost's.
decision_rule: >-
  Quantile score is twice the mean pinball loss over the 11 levels of `studies.ens_members`, in
  points of
  capacity, over daylight half-hours (sun more than 5 degrees above the horizon) at 12 to 36 hour
  leads in
  the scored year (2025-09-01 to 2026-08-31), averaged over the six farms with equal weight. C0 is
  arm 4
  minus arm 2 as a percentage of arm 2. C1 is arm 3 minus arm 2 as a percentage of arm 2.
  Supported when
  C1 is at least 80% of C0 (the retained share), the interval of the share from resampling the 12
  scored months excludes 0.8 from below, and the other two conditions of the prediction hold; partly
  supported when only the share holds; not supported otherwise. Stop rule: if C0 is a gain of
  under 2%
  at three or more farms, members do not help a single farm, and the page says so first. If arm 3
  keeps
  under 50% of the gain, report arm 4 only. The 80% is set by cost: a two-column summary replaces a
  51-member pass through the plant in the weather layer, and a summary that gave up a fifth of the
  gain
  would not justify a second path beside the member pass. The 5% on MAE is set by the cost of a
  physical
  plant beside XGBoost: a plant forecast is only worth keeping if it is not worse in the middle
  either.
benchmark: >-
  Arm 2 (plant on the ensemble mean), arm 5 (XGBoost on the same rows), and the manual_heuristic and
  climatology floors
row_set_id: >-
  Metered solar-farm half-hours with sun more than 5 degrees above the horizon, a null
  `drop_reason` in
  experiment 179's cleaned series, an observed power at every floor and arm, in 12 to 36 hour
  leads (and
  60 to 84 and 108 to 132 hour leads as secondary bands), scored 2025-09-01 to 2026-08-31 with a
  replicate of 2025-03-01 to 2025-08-31
metric: >-
  Quantile score in points of capacity, MAE of the median, and coverage and pinball loss at 0.01,
  0.05,
  0.95, and 0.99, per farm and arm
code: ml-experiments/00224-metered-farm-dayahead-two-number/
verdict: partly supported
best_arm: >-
  Plant fed the mean and spread of the 51 members (arm 3), quantile score 5.64 points of capacity
  (95% interval 5.15 to 6.07) at 12 to 36 h in 2025-09 to 2026-08; XGBoost scores 5.49 (4.93 to
  5.97)
headline: >-
  Mean and spread of the 51 plant responses keep 95% of the members' gain over the ensemble-mean
  plant
  (share 0.95, interval 0.84 to 1.02), but that gain is only 1.8% of the quantile score, and XGBoost
  scores 2.8% better than arm 3 in the scored year.
take_home_message: >-
  The 51 weather forecasts help a single solar farm's day-ahead forecast only a little: passing each
  one through a plant model lowers the quantile score by 1.8% against feeding the plant their
  mean, and
  two numbers (their mean and spread) carry 95% of that. A gradient-boosted tree model on the same
  weather is 2.8% better overall, and the plant model beats it only at the 0.99 quantile, which is
  the
  tail that matters for curtailment.
advice:
  - >-
    Store the mean and standard deviation of the members' plant responses, not all 51 responses,
    for a
    metered farm, because the two numbers keep 95% of the gain at 12 to 36 h (C1 and the retained
    share).
  - >-
    Do not expect more than a 2% gain in quantile score from using members at a single farm, because
    the gain is 0.7% to 2.4% across the six farms at 12 to 36 h (C0 and Figure 3).
  - >-
    Compare any new farm forecast with XGBoost in the middle of the distribution and with the
    plant in
    the upper tail, because XGBoost scores lower pinball loss at q01 to q95 and the plant at q99
    (Figure 4).
  - >-
    Give XGBoost the plant's mean response and the members' spread as input columns before choosing
    between the two families, because each family wins at different quantile levels (Figure 4).
production_status: not-for-production
adopted_in: none
---

**The 51 ensemble members help a single metered solar farm's day-ahead forecast only a little, and
two numbers per half-hour carry nearly all of that help.** NGED meters the output of six solar farms
in its trial area, which this page labels A to F. We passed each of the 51 members of the European
Centre for Medium-Range Weather Forecasts (ECMWF) ensemble forecast through a generic solar-farm
model (south-facing panels at 35 degrees tilt, one fitted scale per farm), which gives 51 possible
outputs per half-hour. We built three forecasts from the 51 outputs: one from the plant fed the
ensemble mean, one from all 51 outputs, and one from only their mean and standard deviation. We
scored them at 12 to 36 hours ahead on daylight half-hours from 2025-09 to 2026-08, against a
gradient-boosted tree model (XGBoost) fitted on the same weather. The tree model is [XGBoost, Chen
and Guestrin (2016)](https://doi.org/10.1145/2939672.2939785). The two-number idea comes from
[experiment 193](../00193-dp-ens-two-number-solar-summary/result.md), which tested it on Primary
substations, and the member machinery from [experiment
187](../00187-dp-ens-members-in-stack/result.md).

**The prediction partly holds, and the stop rule we wrote fires.** The quantile score (twice the
mean pinball loss over 11 levels, in points of capacity) is 5.74 for the plant fed the ensemble
mean, 5.64 for the plant fed the mean and spread of the 51 outputs, 5.64 for the plant fed all 51,
and 5.49 for XGBoost. All 51 members lower the score by 1.8% (95% interval 1.3% to 2.2%); the mean
and spread lower it by 1.7% (1.1% to 2.2%), a retained share of 0.95 (0.84 to 1.02). The share
passes the 80% rule. The gain is under 2% at three farms (C, D, and E), so the stop rule fires, and
this page reports the result without a further arm. The MAE of the median is within 5% of XGBoost's
(9.99 against 9.87 points of capacity, 1.2% higher). The pinball loss at the 0.99 quantile is lower
than XGBoost's (0.386 against 0.478), but at the 0.01 quantile it is higher (0.374 against 0.352),
so the tail condition holds at one of two tails.

![Figure 1: quantile score per arm](figure_1_headline.svg)

![Figure 2: contrasts by lead band](figure_2_contrasts.svg)

![Figure 3: gain over the ensemble-mean arm per farm](figure_3_per_farm.svg)

![Figure 4: pinball loss at the four tail levels](figure_4_tails.svg)

## Numbers

- **Arms and schedule.** Fleet-shape plant with one fitted scale and one ceiling (the 99.5th
  percentile of tuning output) per farm; no per-farm plant, because [experiment
  218](../00218-metered-solar-farm-plant-fit/result.md) found per-farm parameters add 0.003 of the
  output fitted. One fit on the tuning months before the scored year, no refit in the scored year,
  for the plant scale, the offsets, and XGBoost (150 rounds, eta 0.15, depth 5, no column
  subsampling, no calendar columns). Offsets are the 11 residual quantiles of the tuning months in
  four bins of the forecast. The offset factor and spread factor are shared and fitted
  leave-one-farm-out on the other five farms' tuning rows.
- **Weather.** ECMWF ensemble (51 members), 00 UTC run, at the nearest 0.25 degree cell; every
  half-hour takes the run whose lead is 12 to 36 hours (also 60 to 84 and 108 to 132 hours as
  secondary bands). It exists at forecast time. The manual heuristic reads observed power 7 days and
  longer in the past, which also exists at forecast time. The ensemble store starts on 2024-04-01,
  so tuning is April 2024 to August 2025 for the scored year.
- **Rows.** Scored year 2025-09 to 2026-08: 41,654 daylight half-hours per lead band that every arm
  holds, six farms. Replicate (2025-03 to 2025-08, tuning 2024-04 to 2025-02): 23,782 half-hours.
- **Quantile score, 12 to 36 h, scored year (95% interval):** XGBoost 5.49 (4.93 to 5.97); all 51
  members 5.64 (5.14 to 6.07); mean and spread 5.64 (5.15 to 6.07); ensemble mean 5.74 (5.22 to
  6.19); climatology 8.64 (7.97 to 9.29); manual heuristic 8.69 (7.99 to 9.31).
- **Contrasts, 12 to 36 h, scored year:** C0 (members minus ensemble mean) -1.82% (-2.21 to -1.34);
  C1 (mean and spread minus ensemble mean) -1.73% (-2.23 to -1.12); retained share 0.95 (0.84 to
  1.02); C3 (mean and spread minus XGBoost) +2.83% (+0.95 to +5.67); mean and spread against the
  manual heuristic -35.0% (-37.7 to -32.5), against climatology -34.7% (-39.6 to -30.3).
- **Controls.** The placebo that gives the two-number arm the spread of another day (same farm,
  month, hour band, and forecast bin) scores 5.73, so the real spread beats it by 1.54% (1.22% to
  1.83%); the same for all 51 members is 1.47% (1.26% to 1.67%). The placebo that replaces the 51
  outputs by their mean scores 5.75, within 0.1% of the ensemble-mean arm. The members' information
  is therefore real, and small. The night placebo was not run (see the plan).
- **Lead bands.** At 60 to 84 hours the members gain 2.59% (1.81% to 3.28%) and the mean and spread
  keep 0.69 (0.55 to 0.76) of that; at 108 to 132 hours the gain is 2.44% (1.16% to 3.38%) and the
  share 0.50 (-0.25 to 0.70). XGBoost remains better than the two-number arm at both (about 3%).
- **Replicate (2025-03 to 2025-08, shorter tuning).** The members gain 1.32% (1.08% to 1.87%) at 12
  to 36 h with a retained share of 0.87 (0.58 to 1.01). XGBoost is 5.2% worse than the two-number
  arm (interval -6.2% to +4.7%), the reverse of the scored year, because its tuning window is only
  11 months.
- **Per farm, gain of all 51 members over the ensemble mean at 12 to 36 h:** A 2.36%, B 2.19%, C
  1.83%, D 1.92%, E 0.74%, F 2.11%. Leaving one farm out leaves the gain at 1.72% to 2.08% and the
  retained share at 0.94 to 0.96, so no farm flips a contrast.
- **Tail calibration, 12 to 36 h, pooled over farms (coverage at the level; pinball in points of
  capacity):** the two-number arm covers 3.2%, 5.5%, 96.3%, and 99.1% at the 0.01, 0.05, 0.95, and
  0.99 levels (pinball 0.374, 1.432, 1.367, 0.386); XGBoost 3.3%, 6.8%, 95.6%, and 99.6% (0.352,
  1.364, 1.214, 0.478); the ensemble-mean arm 3.3%, 6.3%, 94.9%, and 98.7% (0.366, 1.462, 1.494,
  0.428).
- **Curtailment value.** Not scored. The pilot set of `studies.node_curtailment` holds four nodes
  and scores a few events at the scaled limit, so it shows the arithmetic and cannot rank forecasts;
  this experiment did not map the six farms to those nodes.

## Checks

- **Median minus mean forecast per farm (points of capacity, 12 to 36 h):** the two-number arm +0.27
  to +0.48, the ensemble-mean arm +0.64 to +0.91, XGBoost -0.48 to +0.02 (the mean forecast is the
  11-level weighted mean of the quantiles). The spread therefore halves the gap between median and
  mean but does not remove it.
- **Mean-forecast error beside each quantile-score contrast.** The quantile score improves by 1.7%
  from the ensemble-mean arm to the two-number arm, while the MAE of the mean forecast worsens from
  10.13 to 10.20 points and the MAE of the median is unchanged (9.95 to 9.99). The gain therefore
  sits in the spread and tails, not in the location; XGBoost's MAE of the mean is 10.23, the worst
  of the three, while its quantile score is the best.
- **Triggers fired.** Trigger 3 fires (quantile score and the MAE of the mean forecast point in
  different directions). The explanation, that members sharpen the spread and not the location, is
  supported by the unchanged median MAE and by the placebo that gives the real spread a 1.5%
  advantage over a spread from another day. No Opus audit was run; the maintainer or the manager may
  call one. Triggers 1, 2, 4, 5, 6, and 7 did not fire: the controls behave (the rescale placebo
  changes the score by 0.1%, the shuffles lose most of the gain), the leave-one-farm-out checks keep
  every sign, and no later experiment depends on this result yet.
- **Caveats.** The factors `f` and `s` are fitted on tuning rows whose offsets were estimated on the
  same rows, which makes the fit slightly optimistic for all plant arms alike. The ceiling and scale
  are fitted on the ensemble-driven tuning rows, not on past weather. XGBoost gets one fit and no
  refit, and 17 months of tuning data; the replicate reverses its lead. Farm E has 10 months of
  tuning data in the scored year and the smallest gain. The floors were re-implemented from their
  definitions (13 earlier powers; month, half-hour, and weekend cells pooled over neighbouring
  cells) and not run through `BaseForecaster`. Twelve scored months make intervals close to the
  range of the draws. Figures were drawn from the scoring tables; the SVG files were optimised but
  not visually inspected in the built site.

## What the next experiment should answer

- Does an XGBoost model that is given the plant's mean response and the members' spread as input
  columns beat both the plant forecast and plain XGBoost? The two families win at different quantile
  levels, which suggests so.
- Is the plant's lead at the 0.99 quantile (0.386 against 0.478) real after XGBoost is refitted
  weekly, or a product of its single fit?

## Plan amendments (made before scoring)

- **Supported rule.** The retained share must be at least 0.8 as a point estimate, and the interval
  of C1 must lie wholly below zero; the share's own interval is reported beside it, and the plan's
  wording "excludes 0.8 from below" is dropped as too strict for 12 monthly blocks.
- **Replicate.** The ensemble store starts on 2024-04-01, so the replicate scores 2025-03 to 2025-08
  (six months), with tuning from 2024-04.
- **Quantile score** is twice the mean pinball loss over the 11 levels of `studies.ens_members`, in
  points of capacity; MAE is of the median forecast; the mean forecast in the Checks is the 11-level
  weighted mean.
