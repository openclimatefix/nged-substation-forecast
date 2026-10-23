---
id: '00311'
slug: primary-demand-mixture-density-band-head
title: >-
  Does a mixture-density neural network for the forecast bands lower the quantile score of Primary
  demand research default 1, with the gain in the protected tail groups?
kind: change
theme: calibration-and-uncertainty
also_relevant_to:
- production-reference-and-best-guess
status: done
size: >-
  medium. Stored: private files under
  data/ml-experiments/00311-primary-demand-mixture-density-band-head/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is
  defensible (the inputs, the loss, and what the mixture is centred on); the plan pre-registers one
  and a second arm that keeps the cold-day widening. Callers: this folder's scripts, which import
  the feature builder of experiment 278 and the scoring pattern of experiment 251. Trains a new
  neural head on the CPU; no download. Sonnet only, by the maintainer's instruction on 2026-10-23.
hypothesis: >-
  Primary demand research default 1 (the GAM stack A+H+S of experiments 220 and 251) gets its bands
  from a smooth spread table and a cold-day widening of the upper offsets, around a median shifted
  by the neural head. A mixture-density neural network, which reads the weather state and may be
  skewed, gives bands with a lower quantile score, and the gain sits in the protected tail groups.
prediction: >-
  Written before any score is read; each can fail. (P1) The default plus the mixture-density band
  head lowers the six-Primary quantile score by at least 0.02 points. (P2) At least 60% of that
  gain sits in the protected groups (the 0.01 and 0.99 tails and cold winter nights), counted as the
  share of the total gain that comes from rows in those groups. (P3) The 0.95 coverage of the
  mixture-density bands on ordinary rows (rows in no protected group) is within 2 points of 95%.
  (P4) The gain at D8-like misfit Primaries is not applicable, because D8 is not in the row set, so
  the page says nothing about D8. Stop rule: if the arm worsens the headline by more than 0.02
  points on the first six months of the scored year (2025-07 to 2025-12), stop the fits and report.
decision_rule: >-
  Primary demand research default 1 is the default; the two pre-registered arms are the
  mixture-density band head (arm M) and the same with the cold-day widening kept on top (arm MW),
  so k = 2 and the gain
  interval is read at the level 1 - 0.01/2 (99.5%). At most one arm is adopted. The change is a new
  head, so the minimum gain is 0.03 points by the research-default rule page (the brief for this
  experiment quoted 0.02 for a new fitted step; the rule page owns the value, and P1 keeps the
  brief's 0.02 so the prediction can be scored on either). Condition 1 (gain): the month-block
  interval of (arm minus default) on the six-Primary quantile score lies wholly below zero and the
  point estimate is at most -0.03. Condition 2 (no harm, one-sided 95% upper bound of the harm on
  the pooled six Primaries): at most 0.02 points on the headline; at most 0.05 points on each of the
  0.01 and 0.99 pinball tails (pinball loss of the single level, not doubled), on cold winter nights
  and on sunny daylight half-hours; at most 2% relative on twCRPS at the 95th percentile. Each
  Primary alone is reported with no pass or fail. Condition 3 (replication, replicate year
  2024-25): the same sign and a replicate estimate of at least half of the smaller of the main
  estimate and 0.03. Condition 4 (controls): the controls below pass and no audit trigger is open.
  Condition 5 (serving): every input is available at forecast time. Verdict `supported` if an arm
  meets conditions 1 to 5, `small gain` if the interval is below zero but the estimate is under the
  threshold, `not supported` if the interval includes zero, `refuted` if it lies wholly above zero.
benchmark: >-
  Primary demand research default 1 (stored forecasts of experiments 220 and 251, arm 1), with
  `manual_heuristic`, `climatology`, and BG2 (the XGBoost research benchmark `p_s6_bg2` of
  experiment 87, which is not the production forecast) beside every table.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score over 11 levels (twice the mean pinball loss), in points of effective capacity;
  twCRPS at the 95th percentile; pinball loss and coverage at q01, q05, q95, q99.
code: ml-experiments/00311-primary-demand-mixture-density-band-head/
verdict: partly supported
best_arm: >-
  Arm M, the mixture-density band head on the default's median: quantile score 2.542 points against
  2.579 for Primary demand research default 1 (difference -0.037, 99.5% interval -0.070 to -0.005).
headline: >-
  The band head lowers the six-Primary quantile score by 0.037 points (99.5% interval -0.070 to
  -0.005) and a similar gain (-0.034) appears in the replicate year, but it harms sunny daylight
  half-hours
  (+0.029, one-sided 95% bound +0.074 against a 0.05 tolerance) and D10 alone carries most of the
  gain, so it is not adopted.
lessons:
  - >-
    The gain is in the middle quantiles (0.05 to 0.25), not in the extreme tails: the 0.01 and 0.99
    levels give 9% of it, against the 60% predicted.
  - >-
    The head's lower band is too narrow on sunny daylight half-hours, where the 0.05 quantile
    covers 12.6% of observations (the default 8.1%, nominal 5%).
  - >-
    Without D10 the gain is -0.012 (95% interval -0.028 to +0.005), so the headline rests on one
    Primary.
production_status: candidate
adopted_in: none
take_home_message: >-
  A neural network that sets the forecast bands from the weather state scores 0.037 points better
  than the current Primary demand bands over the scored year, and 0.034 better in the year before,
  but the gain comes almost entirely from one substation (D10) and the new bands are worse on sunny
  daylight half-hours. The band head is a candidate, not an adoption; the next test should keep the
  current
  bands on sunny daylight half-hours or train the head on the quantile score directly.
advice:
  - >-
    Do not judge a Primary-demand band change on the pooled score alone: check the sunny daylight
    half-hours and leave out D10, because the mixture-density head's gain shrinks to -0.012 points,
    with an interval that includes zero, when D10 is left out and
    the head's score on sunny daylight half-hours is worse (leave-one-out and sunny daylight
    bullets).
  - >-
    Try a band head trained on the pinball loss of the 11 levels instead of the likelihood, because
    the likelihood-trained head put its gain in the 0.05 to 0.25 levels and not in the tails (figure
    3).
  - >-
    Keep the cold-day widening off the mixture-density head: arm M and arm MW differ by 0.001
    points (M minus MW -0.001, interval -0.003 to +0.000), so keeping the cold-day widening changes
    the score by at most 0.003 points here (arm MW row).
date: 2026-10-23
---

**Replacing the forecast bands of Primary demand research default 1 with bands from a neural network
lowers the quantile score by 0.037 points of capacity over the scored year, but the gain comes
almost entirely from one Primary, D10, and the new bands are worse on sunny daylight half-hours, so
we do not adopt them.** A similar gain, 0.034 points, appears in the even months of 2024-25,
measured against the stored bands of experiment 169 and not against Primary demand research default
1. The next test should keep the
current bands on sunny daylight half-hours, or train the head on the quantile score itself, because
the head's gain sits in the 0.05 to 0.25 quantiles and not in the extreme tails.

A band is the range between two quantile levels of the forecast, and a forecast here is 11 quantile
levels from 0.01 to 0.99 for each half-hour. The quantile score is twice the mean pinball loss over
those 11 levels, in points, meaning percentage points of the substation's effective capacity (its
rated demand); lower is better. A Primary is a primary substation, D5 to D16 are anonymised
Primaries, and the scored year runs from 2025-07-02 to 2026-06-30. Sunny daylight half-hours are
daylight half-hours that a solar-radiation threshold marks as sunny, and ordinary rows are all rows
that are neither sunny daylight half-hours nor cold winter nights (days with a mean temperature
below 2 °C, night hours only).

We asked whether the bands of Primary demand research default 1 (the GAM stack A+H+S of experiments
[220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md) and
[251](../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md)) can be improved
by a small neural network that outputs a mixture of three bell curves. The default's median stays as
it is, so only the width and skew of the bands change. We trained the network on the stack's own
past errors, refitted it every month on the preceding 24 months, and scored it on the same 561,207
half-hours as experiment 251. The mixture-density design follows [experiment
287](../00287-metered-farm-spread-binned-plant-heads/result.md) and [experiment
300](../00300-metered-farm-best-guess-assembly/result.md) for metered solar farms, and the input
features and monthly cuts are those of [experiment
278](../00278-stack-retraining-cadence/result.md). **The quantile score of the six Primaries fell
from 2.579 to 2.542 points of capacity (lower is better), in 9 of 12 months, but the condition on
harm to sunny daylight half-hours failed, so the result is a candidate under the [research default
rule](../research-default.md).**

![Figure 1. Quantile score minus the default, by group of rows](figure_1_gain_by_group.svg)

![Figure 2. Quantile score minus the default, by Primary](figure_2_gain_by_primary.svg)

![Figure 3. The gain by quantile level](figure_3_gain_by_level.svg)

![Figure 4. Coverage at the four tail quantiles](figure_4_tail_coverage.svg)

![Figure 5. A cold winter week at D10](figure_5_cold_night_week.svg)

![Figure 6. A sunny week at D5](figure_6_sunny_week.svg)

![Figure 7. Mean bands by local half-hour in winter at D10](figure_7_economy7_profile.svg)

**Verdict `partly supported`.** Conditions 1 (gain), 3 (replication), and 5 (serving) hold.
Condition 2 (no harm) fails on sunny daylight half-hours. Condition 4 (controls, with no audit
trigger open) is also not met, because trigger 4 fired and no Opus audit was run, so no arm is
adopted. The
plan's verdict rule named no wording for this case, so `partly supported` is the closest of the six
verdict words. Nothing is added to the [log of versions](../research-default-log.md).

## Design

- **Arms.** Default: arm 1 of experiments 220 and 251 (stored; median, bands, and solar summary
  unchanged). Arm M: the default's median, with every other quantile set by the mixture-density head
  (average of three seeds). Arm MW: arm M with the default's cold-day widening applied to the head's
  upper offsets (multipliers 1.653, 1.363, 1.261, and 1.162 at q75, q90, q95, and q99 on rows whose
  ensemble-mean 24-hour temperature is below 2 C, read from experiment 169's stored arms). At most
  one arm may be adopted, so the gain is read at 99.5%.
- **The head.** A three-component Gaussian mixture on the residual of arm D's mean forecast (arm D
  is the convex model with a Kalman filter of experiment 155), in
  units of the stored 5% to 95% width over 3.29, with the inputs of experiment 278's arm 2: arm D's
  mean, the Kalman filter's level and anomaly, the stored width, lead, ensemble-driver weather,
  calendar flags, and the Economy 7 flag. The head is trained on the negative log-likelihood, and
  its
  quantiles are shifted so that its median equals the default's median. The shift keeps the median
  unchanged and avoids training on the in-sample residuals of the neural median head.
- **Cadence.** A refit at each month start on the rows of the preceding 24 months whose longest
  target lies before the refit date, with every fourth month held out for early stopping, as
  experiment 278 does.

## Results

- **Kind and tier.** `kind: change`, family Primary demand, starting from Primary demand research
  default 1. The change is a tier 2 change (a new fitted head, with the feature code of experiment
  278
  imported), and the new-head minimum gain of 0.03 points applies. The brief for this experiment
  quoted 0.02 points; the rule page owns the value, and the point estimate (-0.037) clears both.
- **Absolute scores, six Primaries (quantile score; twCRPS at the 95th percentile of earlier
  power).** Default 2.579; 0.1708. Arm M 2.542; 0.1703. Arm MW 2.543; 0.1705. BG2 2.715; 0.2024 (BG2
  is the XGBoost research benchmark, not the production forecast). Manual heuristic 3.397; 0.2463.
  Climatology 3.665; 0.2627. The default scored 2.579 (six) and 2.617 (five decision Primaries),
  matching experiment 251 within 0.005, so the reproduction gate passed.
- **Arm M minus default, quantile score.** -0.037 (95% interval -0.060 to -0.014; 99.5% interval
  -0.070 to -0.005), 9 of 12 months better. The three worse months are February, March, and April
  2026 (+0.011, +0.031, +0.011). Arm MW minus default is -0.036 (99.5% interval -0.070 to -0.004).
  Arm M minus arm MW is -0.001 (-0.003 to +0.000), so keeping the cold-day widening changes the
  score by at most 0.003 points here.
- **Against BG2 and the manual heuristic.** Arm M minus BG2 is -0.173 (95% interval -0.314 to
  -0.040), against -0.136 (-0.278 to -0.005) for the default. Arm M minus the manual heuristic is
  -0.855 (-1.123 to -0.613).
- **Gate on harm (one-sided 95% upper bound of arm M minus default, pooled six Primaries).**
  Headline -0.018, passes (tolerance 0.02). Pinball loss at q01 -0.006 and at q99 +0.001, passes
  (0.05). Cold winter nights -0.064 (estimate -0.136), passes (0.05). twCRPS at the 95th percentile
  +1.3% relative, passes (2%). **Sunny daylight half-hours +0.074 (estimate +0.029, 95% interval
  -0.017 to +0.084), fails (0.05).** Arm MW gives the same pattern (sunny daylight bound +0.077;
  twCRPS +1.8%).
- **Replicate year (condition 3).** The even months of 2024-25, 360,122 rows. Research default 1's
  neural median head and solar summary exist for the scored year only, so the comparison is the head
  against the stored bands of arm A (experiment 169) around arm A's own median. Arm M minus the arm
  A
  bands is -0.034 (95% interval -0.052 to -0.016, resampling origin days, not months, so it is
  narrower than a month-block interval would be); arm MW -0.034. The sign matches and the size
  exceeds half of 0.03.
- **P1 and P3 held, P2 failed, and P4 does not apply.** P1 held (-0.037 against at least -0.02). P2
  failed: the 0.01 and 0.99 levels give
  9% of the gain and cold winter night rows give 5%, against 60% predicted. The prediction P2 mixed
  levels and row groups, so we scored it both ways before reading any score: the 0.01 and 0.99
  levels give 9% of the gain, and the cold winter night rows give 5%, and both fall short of 60%. P3
  held: the 0.95 coverage on ordinary rows is 0.935 against 0.945
  for the default, within 2 points of 95%. P4 is not applicable. The stop rule did not fire: the
  first six months improved by 0.059.
- **Where the gain is.** By Primary: D10 -0.157 (-0.305 to -0.049), D14 -0.024 (-0.033 to -0.013),
  D16 -0.017, D9 -0.011, D11 -0.005, D5 -0.001 (none of the six Primaries has a pass or fail).
  Leaving D10 out
  gives -0.012 (-0.028 to +0.005); leaving any other Primary out keeps -0.039 to -0.043. By level
  the gain is largest at q05 (-0.007), q10 (-0.008), and q25 (-0.006). By group: ordinary rows
  -0.048, Economy 7 nights -0.083, Christmas -0.068, leads 12 to 36 hours -0.025, bank holidays
  +0.014 (-0.013 to +0.028).
- **Tails.** The table below gives coverage and pinball loss at the four tail levels. On cold winter
  nights the q95 coverage is 0.970 for the default, 0.941 for arm M, and 0.973 for arm MW. On sunny
  daylight half-hours the q05 coverage is 0.081 for the default and 0.126 for both new arms (nominal
  0.05). q01 means the 0.01 quantile level, and so on.
- **Operator-visible behaviour.** Figure 7 shows the winter profile at D10: the Economy 7 step at
  about 00:30 local time stays inside both sets of bands, and arm M's bands are narrower than the
  default's overnight. Figure 5 shows the narrower cold-night bands in one January week, and figure
  6 shows the narrower lower band at midday in one June week, and the q05 coverage on sunny daylight
  half-hours rises above its nominal 5% at those midday half-hours.
- **The head makes the bands skewed upward.** The mean of arm M's 11 quantiles sits 0.20 points
  above its median, against 0.02 for the
  default, so the head makes the bands skewed upward.
- **Rows and schedule.** `fold_a_daily_gap_v1`, six Primaries D5, D9, D10, D11, D14, D16; scored
  year 2025-07-02 to 2026-06-30; leads 0.5 to 168 hours, so leads of 7 to 10 days are not scored and
  NGED's horizon is 1 to 10 days. Outside hand-filtered events. The head is refitted monthly on 24
  rolling months, three seeds averaged, on the CPU (about 25 s per fit). The default's stored
  forecasts are not refitted. Weather at lag times is the ensemble weather available at forecast
  time. The GAM stack still trails BG2 on sunny April to June half-hours (experiment 251); this
  experiment did not change that.
- **Weather in the training rows.** The 24-month pool of the earliest scored refits holds up to
  about a year of past-weather rows (origins before 2024-07), while the scored rows use ensemble
  weather, as in experiment 278.
- **Code reused.** `ml-experiments/00278-stack-retraining-cadence/` (`e00278_h_data.py` for the rows
  and inputs) and the scoring pattern of experiment 251, copied into this folder. No earlier folder
  was edited.
- **What to try next.** (1) Train the head on the pinball loss of the 11 levels, which is the
  quantity the quantile score averages (twice the mean),
  and not on the likelihood. (2) Keep the default's bands on sunny daylight half-hours and use the
  head elsewhere, as a new pre-registered experiment, because the research default rule does not let
  a failed arm be adjusted after its result is known. (3) Find why D10 gains 0.157 points:
  experiment 220 also located the gap between the neural median head's own bands and the default's
  at D10.

## Tails, six Primaries

| Level | Default coverage | Default pinball | Arm M coverage | Arm M pinball | BG2 coverage | BG2 pinball |
|---|---|---|---|---|---|---|
| q01 | 0.013 | 0.172 | 0.012 | 0.159 | 0.010 | 0.211 |
| q05 | 0.049 | 0.622 | 0.060 | 0.585 | 0.041 | 0.623 |
| q95 | 0.951 | 0.692 | 0.943 | 0.678 | 0.944 | 0.751 |
| q99 | 0.984 | 0.211 | 0.984 | 0.206 | 0.983 | 0.253 |

## Checks

- **Median minus the mean forecast, per Primary (points; the mean is arm D's, and the median is the
  default's, which every arm shares).** D10 +0.72, D11 -0.50, D14 -0.20, D16 -0.84, D5 +0.01, D9
  -0.75.
- **Error of the median beside the quantile-score contrasts.** The median is the default's in every
  arm, so the mean absolute error of the median is identical (4.510) and trigger 3 cannot fire.
- **Controls.** The reproduction gate passed. The assertion that every training target precedes its
  refit date held in all 54 fits (18 monthly refits times 3 seeds). Arm MW equals arm M on every row
  at or above 2 C, and the median
  of arm M equals the default's on every row.
- **Triggers.** Trigger 4 fired: leaving D10 out moves the interval of arm M minus default across
  zero (-0.012, -0.028 to +0.005). No Opus audit was run, by the maintainer's instruction on
  2026-10-23; a fresh Sonnet reviewer re-derived the headline contrasts from the stored arrays and
  found no difference. Trigger 5 did not fire (arm M differs from its nested arm MW by 0.001, and
  the interval includes zero).
  Triggers 1, 2, 6, and 7 did not fire.
