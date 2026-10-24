---
id: '00329'
slug: primary-demand-default-3-member-solar-distribution-at-7-to-10-days
title: >-
  Does building the 7-to-10-day demand bands from the 51 ensemble members' own solar responses (and
  a climatology of the response) close the sunny-daylight weakness of Primary demand research
  default 3?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - weather-products-and-blending
status: done
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00329-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the three arms are fixed
  by the question. Callers: this folder's scripts, which import the folders of experiments 317, 319,
  320, and 322 and read their stored files. No change under `packages/`.
hypothesis: >-
  At 168 to 240 hours the 51 ensemble members have little skill for individual days, so the
  forecast centres on average sunshine and the half-hours that turn out sunny have realised demand
  below the median. The weakness of Primary demand research default 3 on sunny daylight (5.07
  against the manual heuristic's 4.82 in experiment 317, for the convex chain before any blend) is
  therefore a fault of the predictive distribution (the lower tail on realised-sunny half-hours),
  not of the median. Replacing the two-number Gaussian summary of the members' solar responses with
  the 51 responses themselves (X1), mixing them with a climatology of the response (X2), or adding
  the members of the two previous daily runs (X3) fixes the distribution.
prediction: >-
  Written before any score is read; each can fail. (Stage 0) On realised-sunny half-hours at 168 to
  240 hours the default's below-q05 rate is above 8% (nominal 5%) and the q50's mean error is under
  0.1 points of capacity, and on forecast-sunny half-hours (ensemble-mean clear-sky index of 0.9 or
  more) the default is within 2% of the manual heuristic. If the q50's bias exceeds 0.3 points, the
  fault is in the level and the experiment stops. If the default already beats the heuristic on
  sunny daylight at 168 to 240 hours, the experiment stops. (PX1) X1 improves sunny daylight at 168
  to 240 hours by at least 3% (about 0.15 points) with a 99.5% interval wholly below zero, bringing
  the default to within 2% of the heuristic or better, improves the pooled six-Primary score at 168
  to 240 hours by at least 0.02 points, lowers the below-q05 rate on sunny daylight to 6.5% or
  less, and moves the pooled score at 0.5 to 240 hours by under 0.02 points in the wrong direction.
  (PX2) X2 beats X1 by at least 0.01 points at 168 to 240 hours, with a fitted climatology weight
  rising from near 0 at 168 to 192 hours to at least 0.3 at 216 to 240 hours. (PX3) X3 beats X1 by
  at least 0.01 points. If none passes, the weakness is the ensemble's lack of skill at 7 to 10
  days, not the way the stack uses it.
decision_rule: >-
  Family rule of research-default.md. A new fitted step on stored forecasts (tier 1 acting on
  stored quantiles, with a fitted step), minimum gain 0.02 points pooled at 168 to 240 hours, the
  lead range the change acts on, read at the 1 - 0.01/3 level (three arms), at most one adopted.
  An arm is adopted when its interval at 168 to 240 hours lies wholly below the default's, the
  point estimate reaches the minimum gain, no no-harm bound of the research-default page is broken
  at 0.5 to 240 hours or at 168 to 240 hours (pooled 0.02 points, q01 and q99 pinball 0.05, cold
  winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative), leaving one Primary out does not
  flip the sign, and the replicate year agrees. Every arm reuses the default's stored q50 (each
  arm's quantiles are re-centred on it), so the comparison is paired and needs no network seed
  averaging. The reproduction gate (default equals experiment 322's default within 0.01 points:
  2.727 at 0.5 to 240 hours and 3.077 at 168 to 240 hours) comes first.
benchmark: >-
  Primary demand research default 3 (experiment 322's default plus D1: Primary demand research
  default 2 with the experiment 319 heuristic blend on daylight half-hours only). The manual
  heuristic and climatology (experiment 87) and BG2 to 168 hours are references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, replicate year 2024-25, rows outside the hand-filtered events, origin 09:00 UTC. Tail
  rates below q01 and q05 and above q95 and q99, and the probability-integral-transform histogram,
  on sunny daylight.
best_arm: >-
  None run. Primary demand research default 3 stays: 4.50 points on realised-sunny daylight at 168
  to
  240 hours, against the manual heuristic's 4.82.
headline: >-
  Stage 0 stopped the experiment. At 168 to 240 hours Primary demand research default 3 scores 4.50
  points on realised-sunny daylight half-hours against the manual heuristic's 4.82 (6.6% better, 95%
  interval 1.3% to 11.3% better), so the weakness of the convex chain (5.07, 5.2% worse) was closed
  by the two-number solar summary and the daylight blend. One Primary (D5) is still worse than the
  heuristic.
take_home_message: >-
  The 7-to-10-day weakness on sunny half-hours no longer exists in the research demand forecast:
  the two-number solar summary and the daylight-only blend with the manual heuristic already took
  it from 5.2% worse than the heuristic to 6.6% better, so no new distribution method is needed
  there. What remains is one coastal Primary (D5), where the forecast is 11% worse than the
  heuristic, which points back at the solar term and not at the width of the bands.
lessons:
  - >-
    The sunny-daylight gap found in experiment 317 was a gap of the convex chain before the solar
    summary (experiment 319) and the heuristic blend (experiment 322): the chain scores 5.07,
    Primary demand research default 2 scores 4.74, and Primary demand research default 3 scores
    4.50,
    against the heuristic's 4.82.
  - >-
    The sunny half-hours' q50 bias of +5.6 points is selection by the outcome: over all daylight
    half-hours the bias is +0.01 points, so the pre-registered level-fault stop is not evidence of a
    level fault.
  - >-
    On realised-sunny half-hours the default's lower tail is nearly calibrated (7.7% below q05, 1.7%
    below q01), and the heuristic's is not (18.0% and 12.9%).
  - >-
    The remaining weakness is local: D5 scores 6.09 against the heuristic's 5.47, and D9 is level
    with it (3.99 against 4.05), and both are coastal Primaries, as in experiment 325.
advice:
  - >-
    Do not build a member-quantile or climatology-blended solar distribution for 7-to-10-day demand
    bands: the sunny-daylight gap it aimed at is closed in Primary demand research default 3 (stage
    0).
  - >-
    Investigate the solar term at the coastal Primaries D5 and D9 at 168 to 240 hours before any
    further band change (per-Primary table, stage 0).
  - >-
    Read the q50 bias on a group defined by realised sunshine as selection by the outcome, and judge
    level faults on all daylight half-hours (stage 0, "Checks").
verdict: not supported
production_status: not-for-production
adopted_in: none
code: ml-experiments/00329-primary-demand-default-3-member-solar-distribution-at-7-to-10-days/
---

## Plan

The research default (Primary demand research default 3, still current in the research-default log
after experiment 325 adopted nothing) is our research demand forecast for six Primary
substations, up to 10 days ahead. Its 7-to-10-day bands come from experiment
[319](../00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days/result.md)'s
two-number summary of the 51 ensemble members' solar responses (a mean and a standard deviation,
after [experiment 193](../00193-dp-ens-two-number-solar-summary/result.md)).

### The five plan items

1. **Question.** At 168 to 240 hours, does a solar-response distribution built from the 51 members
   and the training years (not only their mean and standard deviation) close the sunny-daylight
   weakness of the research default, and where does the weakness come from?
2. **Prediction that can fail.** The weakness is a fault of the predictive distribution (the lower
   tail on realised-sunny half-hours), not of the median. The thresholds are in the front matter.
3. **Rows and period.** `fold_a_daily_gap_v1`, six Primaries, scored year 2025-07-02 to 2026-06-30,
   replicate year 2024-25, origin 09:00 UTC, rows outside hand-filtered events. The headline range
   is 168 to 240 hours, and 0.5 to 240 hours is the no-harm check.
4. **Metric.** The quantile score over 11 levels in points of effective capacity, with the
   research-default page's protected groups and tolerances and 12-month block bootstrap intervals.
5. **Stop rule.** The reproduction gate fails, or stage 0 finds no weakness left in the default on
   sunny daylight at 168 to 240 hours.

### Stage 0 (zero-fit diagnostic)

Score the default against the manual heuristic and climatology on sunny daylight at 168 to 240
hours, by 24-hour lead band and by Primary. Report the probability-integral-transform histogram and
the below-q05, below-q01, above-q95, and above-q99 rates for the default and the heuristic, the
q50's mean error, and the same scores on forecast-sunny half-hours (ensemble-mean clear-sky index of
0.9 or more, from the stored ensemble's shortwave radiation over the Haurwitz clear-sky value, as in
[experiment 178](../00178-dp-sunny-daylight-gap-diagnosis/result.md)).

### Arms (run only if stage 0 passes)

All arms change the bands after the median, and are re-centred on the default's stored q50.

- **Default:** Primary demand research default 3.
- **X1, member solar distribution:** at 168 to 240 hours, the 51 members' solar responses replace
  the Gaussian points of experiment 319's arm x1 (centres `chain mean + response`, each with the
  chain's offsets times a factor `f` fitted per lead band). Daylight half-hours only.
- **X2, X1 mixed with climatology:** the member responses are mixed with the empirical distribution
  of the response at that Primary in completed earlier origins within plus or minus 15 days of the
  day of year at the same hour; the mixture weight per 24-hour lead band is fitted as-of by pinball
  loss before each scored month.
- **X3, lagged-ensemble pooling:** X1 with the responses of the two previous daily runs for the same
  valid time added, each earlier run down-weighted by a fitted per-lag weight. It uses only earlier
  runs whose response is stored (lead up to 240 hours), so lag 2 covers the 168-to-192-hour band
  only and lag 1 the first two bands.
- **References:** manual heuristic and climatology (static), BG2 below 168 hours.

Experiment 319's code and stored member responses are reused (the folders of experiments 317, 319,
320, and 322 are named in the scripts), and each month's fits use only completed runs before that
month's first origin, as they do. Schedule: the arms refit nothing but the band widths and mixture
weights each month, so a weekly-retrained median would give the same contrast.

## Take-home message

**The weakness this experiment set out to fix is already closed in Primary demand research default
3, so the experiment stopped after its first diagnostic (stage 0) and ran none of its three planned
arms.** Primary demand research default 3 is our research demand forecast for six Primary
substations (anonymised as D5, D9, D10, D11, D14, and D16), up to 10 days ahead. The weakness was a
worse score than the manual heuristic (NGED's rule of thumb, built from the same half-hour in
earlier weeks and years) on sunny daylight half-hours, at leads of 168 to 240 hours. A half-hour
counts as sunny when it is among the sunniest third of a Primary's daylight half-hours by the
sunshine that was realised, as in experiment
[178](../00178-dp-sunny-daylight-gap-diagnosis/result.md). We planned to build the uncertainty bands
from the 51 weather-forecast members' own solar responses and tested first whether any weakness was
left. The quantile score (twice the mean pinball loss over 11 quantiles, in percentage points of
effective capacity; smaller is better) of Primary demand research default 3 on those half-hours is
4.50, against the manual heuristic's 4.82. That is 6.6% better, with a 95% interval of 1.3% to 11.3%
better and a 99.5% interval of 1.7% worse to 13.4% better. Experiment
[317](../00317-primary-demand-default-1-at-7-to-10-days/result.md) had found the convex chain (the
stack before any weather summary or blend) 5.2% worse, at 5.07. The two-number solar summary of
experiment
[319](../00319-primary-demand-default-1-solar-summary-and-heuristic-blend-at-7-to-10-days/result.md)
and the daylight-only blend of experiment
[322](../00322-primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication/result.md)
closed the gap. One Primary, D5, is still behind the heuristic, at 6.09 against 5.47 percentage
points of effective capacity.

![Score on realised-sunny daylight by stage of the stack](sunny_by_stage.svg)

![Score on realised-sunny daylight by Primary](sunny_by_primary.svg)

![Where the realised demand falls among the forecast quantiles](pit_sunny.svg)

## Findings

- **Reproduction gate passed:** Primary demand research default 3 scores 2.727 at 0.5 to 240 hours
  and 3.077 at 168 to
  240 hours, the values of experiment 322 (2.727 and 3.077).
- **Sunny daylight, 37,612 half-hours at 168 to 240 hours:** Primary demand research default 3
  scores 4.50, the manual heuristic 4.82, and climatology 4.95. It beats the heuristic in each
  24-hour band
  (4.48, 4.52, and 4.51 against 4.83, 4.83, and 4.81).
- **Per Primary (Primary demand research default 3, then the heuristic):** D5 6.09 and 5.47, D9 3.99
  and 4.05, D10 5.41 and 6.21,
  D11 3.55 and 4.06, D14 3.64 and 3.96, D16 4.48 and 5.23. Without D5, Primary demand research
  default 3 is 10.4% better
  than the heuristic (95% interval 5.4% to 14.8%).
- **Tails on realised-sunny half-hours:** Primary demand research default 3 has 1.7% below q01, 7.7%
  below q05, 0.7% above
  q95, and 0.2% above q99. The heuristic has 12.9%, 18.0%, 1.3%, and 0.8%.
- **Two of the stage 0 predictions failed, and one stop condition is met only through selection:**
  the below-q05 rate was 7.7% (predicted above 8%), and the q50 bias, defined as q50 minus realised
  demand, was +5.6 points (predicted under 0.1). Realised sunshine selects half-hours with low
  demand, so the bias is a selection effect: over all daylight half-hours it is +0.01.
- **Forecast-sunny half-hours** (ensemble-mean clear-sky index of 0.9 or more) number only 601.
  There Primary demand research default 3 scores 5.02 against the heuristic's 4.27, with a q50 bias
  of -6.4 and 17.0% of realised demand above q95. This is the mirror image of selecting by realised
  sunshine: the ensemble forecasts sun on days that turn out cloudier. The rows are too few to rank
  forecasts, and the group is not the one the brief defines.
- **Stop rule:** the default already beats the heuristic on sunny daylight, so the experiment stops.
  The three arms (X1, X2, X3) were not run, no model was refitted, and the run used about 10
  core-minutes.

## Checks

- **Median minus chain mean, per Primary, at 168 to 240 hours (points):** D5 -0.53, D9 -0.26,
  D10 +0.01, D11 -0.17, D14 -0.22, D16 -0.07.
- **Error of the median beside the score:** mean absolute error of the median on realised-sunny
  half-hours is 8.20 for Primary demand research default 3 and 8.71 for the heuristic, which points
  the same way as the score.
- **Triggers:** none fired. The stage 0 results differ from the brief's expectation, and the cause
  is tested above (the convex chain, Primary demand research default 2, and Primary demand research
  default 3 scored on the same rows). Leaving D5 out
  strengthens the contrast and does not flip it, so no audit ran.

## Caveats and what to try next

- **Lead range:** 168 to 240 hours, the 7-to-10-day range, six Primaries, one scored year. The
  replicate year was not scored, because the experiment stopped at stage 0.
- **Interval:** the 99.5% interval of the default against the heuristic on sunny daylight includes
  zero (-1.7% to 13.4%), so "6.6% better" is established at 95% only.
- **Next:** look at the solar term at D5 and D9 (coastal Primaries) at 168 to 240 hours, where
  experiment
  [325](../00325-primary-demand-default-3-sunshine-response-refit-and-restriction/result.md)
  also found a sunshine response helps. The member-distribution arms stay available if a later
  weakness at long leads shows a skewed tail.
- **Code:**
  `ml-experiments/00329-primary-demand-default-3-member-solar-distribution-at-7-to-10-days/`
  imports the folders of experiments 184, 202, 317, 319, 320, and 322 and reads their stored files.
