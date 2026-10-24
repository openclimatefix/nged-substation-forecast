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
status: planned
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
---

## Plan

The research default (Primary demand research default 3, still current in the research-default log
on 2026-10-25 after experiment 325 adopted nothing) is our research demand forecast for six Primary
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

### Results

To be written after the run.
