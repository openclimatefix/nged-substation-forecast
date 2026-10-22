---
id: '00172'
slug: dp-error-budget-worst-days
part: BG-EB
title: >-
  Is the forecast's error spread over ordinary days, or does a small share of labelled days carry
  most of it?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- switching-labels-in-the-forecast
- calibration-and-uncertainty
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00172-dp-error-budget-worst-days/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (which lead band, which day boundary, how a label is defined), and this plan pre-registers one of
  each and reports the alternatives as labelled sensitivities. Callers: this folder's scripts, which
  read experiment 155's stored arm D forecasts and the stored outputs of experiments 147, 156, and
  159, and call no new modelling code. No model is fitted.
hypothesis: >-
  The error of the stage 2 assembly (experiment 155's arm D) is concentrated on a small number of
  days that the project already knows how to label. The worst 5% of days, ranked by their pinball
  loss above the site's median day, carry at least 30% of the total excess loss, and at least half
  of those days carry one of six labels: a switching event, a metering error or flagged reading, a
  cold snap, an Economy 7 step, a holiday, or a combination.
prediction: >-
  Written before any score is read. The worst 5% of days carry 25% to 45% of the total excess loss
  (a flat distribution would give 5%). At least half of them carry a label: switching events and
  flagged readings are expected to dominate at D9 and D14, cold snaps at D5 and D10. Cold nights
  (day mean ERA5 temperature below 2 C, local hours 18:00 to 06:00) carry a larger share of the
  excess loss than their share of nights, by at least a factor of 1.5.
decision_rule: >-
  Per site, the excess loss of a local day is the day's mean quantile score (twice the mean pinball
  loss over the 11 quantiles, in points of effective capacity, over the D+1 rows with lead 15 to 39
  hours, so each valid half-hour appears once) minus the median of the site's daily scores. Days are
  ranked within a site, the worst 5% of days of each site (rounded up) are taken, and their share is
  the sum of their excess loss divided by the sum of the positive excess loss over all days of the
  site, pooled by summing across the five decision sites D5, D10, D11, D14, and D16 (D9 beside
  them in the tables). The hypothesis is `supported` when the pooled share is at least 30% and at
  least half of the worst days carry a label, `partly supported` when exactly one of the two holds,
  and `refuted` when neither holds. The 95% interval of the share is a bootstrap over the 12 calendar
  months of the scored year (20,000 resamples). The cold-night question is answered by the ratio of
  the cold nights' share of the excess loss to their share of nights, with a month-resampled
  interval; a ratio whose interval lies wholly above 1 says the loss sits on cold nights. A random
  control draws 1,000 sets of days of the same number per site and reports where the real share
  falls in the draws. The verdict is capped at `inconclusive` if the control, which injects a
  known excess of 5 points of loss into 10 random days per site, does not put those days in the
  worst 5% of days at least 90% of the time.
benchmark: >-
  Randomly chosen days (the 1,000-draw control above), and the flat-loss share of 5%
row_set_id: fold_a_daily_gap_v1
metric: >-
  Daily quantile score above the site's median day, in points of effective capacity, over the D+1
  rows (lead 15 to 39 hours) outside the hand-filtered switching masks, 2025-07-02 to 2026-06-29;
  the same on all leads and with the switching masks included, as labelled sensitivities
code: ml-experiments/00172-dp-error-budget-worst-days/
---

# Experiment 172: where the forecast's error sits

*Planned; the write-up replaces this section when the scores exist.* The experiment tallies the
pinball loss of experiment 155's arm D by day, ranks the days, and labels the worst 5% with the
detections the project already has. The labels are:

- **Switching event:** the day overlaps an interval of the refit-offset neighbour-count detector
  (RO-NC, [experiment 147](../00147-causal-live-end-rule-for-ro-nc/result.md), setting R0) or a
  pair alarm ([experiment 156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md),
  window of 8 half-hours) on the site's series. The D8 multi-day drops of
  [experiment 159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md) apply to
  D8 only, which is not one of the six sites.
- **Metering error or flagged reading:** the cleaned power table flags at least one reading of the
  site on the day, or at least 6 of the day's 48 half-hours have no reading.
- **Cold snap:** the day's mean ERA5 temperature is below 2 C. ERA5 is used to label days after the
  fact and is not a forecast input, so the availability lag does not matter here.
- **Economy 7 step:** a winter day (November to March) on which the four half-hours from 00:00 to
  01:30 UTC carry at least 35% of the day's loss above the median (they hold 8% of a flat day).
  The 35% is an assumption fixed here, before any score is read.
- **Holiday:** the day falls in a holiday period of `studies.dp_calendar`.

The code is a copy of the labelling code of experiments 155 and 170, reading their stored forecasts.
