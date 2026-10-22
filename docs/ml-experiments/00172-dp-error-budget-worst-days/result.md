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
status: done
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
verdict: partly supported
best_arm: >-
  Not applicable (descriptive). Arm D's worst 5% of days carry 39.9% of the excess loss (95%
  interval 33.3% to 44.2%), and 48.6% of those days carry a label against 15.2% of all days.
headline: >-
  The worst 5% of days carry 39.9% of the excess loss, above the 30% threshold. Of those days, 48.6%
  carry one of the project's labels, just under the pre-registered half. Switching events label 28
  of 72 days, and cold snaps carry 5.5% of the excess.
lessons:
  - >-
    The error is concentrated, not spread thinly: the worst 10% of days carry 57.7% of the excess
    loss, and no random choice of days came near the worst 5%'s share (best of 1,000 draws: 8.4%).
  - >-
    Switching is the largest single label (39% of the worst days against 9% of all days), and
    keeping the hand-filtered switching rows in the scores lifts the share to 45.4% and the
    labelled fraction to 64.0%.
  - >-
    Cold snaps are over-represented per day (2.7 times) but carry only 5.5% of the excess, because
    the scored winter had 28 such days in 1,391 substation-days.
take_home_message: >-
  A small number of bad days carries a large share of the forecast's error: the worst 5% of days
  carry about 40% of the excess loss. About half of those days coincide with a switching event, a
  holiday, or a cold snap. Switching events are the largest group, on days that carry 26.1% of the
  excess loss, so a better mean model must find the rest. Cold nights
  are not where most of the loss sits in this mild scored winter, so the cold-night drift idea of
  experiment 170 has little to gain here.
advice:
  - >-
    Spend fine-tuning effort on days with a switching event first: they label 28 of the 72 worst
    days and carry 26.1% of the excess loss (figure 2, label table).
  - >-
    Do not treat cold nights as the main place the loss sits for the scored year: cold snaps carry
    5.5% of the excess, and a colder winter is needed to test a cold-night term (cold-night section).
  - >-
    Expect about half of the worst days to carry no label, spread over every month but February,
    and look for a further label in them before building another
    event rule (unlabelled-days bullet).
  - >-
    Score switching rows as well as the unflagged rows when ranking days by loss, because dropping
    them hides the days that the detectors label (sensitivity bullets).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

**A small number of bad days carries a large share of the forecast's error.** We took the forecasts
of experiment 155's arm D, the convex demand model with a Kalman filter on recent errors and two
corrections, for five substations (D5, D10, D11, D14, and D16) over 2025-07-02 to 2026-06-29. For
each substation we scored every day, subtracted the substation's median day, and ranked the days.
The worst 5% of days (72 days of 1,391) carry 39.9% of the total excess loss, and 48.6% of them
coincide with something the project already detects. That is the main answer: the error is
concentrated enough to target, but about half of the worst days have no label yet. Verdict: partly
supported, because the share clears its 30% threshold and the labelled fraction misses its 50%
threshold by 1.4 percentage points.

Excess loss is a day's mean quantile score (twice the mean pinball loss over 11 quantiles, in
points of effective capacity, over forecasts made 15 to 39 hours ahead) minus the substation's
median day. Days are local calendar days. The rows exclude the hand-filtered switching events, as in
experiment 155. D9 is scored beside the five and sits in no headline number.

![Cumulative share of the excess loss against the fraction of days, worst first](fig1_cumulative_share.svg)

![Share of the worst days and of all days that carry each label](fig2_labels.svg)

![Four typical worst days, forecast against actual](fig3_worst_days.svg)

## Worst days

The 20 worst days of the 72 are below. The full list is in
[`worst_days.csv`](worst_days.csv).

| Substation | Local date | Day's score | Excess over median day | Labels |
|---|---|---|---|---|
| D16 | 2026-05-12 | 15.6 | +13.8 | switching |
| D5 | 2026-01-16 | 13.4 | +10.6 | switching |
| D16 | 2026-01-16 | 11.2 | +9.3 | switching |
| D10 | 2026-06-11 | 11.0 | +8.6 | switching |
| D16 | 2026-05-11 | 9.5 | +7.6 | switching |
| D10 | 2025-11-20 | 9.1 | +6.8 | cold snap |
| D16 | 2026-01-17 | 8.5 | +6.6 | none |
| D16 | 2026-05-16 | 8.2 | +6.3 | switching |
| D16 | 2026-05-14 | 8.1 | +6.3 | switching |
| D16 | 2026-04-20 | 7.9 | +6.1 | switching |
| D11 | 2026-01-04 | 7.9 | +5.9 | switching, cold snap |
| D10 | 2026-05-26 | 8.1 | +5.8 | holiday |
| D16 | 2026-05-15 | 7.6 | +5.7 | switching |
| D10 | 2025-12-01 | 7.5 | +5.2 | switching |
| D16 | 2026-01-19 | 6.4 | +4.6 | none |
| D5 | 2026-01-17 | 7.2 | +4.5 | switching |
| D5 | 2026-01-24 | 7.1 | +4.3 | switching |
| D16 | 2025-12-27 | 6.1 | +4.2 | switching, holiday |
| D11 | 2026-01-02 | 6.1 | +4.2 | cold snap, holiday |
| D10 | 2026-01-02 | 6.4 | +4.1 | switching, cold snap, holiday |

## Results

- **Share of the excess in the worst 5% of days:** 39.9% (95% interval 33.3% to 44.2%, resampling the
  12 months of the scored year). By substation: D5 36.2%, D10 37.6%, D11 33.8%, D14 37.4%, D16 51.7%,
  and D9 35.3%. Leaving one substation out gives 36.3% to 41.3%. The worst 10% carry 57.7%, the worst
  25% carry 86.2%.
- **Labels on the worst 72 days (against all 1,391 days):** switching event 28 (9.3% of all days),
  holiday 9 (5.2%), cold snap 7 (2.0%), flagged reading 0 (0.0%), Economy 7 step 0 (0.9%), any
  label 35 (15.2%). A day can carry several labels. The labelled fraction by substation is 27% (D14)
  to 71% (D16).
- **Share of the total excess on labelled days:** any label 35.1%, switching 26.1%, holiday 9.8%,
  cold snap 5.5%.
- **Unlabelled worst days:** 37 of 72, spread across 11 of the 12 months (every month but February).
  We did not look for a further label in them.
- **Cold nights.** The pre-registered ratio (cold nights' share of the row-level excess over their
  share of rows) is 2.99 with a 95% interval of 0.12 to 5.88, so it cannot say that the loss sits
  on cold nights. Post hoc, two stabler ratios agree on the direction. Cold days (day mean ERA5
  temperature below 2 C) carry 2.7 times their share of the day excess (95% interval 1.9 to 3.7),
  and the mean night score on cold days is 1.7 times that on other winter days (1.2 to 1.9). The
  cold days are only 28 of 1,391 days and carry 5.5% of the excess. At 3 C and 5 C the day ratios
  fall to 1.6 and 1.4.
- **Controls.** The best of 1,000 random draws of the same number of days carried 8.4% of the excess
  (mean 3.1%), so the real share of 39.9% is not an accident of the ranking. A known excess of 5
  points injected into 10 random days per substation put 92.2% of those days in the worst 5%
  (needed 90%), so the control passed.

## Checks

- **Median forecast minus mean forecast, per site:** 0.0 points at all six substations. Arm D's
  median forecast equals its mean forecast at every row, so a spread that shifts the forecast cannot
  explain the loss.
- **Mean forecast's error beside the quantile score:** ranking days by the mean forecast's mean
  absolute error gives a worst-5% share of 35.3% (39.9% for the quantile score), and 77 of the 87
  worst pinball days (six substations) are among the worst mean-error days. Both measures point the
  same way.
- **Audit triggers:** none fired. Trigger 2 was considered, because the labelled fraction came in
  at 48.6% against a predicted half. The explanation is tested in the sensitivities below: the
  scored rows exclude the hand-filtered switching events, so the days they label are partly hidden.
  Dropping one substation moves the share by at most 3.6 points and never flips the verdict. No
  Opus audit ran.

## Sensitivities (post hoc)

- **Switching rows kept:** the share rises to 45.4% and 64.0% of the worst days are labelled.
- **All leads (0 to 168 hours) instead of the D+1 band:** the share is 45.0% and 48.9% of the worst
  days are labelled.

## Method and caveats

- **Row set:** `fold_a_daily_gap_v1`, the scored year of experiment 155, arm D, `d_scored_*`
  forecasts. Days with fewer than 24 rows in the D+1 band are dropped. No model was fitted. The run
  used one CPU core and about a minute.
- **Switching label:** the day overlaps an interval of the refit-offset neighbour-count detector
  (RO-NC, setting R0, [experiment 147](../00147-causal-live-end-rule-for-ro-nc/result.md)) or a pair
  alarm ([experiment 156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md),
  window 8) on the substation's series. The intervals use hindsight end rules, as in a lab test,
  and the alarm table may cover only part of the year. The detectors of
  [experiment 141](../00141-switching-detector-fed-into-gam-k/result.md) are the earlier versions of
  the same detector. The D8 drops of
  [experiment 159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md) do not
  apply, because D8 is not among the six substations.
- **Flagged reading:** the cleaned power table flags a reading on the day, or the day has fewer than
  42 of 48 readings. Only D9 has such a day among the worst days; none of the decision substations
  does.
- **Cold snap:** the day's mean ERA5 temperature is below 2 C. ERA5 labels days after the fact and
  is not a forecast input.
- **Economy 7 step:** a winter day on which the half-hours 00:00 to 01:30 UTC carry at least 35% of
  the day's loss above the median. The 35% is an assumption fixed before any score was read, and
  fits 0.9% of days. A different definition could label more days.
- **Holiday:** a holiday period of `studies.dp_calendar`.
- **One year:** the scored year has one winter, and 17 of the 72 worst days fall in January 2026.
  The shares describe that year.

## What to try next

- Look for a label for the 37 unlabelled worst days, starting with the D16 days of January 2026.
- Test a cold-night term on a winter with more cold days, since this year cannot rank it.
