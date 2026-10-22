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
  rows (lead 15 to 39 hours) outside the hand-filtered switching masks, 2025-07-03 to 2026-06-30;
  the same on all leads and with the switching masks included, as labelled sensitivities
code: ml-experiments/00172-dp-error-budget-worst-days/
verdict: inconclusive
best_arm: >-
  Not applicable (descriptive). Arm D's worst 5% of days carry 41.7% of the excess loss (95%
  interval 33.5% to 46.9%), and 48.6% of those days carry a label against 14.5% of all days.
headline: >-
  The worst 5% of days carry 41.7% of the excess loss, above the 30% threshold, and 48.6% of them
  carry a label, just under half. The formal verdict is capped at inconclusive because the injected
  known-answer control recovered 89.2% of injected days against the 90% required.
lessons:
  - >-
    The error is concentrated, not spread thinly: the worst 10% of days carry 59.2% of the excess
    loss, and none of 1,000 random draws of days came near the worst 5%'s share (the best drew 8.5%).
  - >-
    Switching is the largest single label (40% of the worst days against 9.5% of all days), and
    keeping the hand-filtered switching rows in the scores lifts the share to 45.6% and the
    labelled fraction to 62.7%.
  - >-
    Cold snaps carry 5.3% of the excess, because the scored winter had 28 such days in 1,392
    substation-days. Post hoc, a cold day carries 2.6 times its share of the excess.
  - >-
    A known-answer control that injects 5 points fails at the substation with the heaviest tail
    (D16 recovered 52% of injected days), and passes at every substation at 10 points.
take_home_message: >-
  A small number of bad days carries a large share of the forecast's error: the worst 5% of days
  carry about 42% of the excess loss, and about half of those days coincide with a switching event,
  a holiday, or a cold snap. Switching events are the largest group, on days that carry 29% of the
  excess loss, so a forecast that improves only ordinary days leaves most of the loss in place.
  Cold snaps carry 5% of the excess in this mild scored winter, so this year cannot show whether a
  cold-night term pays. The formal verdict is inconclusive because one control narrowly failed.
advice:
  - >-
    Put fine-tuning effort first into days with a switching event: 29 of the 72 worst days overlap
    one, and they carry 29.2% of the excess loss (figure 2, label table).
  - >-
    Do not read the scored year as showing that cold nights are the main place the loss sits: cold
    days carry 5.3% of the excess, and the pre-registered cold-night ratio was 2.9 with an interval
    of -0.4 to 5.8 (cold-night bullet). A winter with more cold days is needed to test a
    cold-night term.
  - >-
    Look for a further label in the 37 unlabelled worst days before building another event rule:
    they fall in every month but February (unlabelled-days bullet).
  - >-
    Score the switching rows too when ranking days by loss, because dropping them hides the days
    that the detectors label (the sensitivity bullets; post hoc).
  - >-
    Size an injected-excess control to the heaviest-tailed substation: 5 points was too small for
    D16 and 10 points recovered every injected day (controls bullet; post hoc).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
pr: 1477
---

**A small number of bad days carries a large share of the forecast's error.** We asked whether the
forecast's error is spread over ordinary days or sits on a few days that the project already knows
how to label. We took the forecasts of experiment 155's arm D (a convex demand model with a Kalman
filter on recent errors, plus a winter night offset and a 14-day mean-error correction) for five
substations, D5, D10, D11, D14, and D16, over 2025-07-03 to 2026-06-30. D9 is scored beside them
and sits in no headline number. For each substation we scored every day, subtracted the
substation's median day, and ranked the days by this excess loss. The worst 5% of substation-days
(72 of 1,392) carry 41.7% of the total excess loss. Of those 72 days, 48.6% overlap a switching
event, a holiday, a cold snap, a flagged reading, or an Economy 7 step (the night-time step in
demand from off-peak storage heating), against 14.5% of all days. The main answer is that the error
is concentrated enough to target, and that about half of the worst days have no label yet.

**The pre-registered verdict is `inconclusive`.** The share clears its 30% threshold and the
labelled fraction misses its 50% threshold by 1.4 percentage points, which would give `partly
supported`. The decision rule caps the verdict at `inconclusive` when the injected known-answer
control recovers fewer than 90% of injected days, and it recovered 89.2%. The control's failure sits
at D16 alone (see Results). The labelled fraction is 35 of 72 days, so one more labelled day would
reach 50%.

Excess loss is a day's mean quantile score minus the substation's median day. The quantile score is
twice the mean pinball loss over 11 quantiles, in points of effective capacity (1% of a series' own
99th percentile of cleaned demand), over forecasts made 15 to 39 hours ahead. Days are local
calendar days. The rows exclude the hand-filtered switching events, as in experiment 155.

![Cumulative share of the excess loss against the fraction of days, worst first](fig1_cumulative_share.svg)

![Share of the worst days and of all days that carry each label](fig2_labels.svg)

![Four typical worst days, forecast against actual](fig3_worst_days.svg)

## Worst days

The 20 worst days of the 72 are below. The full list is in
[`worst_days.csv`](worst_days.csv).

| Substation | Local date | Day's score | Excess over median day | Labels |
|---|---|---|---|---|
| D5 | 2026-01-15 | 21.0 | +18.3 | switching |
| D16 | 2026-05-12 | 15.6 | +13.8 | switching |
| D5 | 2026-01-16 | 13.2 | +10.5 | switching |
| D16 | 2026-01-15 | 11.4 | +9.5 | switching |
| D16 | 2026-01-16 | 11.3 | +9.4 | switching |
| D10 | 2026-06-11 | 11.1 | +8.7 | switching |
| D16 | 2026-05-11 | 9.9 | +8.0 | switching |
| D10 | 2025-11-20 | 9.0 | +6.7 | cold snap |
| D16 | 2026-01-17 | 8.5 | +6.6 | none |
| D16 | 2026-05-14 | 8.1 | +6.3 | switching |
| D16 | 2026-05-16 | 8.1 | +6.3 | switching |
| D16 | 2026-04-20 | 7.9 | +6.0 | switching |
| D11 | 2026-01-04 | 7.8 | +5.8 | switching, cold snap |
| D10 | 2026-05-26 | 8.2 | +5.8 | none |
| D16 | 2026-05-15 | 7.6 | +5.8 | switching |
| D10 | 2025-12-01 | 7.6 | +5.3 | switching |
| D16 | 2026-01-19 | 6.5 | +4.6 | none |
| D5 | 2026-01-17 | 7.4 | +4.6 | switching |
| D16 | 2025-12-27 | 6.2 | +4.4 | switching, holiday |
| D5 | 2026-01-24 | 7.0 | +4.3 | switching |

## Results

- **Share of the excess in the worst 5% of days:** 41.7% (95% interval 33.5% to 46.9%, resampling the
  12 months of the scored year). By substation: D5 42.2%, D10 37.6%, D11 34.4%, D14 37.3%, D16 52.9%,
  and D9 35.7%. Leaving one substation out gives 38.1% to 43.3%. The worst 10% carry 59.2%, the worst
  25% carry 86.7%. The excess of a site is summed over its days with positive excess.
- **Labels on the worst 72 days (against all 1,392 days):** switching event 29 (9.5% of all days),
  holiday 8 (4.2%), cold snap 7 (2.0%), flagged reading 0 (0.0%), Economy 7 step 0 (0.9%), any
  label 35 (14.5%). A day can carry several labels. The labelled fraction by substation is 27% (D14)
  to 71% (D16).
- **Share of the total excess on labelled days:** any label 35.8%, switching 29.2%, holiday 7.2%,
  cold snap 5.3%.
- **Unlabelled worst days:** 37 of 72, spread across 11 of the 12 months (every month but February).
  We did not look for a further label in them. January 2026 holds 19 of the 72 worst days.
- **Cold nights.** The pre-registered ratio is the cold nights' share of the row-level excess over
  their share of all rows. It is 2.88 with a 95% interval of -0.42 to 5.81, so the interval is too
  wide to say that the loss sits on cold nights. Post hoc, two stabler ratios agree on the
  direction. Cold days (day mean ERA5 temperature below 2 C) carry 2.6 times their share of the day
  excess (95% interval 1.9 to 3.6), and the mean night score on cold days is 1.7 times that on other
  winter days (1.2 to 2.0). The cold days are only 28 of 1,392 days and carry 5.3% of the excess. At
  3 C and 5 C the day ratios fall to 1.7 and 1.4.
- **Controls.** The best of 1,000 random draws of the same number of days carried 8.5% of the excess
  (mean 3.1%), so the real share of 41.7% is not an accident of the ranking. The pre-registered
  injected control adds 5 points to 10 random days per substation (the plan first said 18 days, and
  was corrected to 10 before any score was read, because 18 exceeds the 14 to 15 slots of the worst
  5%). It put 89.2% of the injected days in the worst 5%, below the required 90%. By substation it
  recovered 94% (D5), 100% (D10, D11, D14), and 52% (D16). D16 has the heaviest tail, so its
  worst-5% slots are already taken by real days with more than 5 points of excess. Post hoc, 10
  injected points recovered 100% at every substation.

## Checks

- **Median forecast minus mean forecast, per site:** 0.0 points at all six substations, and the
  largest absolute difference at any row is 0.0. Arm D's median forecast equals its mean forecast,
  so a quantile spread that shifts the median away from the mean cannot explain the loss.
- **Mean forecast's error beside the quantile score:** ranking days by the mean forecast's mean
  absolute error gives a worst-5% share of 36.7% (41.7% for the quantile score), and 76 of the 87
  worst pinball days (six substations) are among the worst mean-error days. Both measures point
  the same way.
- **Audit triggers:** trigger 1 (a control fails) fired, because of the injected control. An Opus
  audit ran, and the "Audit" section below gives its findings and their triage. Trigger 2 was also considered, because
  the labelled fraction (48.6%) came in just under the predicted half: the sensitivity with the
  switching rows kept (62.7%) tests the explanation that the scored rows hide switching days.
  Dropping one substation moves the share by at most 3.6 points.

## Sensitivities (post hoc)

- **Switching rows kept:** the share rises to 45.6% and 62.7% of the worst days are labelled.
- **All leads (0 to 168 hours) instead of the D+1 band:** the share is 44.9% and 47.8% of the worst
  days are labelled.

## Audit

An Opus agent re-derived the numbers from the stored forecasts with its own code. Its report is
private (`data/ml-experiments/briefs/e172_audit_report.md`). No headline conclusion changed.

- **Reproduced:** the worst-5% share (41.73%), the per-site shares, the leave-one-out range, and the
  month bootstrap interval match. The audit's day table matches ours on every day, and its labels
  for switching, cold snap, holiday, flagged reading, and any label agree on all 1,392 days.
  Economy 7 was not re-derived.
- **The share is robust:** 39.6% to 42.5% across 18 day definitions (local or UTC days, start or end
  dating, 1 to 48 minimum rows). Dropping January 2026 gives 38.0%.
- **The labelled fraction sits on the 50% bar:** a one-day pad around switching intervals, or UTC
  dating, gives 52.8%. Padding both switching and holiday days gives 56.9%. RO-NC intervals alone
  give 45.8%. A different window would therefore read as `supported` where ours reads `partly
  supported`, so "about half" is the finding.
- **The switching label is not an artefact:** shifting the switching days by 7 to 28 days labels 2
  to 6 of the worst days, against 29 for the real label. One D10 interval runs 60 days
  (2025-12-12 to 2026-02-10) and covers 17% of D10's days.
- **The control failure is explained:** D16 has 9 days with more than 5 points of excess, and 14
  worst-day slots, so a 5-point injection must beat the fifth-largest real excess (6.6 points) and
  about half of the injected days land. Ranking by excess cannot misorder days, so the control tests
  the size of D16's tail and not the method. The pre-registered cap stays.
- **Fixed:** the detector stamps mark the end of a half-hour and were dated as starts. Correcting
  them changed no day and no number. The flagged-reading day is dated by the reading's own
  timestamp and was not changed.

## Method and caveats

- **Row set:** `fold_a_daily_gap_v1`, the scored year of experiment 155, arm D, `d_scored_*`
  forecasts. Days with fewer than 24 rows in the D+1 band are dropped, which removes about 13% of D5's
  days (252 of 287 remain) because many of its switching rows are excluded, and ranks D5's tail on
  fewer days. No forecasting model was fitted. The run used one CPU core and about a minute.
- **Dating:** a row's day is the local date of the half-hour's start, as in the holiday flag of
  `studies.dp_calendar`.
- **Switching label:** the day overlaps an interval of the refit-offset neighbour-count detector
  (RO-NC, setting R0, [experiment 147](../00147-causal-live-end-rule-for-ro-nc/result.md)) or a pair
  alarm ([experiment 156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md),
  window 8) on the substation's series. The intervals use hindsight end rules, as in a lab test.
  The alarm table may cover only part of the year. The earlier detectors of
  [experiment 141](../00141-switching-detector-fed-into-gam-k/result.md) are not used. The D8 drops
  of
  [experiment 159](../00159-switching-d8-multi-day-drops-recurrence-and-report/result.md) do not
  apply, because D8 is not among the six substations.
- **Flagged reading:** the cleaned power table flags a reading on the day, or the day has fewer than
  42 of 48 readings. Only D9 has such a day among the worst days; none of the decision substations
  does.
- **Cold snap:** the day's mean ERA5 temperature (the ECMWF reanalysis of past weather) is below 2 C.
  ERA5 labels days after the fact and is not a forecast input.
- **Economy 7 step:** a winter day on which the four half-hours starting 00:00 to 01:30 UTC carry at
  least 35% of the day's loss above the median. The 35% is an assumption fixed before any score was
  read, and fits 0.9% of days. A different definition could label more days.
- **Holiday:** a holiday period of `studies.dp_calendar`.
- **One year:** the scored year has one winter, and 19 of the 72 worst days fall in January 2026.
  The shares describe that year.

## What to try next

- Look for a label for the 37 unlabelled worst days, starting with the D16 days of January 2026.
- Test a cold-night term on a winter with more cold days, since this year cannot rank it.
