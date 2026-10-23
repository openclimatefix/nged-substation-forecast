---
id: "00308"
slug: lsc-two-forecast-gap-as-a-detector
title: >-
  Is the gap between a no-lag forecast and a forecast with lags a detector of LSC events?
theme: lsc-detectors
also_relevant_to: []
status: done
size: >-
  medium. Stored: private score tables under data/ml-experiments/00308-lsc-two-forecast-gap-as-a-detector/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible for the alarm episode, the match rule, and the neighbour check, so each is fixed
  below before any score is read. Callers: this folder's scripts only, which read the stored
  harness of experiment 258 and the stored worlds of experiment 141. Asset graph: not touched.
  The maintainer's brief limits the work to Sonnet reviewers.
hypothesis: >-
  The gap between the forecast with lags (K0, the Kalman filter on recent power) and the forecast
  without lags (GAM_ONLY, which forecasts the normal running arrangement) at a forecast origin
  already works as a load shift detector. NGED cannot see the true switching state, so this gap is
  the only signal it has. This is part 1 of the brief
  briefs/sw_lsc_two_forecasts_divergence_and_selection_brief.md.
prediction: >-
  Written before any score is read. Variant (a), the plain gap, or variant (b), the gap with the
  neighbour check, matches at least as many of the 35 real load shifts over 1 day as the learned
  score (LR_b20) at LR_b20's false alarm rate, and at least as many as RO-NC at RO-NC's false
  alarm rate. The prediction fails if the gap is dominated by weather error or misfit days
  (holidays, clock changes, D8 and D9), in which case the page names which.
decision_rule: >-
  Fixed before any score is read. Rows: experiment 258's harness (15 Primary series, whole
  history from 2022-01 to 2026-06, an origin every 6 hours, the frozen past-weather GAM of
  experiment 130 as GAM_ONLY, K0 as in experiment 258). Gap scores at an origin, in units of the
  series' noise (1.4826 times the median absolute deviation of the GAM residual in odd months):
  the mean of K0 minus GAM_ONLY over leads 0 to 6 hours (in MW as well), and the mean residual of
  the readings that ended at least 3.5 hours before the origin, over the last 6, 24, and 72 hours.
  An origin is flagged when the absolute score exceeds a threshold k. Consecutive flagged
  origins within 48 hours form one alarm episode; the episode opens at the first flagged origin.
  The threshold is chosen on odd months of ordinary rows only, without labels, as the smallest k
  at which the false alarm rate equals a target rate, and is applied unchanged to even months.
  The target rates are the false alarm rates, measured by the same rule on the same odd months,
  of LR_b20, RO-NC (RO_b80), and the pair alarm (PG_g70), whose stored interval tables are
  scored by the same match rule. Variants: (a) plain; (b) with the neighbour check: the episode is
  kept when at least one connected metered neighbour has a gap of the opposite sign and at least
  30% of the node's size in MW, and the sum of the node's gap and its opposite-sign neighbours'
  gaps is at most half the node's gap; (c) a minimum duration: flagged at two consecutive
  origins (6 hours) or at five consecutive origins (24 hours). A real shift is matched when an
  alarm episode opens between 2 hours before its onset and its end, on either Primary of its
  pair. The lag is the origin's data cutoff minus the onset. A false alarm is an episode that
  opens in an ordinary row: outside every logged closure that touches the series or a
  connected neighbour, and the 72 hours after it, and outside D8's multi-day drops and the 17
  block days (the clean set). D8 and D9 are reported separately. Scoring sets: the 35 real
  shifts over 1 day (matched count and MW-weighted share, alarms per match, lag); the 411
  shifts injected into 44 worlds of experiment 141 (the neighbour check does not apply, because
  the receivers are unmetered); D8's drops and the 17 block days, separately; false alarms per
  Primary-year. Intervals: whole shifts resampled for matches, whole calendar months for
  false alarm rates, 95%, 20,000 draws. Support: variant (a) or (b) matches at least as many of
  the 35 as LR_b20 and as RO-NC at their respective false alarm rates, in the point estimate;
  the interval is reported beside. Verdict `supported` when both comparisons hold,
  `partly supported` when one holds, `not supported` when neither holds. The page also reports
  which of weather error, holidays and clock changes, D8, D9, and misfit days hold the false
  alarms, because the prediction names them as the failure case.
benchmark: >-
  LR_b20 (the carried learned score), RO-NC, and the pair alarm, scored on the same rows by the
  same match rule at their own false alarm rates.
row_set_id: >-
  Experiment 258's harness rows (15 Primary series, 6-hourly origins); not comparable with the
  leaderboard
metric: >-
  Matched shifts of the 35 real shifts over 1 day, alarms per match, lag in hours, false alarms per
  Primary-year
code: ml-experiments/00308-lsc-two-forecast-gap-as-a-detector/
verdict: supported
verdict_note: >-
  The point estimates meet the bar, but the intervals include zero: 11 of 35 shifts matched
  (6 to 17) against 10 for the learned score at its false alarm rate, and 21 (15 to 27) against 20
  for RO-NC at its rate. Under a rule that lets an alarm open up to 4 days after a shift ends, the
  learned score matches 27 and the gap 16.
best_arm: >-
  Plain gap over leads 0 to 6 hours: 11 of 35 real shifts matched while the shift was on (6 to 17)
  at 1.2 false alarms per Primary-year, with a median lag of 41 hours.
headline: >-
  The gap between the forecast with lags and the forecast without lags works as an alarm for load
  shifts about as well as the learned score and RO-NC. The 24-hour mean of the recent residual is
  the better signal (20 of 35 at the learned score's false alarm rate).
lessons: >-
  The filter's own correction trails the recent residual by about a day, so a detector that reads
  the residual directly finds more shifts at the same false alarm rate. The gap flags ordinary days
  at D9 at 2.7 alarms per Primary-year against 1.2 overall, so the gap alone cannot say a shift
  from a persistent change in demand. The neighbour check cost recall without cutting false alarms,
  because the threshold had to be lowered to keep the same false alarm rate.
take_home_message: >-
  NGED can see load shifts in the gap between the two forecasts, without any switching log: set to
  1.2 false alarms per Primary-year, the gap raises an alarm while the shift is still on for 11 of
  35 real shifts, as many as the learned score does (10), and a median 41 hours after the onset. A
  simple 24-hour mean of the recent residual finds more (20 of 35), so the next experiment should
  test whether choosing between the two forecasts with the gap helps the forecast itself.
advice:
  - >-
    Raise a load shift flag from the 24-hour mean residual against the no-lag forecast, not from
    the filter's correction, when the flag must open while the shift is on (Part 1 table).
  - >-
    Do not add the neighbour check to a gap alarm at a fixed false alarm rate: it matched no more
    shifts than the plain gap and needed a lower threshold (Part 1 table, variant b).
  - >-
    Compare any new detector with the learned score under both match rules, because the learned
    score opens its alarms about 4 days after the step and matches 10 shifts while the shift is on
    but 27 within 4 days of its end (Part 1 table).
  - >-
    Report false alarms per Primary-year for D9 and D8 separately, because one Primary can carry
    3 times the alarms of the others (Part 1 table, false alarm split).
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**This experiment asks whether the gap between two forecasts can find load shifts.** A load shift
moves demand from one Primary substation to its neighbours for hours or days. NGED cannot see the
true switching state, so the only signal it has is how far a forecast that uses recent power (K0,
a Kalman filter on the recent readings) differs from a forecast that uses none (GAM_ONLY, which
forecasts the normal running arrangement). We turned that gap, and simpler averages of the recent
residual, into alarms, set each alarm's threshold on odd calendar months without using any label,
and scored the alarms on the 35 real load shifts over 1 day, on 411 shifts injected into 44 copies
of the record, and on false alarms in ordinary rows. **The gap finds as many real shifts as the
learned score (LR_b20) and RO-NC do at their false alarm rates, with wide intervals.** The
24-hour mean residual finds more.

![Real shifts matched against false alarms](figure_1_matched_shifts_against_false_alarms.svg)

![Example gap and residual](figure_2_example_gap_and_residual.svg)

## Matched real shifts (35 shifts over 1 day)

"While on" means an alarm episode opened from 2 hours before the onset to the end of the shift.
"Within 4 days" (post hoc) also allows the 4 days by which the learned score trails a step. Each gap
row uses the threshold that gives the carried detector's false alarm rate on odd months.

| Detector | Matched while on (95% interval) | Within 4 days (post hoc) | Share of moved MW | False alarms per Primary-year, even months | Alarms per match | Median lag (hours) |
|---|---|---|---|---|---|---|
| LR_b20, learned score | 10 (5 to 15) | 27 | 25% | 0.55 | 4.0 | 96 |
| Plain gap, leads 0 to 6 h, at LR_b20's rate | 11 (6 to 17) | 16 | 47% | 1.21 | 4.5 | 41 |
| Gap with 6 hour minimum, at LR_b20's rate | 15 (9 to 21) | 18 | 54% | 1.09 | 3.6 | 62 |
| Gap with neighbour check, at LR_b20's rate | 11 (6 to 17) | 15 | 35% | 1.64 | 5.0 | 83 |
| 24 h mean residual, at LR_b20's rate | 20 (14 to 26) | 20 | 78% | 0.91 | 2.8 | 29 |
| RO-NC (RO_b80) | 20 (14 to 26) | 27 | 68% | 3.64 | 7.3 | 48 |
| Plain gap, leads 0 to 6 h, at RO-NC's rate | 21 (15 to 27) | 22 | 76% | 4.19 | 7.3 | 47 |
| Gap with neighbour check, at RO-NC's rate | 15 (9 to 21) | 18 | 43% | 4.68 | 10.4 | 49 |
| Pair alarm (PG_g70) | 4 (1 to 8) | 6 | 14% | 0.91 | 8.2 | 7 |
| Plain gap, leads 0 to 6 h, at the pair alarm's rate | 11 (6 to 16) | 16 | 47% | 0.91 | 3.6 | 41 |

## Checks and caveats

- **Prediction (point estimates):** variant (a) matches 11 against LR_b20's 10, and 21 against
  RO-NC's 20. The differences are +1 (-5 to 7) and +1 (-6 to 7) shifts. The prediction holds as
  written but cannot rank the detectors.
- **Where the false alarms fall (plain gap at LR_b20's rate, 39 episodes in 32.8 Primary-years of
  ordinary rows):** D9 5 (2.7 per Primary-year, against 1.2 over all series), D8 0, calendar days 3,
  November to March 8, other 23. Holidays, clock changes, D8, and D9 do not dominate. The report
  cannot separate weather error from other causes inside the "other" group.
- **Other row groups at LR_b20's rate:** D8's 29 multi-day drops, 1 matched; the 17 block days, 0
  (the 24-hour mean residual on those days stays under 2 noise units, and no detector, carried or
  gap, matches one). Injected shifts, 182 of 1,063 site events matched (17%), and only 5% of those
  up to 2 days long, 39% of those over 5 days; the median injected step is under 1 MW.
- **Neighbour check:** 29 of the 35 shifts have both Primaries metered. The check kept the matched
  count at 11 and raised false alarms per Primary-year to 1.64, because its threshold had to fall
  from 3.2 to 2.1 to hold the false alarm rate; it does not apply to injected shifts, whose
  receivers are unmetered.
- **Match rule changes the ranking:** the learned score trails a step by 4 days by design (its mask
  reaches back to the step). Allowed to alarm up to 4 days after the shift ends, it matches 27 and
  the gap 16.
- **Data:** the filter's correction is zero before a series' first reading (series 13 and 7 start
  late), so the gap is undefined there and those rows count for neither alarms nor exposure.
- **Median minus mean forecast, per site, and quantile-score contrasts:** not applicable, because no
  quantile forecast is scored. **Audit triggers that fired:** none of the seven fire on this result.
  Trigger 7 applies to part 2 (experiment 309), which reads the gap, and the maintainer excluded an
  Opus audit.
- **Lab conditions:** the injected worlds, the GAM and the filter hyperparameters are those of
  experiments 130, 141, and 258, with past weather, not ENS weather. The carried detectors' 4-day
  delay and the thresholds' fit on the same real record are shared by every detector.

## What to try next

- **Part 2 (experiment 309):** use the gap, and the 24-hour residual, to choose between the two
  forecasts, scored against observed power and against the true normal running arrangement.
- **Post hoc:** a detector that combines the 24-hour residual with the gap.
