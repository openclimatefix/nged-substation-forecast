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
  (holidays, clock changes, D8, and D9), in which case the page names which.
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
  The point estimates meet the prediction, but the 95% intervals on the differences from the
  benchmarks (-5 to 7 and -6 to 7 shifts) include zero. The plain gap matched 11 of 35 real shifts
  while they were on, against 10 for the learned score at the learned score's odd-month false alarm
  rate, and 21 against 20 for RO-NC at RO-NC's rate. On the held-out even months the gap raised 1.21
  false alarms per Primary-year and the learned score 0.55. When an alarm may open up to 4 days
  after a shift ends, the learned score matches 27 and the plain gap 16.
best_arm: >-
  Plain gap over leads 0 to 6 hours: 11 of 35 real shifts matched while the shift was on (6 to 17),
  with 1.21 false alarms per Primary-year (one Primary observed for one year) on even months and a
  median alarm 41 hours after the onset.
headline: >-
  The plain gap between the forecast with lags and the forecast without lags matched 11 of 35 real
  load shifts while they were on, against 10 for the learned score and 21 against 20 for RO-NC,
  with thresholds set on odd months to those detectors' false alarm rates. The 24-hour mean of
  recent forecast error matched 20 of 35 in the point estimate.
lessons: >-
  The 24-hour mean forecast error matched more shifts than the filter's correction at the same odd-
  month false alarm rate, although the intervals overlap. The gap raised 2.7 false alarms per
  Primary-year at D9, against 1.2 over all series. The neighbour check did not raise the matched
  count at the learned score's rate (11 either way) and raised false alarms from 1.21 to 1.64 per
  Primary-year. At RO-NC's rate it lowered the matched count from 21 to 15.
take_home_message: >-
  The gap between the two forecasts flagged 11 of 35 real load shifts while they were still on,
  about as many as the learned score (10) when both thresholds are set to the same false alarm rate
  on odd months. The median alarm came 41 hours after the onset. On the held-out even months the
  gap raised about twice the learned score's false alarms (1.21 against 0.55 per Primary-year). A
  24-hour mean of recent forecast error flagged 20 of 35. Experiment 309 tests whether the gap can
  choose between the two forecasts.
advice:
  - >-
    Raise a load shift flag from the 24-hour mean of recent forecast error against the no-lag
    forecast, rather than from K0's correction, when the flag must open while the shift is on (table
    of matched real shifts).
  - >-
    Expect no gain from the per-origin neighbour check on a gap alarm at a fixed false alarm rate:
    it matched 11 of 35 shifts either way at the learned score's rate and fewer at RO-NC's (checks
    and caveats, neighbour check).
  - >-
    Score a new detector under both match rules, because the learned score opens its alarms about 4
    days after the step and matches 10 shifts while the shift is on but 27 within 4 days of its end
    (table of matched real shifts).
  - >-
    Report false alarms per Primary-year for D8 and D9 separately, because D9 carried about twice
    the alarm rate of the whole set at the learned score's rate (checks and caveats, false alarms).
production_status: not-for-production
adopted_in: none
date: 2026-10-23
---

**This experiment asks whether the gap between two forecasts can find load shifts.** A load shift
is a switching operation that moves part of one Primary substation's demand (a substation that steps
33 kV down to 11 kV) onto neighbouring substations for hours or days. The network operator NGED has
no access to the system that records these operations. The two forecasts it receives from us are K0
(a Kalman filter on recent power readings, added to the generalised additive model forecast) and
GAM_ONLY (the same model with no recent power, which forecasts how the network is normally
switched). We turned the gap between them, and averages of recent forecast error over 6, 24, and 72
hours (observed power minus GAM_ONLY), into alarms. We set each alarm threshold on odd calendar
months, using no knowledge of where shifts happened, and scored the alarms on the other months and
on all months. The scoring covered the 35 real load shifts lasting longer than 1 day, 411
artificial shifts added to 44 simulated copies of the power history ([experiment
141](../00141-switching-detector-fed-into-gam-k/result.md)), and false alarms in ordinary rows. A
row is one Primary at one 6-hourly forecast origin, and it is ordinary when no logged switching
touched the substation or a neighbour in the 72 hours before. An alarm "matches" a shift when it
opens from 2 hours before the onset until the shift ends. **The plain gap matched 11 of 35 real
shifts, the learned score LR_b20 10, and RO-NC 20 (the refit-offset neighbour-count detector), when
each gap threshold reproduces the benchmark's false alarm rate on odd months.** The learned score
([experiment 227](../00227-lsc-fewer-false-alarms-and-softer-use-part-2/result.md)) re-ranks RO-NC's
candidate alarms. The 24-hour mean forecast error matched 20 of 35.

![Real shifts matched against false alarms](figure_1_matched_shifts_against_false_alarms.svg)

![Example gap and residual](figure_2_example_gap_and_residual.svg)

## Matched real shifts (35 shifts lasting over 1 day)

"While on" means an alarm episode opened from 2 hours before the onset to the end of the shift.
"Within 4 days" (post hoc) counts an alarm that opens up to 4 days after the shift ends, because the
learned score opens its alarms about 4 days after the step. "Share of moved MW" weights each
shift by its size in megawatts. "Alarms per match" is all false alarms plus matched alarms divided
by
matched shifts. "Lag" is the forecast origin minus the onset. The threshold of each gap row
reproduces the false alarm rate named in its row on odd months (1.29 per Primary-year for the
learned score, 4.11 for RO-NC, and 0.86 for the pair alarm); the last column is read on even
months.

| Detector | Matched while on (95% interval) | Within 4 days | Share of moved MW | Alarms per match | Median lag (hours) | False alarms per Primary-year, odd then even months |
|---|---|---|---|---|---|---|
| LR_b20, learned score | 10 (5 to 15) | 27 | 25% | 4.0 | 96 | 1.29, 0.55 |
| Plain gap, leads 0 to 6 h, at the learned score's rate | 11 (6 to 17) | 16 | 47% | 4.5 | 41 | 1.17, 1.21 |
| Gap with 6-hour minimum, at the learned score's rate | 15 (9 to 21) | 18 | 54% | 3.6 | 62 | 1.29, 1.09 |
| Gap with neighbour check, at the learned score's rate | 11 (6 to 17) | 15 | 35% | 5.0 | 83 | 1.04, 1.64 |
| 24 h mean forecast error, at the learned score's rate | 20 (14 to 26) | 20 | 78% | 2.8 | 29 | 1.23, 0.91 |
| RO-NC (RO_b80) | 20 (14 to 26) | 27 | 68% | 7.3 | 48 | 4.11, 3.64 |
| Plain gap, leads 0 to 6 h, at RO-NC's rate | 21 (15 to 27) | 22 | 76% | 7.3 | 47 | 3.93, 4.19 |
| Gap with neighbour check, at RO-NC's rate | 15 (9 to 21) | 18 | 43% | 10.4 | 49 | 3.93, 4.68 |
| Pair alarm (PG_g70) | 4 (1 to 8) | 6 | 14% | 8.2 | 7 | 0.86, 0.91 |
| Plain gap, leads 0 to 6 h, at the pair alarm's rate | 11 (6 to 16) | 16 | 47% | 3.6 | 41 | 0.86, 0.91 |

## Checks and caveats

- **Prediction (point estimates):** the plain gap matches 11 against the learned score's 10, and 21
  against RO-NC's 20. The differences are +1 (-5 to 7) and +1 (-6 to 7) shifts, so the experiment
  cannot rank the detectors. The learned score's false alarm rate was 1.29 on odd months and 0.55
  on even months, while the gap's was 1.17 and 1.21, so on even months the gap raised about twice
  the learned score's false alarms.
- **Where the plain gap's false alarms fall (at the learned score's rate, 39 episodes in 32.8
  Primary-years of ordinary rows):** 5 episodes in D9 rows (2.7 per Primary-year, against 1.2 over
  all series), 0 in D8 rows, 3 on bank holidays or within a day of a clock change, 8 from November
  to March, and 23 elsewhere. Holidays, clock changes, D8, and D9 do not hold most of the false
  alarms. The page cannot say how many of the 23 unattributed episodes are weather error, so the
  prediction's weather-error clause is undetermined.
- **Other shift sets at the learned score's rate:** the plain gap matches 1 of D8's 29 multi-day
  drops and none of the 17 unlabelled single days with block-shaped demand (the 24-hour mean error
  on those days stays under 2 noise units, and no detector, carried or gap, matches any of them). A
  noise unit is 1.4826 times the median absolute value of a score on odd-month ordinary rows. The
  injected shifts add 1,063 events at the six substations, because one transfer moves power at the
  donor and at each receiver. The plain gap matches 182 of them (17%): 5% of those lasting up to 2
  days (380 events) and 39% of those lasting over 5 days (83 events). The median injected step is
  0.87 MW.
- **Neighbour check:** 22 of the 35 shifts name two Primaries that are both metered, so the check
  can apply to those and not to the rest. It confirms an alarm origin when a connected metered
  neighbour has an opposite-signed gap at least 30% the size and the two sum to at most half the
  node's gap. The check is applied at each origin, so it removes alarm origins more strictly than a
  rule that kept a whole episode when any origin was confirmed; the episode-level rule is untested.
  The threshold had to fall from 3.2 to 2.1 to hold the false alarm rate, which explains the extra
  false alarms. The check cannot apply to injected shifts, whose receivers are unmetered.
- **Match rule changes the ranking:** the learned score's mask reaches back to the step stamp
  after an alarm that opens about 4 days late. When an alarm may open up to 4 days after the shift
  ends, the learned score matches 27 and the plain gap 16.
- **Deviations from the plan:** a noise unit is the robust scale of each score, not of the
  half-hourly forecast error; and the lag is measured from the forecast origin, which is 4 hours
  after the last reading the filter used.
- **Data:** K0 has no correction before a series' first reading (series 13 and 7 start late), so the
  gap is not computed there and those rows count for neither alarms nor exposure.
- **Audit rule:** none of the seven triggers in the `ml-experiment` skill fires on this result.
  Experiment 309 reads the gap, and the maintainer excluded an Opus audit of this work. The median
  minus mean forecast and quantile-score contrasts do not apply, because no quantile forecast is
  scored here.
- **Lab conditions:** the records, the generalised additive model, and the filter settings come
  from experiments 130, 141, and 258, with estimates of past weather and not ECMWF ensemble (ENS)
  forecasts. The thresholds of every detector were fitted on the same real record.

## What to try next

- **Experiment 309** tests whether the gap, and the 24-hour mean forecast error, can choose between
  the two forecasts, scored against observed power and against the normal running arrangement.
- **Post hoc:** a detector that combines the 24-hour mean forecast error with the gap could be
  tried.
