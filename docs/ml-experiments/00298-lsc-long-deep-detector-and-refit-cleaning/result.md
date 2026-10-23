---
id: '00298'
slug: lsc-long-deep-detector-and-refit-cleaning
title: >-
  Does cleaning the refit of a no-lag forecast of detected load shifts stop it learning them as
  normal?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: planned
size: >-
  complex by the sizing triggers, run with the reviews the maintainer's brief allows (Sonnet only).
  Stored: private daily-residual panels, alarm tables and per-world bias tables under
  data/ml-experiments/00298-lsc-long-deep-detector-and-refit-cleaning/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for the daily
  statistic, the depth and end rules, and the cleaning cap, so all are fixed below before any score
  is read. Callers: this folder's scripts, which call the reviewed `studies.dp_gam` fit and predict
  functions unchanged and copy experiment 274's and 290's harness patterns.
hypothesis: >-
  A trivial detector on the daily mean residual of the lagless convex additive model (GAM_ONLY)
  finds load shifts that last days to weeks and cover at least 10% of a Primary substation's load,
  which the existing hour-scale detectors miss, and excluding the days it flags from the monthly
  refit stops the forecast learning those shifts as the normal pattern.
prediction: >-
  Written before any score is read. (1) The best detector among A to C finds at least 12 of the 17
  unlabelled block days and at least 15 of D8's 29 drops (or the engine evidence of experiment 291
  says the drops are not load shifts), at no more than 2 false alarms per Primary-year in even
  months. (2) The neighbour-confirmed variants D to F cut false alarms per Primary-year by at least
  half at no more than 20% recall loss among events with metered partners, and reject at least 80%
  of injected permanent steps. (3) Cleaning with the best variant cuts the share of a 30-day, 25%
  shift learned as normal from the no-cleaning share to under 5%, at a cost on clean rows under
  0.01 points. We expect (1) to fail on the block days, which experiment 181 found are 12 hours or
  less and 1 to 4 noise units tall, so that a daily mean cannot see them.
decision_rule: >-
  Pre-registered before any score is read; Bonferroni level 98.75% across the four contrasts below.
  C1: the number of the 17 block days and of the 29 drops found (an alarm span covering any part
  of the event's UTC days, or within one day of it), and false alarms per Primary-year in even
  months, for the variant among A to C that has the highest recall on odd-month injected shifts at
  the threshold that gives 2 false alarms per Primary-year in odd months. C2: for D to F against
  the same variant, the ratio of false alarms per Primary-year and the difference in recall among
  events whose donor has at least one metered neighbour, and the share of injected permanent steps
  rejected. C3: the share of a 30-day, 25% shift learned as normal in the 48 hours after its end,
  oracle-cleaned and detector-cleaned against uncleaned, over 10 Primaries and 4 onset dates. C4:
  the cost on rows with no shift, as the difference in mean absolute error against the clean
  series between the cleaned and the uncleaned forecast. Intervals resample Primaries and whole
  events (percentile bootstrap, 2,000 resamples). The detector is `supported` when predictions 1
  to 3 all hold, `partly supported` when the cleaning prediction (3) holds with some detector
  variant, `not supported` otherwise. The result is a statement about whether long shifts stay a
  stated limit, so no porting cost applies.
benchmark: >-
  Experiment 274's no-cleaning cell for the share learned (19% for 30 days, 25%), recomputed in
  this experiment with unmasked fits and 4 onset dates; experiment 290's range of 9% to 29%;
  RO-NC (the refit-offset neighbour-count detector), the pair alarm and the learned score of
  experiment 195 on the same real events.
row_set_id: >-
  Own rows: 16 Primary substations' daily residuals from 2022-01 to 2026-06 (real record), and 10
  Primaries at 4 onset dates (injected); not comparable with the leaderboard
metric: >-
  Detector: recall by event class, false alarms per Primary-year, onset and end error in hours.
  Cleaning: share learned as normal (forecast trained on the injected series minus the forecast
  trained on the clean series, divided by the mean injected transfer, over the 48 hours after the
  end) and mean absolute error against the clean series.
code: ml-experiments/00298-lsc-long-deep-detector-and-refit-cleaning/
production_status: not-for-production
adopted_in: none
---

**Plan only.** The write-up replaces this body once the run has finished. The plan is the brief at
`data/ml-experiments/briefs/sw_lsc_long_deep_detector_and_refit_cleaning_brief.md` (Parts 1 and 2;
Part 0 is [experiment 290](../00290-lsc-long-shift-cut-date-robustness/result.md)).

## Sizing (the five triggers)

- **What gets stored:** private parquet tables only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (statistic, depth rule, end rule, cleaning cap), so all
  are fixed here.
- **Callers not nameable without searching:** no; the scripts call `studies.dp_gam` unchanged.

## Design

- **Residual.** GAM_ONLY (no lagged power, 24-month window, ridge 0.1, as in experiments 274 and
  290) refitted at the first of each month, predicting the whole month with realised ERA5 and CAMS
  weather (a lab shortcut). The fit is unmasked: the hand-filtered switching rows train too.
  The daily mean residual is the mean over the UTC day, in MW. One noise unit is 1.4826 times the
  median absolute deviation of the daily means over the previous 24 months (experiment 217).
- **Depth rule.** The mean over the candidate window is at least 10% of the Primary's median power
  over the previous 365 days. A span lasts at least 2 days and ends on the first day whose daily
  mean is back inside 1.5 noise units for a day. The hindsight option moves the start back by up
  to 2 days.
- **Variants.** A: mean of the last W = 2, 3, 7 days against a threshold k. B: a CUSUM on the daily
  mean z-score with allowance 0.5. C: the last 3 days minus the preceding 28. D: A or C confirmed
  by a metered neighbour (the adjacency table of experiment 43) with the opposite sign and at least
  30% of the size, whose summed mean with the node's is under 50% of the node's. E: the pair test,
  where the difference of two connected residuals shifts while their sum stays within 1.5 noise
  units. F: up to 3 neighbours whose summed mean cancels the node's, with the share explained.
- **Tuning.** Thresholds are set on odd months only, to give 2 false alarms per Primary-year, and
  scored on even months. A false alarm is a span with no known event within 3 days (a D8 drop, a
  block day, or a logged real transfer); D8 and D9 are counted apart.
- **Injected shifts.** Durations 1, 3, 6, 7, 14, 30 and 90 days, sizes 10%, 25% and 50% of load,
  proportional to own power or a fixed block, at 10 Primaries and onsets in December 2022, June
  2023, December 2023 and June 2024; permanent steps of 10% and 25% starting 3, 6 and 12 months
  before the end of the window. The receiver is unmetered, or a metered neighbour (the injected
  series of the neighbour is fitted too). Each world is refitted monthly with the injected series,
  so the detector sees what the forecast learned.
- **Cleaning arms.** (a) none; (b) oracle (true shift days excluded from the refit); (c) the best of
  A to C; (d) the best of D to F. A day flagged by the end of the previous month is excluded from
  the next refit, with the start moved back 2 days; a run of excluded days is capped at 60. Two
  refit cycles are rerun to see whether the same days are flagged again.
- **Real rows.** The 17 block days of experiment 175, the 29 drops of D8 (reported apart; they
  are probably a gas engine, experiment 291), and the real transfers over 1 day in the labelled set
  of experiment 41. RO-NC, the pair alarm and the learned score are read from the stored alarm
  tables of experiments 195 and 217 at their stored operating points.

## Earlier advice followed

- Quote the share learned as a range, not as 19% (experiment 290).
- Use several end dates, not the first of a month (experiment 290).
- Report the effect on separate row groups (decision program brief).

## Checks, to be filled in at write-up

- Median minus mean forecast per site (not applicable: GAM_ONLY forecasts the mean).
- Error of the mean forecast beside every contrast.
- Which of the seven audit triggers fired.
