---
id: "00181"
slug: switching-block-detector-and-like-for-like-table
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00181-switching-block-detector-and-like-for-like-table/
  and data/ml-experiments/briefs/ only (a label table for NGED review is never committed). Serving
  path: not touched. Degradation rule: not touched. Designs: more than one block detector is
  defensible (window lengths, flank widths, neighbour support), so this plan fixes each setting in
  numbers before any detector is scored. Callers: this folder's scripts only, which copy
  experiment 146's loaders for the frozen convex-model residual panel of experiment 130 and read
  stored tables of experiments 101, 130, 146, 147, 156, 159, and 175.
hypothesis: >-
  The 17 unlabelled worst days that experiment 175 classed as sharp-edged blocks were missed by the
  refit-offset neighbour-count detector (RO-NC) and by the opposite-step pair alarm because most of
  the blocks are shorter than the 24-hour windows those detectors average over and their steps are
  small in noise units, not because the blocks are invisible. A detector that fits a rectangular
  block (a step up, a flat level, a step down) at lengths of 6 hours to 5 days finds most of them
  at the same number of unexplained alarms as RO-NC.
prediction: >-
  Written before the block detector is scored. At RO-NC's matched budget of 80 unexplained
  odd-month alarms, the single-series block detector finds at least 12 of the 17 block days. This
  is the brief's prediction and can fail. The step 1 table (read before this plan was committed, so
  not a prediction) already shows that the median block lasts 8 hours and that the median step is
  2.1 to 2.7 noise units at the 12-hour and 4-hour windows, which makes finding 12 of 17 doubtful.
decision_rule: >-
  Descriptive for part 1 step 1, part 2, and part 3, which carry no verdict. Part 1 step 2 is read as
  follows. `supported` when at least 12 of the 17 block days are found (alarm on the same series
  within 3 hours of the onset) at the odd-month budget of 80 unexplained alarms; `partly supported`
  when 7 to 11 are found; `refuted` when fewer than 7 are found. The threshold is set on odd months
  only. The days are also split by month parity, and the held-out count is the days in even months.
  The detector costs one new pipeline step, so a porting threshold is not set: the experiment
  decides only whether the blocks are findable.
benchmark: >-
  RO-NC on the frozen convex-model residual (arm B of experiment 130) and the pair alarm of
  experiment 156, which fire on 0 of the 17 block days (experiment 175).
row_set_id: >-
  none (scores stored and new detector alarms against the 17 block days of experiment 175 and the
  35 real transfers of experiment 41, with no forecast)
metric: >-
  Block days found out of 17, unexplained alarms on odd and even months, and for part 3 the real
  transfers found out of 35 and by moved MW.
code: ml-experiments/00181-switching-block-detector-and-like-for-like-table/
---

# Experiment 181: why were 17 block days missed, and can a block detector find them?

Planned. The write-up follows once the fits finish.

## Pre-registered design

- **Residual.** The frozen convex-model residual of experiment 130, arm B, on the 15 series, 2022-01
  to 2026-06 (RO-NC's input). Lab shortcuts: past weather (ERA5) and hindsight, as in experiment
  130.
- **Block statistic.** For each series, each half-hour `t`, and each block length `L` in 12, 24, 48,
  96, and 240 half-hours (6 hours, 12 hours, 1 day, 2 days, 5 days): the mean of the residual over
  `[t, t+L)` minus the mean over the `F = min(L, 48)` half-hours before `t`, and the same minus
  the mean over the `F` half-hours from `t+L`. The score is the smaller of the two (up block) or the
  larger of the two (down block), in noise units: the odd-month median absolute deviation (times
  1.4826) of that score for that series and length. A window needs 75% of its readings present.
- **Alarms.** The largest score over the lengths at each `t`, local peaks within 48 half-hours,
  as RO-NC does. An alarm is unexplained when no logged closure on its series lies within 24
  hours and no labelled real transfer on its series contains it, as in the false-alarm check.
- **Budget.** The threshold is the smallest one that leaves at most 80 unexplained alarms in the
  odd-numbered months of 2022-01 to 2026-06, the count RO-NC's threshold was tuned to.
- **Neighbour support (descriptive arm S1).** The own score plus the best opposite-signed score of a
  metered interconnected series at the same `t` and `L`, clipped to 0 to 3 noise units, with weight
  1. S0 (no neighbour term) is the headline.
- **Found.** An alarm on the same series within 3 hours of the block's onset (experiment 175's
  window). The looser count (any alarm within the local day) is also reported.
