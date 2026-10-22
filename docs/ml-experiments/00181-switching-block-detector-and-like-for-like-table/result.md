---
id: "00181"
slug: switching-block-detector-and-like-for-like-table
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
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
  Porting the detector would add one pipeline step, so no porting threshold is set: the experiment
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
verdict: refuted
best_arm: >-
  The block detector with the noise unit set on odd months (S0) finds 0 of the 17 block days at RO-NC's
  budget of 80 unexplained odd-month alarms (threshold 4.37 noise units). The post hoc variant with a
  noise unit taken from the previous 7 days (S2) finds 5 of 17 (4 of the 5 days in even months).
headline: >-
  The block detector finds 0 of 17 block days at 80 alarms (prediction: at least 12). The
  best-fitting block of the 17 days has a median score of 2.3 noise units and a median length of 12
  hours. Injected blocks 5 noise units tall are found 57% to 65% of the time, and blocks 6 noise
  units tall about 90%. On the 35 real transfers, RO-NC on the convex-model residual finds 18
  against 12 on the XGBoost residual, but only 14 against 12 on the 29 transfers that both
  residuals can see.
lessons: >-
  The 17 missed days are mostly short blocks of 1 to 4 noise units on the residual that RO-NC reads,
  so neither a longer or shorter averaging window nor a block-shaped statistic separates them from
  the residual's ordinary variation at 80 alarms. The convex model's lead on real transfers comes
  mostly from readings the XGBoost panel lacks.
take_home_message: >-
  A detector fitted to rectangular blocks finds none of the 17 unlabelled worst days. It was allowed
  the same 80 unexplained alarms in odd-numbered months as RO-NC, the existing detector. The days
  are mostly blocks lasting 12 hours or less and standing 1 to 4 noise units above the residual
  (one noise unit is the usual spread of the forecast residual), too close to its ordinary variation
  for an alarm that reads only the residual. The next step is to send NGED the candidate label
  table, so that NGED can say which of the 17 days were real switching. On the 35 labelled real
  transfers lasting over 1 day, RO-NC finds 18 on the convex-model residual and 12 on the XGBoost
  residual. Four of the 6 extra transfers sit at Primaries where the XGBoost panel has no readings,
  and on the 29 transfers that both residuals cover the counts are 14 and 12.
advice:
  - >-
    Do not expect a residual-only block detector to find the 17 unlabelled worst days at RO-NC's
    budget of 80 alarms: the days sit at 1 to 4 noise units (figures 2 and 3).
  - >-
    Quote the convex-model residual's real-transfer lead as 18 against 12 of 35 with the coverage
    caveat, and as 14 against 12 of 29 where both panels hold readings (figure 1).
  - >-
    Send the candidate label table to NGED before building another detector for short blocks, so
    that the 17 days can be told apart as switching or not (part 2).
  - >-
    Test a noise unit taken from the previous 7 days before ruling out block detection: it found 5
    of 17 at the same 80 alarms (post hoc, figure 2).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

# Experiment 181: why were 17 block days missed, and can a block detector find them?

**A detector built for blocks finds none of the 17 unlabelled worst days.** The days are short
blocks of 1 to 4 noise units, where one noise unit is the usual spread of the forecast residual
(observed minus forecast power) of that series. A block is a switching-like change in the power a
Primary substation draws: it steps up, stays level, and steps back. [Experiment
175](../00175-dp-unlabelled-day-shapes/result.md) found 17 sharp-edged block days among the 37 worst
days that no label explains. Neither RO-NC (the refit-offset neighbour-count detector of [experiment
82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md)) nor the opposite-step
pair alarm of [experiment 156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md)
fired on any of them. We asked why, wrote a rectangular-block detector, and counted how many of the
17 days it finds when it is allowed the same 80 unexplained alarms in odd-numbered months as RO-NC.
It finds 0 (prediction: at least 12), so the verdict is `refuted`. **RO-NC finds more real transfers
on the convex-model residual than on the XGBoost residual, but most of the difference comes from
Primaries that the XGBoost panel does not cover.** A transfer is a labelled period in which
switching moved load from one Primary substation to others. On the 35 labelled transfers lasting
over 1 day, RO-NC finds 18 on the residual of the convex demand model and 12 on the residual of the
XGBoost model. Four of the 6 extra transfers sit at Primaries where the XGBoost panel (its table of
readings and forecasts) has no readings. NGED is National Grid Electricity Distribution; a Primary
substation steps transmission voltage down to distribution voltage. The 35 transfers are those of
the 66 labelled real transfers that last over 1 day.

![Real transfers found by each residual](figure_1_like_for_like.svg)

![Block days found against alarms](figure_2_block_days_found.svg)

![Best block score of each block day](figure_3_block_scores.svg)

![Injected blocks found](figure_4_injected_blocks.svg)

## Results

- **Part 3, like-for-like table.** RO-NC's threshold gives 80 unexplained odd-month alarms on each
  residual. A transfer counts as found when an alarm falls within 2 hours of the labelled start and
  the alarm's end within the larger of 6 hours and 10% of the duration. The same code scored both
  residuals. The MW-weighted figure is the share of the moved megawatts that the found transfers
  carry, and the interval resamples transfers 20,000 times.

| Transfers | XGBoost residual found (MW-weighted) | Convex-model residual found (MW-weighted) | Paired MW-weighted gap (95% interval) |
|---|---|---|---|
| All 35 (42.7 MW) | 12 (0.357) | 18 (0.657) | +0.300 (+0.089 to +0.505) |
| 29 with readings in both panels (31.2 MW) | 12 (0.489) | 14 (0.604) | +0.115 (-0.046 to +0.300) |
| 6 with no readings in the XGBoost panel (11.5 MW) | 0 (0.000) | 4 (0.800) | +0.800 (+0.402 to +1.000) |

- **Injected transfers** (experiment 130, quoted): 0.794 convex against 0.772 XGBoost, a gap of
  +0.023 (-0.001 to +0.046).
- **What differed between experiments 101 and 130:** nothing in the matching rule or the 35
  transfers, because experiment 130 copied experiment 101's scoring code. The two differ in the
  residual, in the 15 series (against 16, with the same 12 found by the XGBoost arm on either), and
  in the 6 transfers at two Primaries that the XGBoost panel lacks. The convex model also trained
  with these real transfers masked, which flatters it on them.
- **Part 1, step 1: why the 17 days were missed.** On the convex-model residual, the best-fitting
  block of each day has a median score of 2.3 noise units (largest 4.4) and a median length of 12
  hours; 13 of the 17 best-fitting blocks last 12 hours or less. RO-NC's 24-hour step statistic has
  a median of 2.2 noise units against its threshold of 4.35, and exceeds it on 2 days (5.2 and 7.2),
  where RO-NC still raises no alarm; why is not tested. All 17 days have at least two metered
  interconnected Primaries, so a missing partner does not explain the misses. No RO-NC or pair alarm
  falls inside any of the 17 local days.
- **Part 1, step 2: block detector.** Threshold at 80 unexplained odd-month alarms: 4.37 noise
  units (78 unexplained alarms in even months). S0 finds 0 of 17 days (0 of the 5 in even months).
  Neighbour support (S1, threshold 5.37) finds 1. The post hoc S2, which divides by the previous 7
  days' spread, finds 5 of 17 (4 of 5 in even months; 84 unexplained even-month alarms). S0 finds 7
  of 17 only at about 1,300 unexplained odd-month alarms.
- **Injected-block control.** Blocks of height 5 noise units are found 57% to 65% of the time at
  lengths of 6 to 48 hours, and 6 noise units 89% to 92%, so the detector works on blocks of the
  size it is tuned for. At 3 to 4 noise units it finds 5% to 29%.
- **Part 2, candidate label table:** 133 rows in
  `data/ml-experiments/briefs/sw_block_detector_label_table.csv` (private, not committed): 17 block
  days (3 of medium confidence), 21 balanced-opposite alarms (all low), 29 drops of D8 (23 of medium
  confidence, with a logged closure within 24 hours), and 66 labelled real transfers (50 of medium
  or high confidence). Columns: onset, end, depth in MW, duration, pair, source, confidence.

## Checks

- **Median minus mean forecast, and the mean-forecast error:** not applicable, because no forecast
  is scored.
- **Trigger 2 (a result contradicts its prediction) fired**: 0 days found against a prediction of at
  least 12. The explanation that the blocks are too small is tested by the threshold curve and the
  injected-block control. The audit that trigger 2 calls for was not run here.
- **Trigger 4** (leaving a site out) was not run. **Triggers 1, 3, 5, 6, and 7** did not fire: the
  injected-block control passes.

## Deviations and caveats

- **Residual.** The block detector reads the convex-model residual of experiment 130 (ERA5 weather,
  frozen), not the forecast residual of arm D (ensemble weather and Kalman filter) on which
  experiment 175 found the days. The blocks are visible in both, but the noise unit differs, and
  the Kalman filter in arm D absorbs a multi-day block. This is the main caveat on "0 of 17".
- **Found rule.** The plan said an alarm within 3 hours of the block's onset. The median block
  length of 8 hours in the plan came from an earlier look at the detector's edges, and 12 hours is
  the best-fitting block's median here. The strongest edge of experiment 175 can be the block's
  return edge, so the headline counts an alarm inside the local day or within 3 hours of it. The
  edge rule finds 0 (S0) and 2 (S2).
- **Peak rule.** Alarms are local peaks within 48 half-hours, as in RO-NC, so a day whose best
  score (D5, 4.38) exceeds the threshold can lose to a larger peak nearby (4.56, outside the day).
  The count would then be 1 of 17.
- **Looser than the plan.** An alarm of either sign, or one that the masks call explained, counts as
  found; the injected-block recall includes chance exceedances. Both favour the detector. The
  noise-unit floor of S2 uses all months. The like-for-like intervals resample transfers, not
  Primaries, and "has readings" means at least half of the transfer's half-hours are present.
- **The 5-of-17 result is exploratory.** S2 was added after the 17 days were seen, so its count is a
  hint for a later pre-registered test and not evidence that block detection works. No Opus audit was
  run, by the manager's decision.
- **Post hoc.** S2 was added after S0 found nothing. Its threshold is set on odd months, but the
  choice of S2 followed the 17 days.
- **Unexplained alarms** may include true unlogged switching, so the budget is conservative for any
  detector that finds real blocks.
- **Two choices that live use would not allow.** The residual uses past weather that arrives about 5
  days late, and block extents use hindsight.

## What to try next

Send the candidate label table to NGED to learn which of the 17 blocks are switching. A block
detector on the arm D residual, which has a smaller noise unit on these days, is the untested
alternative, and needs a longer history than the one year experiment 175 covers.

Code: `ml-experiments/00181-switching-block-detector-and-like-for-like-table/`; the numbers above
are printed by `e00181_like_for_like.py`, `e00181_why_missed.py`, `e00181_block.py`, and
`e00181_block_control.py`, and kept as `printed_*.txt` in that folder.
