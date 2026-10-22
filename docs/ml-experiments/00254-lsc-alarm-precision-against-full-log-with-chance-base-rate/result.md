---
id: "00254"
slug: lsc-alarm-precision-against-full-log-with-chance-base-rate
title: >-
  How many LSC alarms are real load shifts, judged against the full log and a chance base rate?
theme: lsc-detectors
also_relevant_to: []
status: done
size: >-
  simple. Stored: private tables under
  data/ml-experiments/00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one defensible design, because
  the operating points are frozen from experiment 227. Callers: this folder's scripts only; they
  read the alarm tables of experiments 195 and 227 and the log of experiment 37.
hypothesis: >-
  Experiment 195's loose rule (a logged closure within 24 hours "explains" an alarm) overcounts real
  alarms, because a random window is explained about 45% of the time (experiment 159, which counts any
  time inside a closure or within 24 hours of it), and the strict rule (experiment 195's step-and-
  end match) undercounts them, because the log's times are approximate. Once the share expected by
  chance is removed, a large share of the alarms from LR_b20, the learned alarm source, that are not
  strict matches still coincide with a logged closure.
prediction: >-
  Written before any table exists. For LR_b20, at least 30% of the alarms that are not strict
  matches have a logged closure on the same Primary substation or a connected neighbour starting or
  ending within 24 hours of the step, at an excess of at least 15 percentage points over the mean
  of the week-shifted base rate, so its corrected false-alarm rate is under 3 per matched load
  shift (against 4.0 if every non-match were false). It fails if LR_b20's log-near share is inside
  the 2.5% to 97.5% range of the shifted shares; the page then says that the unmatched alarms are
  false at the rate the strict count implies, and that the precision of an event table is about 1
  in 5.
decision_rule: >-
  Pre-registered. Sources, frozen from experiment 227 with nothing tuned: LR_b20 (the headline),
  LR_b80, RO_b80, and PG_g70, on all 16 series and the whole real record, each learned alarm taken
  from the fold that holds out its Primary pair. The unit is a row of experiment 227's interval
  table (alarms on one series that start within 24 hours merge), stamped at its step. Classes, in
  this order: strict match (experiment 195's rule: step within 2 hours of a logged onset of the 35
  labelled load shifts over 1 day, end within the larger of 6 hours and 10% of the duration);
  log-near (a closure of the 1,040-row log of experiment 37, kept at 2 hours to 60 days, on the same
  series or a connected neighbour, starts or ends within 24 hours of the step; 6 and 48 hours are
  reported beside); balanced-opposite candidate; D8 drop edge; block day; bank holiday or clock
  change; none. Chance base rate: every alarm of a series is shifted by the same whole number of
  weeks (circular within the record, 200 shifts of 3 to 100 weeks either way, so hour of day and
  weekday are kept), and the log-near share is recomputed. Excess = observed share minus the mean of
  the shifted shares, reported with the 2.5% and 97.5% points of the shifts. Corrected precision =
  (strict matches + excess log-near alarms among the non-matches) / all alarms; its interval comes
  from resampling whole calendar months of alarms (1,000 resamples) with each alarm's own chance
  probability. False alarms per matched load shift = (all alarms minus estimated real alarms) /
  strict matches. The same table is made for each Primary substation, with D8, D9, and D10 shown
  separately. The verdict is `supported` if LR_b20 meets both bars of the prediction, `not
  supported` if its log-near share is inside the shifted range, and `partly supported` otherwise.
  No port threshold applies, because the experiment changes no code in production; the numbers feed
  the event table for NGED and the harm accounting of later experiments.
benchmark: >-
  Experiment 227: LR_b20 matches 21 of 35 logged load shifts at 5.0 alarms per match, which counts
  every alarm that is not a strict match as false.
row_set_id: >-
  Experiment 227's alarm tables on the whole real record; no forecast rows, so no leaderboard
metric: >-
  Share of alarms in each class, excess log-near share over the week-shifted base rate, corrected
  precision, and false alarms per matched load shift.
verdict: supported
best_arm: >-
  LR_b20 (experiment 227's learned alarm score, at the threshold that leaves 20 unexplained odd-month
  candidates): 58.8% of its non-strict alarms (77.3 on average over the folds) start or end within 24
  hours of a logged closure, against 31.2% for the week-shifted alarms, an excess of 27.7 percentage
  points (95% interval 18.5 to 36.0). Corrected precision is 45.3% (33.3% to 55.5%), and false alarms
  per matched load shift fall from 3.68 to 2.66 (1.67 to 4.86).
headline: >-
  An estimated 45% of LR_b20's 102 alarms are real load shifts (25 strict matches plus about 21 excess
  near-closure alarms), against 28% for LR_b80, 25% for RO_b80, and 18% for PG_g70, the gated pair
  alarm. LR_b20 raises 2.7 false alarms per matched load shift (1.7 to 4.9). No source shows an excess
  over the shifted base rate at D10, but the intervals there are wide.
lessons: >-
  Chance alone puts a logged closure edge within 24 hours of about 31% of the alarms, so the loose
  "explained within 24 hours" rule overcounts real alarms, and the strict rule undercounts them by
  about 21 alarms of 102 for LR_b20. Among the four sources, precision falls as the alarm count rises.
  The week-shifted base rate cannot say which individual alarms are real.
take_home_message: >-
  We estimate that about 45% of the alarms from the learned load shift screen at its strictest
  threshold (LR_b20) are real load shifts, with 2.7 false alarms for each logged load shift it finds.
  The looser sources, LR_b80 and RO_b80, which keep 80 candidate alarms, are real about 28% and 25% of
  the time. An event table built from LR_b20 would carry an estimated precision of about 45%.
advice:
  - >-
    State an alarm source's precision as strict matches plus the excess near-closure alarms over a
    week-shifted base rate, not as the share explained by any logged closure within 24 hours, which
    chance alone gives 31% of the time (figure 2).
  - >-
    Expect LR_b20 to have a higher precision than LR_b80 or RO_b80: 45% (33% to 56%) against 28% and
    25% (figure 3).
  - >-
    Do not read the per-Primary precision of any source at D8, D9, or D10 as evidence of real load
    shifts, because the excess there lies inside the shifted range for every source (per-Primary
    figure).
  - >-
    Report any harm accounting for an alarm source with LR_b20's excess near-closure alarms (about 21
    of 102) counted as real, and with strict matches only as the sensitivity row (headline).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
code: ml-experiments/00254-lsc-alarm-precision-against-full-log-with-chance-base-rate/
---

**We estimate that about 45% of the alarms raised by LR_b20 are real load shifts.** LR_b20 is the
learned alarm score of experiment 227, at the threshold that leaves 20 unexplained odd-month
candidates. For each of the 21 logged load shifts that its alarms match, LR_b20 raises 2.7 false
alarms (95% interval from resampling whole calendar months, 1.7 to 4.9). A load shift is demand that
the electricity network operator (NGED, National Grid Electricity Distribution) moves between
neighbouring Primary substations for hours or days. The load shift screen (LSC) raises an alarm when
it detects one. A "strict match" is an alarm whose step and end agree with one of 35 labelled load
shifts of over 1 day, using experiment 195's rule. We checked, for every alarm that is not a strict
match, whether a logged closure on the same or a neighbouring Primary substation starts or ends
within 24 hours. We compared that share with the share after moving every alarm by a whole number of
weeks (200 shifts of 3 to 100 weeks). For LR_b20, 58.8% of the non-strict alarms have such a
closure, against 31.2% after shifting, an excess of 27.7 percentage points. The pre-registered
prediction holds on the point estimates, so the verdict is `supported`. The interval for false
alarms per matched load shift includes the bar of 3, so that half of the prediction is not firmly
met.

![Figure 1: class of each alarm, by alarm source](classes.svg)

![Figure 2: observed share of non-matching alarms near a logged closure against 200 week-shifted shares](observed_against_shifted.svg)

![Figure 3: false alarms per matched load shift, and corrected precision by Primary substation](corrected_precision.svg)

**The loose rule overcounts and the strict rule undercounts, so the table states corrected numbers.**
"Alarms" counts each alarm once per series and stepped time, averaged over the cross-validation folds
that held the series out. Shares are shares of the non-strict alarms. "Corrected precision" is the
strict-match alarms plus the excess near-closure alarms, divided by all alarms. "False alarms per
matched load shift" divides the alarms that are not estimated to be real by the 21 matched load shifts.

| Source | Alarms | Strict-match alarms | Matched load shifts | Near-closure share of non-strict alarms | Shifted share | Excess (percentage points) | Corrected precision | False alarms per matched load shift |
|---|---|---|---|---|---|---|---|---|
| LR_b20 | 102.3 | 25.0 | 21 | 58.8% | 31.2% | 27.7 (18.5 to 36.0) | 45.3% (33.3% to 55.5%) | 2.66 (1.67 to 4.86) |
| LR_b80 | 245.3 | 29.0 | 24 | 49.5% | 31.0% | 18.4 (11.0 to 26.5) | 28.1% (19.7% to 37.1%) | 7.35 (4.77 to 12.50) |
| RO_b80 | 296.0 | 24.0 | 19 | 43.8% | 25.5% | 18.3 (10.5 to 26.3) | 24.9% (16.6% to 33.2%) | 11.70 (7.34 to 21.18) |
| PG_g70 | 85.0 | 3.0 | 3 | 52.4% | 37.4% | 15.1 (0.4 to 29.9) | 18.1% (2.8% to 34.0%) | 23.22 (9.61 to 80.02) |

- **Question and rule.** Does LR_b20's near-closure share beat a week-shifted base rate, and what do
  false alarms per matched load shift become? The pre-registered decision rule is met on the point
  estimates: share 58.8% against 30%, excess 27.7 against 15 percentage points, 2.66 false alarms
  against 3. The share lies above the 97.5% point of the shifts (39.7%).
- **Row set and data.** Experiment 227's four frozen alarm sources on all 16 series and the whole
  record (2022-01 to 2026-06), with no tuning. The log is experiment 37's 629 closures of 2 hours to
  60 days. Weather is not used. D-numbers are the published labels of Primary substations, and
  "series N" names an unlabelled Primary substation.
- **Other windows.** At 6 hours LR_b20's share is 48.5% against 13.7% after shifting; at 48 hours it
  is 65.9% against 46.2%. Shifts of 3 to 12 weeks, which keep the season, give 58.8% against 30.5%.
- **Months not used to set the threshold.** LR_b20's budget of 20 unexplained odd-month candidates was
  set on odd months, where "unexplained" means not near a logged closure. That could inflate the share.
  The even months, which did not set the budget, show a larger share: 72.9% against 33.0% after
  shifting (39.8 percentage points, 27.4 to 51.2), corrected precision 60.2% (46.1% to 72.0%), and 1.54
  false alarms per matched load shift (0.81 to 3.35). Odd months give 48.3% against 29.8% (18.5
  percentage points, 4.9 to 31.8) and precision 31.0% (15.6% to 44.7%). This split was chosen after
  seeing the results. The headline stays on the whole record.
- **Leaving one Primary substation out.** The excess stays between 24.5 and 30.4 percentage points.
  False alarms per matched load shift stay between 2.42 and 3.40, and exceed the bar of 3 without
  series 9, series 13, D16, or D5 (3.28, 3.14, 3.02, and 3.40). D5 has the highest precision (84.8%,
  68.0% to 93.9%).
- **Where the excess is absent.** LR_b20's excess at D10 is -3.8 percentage points (-39.0 to 38.6; 5
  alarms) and at D8 +20.6 (-12.9 to 62.9; 5 alarms). LR_b80's excess is +1.0 at D10 (-17.6 to 22.4; 20
  alarms). RO_b80 gives +4.8 at D9, -1.7 at D10, and +5.8 at D8. The pair alarm gives -10.4 at D10. No
  source shows an excess at D10, and the D8 and D9 values lie inside the shifted range. The intervals
  are wide, so the data neither confirm nor rule out real load shifts there.
- **Where the excess is clear.** LR_b20's excess is above zero at D5 (+70.5, 49.1 to 83.6), D16
  (+36.7), and series 15 (+40.2). The other Primary substations have too few alarms to say.
- **Limits.** "Not near a logged closure" is not proof of a false alarm, because the log covers only
  the trial area and only what NGED logged. Each of the 16 series gets its own independent shift. The
  shifted alarms keep weekday and hour of day but not season, except in the 3-to-12-week row. The
  month-resampled intervals hold each alarm's chance probability at its mean over the 200 shifts, so
  they leave out the shift-sampling noise and are slightly too narrow. The pair alarm is counted as 85
  alarm events, whereas experiment 227's interval table has 213 rows that include partner rows, so
  its false alarms per matched load shift are not comparable with experiment 227's. The balanced-
  opposite class says nothing about the pair alarm, because the pair alarm is defined by opposite
  steps; 39 of its 85 alarms land in that class after the strict and near-closure classes have taken
  theirs. The cleaning flag (`substation_zero`) is not used as a class, because the cleaned power
  table is not part of the stored inputs. The false-alarm rate divides by the 21 matched load shifts,
  not by the 25 strict-match alarms (which would give about 2.2).
- **What to try next.** Read the 45 or so LR_b20 alarms that are near a closure but not strict matches
  (about 21 of them excess over chance) with NGED, and report any harm accounting with those excess
  alarms counted as real.

## Checks

- **Median forecast minus mean forecast, per site.** Not applicable: the experiment forecasts nothing.
- **Mean-forecast error beside each quantile-score contrast.** Not applicable, for the same reason.
- **Triggers.** None fired: no control exists, the result agrees with its prediction, leaving one
  Primary substation out keeps the excess above 24 percentage points, and no nested arm exists.
- **Code review (Sonnet).** The review found four defects, all fixed: the threshold might inflate the
  share (the even-month check shows it does not); per-Primary false-alarm denominators mixed events and
  alarms, so the per-Primary table shows precision and excess only; the all-false baseline used a
  different definition of real alarms from the corrected count, so both now count 25 strict-match
  alarms as real; and the week shift wrapped by 4 days, so it now wraps by 234 whole weeks.
