---
id: "00132"
slug: ro-nc-rescored-with-p12-and-p13-residuals-in-2025
theme: switching-detectors
also_relevant_to: []
status: planned
size: medium
hypothesis: >-
  Post hoc, and descriptive: the 6 real load transfers over 1 day at P12 and P13 that the refit-offset
  neighbour-count detector (RO-NC) could not score in experiment 101 are found about as often as
  injected transfers of the same step size and metering, once P12 and P13 have a residual in 2025,
  so that their absence from the residual panel, and not a property of those transfers, explains
  the six misses.
prediction: >-
  Written before any result exists. With a residual for P12 and P13, RO-NC finds between 2 and 4 of
  the 6 transfers, which raises its count on the 35 transfers over 1 day from 12 to between 14 and
  16 (recall 0.40 to 0.46). The residual noise at P12 and P13 is larger than at the other Primaries
  because each of the two series has under 17 months of history, so step sizes in noise units are
  smaller than the 4 noise units the detector needs for most of the 6 transfers.
decision_rule: >-
  Descriptive only, because 6 transfers cannot rank anything. The write-up reports how many of the 6
  are found, the recall on the 35 with a 95% Wilson interval, and the injected-transfer recall at
  P12 and P13. The threshold for porting is not applicable: no production change follows.
benchmark: >-
  RO-NC in experiment 101 (12 of 35 found over 1 day; 0.357 of the moved MW), and injected-transfer
  recall in experiment 82 (0.770 of the moved MW).
row_set_id: >-
  none (scores stored detector alarms against the 66 labelled real transfers of experiment 41, with
  no forecast)
metric: >-
  Events found (an alarm within 2 hours of the labelled start and an end within the larger of 6
  hours and 10% of the duration), MW-weighted recall, injected-transfer recall at P12 and P13,
  residual noise scale in MW and in the share of capacity.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00132-ro-nc-rescored-with-p12-and-p13-residuals-in-2025/
---

# Experiment 132: does RO-NC find the 6 real transfers at P12 and P13 with a residual?

**Plan only; no result yet.**
[Experiment 101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md) found
that 6 of the 35 real transfers over 1 day lie at two Primary substations, P12 and P13, whose
residual panel holds no readings in the transfer windows. The task was written as "restore the
missing readings". The diagnosis changed the question.

## What was missing

- **The readings are not missing.** In the raw `power_time_series` table and the
  `cleaned_power_time_series` table, both series start on 2025-02-06 and cover all 6 transfers
  (March to November 2025). Cleaning flags `substation_zero` on 0% to 12% of the half-hours of a
  month. The experiment counts the flagged half-hours inside each transfer window and prints them.
- **The residual is missing.** The panel of experiment 37 reads the out-of-sample fits of experiment
  17, which start a series only when it has 17,520 observed half-hours (one year) before the
  purge. P13 therefore enters in February 2026 and P12 in March 2026, after all 6 transfers.
- **No raw source holds earlier readings.** The two series have no readings before 2025-02-06, so
  there is nothing to re-read, and no cleaning rule to relax.

## What the experiment does

1. **Build a residual for P12 and P13 over 2025-02-06 to the start of each series' existing
   panel, with blocked out-of-sample refits.** Each calendar month (and, for the offset panel, each
   half-month starting on the 16th) is held out in turn. The XGBoost model of experiment 17's
   arm H0.5 is fitted on the series' own rows outside the held-out block, a 7-day purge on each
   side, and every row from 2026-07-01 onwards is excluded. The weights are experiment 17's
   half-life of 0.5 years, measured from the held-out block in either direction. This uses data from
   after the transfer, which is a lab shortcut of the kind the switching experiments allow.
2. **Check that nothing leaks from the transfer window.** For each of the 6 transfers, the refit
   that scores the transfer's month holds that month's block out; a check script prints that no
   training row lies inside the held-out block or its purge, and that the stored residual of every
   transfer half-hour is the held-out prediction's. The existing 2026 panel rows are left as they
   are.
3. **Rescore RO-NC unchanged.** The logistic scores and thresholds are experiment 82's stored
   files, loaded and not refitted, and the injection draw uses experiment 82's seeds. Adding
   residuals for P12 and P13 also changes the neighbour counts that LOGIT_SAME and RO-NC use at
   P10, P14, P16, and P25, so all 35 transfers are rescored, and the write-up lists any transfer
   whose outcome changed.
4. **Report** how many of the 6 are found, the recall on the 35 with intervals, injected-transfer
   recall at P12 and P13, and how the residual noise at P12 and P13 compares to the other Primaries.

## Sizing

- **What gets stored:** private tables in `data/ml-experiments/00132-*/` only; nothing under `src/`
  or `packages/`.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The residual could come from a forward-only fit with a
  shorter history, from blocked refits, or from an in-sample fit. Blocked refits keep the scored
  residual out of sample without leaving the 6 transfers uncovered.
- **Callers not nameable without searching:** no; the scripts are copies of experiment 101's.

## Caveats written in advance

- **The new residuals are not comparable one-for-one with experiment 17's.** They come from under 17
  months of history, against 4 to 6 years at the other Primaries, and from blocked rather than
  forward fits. The write-up compares the residual noise in MW and as a share of capacity.
- **Six transfers are descriptive only.** A count of 2 or 3 found cannot be told from 2 or 3 found at
  the injected rate.
