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
  injected transfers of the same step size and metering, once P12 and P13 have a residual in 2025
  and the zero readings inside the transfers are no longer flagged as faults, so that the missing
  readings, and not a property of those transfers, explain the six misses.
prediction: >-
  A guess, written before any result exists, because the step sizes in noise units cannot be known
  before the residuals exist. The rule that replaces the guess is fixed here: for each of the 6
  transfers, the expected number found is the injected recall of experiment 82 at that transfer's
  measured step band and metering, as in experiment 101. At P12 and P13 one transformer reads zero
  for the whole of 5 of the 6 transfers, so the guess is that most of the 6 have steps over 4 noise
  units and RO-NC finds 3 to 5 of them.
decision_rule: >-
  Descriptive only, because 6 transfers cannot rank anything. The hypothesis is contradicted if RO-NC
  finds at most 1 of the 6 when the rule above expects 3 or more, and is consistent with the
  injected recall if the count found is within 1 of the expected count. The write-up reports the 6
  and the other 29 transfers separately, with a 95% Wilson interval on each count. No production
  change follows, so no porting threshold applies.
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

# Experiment 132: does RO-NC find the 6 real transfers at P12 and P13 with their readings restored?

**Plan only; no result yet.**
[Experiment 101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md) found
that 6 of the 35 real transfers over 1 day lie at two Primary substations, P12 and P13, whose
residual panel holds no readings in the transfer windows. The two series are the two transformers of
one Primary substation. The task was written as "restore the missing readings", and the diagnosis
found two separate causes.

## What was missing

- **Cause 1: the residual panel starts after the transfers.** In the raw `power_time_series` table,
  both series start on 2025-02-06 and cover all 6 transfers (March to December 2025). Experiment
  17's fits, which the panel of experiment 37 reads, start a series only when it has 17,520 observed
  half-hours (one year) before the purge, so P13 enters the panel in February 2026 and P12 in March
  2026.
- **Cause 2: the cleaning rule `substation_zero` flags the transfers' own readings.** In 5 of the 6
  transfers, one of the two transformers reads exactly 0 MW for all or most of the window (E44: 10
  of 99 half-hours, E49: 102 of 103 at P12, E50: 103 of 109 at P13, E52: 99 of 103 at P13, E54: 100
  of 103 at P13). The other transformer carries the load: in E52 P13 falls from 2.99 MW to 0.03 MW
  and P12 rises from -0.92 MW to 1.93 MW, so the sum stays near 2 MW. The rule flags a reading of 0
  at a Primary substation as a telemetry fault, so these half-hours are NaN in `features.parquet`
  and never reached the panel.
- **E59 has no readings at P12 at all.** The raw table holds none for the 52 half-hours of the
  window, so no route restores them. P13 is present.

## What the experiment does

1. **Restore the zero readings by the legitimate route.** For each of the 6 transfers, the
   half-hours flagged `substation_zero` at P12 and P13 within 2 days either side of the labelled
   window are re-read from the raw `power_time_series` table and written to a private table under
   `data/ml-experiments/00132-*/`. No value is interpolated or invented, and neither the raw store
   nor the cleaned table nor any code under `src/` or `packages/` changes. The write-up states how
   many half-hours were restored and which window had none to restore (E59 at P12).
2. **Build a residual for P12 and P13 up to the start of their stored panels, with blocked
   out-of-sample refits.** Calendar months (main residual, 1st to 1st) and half-months (offset
   residual, 16th to 16th) are held out in turn. The XGBoost model of experiment 17's arm H0.5 is
   fitted on the series' own cleaned rows outside the block with a 7-day purge on each side, before
   2026-07-01, and each row's weight halves every 0.5 years of distance from the block. The fits use
   data from after the block, a lab shortcut of the kind the switching experiments allow. The
   restored zeros are scored but never trained on.
3. **Check for leakage and for a different noise structure.** The fit script records, per block,
   the training rows inside the block or its purge (it must be 0) and the block edges, printed for
   each of the 6 transfers. The write-up shows the residual's median absolute deviation by month at
   P12 and P13 beside the other Primaries, in MW and as a share of capacity. A sensitivity run
   fits each block with equal weights (no half-life), and a second sensitivity run also masks the
   other labelled transfer windows from the training rows. The 6 outcomes are reported under all
   three variants.
4. **Count the half-hours at each stage.** Per transfer window and series, the write-up lists
   half-hours present in the raw table, flagged `substation_zero`, null in `features.parquet`, and
   restored.
5. **Rescore RO-NC unchanged.** The end rule, the 2-hour and 10% matching window, experiment 82's
   thresholds, and its stored logistic scores are loaded and not refitted. Because P12 and P13 are
   neighbours of each other and of P10, P14, P16, and P25, the neighbour counts at those series
   change, so the write-up reports: the 6 transfers and the other 29 separately; every transfer
   whose outcome changed in either direction; and the unexplained-alarm count over odd months and
   over the whole history, before and after, at the stored thresholds.
6. **Score injected transfers on the old and the new panel.** Experiment 82's fresh draw, with its
   seeds, is scored on experiment 101's panel (which must reproduce 0.770 of the moved MW) and on
   the new panel. The new panel's draw differs, because the draw accepts and rejects events by
   which series hold power, so the write-up gives the injected recall on the new panel, on the new
   panel restricted to events that touch neither P12 nor P13, and at P12 and P13 alone. The expected
   count for the 35 is recomputed from the new rates.

## Sizing

- **What gets stored:** private tables in `data/ml-experiments/00132-*/` only; nothing under `src/`
  or `packages/`.
- **Production serving path:** not touched.
- **Degradation rule:** not touched. The write-up reports that `substation_zero` flags real zero
  readings at a Primary substation whose transformer is switched out, which is for the maintainer to
  weigh.
- **More than one defensible design:** yes. The residual could come from a forward-only fit with a
  shorter history, from blocked refits, or from an in-sample fit. Blocked refits keep the scored
  residual out of sample without leaving the transfers uncovered.
- **Callers not nameable without searching:** no; the scripts are copies of experiment 101's.

## Caveats written in advance

- **The new residuals are not comparable one-for-one with experiment 17's.** They come from under 17
  months of history, against 4 to 6 years at the other Primaries, and from blocked rather than
  forward fits, so seasons with no other year in the training rows are extrapolated. The noise scale
  of each series is still estimated on its own odd months.
- **The two series are one site.** A step at one is likely to appear at the other, so two
  same-sign neighbours at P12 and P13 are close to one piece of evidence counted twice.
- **The transfers at P12 and P13 swap load between the two transformers of one Primary.** They are
  labelled transfers, but they move load inside the substation, which differs from the injected
  transfers between Primaries.
- **Six transfers are descriptive only.**
