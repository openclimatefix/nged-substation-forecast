---
id: "00132"
slug: ro-nc-rescored-with-p12-and-p13-residuals-in-2025
theme: switching-detectors
also_relevant_to: []
status: done
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
verdict: partly supported
best_arm: >-
  With the readings restored and blocked residuals, RO-NC finds 4 of the 6 transfers at P12 and P13
  (0.67; 95% interval 0.30 to 0.90), against 0 of 6, and 17 of the 35 transfers over 1 day (0.49;
  0.33 to 0.64), against 12 of 35 (0.34; 0.21 to 0.51). MW-weighted recall on the 35 is 0.631
  (0.433 to 0.787), against 0.357 (0.169 to 0.554).
take_home_message: >-
  The 6 real transfers at P12 and P13 were missed because the readings that show them were removed,
  not because the detector cannot see them. Once the zero readings of the switched-out transformer
  are restored from the raw store, the detector finds 4 of the 6, and 17 of all 35 transfers over 1
  day instead of 12. The cleaning rule that removes a reading of exactly 0 MW at a Primary substation
  also removes real periods when a transformer is switched out.
lessons: >-
  In 5 of the 6 transfers one of the two transformers of the Primary reads exactly 0 MW, and the
  `substation_zero` cleaning rule flagged those half-hours. Restoring 426 half-hours from the raw
  store, and building residuals for 2025 with held-out blocks, lets RO-NC find the 4 transfers whose
  steps exceed 10 noise units. The 2 it misses have steps under 1 noise unit. The restoration also
  raised the unexplained alarms at the stored thresholds by 7 to 15 on odd months.
advice:
  - >-
    Score a detector on real transfers at P12 and P13 only with the zero readings restored, because
    the cleaned table removes the transformer-off readings that make up the transfer (restored
    half-hours table).
  - >-
    Count the half-hours that `substation_zero` flags inside each labelled transfer before
    concluding that a detector missed it, because the rule flagged 426 half-hours in 5 of the 6
    transfers at P12 and P13 (restored half-hours table).
  - >-
    Re-tune the thresholds after adding series with new residuals, because odd-month unexplained
    alarms rose from 80 to 95 at PF14's stored threshold when P12 and P13 gained residuals
    (alarm counts).
  - >-
    Quote RO-NC's real-transfer recall as 17 of 35 (0.49; 0.33 to 0.64) only together with the
    restoration and the blocked residuals for P12 and P13, and keep 12 of 35 for the panel without
    them (figure 1).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00132-ro-nc-rescored-with-p12-and-p13-residuals-in-2025/
---

# Experiment 132: does RO-NC find the real transfers at P12 and P13 with their readings restored?

**The detector missed the 6 real transfers at P12 and P13 because the readings that show them had
been removed, and with those readings restored it finds 4 of the 6.** The two series are the two
transformers of one Primary substation. In 5 of the 6 transfers, one transformer reads exactly 0 MW
for the whole transfer while the other carries the load, and the cleaning rule `substation_zero`
flags a reading of 0 at a Primary substation as a telemetry fault. We re-read the 426 flagged
half-hours from the raw power table, built residuals for P12 and P13 over 2025 with held-out
blocks (experiment 17's XGBoost model, which has no residual before 2026 because it needs a year
of history), and rescored the refit-offset neighbour-count detector (RO-NC) of
[experiment 82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md) with its
stored thresholds and logistic scores. Experiment 101 had found 12 of the 35 labelled real
transfers over 1 day. The detector now finds 17 of the 35 (0.49; 95% Wilson interval 0.33 to 0.64).

![Share of the transfers found, with and without the restoration](figure_1_found_by_panel.svg)

## Results

- **At P12 and P13:** 4 of the 6 transfers are found (E49, E50, E52, E54), against 0 of 6. Their
  steps are 10.4 to 17.1 noise units. The 2 not found are E44, where only 10 half-hours of the
  99-half-hour window read 0, and E59, where P12 has no raw readings at all. Their steps are 0.6 and
  0.9 noise units.
- **Elsewhere:** 13 of the other 29 transfers are found (0.45; 0.28 to 0.62), against 12. The extra
  one is E55 at P13 and P14, whose step rises from 1.9 to 11.4 noise units now that P13 has
  readings. No transfer found before is lost.
- **MW-weighted recall on the 35:** 0.631 (0.433 to 0.787), against 0.357 (0.169 to 0.554).
- **Injected transfers:** the old panel reproduces experiment 82's 31,200 stored rows exactly. On
  the new panel's draw, headline-band recall is 0.772 (0.669 to 0.858), against 0.770. Transfers
  that touch P12 or P13 are found 0.196 of the time, MW-weighted 0.767 (0.645 to 0.869), over 56
  injected transfers and 3 shapes.
- **Restored half-hours:** 426, all exactly 0 MW, and none invented: E44 10 at each series, E49 102
  at P12, E50 103 at P13, E52 1 at P12 and 100 at P13, E54 100 at P13. E50 at P13 has 43
  half-hours that are absent from the raw table, and none are restored.
- **Variants:** blocked fits with equal weights, and with the six transfer windows masked from the
  training rows, give the same 17 found and the same outcome on every transfer.

![Raw readings of the two transformers around each transfer](figure_2_readings_of_the_six_transfers.svg)

## Checks

- **No leakage:** every blocked fit has 0 training rows inside its held-out block or the 7-day
  purge, for all 3 variants and both residuals (printed per block by `e00132_report.py`). The
  transfers' blocks are listed there.
- **Noise:** the noise unit at P12 is 0.431 MW, 0.143 of capacity, against a median of 0.030 over
  the other 14 Primaries (4.8 times). The noise unit at P13 is 0.037 of capacity (1.2 times). The 4
  transfers found have steps over 10 noise units, so the larger noise unit at P12 does not decide
  them.
- **Alarm rate:** at the stored thresholds, unexplained alarms on odd months rise from 80 to 95
  (PF14), 75 to 88 (X), 80 to 87 (LOGIT_SAME), and 80 to 87 (RO-NC). The thresholds were not
  re-tuned. All 5 newly found transfers have steps over 10 noise units, so a threshold raised to
  restore 80 alarms would keep PF14's and X's finds, and RO-NC's was not re-checked.
- **Audit triggers:** none fires. The control, experiment 82's injected rows on the old panel,
  reproduces exactly. The experiment fits no quantile model, so the mean-versus-median check does
  not apply.

![Noise unit of each series as a share of its capacity](figure_3_noise_by_series.svg)

## Caveats

- **Six transfers, descriptive only.** Five of the 6 have readings at both transformers, and the
  load moves between the two transformers of one Primary, which differs from the injected transfers
  between Primaries. E59 has no readings at P12.
- **The expected count is not informative.** The pre-registered rule, injected recall at each
  transfer's step band and metering, gives 0.8 expected of the 6, because the injected draw holds
  no transfer of 10 noise units or more without a metered receiver, and the rule fell back to a band
  of 4 injected transfers. Using the metered cell (0.949) gives about 3.8. The count found, 4,
  matches the second figure, so the data cannot tell the two rules apart.
- **The residuals use data from after each transfer.** Held-out blocks are out of sample but use
  future rows, and have under 17 months of history.
- **The injected draw changed.** More series hold power, so the draw accepts different transfers.
  Recall on long injected transfers (S1, even months) is 0.449, against 0.519.

## What the next experiment should answer

The unexplained alarms rose by 7 to 15 on odd months, so the next step is to re-tune the thresholds
on the new panel and confirm that the 4 found transfers survive. Whether the 2 transfers with steps
under 1 noise unit can be found is a question for the detector's steps, not for the data.

Code: `ml-experiments/00132-ro-nc-rescored-with-p12-and-p13-residuals-in-2025/`; every number
above is printed by `e00132_restore.py`, `e00132_report.py`, and `e00132_inj_report.py`.
