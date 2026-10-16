---
id: "00068"
slug: sw2-5-train-detect-retrain-loop
status: planned
size: medium
hypothesis: >-
  When the lagless XGBoost model (the model whose forecast residual the switching detector reads)
  is refitted with zero weight on the half-hours inside detected switching intervals, and the
  detector is then run again on the new residual, three times over, the detector finds a larger
  share of the MW moved by load transfers at the same number of unexplained alarms. The refit also
  lowers the forecast error on half-hours outside logged switching. The mechanism is that a
  lagless XGBoost model trained through a transfer partly learns the transfer. That shrinks the
  transfer's step in the residual and biases the forecast.
prediction: >-
  On the world where the injected transfers are part of the training data (world B, defined in the
  plan), MW-weighted recall (the share of the moved MW that sits in found transfers) on even months,
  pooled over the injection shapes S2, S3, and S4, rises from round 0 to round 3 by +0.00 to +0.03
  (C1). The gain is expected mostly on transfers that last over 7 days, because a recency half-life
  of 0.5 years and a monthly refit limit how much one transfer can teach the model. The
  mean absolute error of the forecast on even-month half-hours outside logged and labelled
  switching, in the world of real data (world A), changes by -1.0% to 0.0% from round 0 to round 3
  (C2). Experiment 61 found the benchmark detector at 0.63 to 0.65 on injected transfers that were
  not in the training data. A gain of +0.02 or more in recall, and a fall of 1% or more in error,
  are not the expected outcome. The likeliest verdicts are `refuted` and `small gain`.
decision_rule: >-
  Two planned contrasts, each read at 97.5% (Bonferroni across C1 and C2) from 20,000 resamples of
  whole calendar months. C1 is the paired difference in MW-weighted recall, round 3 minus round 0,
  in world B, averaged with equal weight over S2, S3, and S4, on the injected transfers that start in
  even months. C2 is the relative change in the mean absolute error of the forecast, round 3 minus
  round 0, in world A, over even-month half-hours that lie more than 24 hours from every logged
  closure on the series and from every labelled real transfer on the series, with equal weight per
  series. The verdicts are exclusive and are tried in this order. `invalid` when the round-0 check in
  the plan fails. `refuted` when the upper end of the C1 interval lies below +0.02 and the lower end
  of the C2 interval lies above -1.0%, which also covers a loop that makes either metric worse.
  `supported` when the C1 interval lies wholly above zero with a point value of at least +0.02, the
  C2 interval lies wholly below zero with a point value of at most -1.0%, and the even-month
  unexplained-alarm count of every round is within 10% of round 0's. `partly supported` when exactly
  one of the C1 condition and the C2 condition holds in full and the alarm-count condition holds.
  `small gain` when no earlier verdict applies and an interval excludes zero in the predicted
  direction. `inconclusive` otherwise, including every case where the alarm-count condition fails.
  Both thresholds, +0.02 and -1.0%, are judgements: the loop adds one detector run and one refit per
  round to the training pipeline and nothing to the serving path, and a gain smaller than these
  would not justify three extra refits per release. The by-round differences (round 1 minus round 0,
  round 2 minus round 1, round 3 minus round 2) in both metrics, the comparison of round 1 with a
  random-mask control of equal volume (C3), world A's recall, and the labelled real transfers are
  reported without changing the verdict.
benchmark: >-
  Round 0: the refit with no mask, scored with the PE detector of experiment 61 (D0M's alarms at
  the 24-hour window, each interval ending at the strongest opposite step within 14 days), whose
  threshold is matched to at most 80 odd-month unexplained alarms in every round.
row_set_id: >-
  none (the experiment scores detector alarms against injected transfers and the lagless XGBoost
  model's own forecast error)
metric: >-
  MW-weighted recall (headline) as in experiments 55 to 62; mean absolute error of the lagless
  XGBoost forecast, in points of series capacity; the share of the moved MW that is missed;
  unweighted recall; recall by donor step in noise units split by metered receiver; recall by
  transfer duration; per round, the noise scale, the threshold, the odd-month and even-month
  unexplained-alarm counts, and the masked share of training weight; the count of the 35 labelled
  real transfers of more than 24 hours that are found.
code: ml-experiments/00068-sw2-5-train-detect-retrain-loop/
verdict: null
best_arm: null
lessons: null
advice: []
production_status: null
adopted_in: null
---

# Experiment 68 (SW2-5, part of the switching-event detector series): does refitting without the detected switching help the detector?

*Status: planned. This page is the plan; the write-up replaces it when the run finishes.*

## Plan

**The lagless XGBoost model.** The detector reads the residual (measured power minus forecast) of
[experiment 17](../00017-training-window-and-retraining-rule-drift-demand/result.md)'s
`H0.5` arm. That
XGBoost model uses weather, clock, and calendar inputs and no past power. It is refitted monthly on
rows that end 7 days before the month, with recency weights of half-life 0.5 years. The 16 Primary
substation series are fitted one at a time.

**The loop.** Round 0 is the refit with no mask. Round `k` is the same refit with zero weight on the
half-hours inside the intervals that the PE detector found on round `k-1`'s residual. After each
refit the detector is run again on the new residual. Three rounds are run. The mask for the refit of
month `M` uses only intervals whose end lies at least 15 days before the refit's training end (the
14 days that the end search looks ahead, plus the 24 hours that the step window needs). The
detections themselves use the whole span of the previous round's residual and a threshold tuned on
all odd months, so the loop is training-time hindsight for research. A production loop would tune on
the months before each refit, and the write-up says what that changes only if the result is
supported.

**Matched alarm rate.** In each round the noise scale is re-estimated on odd months from that round's
residual, and the threshold is lowered from a high value in steps of 0.05 noise units to the lowest
threshold that leaves at most 80 odd-month unexplained alarms. An alarm is unexplained when it lies
more than 24 hours from every logged closure on its series. The report prints, for every round, the
noise scale, the threshold, and the unexplained-alarm counts for odd and even months, and the recall
at round 0's threshold as a sensitivity.

**Two worlds.** World A is the real data. It gives the forecast error (C2), the alarm counts, and the
labelled real transfers, which are in the training data and in the mask by construction, so the
count of those found is descriptive. World B puts injected transfers into the measured power before
the lagless XGBoost model is fitted, so the model can learn them as it would learn real ones. The
injected transfers of world B are a fresh draw of 4,000 headline-band transfers (25% or more of the
receiving load, over 1 day) from experiment 58's generator, with onsets in even months, reduced
greedily to a set with no two transfers on one series within 1 day of each other. (Experiment 58's
own 620 even-month transfers overlap too much on one series: only 103 of them fit.) Each
transfer takes shape S2, S3, or S4 in turn, so the pooled recall covers the three shapes the
detector does not assume. World B contains no injected transfers in odd months, so the odd-month
tuning is unaffected. The loop runs in world B on its own residuals and masks. World A with injected
transfers added to the residual only (experiment 61's scoring, in which the model never saw them) is
reported as a secondary result, because an injected transfer that is not in the training data cannot
test the stated mechanism.

**Control.** The random-mask control (RC) is round 1 with a mask of equal volume on each series: each
detected interval of world A is moved to a random start in the same series and calendar month, with
the same duration, avoiding the detected intervals. The write-up gives the masked share of
recency-weighted training mass for round 1 and RC. RC is compared with round 1 in both metrics (C3).
Only round 1 has a control.

**Round-0 check.** The round-0 refit must match experiment 17's stored `H0.5` predictions (a
difference of 0 for the first month tested). PE on the stored panel must reproduce the 0.63 to 0.65
of experiment 61. A failure of either makes the verdict `invalid`.

**Sizing, one line per trigger.** (1) What gets stored: private predictions and tables under `data/`
and no Delta table. (2) Production serving path: not touched. (3) Degradation rule: not touched.
(4) More than one defensible design: yes, for the mask rule and the number of rounds, and the plan
fixes both. Any other mask rule is a post hoc arm. (5) Code whose callers are unfamiliar: no, because
the code copies the harness of experiment 61 and the fit of experiment 17. The size is medium. One
Sonnet agent reviewed this plan, one Sonnet agent reviews the code, one Sonnet agent checks the
write-up against the printed report, and one Sonnet agent reviews the prose. No Opus agent is used,
because the maintainer has asked for Opus only where vital.

**Earlier advice.** The plan follows [experiment 61](../00061-sw2-6-paired-step-intervals/result.md)
by ending each interval at the strongest opposite step within 14 days and by holding unexplained
alarms to 80. [Experiment 66](https://github.com/openclimatefix/nged-substation-forecast/pull/1297)
recommends confirming its post hoc first-return rule on a fresh draw before adopting it, so the plan
keeps the 14-day rule. [Experiment 38](../00038-switching-mask-value/result.md) refuted masking
detected switching from the error correction's history. This experiment masks training rows of the
lagless XGBoost model, which experiment 38 did not test. The standard tables of experiments 55 to 62
(recall by donor step split by metered receiver, size-weighted headline, MW missed) are reported.

## What the next experiment should answer

*Written after the results are read.*
