---
id: "00147"
slug: causal-live-end-rule-for-ro-nc
theme: switching-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00147-causal-live-end-rule-for-ro-nc/ only
  (per-transfer outcome tables), no prediction set. Serving path: not touched. Degradation rule:
  not touched. Designs: three defensible causal end rules, all tested as arms. Callers: this
  folder's scripts only, which copy experiment 138's code and read its stored arm B score and
  threshold without changing them.
hypothesis: >-
  The end of a switching interval that RO-NC has alarmed on can be called from data available at the
  time, with no lookahead, at a small loss of the share of moved MW found, a low rate of false ends
  (an end called before the transfer has ended), and a lag of a few days at most. The offline end
  rule of experiments 61 and 130 picks the largest opposite step within 24 hours after the first
  return step, which needs data from up to 48 hours after that first step, so its "known from"
  time is up to 24 hours too early.
prediction: >-
  Written before any result. The rule that ends at the first return step (R1) loses 0.02 to 0.08 of
  the share found against the offline end rule and calls 0.05 to 0.15 of the found onsets too
  early; the rule that waits for the return step to hold for 6 hours (R3) halves the false ends of R1
  and loses 0.01 to 0.04 of the share found; the rule that waits for the largest step to pass (R2)
  gives the same ends as the offline rule on at least 0.9 of transfers, with a median lag of 48 hours
  after the true end.
decision_rule: >-
  Three planned contrasts, each live rule (R1, R2, R3) minus the offline end rule (R0) in the
  MW-weighted share found (onset within 2 hours and end within the larger of 6 hours and 10% of the
  duration), pooled over shapes S2, S3, and S4, on experiment 138's fresh headline-band draw, for
  transfers that start in even months, with RO-NC arm B alarms (experiment 138's stored score and
  threshold) and no refitting. The 98.33% interval (Bonferroni for three contrasts) resamples whole
  calendar months 20,000 times. Lag is the stamp at which the end is called minus the true end, in
  hours, for transfers whose end is found (median and 90th percentile). A false end is a call,
  among transfers whose onset is found, of an end earlier than the true end by more than the end
  tolerance. A late end is later than the true end by more than that tolerance. A live rule is
  `supported` when the lower end of its contrast is above -0.05, its false-end share is at most
  0.10, and its median lag is at most 72 hours. The hypothesis is `supported` if at least one rule is
  supported, `refuted` when the upper end of every contrast is under -0.05, and `inconclusive`
  otherwise. R0's honest call time is reported beside the live rules, and the 35 real transfers
  over 1 day are descriptive.
benchmark: >-
  R0, the offline end rule of experiment 130 (arm B, share found 0.794 on the same draw).
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, not
  forecasts)
metric: >-
  MW-weighted share found, end lag in hours, share of false ends, share of late ends.
code: ml-experiments/00147-causal-live-end-rule-for-ro-nc/
production_status: not-for-production
adopted_in: none
---

# Experiment 147: can the end of a switching interval be called without looking ahead?

**This page is the plan. The result is not in yet.** After RO-NC alarms on the start of a load transfer,
the offline end rule searches up to 14 days for the first step in the opposite direction, then takes
the largest opposite step within 24 hours after it. A live service cannot do that, because the
24-hour step statistic at a stamp needs the 24 hours after the stamp. This experiment tests three
rules that use only data available when the end is called, and measures how late and how often
wrong each is against the offline rule.

## Arms

- **R0 (reference):** the offline end rule of [experiment
  130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md). Its honest call time is the first
  return step plus 96 half-hours (48 hours), because the largest step within the next 24 hours is
  only known then.
- **R1, first return step:** the end is the first stamp whose opposite step reaches the offline
  threshold (half the onset step, at least 1.5 noise units), called 24 hours later, when that
  stamp's step can be computed.
- **R2, running peak:** after the first return step, the end is the stamp of the largest opposite
  step so far. It is called once the step falls below 0.8 of that peak, or 24 hours after the first
  return step if it has not.
- **R3, hold:** the end is the first stamp from which the opposite step stays above the threshold
  for 12 half-hours (6 hours), called when the 12th stamp's step can be computed.
