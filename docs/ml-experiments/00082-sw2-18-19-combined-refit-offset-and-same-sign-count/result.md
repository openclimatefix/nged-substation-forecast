---
id: "00082"
slug: sw2-18-19-combined-refit-offset-and-same-sign-count
status: planned
size: simple
hypothesis: >-
  Experiment 71's refit-offset detector X (the 24-hour step statistic taken from whichever of two
  monthly-refit residual models did not refit within 24 hours of the stamp) and experiment 70's
  logistic score with the same-sign neighbour count (LOGIT_SAME) fix different misses of the
  benchmark PF14. Applying the logistic score to the candidates of X (XL) therefore finds more of
  the moved MW than PF14, and more than either part alone, at the same count of unexplained alarms,
  on a fresh draw of injected transfers.
prediction: >-
  MW-weighted recall, even months, shapes S2, S3, and S4 pooled, first-return end rule. PF14 finds
  about 0.67 (experiments 69, 70, and 71: 0.665 on the same seed). X gained +0.050 and LOGIT_SAME
  +0.048 against PF14. The first gain sits in the 13% of transfers within 24 hours of a refit, and
  the second in transfers with no metered receiver and a donor step of 2 to 6 noise units, so the
  two overlap little. C1 (XL minus PF14) is +0.07 to +0.10, C2 (XL minus X) +0.03 to +0.05, and C3
  (XL minus LOGIT_SAME) +0.03 to +0.05. The gain is smaller than the sum of the parts if the
  logistic score, refitted on X's candidates, loses some of the benefit of the refit-zone alarms.
  The long draw gains nothing (as in experiments 70 and 71).
decision_rule: >-
  Three planned contrasts, each the paired difference in MW-weighted recall, pooled with equal
  weight over S2, S3, and S4, on the fresh headline-band draw, for transfers that start in even
  months. Intervals are 98.33% (Bonferroni across the three contrasts), from 20,000 resamples of
  whole calendar months. C1 is XL minus PF14, C2 is XL minus X, and C3 is XL minus LOGIT_SAME. Every
  arm is held to 80 unexplained odd-month alarms. The verdict is `refuted` when the upper end of C1
  is under +0.02; otherwise `supported` when the lower end of C1 lies above zero, its point value is
  at least +0.02, and the point values of C2 and C3 are both above zero; otherwise `small gain` when
  the lower end of C1 lies above zero; and `inconclusive` in every other case. The threshold of
  +0.02 is the one experiment 70 used, because the combined detector needs the second residual
  model and the neighbour steps of every candidate. Requiring C2 and C3 above zero asks whether
  the combination beats each part, which decides whether the second part is worth porting once the
  first is. The re-scored X (sanity arm 2 of SW2-18) and LOGIT_SAME (sanity arm 2 of SW2-19), the
  fresh long draw, and the labelled real transfers are descriptive and carry no verdict.
benchmark: >-
  PF14: D0M at 4.0 noise units (80 unexplained odd-month alarms) with the first-return end rule of
  experiment 69.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 43, 55, 58, 60, 61, 65, 66, 69, 70, and 71 do, not forecasts)
metric: >-
  MW-weighted recall (headline), pooled over S2, S3, and S4 on even months at 80 odd-month
  unexplained alarms. The share of the moved MW missed, recall by donor step in noise units split
  by metered receiver, each shape separately, and the count of the 35 labelled real transfers over
  1 day found follow.
code: ml-experiments/00082-sw2-18-19-combined-refit-offset-and-same-sign-count/
---

# Experiment 82 (SW2-18 and SW2-19 combined, part of the switching-event detector series): do the refit offset and the same-sign neighbour count add up?

## Plan

- **Arms.** PF14 (benchmark); X ([experiment 71](../00071-sw2-18-refit-offset-and-gap-bridging/result.md)
  re-scored); LOGIT_SAME ([experiment 70](../00070-sw2-19-same-sign-neighbour-count/result.md)
  re-scored); XL, the logistic score with the same-sign count fitted on the candidates of X.
- **Data.** A fresh draw: new seeds for the headline-band and long draws, not those of experiments
  69, 70, and 71. Same panel of 16 Primary substations, same injection shapes S2 to S4, same 80
  odd-month unexplained alarms, same matching rule as experiments 70 and 71.
- **Reuse.** The harness of experiment 71 and the scores of experiment 70, copied into this folder.
  The PF14 arm must reproduce experiment 69's `found` flags on a sample before any score is read.
- **Hindsight caveat.** The alarm ends use up to 14 days of hindsight, so the detector serves
  training-time masks and the NGED event table, not live forecast inputs.

## Size, by the five triggers

- **What gets stored:** nothing in production; detector tables and parquet results in the private
  data folder only.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **Designs:** one defensible design, the logistic score refitted on X's candidates; refitting
  on PF14's candidates would not exercise the refit-zone alarms.
- **Callers:** none outside this folder, which copies the code it reuses.

## Earlier advice followed

- Stop suppressing alarms near the refit (experiment 71); XL uses a step statistic never computed
  across a refit.
- Use LOGIT_SAME for headline-band transfers (experiment 70); XL adds it to X.
- Report recall by donor step split by metered receiver, MW-weighted headline, and the share of
  MW missed (the switching design steers).
