---
id: "00082"
slug: sw2-18-19-combined-refit-offset-and-same-sign-count
status: done
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
verdict: supported
best_arm: >-
  XL (the logistic score with the same-sign neighbour count, applied to the candidates of the
  refit-offset detector X) finds 0.770 of the moved MW on fresh headline-band transfers (98.33%
  interval 0.665 to 0.857), against 0.656 for PF14 (0.530 to 0.727). C1, XL minus PF14, is +0.114
  (+0.074 to +0.182); C2, XL minus X, is +0.058 (+0.034 to +0.096); C3, XL minus LOGIT_SAME, is
  +0.064 (+0.028 to +0.127).
lessons: >-
  The refit offset and the same-sign neighbour count fix different misses, so their gains mostly
  add: +0.114 together against +0.056 and +0.050 alone on this draw (the sum is +0.106). The
  share of the moved MW that no alarm starts on falls from 0.300 (PF14) to 0.195 (XL), and the
  share of the moved MW missed from 0.344 to 0.230. Long transfers (15.6 to 60 days) gain nothing,
  as in experiments 70 and 71. On the 35 labelled real transfers of more than 1 day, XL found 12
  and PF14 8.
advice:
  - >-
    Build the detector from the refit-offset step statistic and score its candidates with the
    same-sign neighbour count, rather than either change alone; together they found +0.114
    (+0.074 to +0.182) more of the moved MW than PF14 (C1).
  - >-
    Port the neighbour-count score on top of the refit-offset detector, not instead of it: XL
    beats the refit-offset detector alone by +0.058 (+0.034 to +0.096) and the neighbour-count
    score alone by +0.064 (+0.028 to +0.127) (C2, C3).
  - >-
    Do not expect either change to help transfers lasting 15.6 to 60 days; XL minus PF14 on the
    long draw is +0.000 (-0.026 to +0.026).
  - >-
    Look for the remaining misses among sharp steps of 2 to 4 noise units with no metered
    receiver, where XL still finds only 149 of 550 transfers (recall by donor step).
production_status: candidate
adopted_in: none
---

# Experiment 82 (SW2-18 and SW2-19 combined, part of the switching-event detector series): do the refit offset and the same-sign neighbour count add up?

**Yes: the two fixes find different missed load transfers, so using both finds about twice as much
extra moved power as either alone. Of the megawatts (MW) moved by injected transfers, the combined
detector XL finds 0.770 and the benchmark PF14 finds 0.656.** The remaining misses are mostly
transfers whose donor substation steps by only 2 to 4 noise units and that have no metered
receiver, so a stronger signal for those is the next thing to try. The verdict is `supported`.

We asked whether two earlier improvements to a detector for load transfers combine. A load transfer
is a switching operation that moves part of one Primary substation's demand to neighbouring
substations for hours or days. [Experiment 71](../00071-sw2-18-refit-offset-and-gap-bridging/result.md)
stopped the detector ignoring alarms near the monthly refit of its forecasting model.
[Experiment 70](../00070-sw2-19-same-sign-neighbour-count/result.md) scored each alarm candidate
partly by how many neighbouring substations stepped the same way. We added known transfers to the
history of 16 Primary substations, combined the two changes, and counted the share of the moved MW
that each detector found. Every detector was held to 80 alarms not explained by the switching log
on odd months.

![Share of moved MW found, by detector](figure_1_recall_by_arm.svg)

![Share of transfers found by donor step](figure_2_share_found_by_step.svg)

## Results

- **Arms.** PF14: alarm when the 24-hour residual step reaches 4.0 noise units, end at the first
  opposite step of at least half the onset step. X: PF14 with the step taken from whichever of two
  residual models, refitted half a month apart, did not refit within 24 hours. LOGIT_SAME: PF14's
  candidates scored by a logistic regression that includes the same-sign neighbour count. XL: the
  same logistic score fitted on X's candidates.
- **Row set.** 611 fresh headline-band transfers per injection shape (25% or more of the receiving
  load, over 1 day) starting in even months, shapes S2, S3, and S4 pooled (S2 a constant MW, S3 a
  share of the receiver's profile, S4 a mix fitted to the labelled real transfers); 214 long
  transfers (15.6 to 60 days). Fresh seeds, not those of experiments 69 to 71.
- **MW-weighted recall (98.33% interval, resampling whole calendar months):** PF14 0.656 (0.530 to
  0.727); X 0.713 (0.596 to 0.803); LOGIT_SAME 0.706 (0.591 to 0.776); XL 0.770 (0.665 to 0.857).
  The sanity arms X and LOGIT_SAME reproduce their experiments' gains: +0.056 and +0.050 over PF14.
- **Planned contrasts:** C1 XL minus PF14 +0.114 (+0.074 to +0.182); C2 XL minus X +0.058 (+0.034 to
  +0.096); C3 XL minus LOGIT_SAME +0.064 (+0.028 to +0.127). All three lower ends are above zero,
  C1 is above the threshold of +0.02, and the sum of the single gains (+0.106) is close to C1.
- **Share of the moved MW missed:** PF14 0.344, X 0.287, LOGIT_SAME 0.294, XL 0.230. For XL, 0.195
  is transfers with no alarm at the start and 0.035 is transfers with a wrong end.
- **Shapes:** XL minus PF14 is +0.107 to +0.116 for S1, S2, S3, and S4, and +0.096 for S5; all
  lower ends are above zero.
- **By donor step, transfers found (S2 to S4 summed):** with a metered receiver and a step of 2 to
  4 noise units, XL 109 of 151, PF14 81. With no metered receiver and a step of 2 to 4, XL 149 of
  550, PF14 42. With no metered receiver and a step of 4 to 6, XL 175 of 255, PF14 131.
- **Long transfers (descriptive):** XL minus PF14 +0.000 (-0.026 to +0.026).
- **Labelled real transfers of more than 1 day (35):** found XL 12, X 10, PF14 8, LOGIT_SAME 7; an
  alarm started within 2 hours of the labelled start for XL 15, X 12, LOGIT_SAME 12, PF14 11. The
  35 are too few to rank the arms.
- **Alarms:** unexplained alarms on even months were PF14 80, X 80, LOGIT_SAME 61, XL 67, against
  at most 80 on odd months for all arms (X 75, at its grid threshold of 4.1).

## Caveats

- The alarm ends use up to 14 days of hindsight, so the detector serves training-time masks and
  the NGED event table, not live forecast inputs.
- The logistic score is fitted on odd-month injections, so the odd-month contrast (+0.141) is
  in-sample and decides nothing.
- The harness reproduces experiments 71 and 70 for PF14, X, and LOGIT_SAME on their draw; there is
  no earlier reference for XL.
- The 35 real transfers give a weak signal only (12 against 8 found).

## What the next experiment should answer

The remaining misses sit in sharp steps of 2 to 4 noise units with no metered receiver (149 of 550
found by XL). A stronger signal there, such as a longer window or a neighbour-balance check, is
the next thing to test.
