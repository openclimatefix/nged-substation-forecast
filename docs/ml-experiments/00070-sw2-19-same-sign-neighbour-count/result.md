---
id: "00070"
slug: sw2-19-same-sign-neighbour-count
status: planned
size: simple
hypothesis: >-
  One alarm score that adds the count of neighbouring substations stepping the same way as the
  alarming substation to the substation's own 24-hour step and the largest opposite neighbour step
  (LOGIT_SAME) finds more of the MW moved by injected load transfers than the single-series
  detector D0M, with the same first-return end rule (PF14), at the same count of unexplained alarms
  on odd months, on a fresh draw of injected transfers.
prediction: >-
  MW-weighted recall (the share of the moved MW that sits in found transfers), even months, shapes
  S2, S3, and S4 pooled, PF14 end rule. PF14 finds about 0.67 on a fresh headline-band draw
  (experiment 69: 0.668). LOGIT_SAME minus PF14 (C1) is +0.01 to +0.04, smaller than experiment
  65's post hoc +0.034 because a variant chosen after reading one draw does better on that draw
  than on a new one, and because PF14 already finds more than the end rule of experiment 65. On
  the fresh long set (C2) the difference is -0.04 to +0.03, because the long-set intervals are wide
  and the neighbour evidence acts on the onset only. The gain sits in transfers with a donor step
  of 2.5 to 6 noise units.
decision_rule: >-
  Two planned contrasts, each the paired difference in MW-weighted recall, pooled with equal weight
  over S2, S3, and S4 and read on transfers that start in even months. Intervals are 97.5%
  (Bonferroni across the two contrasts), from 20,000 resamples of whole calendar months. C1 is
  LOGIT_SAME minus PF14 on a fresh headline-band draw, C2 the same on a fresh long draw. Both arms
  are held to 80 unexplained odd-month alarms. The verdict is `supported` when the lower end of C1
  lies above zero, its point value is at least +0.02, and the point value of C2 is at least -0.02.
  Otherwise, in this order, it is `refuted` when the upper end of C1 is under +0.02; `small gain`
  when the lower end of C1 lies above zero, its point value is under +0.02, and the point value of
  C2 is at least -0.02; and `inconclusive` in every other case. The threshold of +0.02 is set by
  the work of porting: the score needs the neighbour steps of every candidate and a logistic
  regression fitted on injected transfers, in a detector that serves training-time masks and the
  event table. The logistic score without the same-sign count (LOGIT) and the labelled real
  transfers are descriptive and carry no verdict.
benchmark: >-
  PF14: D0M at 4.0 noise units (80 unexplained odd-month alarms) with the first-return end rule of
  experiment 69.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 43, 55, 58, 60, 61, 65, 66, and 69 do, not forecasts)
metric: >-
  MW-weighted recall (headline), pooled over S2, S3, and S4 on even months at the matched
  unexplained-alarm count. The share of the moved MW missed, recall by donor step in noise units
  split by metered receiver, each shape separately, and the count of the 35 labelled real transfers
  over 1 day found follow.
code: ml-experiments/00070-sw2-19-same-sign-neighbour-count/
---

# Experiment 70 (SW2-19, part of the switching-event detector series): does the count of neighbours that step the same way find more moved load?

*Plan, written before any fresh transfer is scored.*

**Question.** Experiment 65 found, after reading its results, that adding the count of neighbouring
substations stepping the same way to a logistic alarm score raised the share of moved MW found by
+0.034 over the single-series detector (+0.013 to +0.091). This experiment repeats that arm as a
planned contrast, against the first-return end rule PF14 that experiment 69 made the benchmark,
on injected transfers drawn with new seeds.

**Design.**

- **Scores.** A candidate is a local peak of a substation's 24-hour step statistic at or above 2.5
  noise units. LOGIT_SAME is a logistic regression on the candidate's own step, the largest
  opposite-sign neighbour step, the number of neighbours, and the number of neighbours whose
  same-sign step exceeds 2.0 noise units. LOGIT is the same without the same-sign count. Both are
  fitted as in experiment 65: positives are the candidates within 2 hours of the onset of an
  injected headline-band transfer that starts in an odd month (experiment 58's draw), negatives
  are the unexplained odd-month candidates of the real residual. Each threshold is set so that
  80 odd-month unexplained real alarms remain.
- **End rule.** Every arm uses PF14: the first opposite step of at least half the onset step,
  otherwise the tracked end.
- **Test transfers.** A fresh headline-band draw of 1,160 transfers and a fresh long-set draw, both
  with seeds that experiments 58 and 69 did not use, scored under the five injection shapes.
  The headline pools S2, S3, and S4, which the scores do not assume.
- **Real transfers.** The 35 labelled real transfers over 1 day are scored for each arm.
- **Caveat.** Interval ends use up to 14 days of hindsight, so the detector serves training-time
  masks and the NGED event table, not live forecast inputs.

**Size.** Simple: (1) nothing new is stored beyond private score tables; (2) no production serving
path; (3) no degradation rule; (4) the design repeats experiment 65's score under experiment 69's
benchmark, with one defensible reading; (5) the code is a copy of the two experiments' harnesses,
whose callers are all inside this folder.

**Earlier advice followed.** Experiment 65: pre-register the score with the same-sign count
(done), use D0M as the benchmark for neighbour evidence (done, with experiment 69's end rule).
Experiment 69: use PF14 as the end rule (done); keep a matched-alarm-count control (the matched
count of 80 applies to every arm).
