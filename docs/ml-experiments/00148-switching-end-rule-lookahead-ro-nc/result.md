---
id: "00148"
slug: switching-end-rule-lookahead-ro-nc
theme: switching-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00148-switching-end-rule-lookahead-ro-nc/
  only (per-transfer outcome tables and an interval table), no prediction set. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible lookahead rule (fix the end at the
  largest opposite step that no later step beats within the lookahead), tested at four lookaheads.
  Callers: this folder's scripts only, which copy experiment 147's code and read experiment 138's
  stored arm B score and threshold without changing them.
hypothesis: >-
  The end of a switching interval that RO-NC has alarmed on can be fixed by waiting a bounded
  lookahead after the candidate end, with no loss of recall against the offline end rule. The
  candidate end is the largest opposite step seen so far, and the end is fixed once no later step
  beats it within the lookahead. A lookahead of 24 hours or less is enough, so a live service need
  not wait 2 days.
prediction: >-
  Written before any result. A lookahead of 0 hours is the same rule as experiment 147's R1, so it
  reproduces R1's share found of 0.298 and its false-end share of about 0.6 (a check, not a
  finding). A lookahead of 6 hours loses 0.10 to 0.40 of the share found against the offline rule
  R0, with a false-end share of 0.2 to 0.5. A lookahead of 24 hours loses under 0.03, with a
  false-end share under 0.10. A lookahead of 48 hours loses under 0.01, with a false-end share
  under 0.06. The call lag is the lookahead plus the 24 hours the step statistic needs, plus the
  hours from the true end to the candidate end, so about 24, 30, 48, and 72 hours.
decision_rule: >-
  Four planned contrasts: the MW-weighted share found of the rule at a lookahead of 0, 6, 24, and 48
  hours minus the offline end rule R0, pooled over shapes S2, S3, and S4, on experiment 147's fresh
  headline-band draw (same seeds, same alarms, no refitting), for transfers that start in even
  months, with no merging of alarms. The 98.75% interval (Bonferroni for four contrasts) resamples
  whole calendar months 20,000 times. A lookahead is `adequate` when the lower end of its contrast is
  above -0.02, its false-end share is at most 0.10, and its late-end share is at most 0.10 (false
  end: an end called earlier than the true end by more than the end tolerance, among transfers whose
  onset is found). The recommended lookahead is the smallest adequate one. The hypothesis is
  `supported` when a lookahead of 24 hours or less is adequate, `partly supported` when only 48
  hours is adequate, `refuted` when the upper end of every contrast is under -0.02, and
  `inconclusive` otherwise. The smallest gain worth porting is zero lost share: the rule costs
  nothing in production beyond the wait, so the threshold is the -0.02 margin on recall. Reference
  rows, reported but not in the verdict: experiment 147's running-peak rule R2, and the hindsight
  rule H14 (largest opposite step within 14 days, known at 14 days). The merge of overlapping
  alarms within 24 hours (M48) is reported for every rule. The 35 real transfers over 1 day are
  descriptive.
benchmark: >-
  R0, the offline end rule of experiment 130 (arm B, share found 0.794 on the same draw).
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, not
  forecasts)
metric: >-
  MW-weighted share found, false-end share, late-end share, end error in hours, call lag in hours.
code: ml-experiments/00148-switching-end-rule-lookahead-ro-nc/
production_status: not-for-production
adopted_in: none
---

# Experiment 148: how long must a live service wait to fix the end of a switching interval?

**Planned.** Experiment 147 found that a running-peak rule fixes the end of an RO-NC interval with no
lost recall and a median call lag of 29 hours. This experiment scores the end rule at four fixed
lookaheads (0, 6, 24, and 48 hours), the range the maintainer said is acceptable, instead of only
"causal" against 14-day hindsight. Pre-registered verdict rule: see `decision_rule` above.

Lookahead rule: after the first return step, the candidate end is the largest opposite step so far.
The end is fixed at the candidate once no step in the lookahead after it is larger. The call is
made when the 24-hour step statistic of the last stamp of the lookahead can be computed.
