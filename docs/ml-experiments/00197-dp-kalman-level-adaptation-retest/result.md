---
id: '00197'
slug: dp-kalman-level-adaptation-retest
part: KLA2
theme: differentiable-physics-forecast-engine
also_relevant_to:
- lagged-power-and-error-correction
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00197-dp-kalman-level-adaptation-retest/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one gate and
  more than one shared-level design are defensible, so this plan pre-registers one gate and one
  selection rule. Callers: this folder's scripts, which copy the filter and gate code of experiment
  190; they call the reviewed `studies.*` modules unchanged. No new data and no new forecaster fit.
hypothesis: >-
  Experiment 190's known-answer gate could not be passed by a filter with one state per hour band,
  because the target's band is usually unread since an offset began. Scored only on pairs the
  filter can know about, (a) a faster per-band filter (A1) recovers at least 50% of a 0.10-size,
  16-hour offset on target bands already read, and (b) a filter whose bands share one level while a
  same-sign run lasts (A3) recovers at least 50% on target bands not yet read, because it moves
  bands that no reading has reached.
prediction: >-
  Written before any score is read. P1: A1 at the selected setting recovers at least 50% of the
  excess loss on band-read pairs, for the 0.10 size and 16 hours. P2: A3 recovers at least 50% on
  band-unread pairs for which another band has been read since the onset, for the same offset. P3:
  on the left-alone days, A1 and A3 each score within +0.005 points of A0, upper 95% bound. P4: on
  the scored year's real rows A3 minus A0 lies within -0.03 to +0.01 points with an upper 95% bound
  of at most +0.01. P5: the share of pairs that no filter can learn about (no band read since the
  onset) is large in the first 6 hours after onset, which shows the structural limit.
decision_rule: >-
  The redesigned gate comes first. It passes for an arm when the P1 (A1) or P2 (A3) recovery is at
  least 50% with a 95% case-resampled lower bound above 25%, and the arm's left-alone harm has an
  upper 95% bound of at most +0.005 points. A1's setting is selected from the grid m in {3, 5, 10,
  20, 50}, k in {2, 4, 6}, z = 0.5 on the injected cases of sizes 0.05 and 0.20 (band-read pairs,
  all durations, highest recovery), and tested at size 0.10, so the test cases do not choose the
  setting. A3 is selected the same way. If A3 fails the gate or P4, level adaptation is dropped and
  the 15 smooth-offset days are recorded as not a filter problem. If A3 passes the gate and fails
  P4, the verdict is `not supported` and the note says the shared level moves bands wrongly on
  real days. Experiment 190's finding stays on the page: the always-on boost scored 0.023 points
  worse, so a faster filter alone is not the answer. The smallest gain worth porting is 0.02 points
  over all rows, because a shared-level counter adds state to the serving path.
benchmark: >-
  Arm D of experiment 155 (A0) as rebuilt by experiment 190, 2.7562 points over leads 0.5 to 168
  hours at the five decision Primaries in the scored year.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Recovery of the injected offset's excess quantile score (1 minus the arm's excess over the clean
  A0, divided by A0's excess over the clean A0), pooled over pairs, by hours since onset and by
  whether the target band has been read since the onset; and, on real rows, the quantile score in
  points of effective capacity.
code: ml-experiments/00197-dp-kalman-level-adaptation-retest/
---

# Retest of the faster Kalman level state with a fair known-answer gate

Planned; the write-up follows.
