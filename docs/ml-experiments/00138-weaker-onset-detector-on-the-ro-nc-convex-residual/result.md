---
id: "00138"
slug: weaker-onset-detector-on-the-ro-nc-convex-residual
theme: switching-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00138-weaker-onset-detector-on-the-ro-nc-convex-residual/
  only (candidate tables, fitted logistic scores, alarm and outcome tables), no prediction set.
  Serving path: not touched. Degradation rule: not touched. Designs: three defensible ways to catch
  weaker, slower onsets (a lower candidate bar, a persistence requirement, a slow second window), all
  tested as arms. Callers: this folder's scripts only, which copy experiment 130's detector code and
  call the reviewed `studies.*` modules without changing them.
hypothesis: >-
  A detector that also considers weaker and slower onsets finds a larger share of the moved megawatts
  (MW) of injected load transfers than the refit-offset neighbour-count detector (RO-NC) on the
  frozen convex residual (experiment 130, arm B), at the same 80 unexplained odd-month alarms. In
  experiment 130, most of the remaining misses were donor steps of 2 to 4 noise units with no
  metered receiving substation, which the 24-hour step statistic cannot separate from noise.
prediction: >-
  Each weaker-onset arm minus arm B, in MW-weighted recall at 80 unexplained odd-month alarms, pooled
  over shapes S2, S3, and S4 on even months: +0.02 to +0.06 for the best arm. The prediction is
  modest because every extra weak candidate also adds odd-month false alarms, and the alarm budget is
  fixed. The odd-month alarm count at the arm's own threshold is held at 80 by construction, so the
  cost shows as lost recall of the strong steps, not as extra alarms. Experiment 132 found that
  alarms rise by 7 to 15 on odd months at fixed thresholds, so the fixed-budget comparison is the
  fair one.
decision_rule: >-
  Three planned contrasts, each a weaker-onset arm (W1 lower candidate bar, W2 persistence feature,
  W3 slow second window) minus arm B, the paired difference in MW-weighted recall pooled with equal
  weight over shapes S2, S3, and S4, on experiment 82's fresh headline-band draw, for transfers that
  start in even months, on the 15 series with convex-model residuals. Every arm is held to 80
  unexplained odd-month alarms, estimates its noise unit from its own odd-month residual, and fits
  its own logistic score on injections from odd months only. The 95% interval resamples whole
  calendar months 20,000 times, at the Bonferroni level for three contrasts (98.33%). The verdict is
  `supported` when the lower end of the best arm's contrast is above +0.03; `small gain` when the
  lower end is above zero but under +0.03; `refuted` when the upper ends of all three are under
  +0.02; and `inconclusive` otherwise. The bar of +0.03 matches experiment 130's, because every arm
  needs a longer feature window or an extra candidate source, which is a new pipeline step. The
  recall of arm B is re-scored here as a control and must reproduce experiment 130's 0.794 to within
  0.005. Descriptive and verdict-free: odd-month alarms per month for each arm (the cost to watch),
  recall by donor step in noise units split by metered receiver, and the 35 real transfers over 1
  day found.
benchmark: >-
  Arm B of experiment 130, RO-NC on the frozen convex residual (0.794, 95% interval 0.719 to 0.855),
  re-scored here on the same rows.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 82 and 130 do, not forecasts)
metric: >-
  MW-weighted recall, pooled over S2, S3, and S4 on even months at 80 odd-month unexplained alarms.
  Descriptive: unexplained alarms per odd month, recall by donor step in noise units and metering,
  and the number of the 35 labelled real transfers over 1 day found.
code: ml-experiments/00138-weaker-onset-detector-on-the-ro-nc-convex-residual/
production_status: not-for-production
adopted_in: none
---

# Experiment 138: does a detector for weaker, slower onsets find more of the moved load?

**This page is the plan. The result is not in yet.** A load transfer is a switching operation that
moves part of one Primary substation's demand onto neighbouring substations for hours or days. The
refit-offset neighbour-count detector (RO-NC) on the frozen convex-demand-model residual found 0.794
of the injected moved megawatts (MW) in [experiment
130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), and most of what it missed was a first
step of 2 to 4 noise units with no metered receiving substation. This experiment tests three ways of
catching those weaker onsets, each at the same budget of 80 false alarms on odd-numbered months.

## Arms

- **B (control):** RO-NC on the frozen convex residual, as in experiment 130 (candidates at 2.5
  noise units or more on the 24-hour step, logistic score on the step, the largest opposite-sign
  neighbour step, and topology counts).
- **W1, lower candidate bar:** candidates from 1.5 noise units, the same logistic score. Tests
  whether the score can already rank weak candidates.
- **W2, persistence requirement:** candidates from 1.5 noise units, and the logistic score gains a
  feature, the smaller of the 24-hour step statistic at 1 day and at 2 days after the candidate, in
  noise units with the candidate's sign. A transfer persists, and noise does not.
- **W3, slow second window:** the candidates of arm B plus candidates from a 72-hour step statistic
  (its own noise unit, candidates from 3.0), then the same logistic score with the window as a
  feature. Tests whether a slower window lifts small steps clear of the noise, as [experiment
  59](../00059-sw2-13-several-windows-in-one-detector/result.md) tried on the lagless residual.

All arms train their logistic score on injections from odd months only and tune the alarm threshold
on odd-month alarms, then score even months, as the SW2 iterations did. Residuals are the ones
experiment 130 stored, so no model is refitted. Lab shortcuts, as the switching brief allows: ERA5
weather in the residual, hindsight end rule, true injected MW.

## Checks to run

- Arm B reproduces experiment 130's 0.794 within 0.005.
- Median forecast minus mean forecast, and the mean-forecast error contrast, do not apply (no
  forecast is scored). The audit triggers are checked on the control, on nested arms (W2 contains
  W1's candidates), and on leaving one series out.
