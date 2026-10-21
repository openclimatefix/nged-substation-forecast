---
id: "00138"
slug: weaker-onset-detector-on-the-ro-nc-convex-residual
theme: switching-detectors
also_relevant_to: []
status: done
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
  fixed. The odd-month alarm count at the arm's own threshold is held at 80 by construction, so extra
  weak candidates show up as lost recall of the strong steps, not as extra alarms.
  Experiment 132 found that
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
  Arm B of experiment 130, RO-NC on the frozen convex residual (0.794, 95% interval 0.719 to 0.856),
  re-scored here on the same rows.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 82 and 130 do, not forecasts)
metric: >-
  MW-weighted recall, pooled over S2, S3, and S4 on even months at 80 odd-month unexplained alarms.
  Descriptive: unexplained alarms per odd month, recall by donor step in noise units and metering,
  and the number of the 35 labelled real transfers over 1 day found.
code: ml-experiments/00138-weaker-onset-detector-on-the-ro-nc-convex-residual/
verdict: refuted
best_arm: >-
  None beat the control. Arm B (RO-NC as in experiment 130) finds 0.794 of the moved MW (95%
  interval 0.719 to 0.856); W2, the best weaker-onset arm, finds 0.796 (0.725 to 0.855).
headline: >-
  W1, W2, and W3 minus B are -0.005 (-0.017 to +0.006), +0.001 (-0.009 to +0.018), and -0.008
  (-0.028 to +0.002) in the share of the moved MW found, at 80 unexplained odd-month alarms and a
  98.33% interval. All three upper ends are under the +0.02 refutation line.
lessons: >-
  Lowering the candidate bar, requiring the step to persist, and adding a 72-hour window do not
  lift the share of moved power found on injected transfers: at a fixed budget of 80 false
  alarms, the added weak candidates appear to take the place of stronger ones (the unmetered
  4 to 6 noise-unit band loses 233 to 225 found). On the 35 real transfers over 1
  day the three arms find 21 to 23,
  against 18, a difference the injected transfers do not show.
take_home_message: >-
  Making the detector more willing to fire on weak, slow starts did not help on injected
  transfers: all three changes found the same 0.79 of the moved power as before (differences of
  -0.008 to +0.001, no interval above +0.018) at the same 80 false alarms. Most of the weak starts
  that remain missed (a first step of 2 to 4 noise units at a donor with no metered receiving
  substation) look too much like noise for any of the three logistic scores to separate them
  from noise. On the 35 real transfers lasting
  over 1 day the new arms found 21 to 23 against 18, which is a possible gain worth checking on a
  bigger labelled set, since the injected transfers show no such gain.
advice:
  - >-
    In experiment 138, lowering the candidate threshold of RO-NC from 2.5 to 1.5 noise units
    changed the share of moved MW found by -0.005 (-0.017 to +0.006) at 80 false alarms, so do
    not expect a lower threshold to find weak transfers (W1, figure 2).
  - >-
    Do not expect a 72-hour step statistic to help RO-NC on injected transfers: it changed the
    share found by -0.008 (-0.028 to +0.002) (W3, figure 2).
  - >-
    Treat the persistence feature as a possible gain on real transfers, not a proven one: it left injected
    recall unchanged (+0.001; -0.009 to +0.018) but found 23 of 35 real transfers over 1 day
    against 18 (figure 4).
  - >-
    Look for the weak misses outside the step statistic: transfers with a first step of 2 to 4
    noise units and no metered receiver were found 59 of 365 times by arm B and no more often by
    any weaker-onset arm (figure 3).
production_status: not-for-production
adopted_in: none
---

# Experiment 138: does a detector for weaker, slower onsets find more of the moved load?

**Making the load-transfer detector fire on weaker and slower starts did not find more of the moved
power on injected transfers.** A load transfer is a switching operation that moves part of one
Primary substation's demand onto neighbouring substations for hours or days. We added synthetic
transfers to real power history, and counted the share of the moved megawatts (MW) on which a
detector raised an alarm. The detector, the refit-offset neighbour-count detector (RO-NC), flags a
step in one Primary substation's forecast errors, helped by an opposite step at its neighbours. A
noise unit is the usual size of that step in odd-numbered months. The forecast is the convex demand
model of [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), with its
coefficients held fixed. Three changes aimed at weak first steps (2 to 4 noise units) were a lower
threshold for what counts as a candidate, a persistence feature, and a 72-hour window. Each found
0.79 of the moved MW, the same share as the unchanged detector (arm B), at the same budget of 80
false alarms, counted on odd-numbered months. The verdict is `refuted`, because no change had an
interval reaching +0.02. Most of the weak first steps that stay missed are at a donor substation
whose load moves to a receiving substation with no power meter, and none of the three changes
separates them from noise. On the 35 real transfers lasting over 1 day, the three changes find 21 to
23 and arm B finds 18. The injected transfers show no such gain.

![Share of moved MW found, by arm](figure_1_recall_by_arm.svg)

![Planned contrasts](figure_2_contrasts.svg)

![Share of transfers found by donor step, no metered receiver](figure_3_share_found_by_step.svg)

![Real transfers found](figure_4_real_transfers_found.svg)

## Arms and rows

- **B (control):** RO-NC on the frozen convex residual, candidates at 2.5 noise units or more on the
  24-hour step, logistic score on the step, the largest opposite-sign neighbour step, and topology
  counts. It reproduces experiment 130 exactly: 7,745 rows of detector outcomes per transfer and
  shape, none differ, and recall is 0.794.
- **W1:** candidates from 1.5 noise units, same score features.
- **W2:** candidates from 1.5 noise units, plus the step still present 1 to 2 days after the
  candidate (a 24-hour mean after a one-day gap, minus the day before).
- **W3:** B's candidates plus peaks of a 72-hour step at 3.0 noise units or more, and the 72-hour
  step as a score feature.
- **Rows:** experiment 82's fresh headline-band draw, 602 transfers per shape in even months, shapes
  S2 to S4 pooled, 15 series. Each arm fits its own logistic score on odd-month injections and sets
  its threshold at 80 unexplained odd-month alarms. Interval: whole calendar months resampled 20,000
  times.
- **Lab shortcuts:** the residual uses ERA5 weather that arrives about 5 days late, the end of each
  alarm is chosen with hindsight, and the moved MW is the injected amount.

## Results

- **Recall (95% interval):** B 0.794 (0.719 to 0.856), W1 0.789 (0.711 to 0.852), W2 0.796 (0.725 to
  0.855), W3 0.787 (0.707 to 0.848).
- **Planned contrasts (98.33% interval):** W1 minus B -0.005 (-0.017 to +0.006), W2 minus B +0.001
  (-0.009 to +0.018), W3 minus B -0.008 (-0.028 to +0.002). The best lower end is under zero, and
  every upper end is under +0.02.
- **Alarm counts:** every arm has exactly 80 unexplained odd-month alarms by construction, and 65 to
  68 on even months, so weak candidates show up as lost recall, not extra alarms. In the unmetered 4
  to 6 noise-unit band W1 finds 225 of 380, W2 226, W3 213, against 233 for B.
- **Weak steps stay missed:** for transfers with no metered receiver and a donor step of 2 to 4
  noise units, B finds 59 of 365, W1 44, W2 50, and W3 59.
- **Long transfers (15.6 to 60 days):** each weaker-onset arm minus B is +0.005 (+0.000 to +0.026,
  95% interval).
- **Real transfers over 1 day (descriptive):** B 18 of 35, W1 21, W2 23, W3 22. The scores were
  fitted on injected transfers only, and the real set is too small to rank the three arms.

## Checks

- **Control:** arm B reproduces experiment 130 row for row.
- **Forecast checks:** no forecast is scored here, so the median-minus-mean and mean-forecast checks
  do not apply.
- **Audit triggers:** none of the seven conditions in the `ml-experiment` skill fires. The result
  contradicts the prediction (+0.02 to +0.06), and the by-step table is the tested explanation:
  added weak candidates replace stronger ones at the fixed 80-alarm budget. The nested arm W2 does
  not beat W1 by more than its interval. The real-transfer lead is descriptive, and no later design
  depends on it.

## Caveats

- **Injected transfers are rectangular steps,** so a persistence or slow-window feature has little
  to add there. Real transfers are less regular, which may explain why the real count rises while
  the injected recall does not. This is a guess that the experiment did not test.
- **The real-transfer count is 35 transfers at one threshold,** and arms W1 to W3 were never tuned
  on them.

## What to try next

- Check the real-transfer lead on a larger labelled set before porting W2.
- Look for a signal other than the residual step for weak transfers with no metered receiver.

Code: `ml-experiments/00138-weaker-onset-detector-on-the-ro-nc-convex-residual/`; every number above
is printed by `e00138_contrasts.py`.
