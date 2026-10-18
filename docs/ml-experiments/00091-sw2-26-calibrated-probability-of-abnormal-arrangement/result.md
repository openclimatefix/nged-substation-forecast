---
id: "00091"
slug: sw2-26-calibrated-probability-of-abnormal-arrangement
status: done
size: medium
hypothesis: >-
  The refit-offset neighbour-count detector (RO-NC, called XL on the pages of experiments 70, 71,
  and 82), run causally (with only the data before a forecast origin), can give for each past
  half-hour in the lag window a probability that the half-hour lies inside an abnormal running
  arrangement (ARA). The probability is calibrated, and it separates better than the detector's
  yes-or-no flag.
prediction: >-
  Isotonic calibration fitted on injected transfers that start in odd calendar months gives a
  reliability curve within 0.1 of the diagonal on injected transfers that start in even months. The
  Brier score of the calibrated score of the latest unreturned candidate (arm 2) is at least 20%
  below the Brier score of the binary causal flag (arm 1). The prediction fails if any probability
  bin that holds at least 5% of the scored half-hours misses the diagonal by more than 0.1. In the
  first 24 hours after an onset the score does not separate, so every arm is useful from day 2.
decision_rule: >-
  Two planned contrasts on the even-month rows of the injected transfers, shapes S2, S3, S4, and S6
  pooled with equal weight: C1 is arm 2 minus arm 1 in Brier score, and C2 is arm 3 minus arm 1,
  each as a share of arm 1's Brier score. Intervals are 97.5% (Bonferroni across two), from 10,000
  resamples of whole injection batches. The verdict is `supported` when C1 is at least 20% below
  arm 1 with its upper end below zero and the reliability condition of the prediction holds for arm
  2 on even months; `small gain` when the upper end of C1 is below zero but one of the other two
  conditions fails; `refuted` when the lower end of C1 is above zero; and `inconclusive` otherwise.
  The 20% threshold is the cost of porting: arm 2 needs the detector to keep every candidate above
  a low score floor, and a calibration table, on the serving path of Flexpectation v2, where arm 1
  needs only the detector's present output. Shape S1 (a share of the donor's profile, which matches
  the harness), the bins by hours since onset, area under the receiver operating characteristic
  curve, and the real-data check are descriptive.
benchmark: >-
  Arm 1, the binary causal RO-NC flag (the half-hour lies inside an interval the causal detector
  reports, open or closed), with its two values replaced by their hit rates on odd months so that
  its Brier score is also calibrated. The uncalibrated 0 and 1 flag is reported too.
row_set_id: >-
  none (lag times of the injected cases of experiment 88, drawn from the 292 daily runs of
  fold_a_daily_gap_v1, and the real half-hours of those runs; not comparable with the leaderboard)
metric: >-
  Brier score, reliability curve, and area under the receiver operating characteristic curve of the
  probability that a lag-window half-hour lies inside an ARA. No forecast, so no tail scores.
code: ml-experiments/00091-sw2-26-calibrated-probability-of-abnormal-arrangement/
verdict: inconclusive
best_arm: >-
  Arm 3 (the calibrated score of the latest candidate with the step size) has a Brier score 4.0%
  below the yes-or-no flag on the pre-registered rows (97.5% interval -8.2% to +0.8%). Arm 2 is
  3.3% below (-7.6% to +1.6%), against the 20% target.
lessons: >-
  The causal detector's score is a weak guide to whether a past half-hour lies inside a switching
  period. A calibrated probability beats the yes-or-no flag only on transfers large enough to see
  (11.2% lower Brier score, -20.3% to -2.0%), and against NGED's logs it has almost no skill (area
  under the curve 0.52 to 0.53). Probabilities above 0.5 are too high.
advice:
  - >-
    Do not build a forecast-widening rule on this detector's calibrated probability: against the
    flag it lowers the Brier score by 3.3% (-7.6% to +1.6%), not the 20% the plan required (C1).
  - >-
    Expect the detector to mark few of the half-hours that NGED's logs mark: it flags 1,607 of
    19,099 logged half-hours at the 292 origins, and 92% of logged half-hours stay unflagged (real
    half-hours).
  - >-
    Test a lag-unusualness input that needs no switching label instead of a detector probability,
    since the detector cannot see most real switching (real half-hours, figure 2).
  - >-
    When injecting transfers to score a detector, draw durations of 1 day and longer as well: the
    detector needs 24 hours of data, and its area under the curve is 0.81 to 0.86 for transfers
    under 14 days and 0.67 over 14 days (figure 2).
production_status: not-for-production
adopted_in: none
---

# Experiment 91: can the causal switching detector give a calibrated probability of an abnormal running arrangement?

**Take-home message: under ideal laboratory conditions, a probability built from the causal
refit-offset neighbour-count detector (RO-NC, called XL on the pages of experiments 70, 71, and 82)
is only slightly better than the detector's yes-or-no flag, and on real switching it has almost no
skill.** Widening a forecast's quantiles by this probability would therefore act on few of the rows
that need it. The probability is useful only for transfers large enough and long enough for the
detector to see. A source of warning that does not depend on a detector label is the next thing to
try.

An abnormal running arrangement (ARA) is a period when a network operator has closed a normally
open switch, so that one Primary substation (a substation that steps down to the local distribution
network) carries load that belongs to a neighbour. A forecast that reads recent power (a lag) is
misled when a lag lies inside an ARA. We asked whether the detector, shown only the data before a
forecast origin, can say for each half-hour of the 7 days before the origin how likely it is to lie
inside an ARA. We injected 839 transfers of known size and timing into the power of 16 Primary
substations (from [experiment 88](../00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md)'s
draw), and a second pass of 183 transfers lasting 1 to 60 days. We then fitted the probability on
transfers that start in odd calendar months and scored it on those that start in even months.

The verdict is **inconclusive**. Arm 2, the calibrated score of the latest candidate, has a Brier
score (mean squared error of a probability; smaller is better) 3.3% below the yes-or-no flag (97.5%
interval -7.6% to +1.6%), against the 20% target. Where the transfers are large enough to see
(a post hoc subset), the gain is 11.2% (-20.3% to -2.0%).

![Change in Brier score against the yes-or-no flag](figure_1.svg)

![Area under the curve by duration of the injected transfer](figure_2.svg)

![Reliability of the probability](figure_3.svg)

- **Arms.** Arm 1: the yes-or-no flag (the half-hour lies inside an interval of a candidate above the
  detector's threshold), replaced by its hit rates on odd months so it is calibrated too. Arm 2: a
  logistic regression on the candidate's score, its age, the time from its onset to the half-hour,
  and any return step, then isotonic regression, for the latest unreturned candidate. Arm 3: arm 2
  plus the candidate's step in noise units. Post hoc arm: the highest-scoring candidate instead of
  the latest.
- **Pre-registered rows** (shapes S2, S3, S4, and S6, even months, 3,358,215 rows of all five
  shapes with 85,650 inside an injected transfer): Brier score flag 0.02321, arm 2 0.02243, arm 3
  0.02229, post hoc arm 0.02239. Area under the curve 0.710, 0.728, 0.729, and 0.718.
- **Contrasts** (change in Brier score against arm 1, 97.5% intervals from resampling 83 injection
  batches): C1, arm 2, -3.3% (-7.6% to +1.6%). C2, arm 3, -4.0% (-8.2% to +0.8%). Arm 3 against arm 2:
  -0.7% (-1.9% to +0.9%). Arm 2 beats the flag in 3 of the 4 even months that hold positive rows.
- **Calibration.** The condition of the plan (no bin holding at least 5% of rows misses by more than
  0.1) is met, but only because 97% of rows fall in the lowest bin. Probabilities of 0.55 to 0.77
  are too high: 0.29 to 0.35 of those rows lie inside a transfer (figure 3).
- **Post hoc rows.** Dropping the rows the same detector already flags on the uninjected power
  (which hold unlabelled real switching): arm 2 -1.6% (-6.1% to +2.9%). Also dropping rows inside
  transfers whose donor step is under 4 noise units: arm 2 -11.2% (-20.3% to -2.0%), arm 3 -11.6%
  (-20.7% to -2.2%), post hoc arm -13.5% (-23.6% to -3.5%); Brier scores 0.00902, 0.00801, 0.00798,
  and 0.00780, area under the curve 0.871, 0.880, 0.880, and 0.879.
- **By duration of the injected transfer** (clean rows, both draws; area under the curve for the
  flag, arm 2, post hoc arm): under 24 hours 0.813, 0.826, 0.829; 1 to 3 days 0.835, 0.855, 0.861;
  3 to 14 days 0.827, 0.828, 0.837; over 14 days 0.675, 0.670, 0.682. The flag marks 64%, 70%, 68%,
  and 37% of the positive rows, and 1.1%, 2.8%, 2.9%, and 2.4% of the negative rows.
- **Real half-hours** (292 origins, 130,816 rows, 19,099 inside a logged closure or labelled event;
  the logs are incomplete, so precision is a lower bound): area under the curve flag 0.521, arm 2
  0.532, arm 3 0.528, post hoc arm 0.531. The flag marks 6,271 rows, of which 1,607 (26%) lie inside
  the logs, and misses 17,492 of the 19,099 logged rows.
- **Laboratory shortcuts.** The residual model uses ERA5 weather, which arrives about 5 days late, so
  a live probability would need a residual built from ENS forecasts. The noise scales, the
  logistic score, and its threshold were fitted on odd months of the whole record, including months
  after an origin. The expected power is floored at 10% of its whole-record median. The injected
  power does not enter the monthly refits of the residual model (the fast path). Rows are every
  sixth hour of the 7-day lag window. A row inside a transfer belongs to the month the transfer
  starts in, and any other row to the month of its lag stamp, so rows at one origin can fall on both
  sides of the odd-even split.
- **Caveats.** The 183 longer transfers are fewer than the 384 asked for, because the quiet-period
  rules left few windows. The isotonic map is fitted on the logistic regression's own training
  rows. Partner intervals take the score and step of the alarm series, so arm 3 adds little on the
  receiver side.
- **What to try next.** A lag-unusualness input that needs no switching label (the question of
  experiment 90, SW2-28), and a detector that sees the real switching the logs hold, since this
  detector finds 8% of it.
