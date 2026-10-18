---
id: "00091"
slug: sw2-26-calibrated-probability-of-abnormal-arrangement
status: planned
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
---

# Experiment 91: can the causal switching detector give a calibrated probability of an abnormal arrangement?

This is the plan. The write-up replaces it when the experiment finishes.

## Question

An abnormal running arrangement (ARA) is a period when a network operator has closed a normally
open switch, so that one Primary substation carries load that belongs to a neighbour. A demand
forecast that reads recent power as input (a lag) is misled when a lag lies inside an ARA. Experiments
86 and 88 showed that correcting such a lag by the detector's MW does not help, because the MW is
about 1.8 times too large. They also showed that the 5% to 95% band of the forecast covers only 78%
of such rows. The next step is to widen the band when a lag may be inside an ARA, which needs a
probability and no MW. This experiment asks whether the RO-NC detector of [experiment
82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md), run causally, can give
that probability, and whether the probability is calibrated.

## Design

- **Cases.** Experiment 88's injected transfers (events from 2025-07 to 2026-06, shapes S1, S2, S3,
  S4, S6, 83 batches of events on disjoint series), redrawn with experiment 88's seeds. For each of
  the 292 daily origins of `fold_a_daily_gap_v1` with an injected event active in the lag window,
  the detector is run causally with the cut at the newest usable half-hour (the origin minus 3.5
  hours) for all 16 Primary series. Every sixth hour of the lag window (the 7 days before the cut)
  on every series is a row. A row is positive when the half-hour lies inside an injected interval
  on that series, and negative otherwise, except that rows inside a logged closure or an
  oracle-labelled event are dropped.
- **Odd and even months.** Transfers that start in odd calendar months fit the calibration. Transfers
  that start in even months score it. A row inside an injected transfer takes the month the
  transfer starts in, and any other row takes the month of its lag stamp. Experiment 82 fitted its
  logistic score and threshold on odd months too, so even-month rows stay out of every fit.
- **Candidates.** The causal detector keeps every candidate above a low score floor (the RO-NC
  threshold minus 4 log-odds), not only those above the threshold. Each candidate gets the
  detector's interval (the tracked level end, then the first return step), with its end known only
  if it lies at least 48 half-hours before the cut.
- **Arms.**
    1. Binary causal RO-NC flag: the row lies inside an interval of a candidate above the RO-NC
       threshold, open or closed.
    2. Calibrated score of the latest unreturned candidate: of the candidates of the row's series
       that start at or before the row and have not returned before it, take the latest. A logistic
       regression on its RO-NC score, the log of its age at the cut, and (when a return step is known
       after the row) the return step in noise units gives a raw score; isotonic regression on odd
       months maps the raw score to a probability. A row with no such candidate gets the odd-month
       rate.
    3. Arm 2 with the candidate's onset step in noise units as a further input to the logistic
       regression.
- **Scores.** Brier score, a reliability curve in deciles of the predicted probability, and area
  under the receiver operating characteristic curve, pooled and by hours since onset at the cut
  (under 24 hours, 1 to 3 days, over 3 days), by shape, by donor or receiver role, and by step size.
- **Real half-hours (descriptive).** The detector run causally at the same 292 cuts on the real
  residuals. Truth is the logged closures plus the oracle-labelled events of experiment 86. The
  logs miss ARAs, so precision is a lower bound.
- **Post hoc variants (labelled as such in the write-up).** The pre-registered rows keep every
  half-hour outside an injected transfer as a negative, but the real data hold unlabelled switching
  that the detector rightly flags. Variant `clean` drops the rows that the same detector, run on
  the uninjected residuals at the same cut, already flags. Variant `clean_large` also drops rows
  inside injected transfers whose donor step is under 4 noise units, which no detector that needs
  24 hours of data can see. A fourth arm takes the highest-scoring covering candidate instead of
  the latest one. The verdict follows the pre-registered rows.
- **Weather in the residual model.** The detector's residual model is the stage 1a model fed with
  ERA5 weather, as in experiment 88. ERA5 arrives about 5 days late. This experiment asks first
  whether a calibrated probability helps at all under ideal laboratory conditions, so it does not
  build the ENS-driven residual that a live probability needs. The write-up lists every laboratory
  shortcut.
- **Compute.** No GPU. About 3 to 6 CPU hours on 4 cores (slot C); workers write one file per
  batch and shape.

## Size (the five triggers)

- **Stored:** a few result tables under `data/`; no Delta table and no contract.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (how to decay a candidate with age), so a plan review runs.
- **Code whose callers cannot be named without searching:** the detector code is copied from
  experiment 88, which is unmerged, so a code review reads the copy.

## Process

Sonnet writes the plan, one Sonnet reviewer reads it, Sonnet implements, one Sonnet reviewer reads
the code for leakage (nothing at or after the cut reaches the detector or a feature, and no
even-month row reaches a fit), Sonnet runs it and writes the result, one Sonnet reviewer checks
the write-up against the printed report, and a Sonnet reviewer runs the prose review.
