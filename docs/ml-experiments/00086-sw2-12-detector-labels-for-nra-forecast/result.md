---
id: "00086"
slug: sw2-12-detector-labels-for-nra-forecast
status: planned
size: medium
hypothesis: >-
  On the 292 daily ENS forecast runs of experiment 48, giving the monthly-refitted (P-roll) Primary demand
  model the lags minus the MW that switching moved, together with that MW as three columns, lowers
  the quantile score on the half-hours whose lag sits inside a switching event, when the events and
  MW come from the XL load-transfer detector of experiment 82 instead of NGED's logs, and the gain
  is smaller than with the oracle labels of experiment 41 by the share of events XL misses.
prediction: >-
  On the in-event-lag rows outside switching, with oracle labels the arm lowers the score by 0.3 to
  0.6 points of capacity (experiment 75, weekly refits: corrected lags alone -0.42 after the mask). With XL labels
  it lowers the score by 0.1 to 0.4 points, so the oracle-minus-XL gap is -0.05 to -0.3. On all
  rows outside switching, neither arm changes the score by more than 0.02 points either way.
decision_rule: >-
  Three planned contrasts on the rows outside switching whose lag is inside an oracle-labelled
  event (the headline rows; the same rows for every arm), quantile score over all leads, 98.33%
  interval (Bonferroni across three), whole validation months resampled. C1, `lagmw_xl` minus
  `plain`; C2, `lagmw_oracle` minus `plain`; C3, `lagmw_xl` minus `lagmw_oracle`. The XL labels are
  supported as a training-time input when the upper end of C1 is below zero. A production use also
  needs the no-harm check: C1 on all rows outside switching has an upper end of at most +0.02
  points (0.34% of the plain arm's score, experiment 75's margin). The threshold is zero because
  the change needs a detector on the serving path, which has to pay for its complexity with a gain
  whose interval excludes zero. If C1 does not exclude zero but the no-harm check passes, the
  answer is that detector labels add nothing outside switching that we can show. C3 is descriptive:
  it is the skill lost without NGED's logs. Every other comparison is descriptive.
benchmark: >-
  The `plain` arm refitted by this experiment's own code. `manual_heuristic` and `climatology`
  from experiment 48's stored arms are descriptive references on the same rows.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity,
  on the rows outside switching (all of them, and the in-event-lag subset)
code: ml-experiments/00086-sw2-12-detector-labels-for-nra-forecast/
---

# Experiment 86: do the detector's switching labels give the same gain as NGED's logs?

Plan only; the write-up replaces this page when the run finishes.

- **Question.** [Experiment 75](../00075-sw2-11-nra-forecast-with-labelled-switching-masks/result.md)
  found that outside switching an oracle mask and oracle MW columns changed the Primary score by
  under 0.06 points. Only the half-hours whose lag sits inside an event gained (corrected lags
  -0.42, -1.51 to +0.02). This experiment gives XGBoost the corrected lags and the MW columns
  together, once with the oracle labels of experiment 41 and once with the labels of the XL
  detector of [experiment 82](../00082-sw2-18-19-combined-refit-offset-and-same-sign-count/result.md)
  (MW-weighted recall 0.770 on injected transfers). The gap is the skill lost without NGED's logs.
- **Schedule.** The cheap P-roll schedule of [experiment 49](../00049-vp1-screening-reference-under-p-roll-demand/result.md):
  a refit on the 1st of each month from 2025-07-01 to 2026-06-01, on the latest 365 days of the ENS
  control member's rows, 150 rounds, learning rate 0.15, recency half-life 0.5 years, seed 0. The
  contrast is not about retraining cadence, so the weekly schedule of experiment 75 is not used.
- **Arms** (experiment 75's N1 recipe and inputs). `plain`: all rows, no label input. `lagmw_oracle`: lags minus
  MW plus the three MW columns, oracle labels, trained on all rows. `lagmw_xl`: the same with XL
  labels. `mask_xl` (descriptive): training rows inside an XL event dropped, nothing else.
- **XL labels.** The XL alarms over the whole history, with the first-return end rule, give each
  alarm series an interval and the signed MW of its 24-hour step. Neighbours whose own step has the
  opposite sign (more than 2.0 noise units, the experiment 82 neighbour rule) get the same interval
  and their own step as a receiver. No switching log is read. The labels use hindsight (the end of
  an event is known 24 hours later), so the result is an upper bound for a service.
- **Rows.** Experiment 75's 292 runs; scored on rows outside oracle-labelled switching and logged
  closures; headline rows have a lag inside an oracle event. Also reported: the number of those
  rows whose lag is inside an XL event, and the number of rows with a lag inside an XL event that
  the oracle does not label.
- **Compute.** Slot A, 8 cores, GPU at most 12 GB. 4 workers of 2 threads each (12 refits of 16
  series and 4 arms at 150 rounds is a small job).
- **Size.** Stored: private predictions only. Serving path: untouched. Degradation rule:
  untouched. More than one defensible design: no, the brief fixes the arms and the labels. Callers
  across code: the XL export script runs inside experiment 82's folder because its pickled score
  can only be loaded there; the fit and scoring scripts are copies of experiment 75's.
- **Earlier advice followed.** Test corrected lags with the MW columns (experiment 75 advice 2);
  judge against a same-code `plain` arm (advice 4). Departed from: none.
