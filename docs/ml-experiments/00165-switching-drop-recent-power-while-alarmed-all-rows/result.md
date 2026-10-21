---
id: "00165"
slug: switching-drop-recent-power-while-alarmed-all-rows
title: >-
  A forecast that ignores recent power whenever a switching alarm is active, scored on every real row
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
  - production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private score tables under
  data/ml-experiments/00165-switching-drop-recent-power-while-alarmed-all-rows/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one defensible definition of
  "inside an event" (by logged transfer, or by the alarm that is active), so both are fixed here
  before any score is read. Callers: this folder's scripts only; they copy the modules of
  experiment 163 and read experiment 156's stored alarm tables, experiment 87's stored best guess 2
  predictions, and experiment 155's arm C.
hypothesis: >-
  A forecast built from a frozen generalised additive model (GAM) and a Kalman filter on its recent
  errors, with the filter's readings set missing from the moment EITHER the pair alarm of
  experiment 156 OR the refit-offset neighbour-count detector (RO-NC) is known until the end rule
  of experiments 147 and 148 fires, forecasts better than the same forecast with no mask while an
  event is running and in the 72 hours after it ends, and costs nothing on the real rows outside
  events, because the filter then restarts and the mask only removes readings that contain moved
  load.
prediction: >-
  Written before any result. Outside events the either-alarm mask scores 0.04 to 0.15 points of
  effective capacity worse than the unmasked stack, because RO-NC's false alarms mask ordinary days
  (experiment 163 measured +0.102 for the union and +0.033 for the pair alarm on rows outside
  logged transfers and hand-filtered switching events). On the 72 hours after a logged transfer
  ends the either-alarm mask scores 5 to 8 points, against 12 to 15 for the unmasked stack and 9 to
  11 for best guess 2 (BG2), as in experiment 163. While a logged transfer runs, against observed
  power that contains the moved load, the mask scores worse than the unmasked stack, so the inside
  test of the decision rule is expected to fail on logged transfers. On rows while an alarm is
  active, whether or not it matches a logged transfer, the mask is expected to score within 1 point
  of the unmasked stack.
decision_rule: >-
  Pre-registered before any score is read. Rows: every row with a reading in the scored year of
  `fold_a_daily_gap_v1` (2025-07-02 to 2026-06-30, 292 runs) at the six decision Primaries (D5,
  D9, D10, D11, D14, D16), shared by every stack arm; BG2 is compared on the rows it shares with the
  stack. Row sets by logged transfer (experiment 163's definitions, by target time, forecast origin
  at or after the logged onset): BEFORE (logged onset to the earliest time either alarm is known),
  INSIDE (alarm known to logged end), AFTER (72 hours after the logged end). OUTSIDE is every other
  row: not within a logged transfer from its onset to 72 hours after its end. Rows are also split by
  the alarm active at the forecast origin (descriptive): ACTIVE (origin between the time the alarm
  is known and the time its end is known) and POST (the 72 hours of target time after that end),
  separately for the pair alarm and the union. Score: twice the mean pinball loss over 11
  quantiles, in points of effective capacity, lower is better. Arms: K0 (the unmasked GAM and
  Kalman filter), MASK_NEW (mask from the pair alarm), MASK_UNION (mask from either alarm, the
  headline), GAM_ONLY (no recent power at all, a reference), ORACLE (the logged interval masked
  from its logged onset, on event rows only), BG2 (experiment 87, `p_s6_bg2`); references: the
  manual heuristic and climatology. Intervals: outside rows are resampled by calendar month, event
  rows by logged transfer, 20,000 draws, 95%. Winter (November to March) first. Contrasts: C1
  MASK_UNION minus K0 on OUTSIDE; C2 MASK_UNION minus K0 on INSIDE; C3 MASK_UNION minus K0 on
  AFTER; C4 the same three for MASK_NEW; C5 MASK_UNION minus BG2 on AFTER and on OUTSIDE (no
  threshold). Verdict: `supported` when MASK_UNION has an upper 95% bound under +0.01 on C1 and a
  negative point estimate with an upper 95% bound under zero on C2 and C3; `partly supported` when
  C3 passes and C1 has an upper bound between +0.01 and +0.10; `not supported` when C1's upper
  bound is +0.10 or more, or C3 has a positive point estimate; `inconclusive` when fewer than 8
  transfer-substation pairs have AFTER rows. The threshold on C1 is +0.01 because the mask adds the
  alarm tables to the serving path, which is worth doing only if ordinary days do not get worse.
  D8 is not a decision Primary and has no frozen GAM here, so its 29 multi-day drops of experiment
  159 are described only: the share of drops whose start has a pair alarm or an RO-NC alarm within
  24 hours.
benchmark: >-
  The unmasked stack (K0) and BG2 (experiment 87, `p_s6_bg2`), on the same rows.
row_set_id: >-
  fold_a_daily_gap_v1 scored year, every row with a reading at six substations; not comparable with
  the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; tail coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99.
code: ml-experiments/00165-switching-drop-recent-power-while-alarmed-all-rows/
---

# Experiment 165: a forecast that ignores recent power while a switching alarm is active

This page is the plan. Experiment 163 scored the mask on logged transfers and on rows outside
transfers, and left out the rows inside hand-filtered switching events. This experiment scores every
real row at the six decision Primaries and reports the cost of false alarms.

- **Method.** Copy the stack of experiment 163. Readings are set missing from the time an alarm is
  known until the end rule of experiments 147 and 148 fires; the filter then restarts.
- **Weather.** The analysis proxy, as in experiment 163: weather up to the forecast origin only.
- **Schedule.** The GAM and filter are frozen, so no refit schedule applies.
