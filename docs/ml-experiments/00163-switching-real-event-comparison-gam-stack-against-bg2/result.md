---
id: "00163"
slug: switching-real-event-comparison-gam-stack-against-bg2
title: >-
  Real logged load transfers: the XGBoost-free stack with a switching mask against best guess 2
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
  - production-reference-and-best-guess
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00163-switching-real-event-comparison-gam-stack-against-bg2/
  only (per-row score tables). Serving path: not touched. Degradation rule: not touched. Designs:
  more than one defensible row-set definition (by target time or by forecast origin), so the
  definition is fixed here before any score is read. Callers: this folder's scripts only; they copy
  the modules of experiment 156 they need and read experiment 156's stored alarm tables, experiment
  87's stored best guess 2 predictions, and experiment 155's stored arm C.
hypothesis: >-
  On the logged real load transfers at the six decision Primaries, a forecast built from a frozen
  generalised additive model (GAM), a Kalman filter on its recent errors, and a mask that sets
  the readings missing from the moment an alarm is known (the pair alarm of experiment 156, with
  the end rule of experiments 147 and 148) forecasts the days after a transfer ends better than
  best guess 2 (BG2), because BG2 takes lagged power as an input and so carries the transfer
  into its forecast, and the mask does not. The stack costs little on the real rows outside
  transfers.
prediction: >-
  Written before any result. On the rows within 72 hours after a logged transfer ends (forecast
  origin after the transfer began), the unmasked stack (STACK-K0) and BG2 both score 4 to 12 points
  of effective capacity, and the mask closes 20% to 60% of the gap between STACK-K0 and the
  oracle mask (the logged interval masked from its true onset), because the pair alarm finds
  only 5 of 35 logged transfers over a day and so most transfers are never masked. The union mask
  (pair alarm plus the refit-offset neighbour-count detector, RO-NC) closes 40% to 75%. BG2 lies
  between STACK-K0 and the oracle mask, closing 0% to 50%. Before the alarm is known, no arm can
  react, so all arms score within 1 point of each other. Outside transfers the pair-alarm mask
  scores within 0.10 points of STACK-K0 (experiment 156 measured +0.064), and BG2 scores
  within 0.10 points of the stack's best arm (experiment 155 found no ranking at the 0.3-point
  interval width of five series).
decision_rule: >-
  Pre-registered before any score is read. Rows: the scored year of `fold_a_daily_gap_v1`
  (2025-07-02 to 2026-06-30, 292 runs), six decision substations (D5, D10, D11, D14, D16, D9), rows
  with a reading, shared by every arm and by BG2. A logged real transfer (the 66 labelled
  `real_operation` transfers of experiment 41 with visible effects) contributes one event for each
  of the six substations it involves. Three row sets per event, defined by the TARGET time and
  restricted to forecast origins at or after the logged onset: BEFORE (target from the logged onset
  to the earliest time either alarm is known, or to the logged end if neither is ever raised),
  INSIDE (target from that alarm time to the logged end), AFTER (target from the logged end to 72
  hours later). Score: twice the mean pinball loss over 11 quantiles, in points of effective
  capacity (lower is better), against the observed power. Only AFTER has a truth free of the
  transfer, because then the observed power is the normal running arrangement; BEFORE and INSIDE are
  scored against observed power that contains the transfer and are descriptive. The clean ceiling
  is ORACLE, the stack with the logged interval masked from its logged onset and the end known at
  once; the gap closed is (STACK-K0 minus arm) divided by (STACK-K0 minus ORACLE), pooled over the
  AFTER rows of events with a positive gap. Arms: STACK-K0, STACK-NEW (the pair alarm with the 4-hour
  window as chosen by experiment 156, headline), STACK-UNION, STACK-RONC, GAM-ONLY (no correction
  from recent power), ORACLE, and BG2 (experiment 87, `p_s6_bg2`); references: experiment 155's arm
  C (the unmasked stack with recipe R), the manual heuristic, and climatology. Intervals: events are
  resampled by logged transfer (cards), 20,000 draws, 95%; the rows outside events are resampled by
  calendar month. Winter (November to March) is reported first, then the other months. Planned
  contrasts: C1 STACK-NEW minus BG2 on AFTER; C2 STACK-UNION minus BG2 on AFTER; C3 the share of
  the STACK-K0-to-ORACLE gap closed by STACK-NEW and by STACK-UNION on AFTER; C4 STACK-NEW minus
  STACK-K0 and STACK-NEW minus BG2 on all real rows outside events and their 72 hours after (no
  harm); C5 the same for STACK-UNION. Verdict: `supported` when C3 shows STACK-NEW or STACK-UNION
  closing at least 40% of the gap and C4 and C5 for that arm have an upper 95% bound of at most
  +0.10 points; `not supported` when both arms close under 20% of the gap; `partly supported` between;
  `inconclusive` when fewer than 8 events have AFTER rows. The BG2 contrasts (C1, C2, and C4 against
  BG2) are reported with their intervals and carry no threshold, because five or six series cannot
  rank the stack against BG2 (experiments 136 and 155).
benchmark: >-
  BG2 (experiment 87, `p_s6_bg2`) and the unmasked stack (STACK-K0), on the same rows.
row_set_id: >-
  fold_a_daily_gap_v1 scored year rows with a reading at six substations, as in experiments 155
  and 156; event row sets defined above; not comparable with the leaderboard
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; share of the gap to the oracle mask closed; tail coverage and pinball loss at 0.01,
  0.05, 0.95, and 0.99.
code: ml-experiments/00163-switching-real-event-comparison-gam-stack-against-bg2/
---

**Planned, not yet run.** This page is the plan; the write-up replaces it when the scores exist.

The question: when an operator moves load between substations and the transfer is logged, which
forecast copes better in the days after, a forecast that sets the readings of a suspected transfer
missing (the stack) or best guess 2 (BG2), which learns from recent power as it is? The stack and
BG2 are scored on the same logged real transfers and the same real rows outside them.
