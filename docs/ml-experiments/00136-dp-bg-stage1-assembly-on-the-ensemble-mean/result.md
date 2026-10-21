---
id: '00136'
slug: dp-bg-stage1-assembly-on-the-ensemble-mean
title: 'Differentiable physics, best guess stage 1: GAM-K with everything that has helped, driven by the ensemble mean'
part: BG-1
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00136-dp-bg-stage1-assembly-on-the-ensemble-mean/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (how the ensemble mean drives the model, whether the spread is refitted per arm), and
  the plan pre-registers one of each. Callers: this folder's scripts, which copy experiments 111,
  114 and 133 and call the reviewed `studies.dp_*` modules without changing them. No new
  modelling.
hypothesis: >-
  A convex demand model with a Kalman filter on recent errors (GAM-K, experiment 111), given
  experiment 114's heating split by day type (T-all) and experiment 133's single shared filter
  noise setting (POOL), and driven by the ensemble mean of the ECMWF ensemble (ENS) instead of its
  51 members, forecasts Primary demand 0.5 to 168 hours ahead at least as well as BG2 (experiment
  87) and better than experiment 111's GAM-K. The mechanism is that the three gains came from
  different parts of the model (the hinge, the filter noise, the weather input) and so add, and that
  experiment 125 found that the members' own spread adds no tail skill, so one forecast on the mean
  loses nothing.
prediction: >-
  C1 (A0, GAM-K on the ensemble mean, minus experiment 111's GAM-K) is within +/-0.03 points.
  C2 (A1 = A0 + T-all, minus A0) is -0.02 to -0.04 points, a share of the -0.039 that experiment 114
  found on past weather, because ensemble weather is noisier. C3 (A2 = A1 + POOL, minus A1) is -0.01
  to -0.03 over all leads (experiment 133 found -0.036 at 12 to 36 hours). C4 (A3 = A2 with weekly
  refits, minus A2) is within +/-0.02. C5 (A2 minus BG2 as stored, 2.78) is -0.03 to -0.12, so A2
  scores 2.66 to 2.75 against BG2's 2.78. C6 (A2 minus experiment 111's GAM-K, 2.800) is -0.06 to
  -0.15. The q95 coverage on cold nights stays at 73% to 85% in every arm, as in experiment 125.
decision_rule: >-
  Six planned contrasts (C1 to C6), each read at the 99% level of the month-resampled paired
  bootstrap (12 calendar months of the scored year, 20,000 resamples). The headline arm is A2,
  which is the production-style monthly refit, with A3 as the check on weekly refits. The hypothesis
  is `supported` when C6's interval lies wholly below zero and C5's interval has an upper bound of
  at most +0.05 points (BG2 is not better by more than 0.05), with the replicate-months sign of
  both the same and at least 4 of 5 decision substations at or below zero on C6. It is `partly
  supported` when C6 is wholly below zero and C5's upper bound is above +0.05, or when C5 passes and
  C6 includes zero. It is `refuted` when C6's interval lies wholly above zero. It is `inconclusive`
  otherwise. The threshold on C6 is zero: every added piece is already built and tested in an earlier
  experiment, so combining them adds no pipeline step beyond the day-type split in the design matrix.
  The non-inferiority margin 0.05 is about a third of the BG2 minus GAM-K spread across lead bands in
  experiment 111, and a smaller difference is not worth a second forecast engine. The verdict is
  capped at `inconclusive` if the reproduction control P1 fails (A0 within +/-0.03 of experiment
  111's GAM-K is a prediction, but the FIX arm of experiment 133 reproducing 111 within 1e-4 on
  111's own mixture path must hold).
benchmark: >-
  BG2 as stored (experiment 87, 2.78 points over leads 0.5 to 168 hours at the five decision
  substations in the scored year); experiment 111's GAM-K as stored (2.800); climatology and
  manual_heuristic as stored by experiment 111, for absolute skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, at the five decision
  substations D5, D10, D11, D14, and D16 (D9 beside them), over leads 0.5 to 168 hours, then under
  24 hours, 24 to 72 hours, 72 to 168 hours, winter (November to March), cold nights, Christmas and
  bank holidays; coverage and pinball loss of the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00136-dp-bg-stage1-assembly-on-the-ensemble-mean/
production_status: not-for-production
adopted_in: none
---

# Experiment 136: does GAM-K with everything that has helped, on the ensemble mean, match BG2?

Plan only; the write-up follows the runs.

## Plan

**Question.** Does the convex model with the Kalman filter (GAM-K, [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)), given the pieces
that helped in later experiments and driven by the ensemble mean, forecast Primary demand at least as
well as BG2 ([experiment 87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)),
and better than experiment 111's GAM-K? No new modelling: every piece exists.

**Arms.** All arms share the rows, the weather calibration and solar shrink of experiment 111, and
the pinned capacity.

| Arm | Fits | Filter noise | Weather input | Spread |
|---|---|---|---|---|
| BG2 | as stored (experiment 87) | | | |
| K111 | as stored (experiment 111's GAM-K) | experiment 111's values | 51 members, mixture | as stored |
| A0 | experiment 111's monthly fits, 24-month window | experiment 111's values (FIX) | ensemble mean | refitted on the tuning months |
| A1 | A0 with T-all | FIX | ensemble mean | refitted |
| A2 | A1 | POOL (experiment 133) | ensemble mean | refitted |
| A3 | A2 refitted weekly | POOL | ensemble mean | refitted |

**Departure from the brief, and why.** The brief's A3 is "A2 with monthly refits", but experiment
111's fits are already monthly with a 24-month window, so A0 to A2 already refit monthly and the
brief's A3 would equal A2. A3 is therefore A2 with weekly refits, the production cadence, which
tests whether the monthly cadence the arms use changes the result.

**Ensemble mean as the input.** The forecast is one pass of the model through the calibrated
ensemble-mean temperature and irradiance, which replaces the 51 member passes of experiment 111. The
quantiles are the corrected mean plus the residual table fitted on the tuning months from the same
arm's own forecast (`e00133_ens.fit_spread_from_frame`), as K0-w does in [experiment
125](../00125-differentiable-physics-n8-ens-members-through-gam-k-and-the-spread-around-them/result.md).
The mean of a model with a hinge on temperature differs from the model at the mean temperature, so
C1 is not zero by construction.

**T-all.** [Experiment 114](../00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/result.md)
fitted T-all at yearly cuts with its own design code. This experiment copies that design code and fits
it at every monthly (and weekly) cut, and predicts on the ensemble-mean-driven cache.

**Filter.** The filter runs on the proxy-driven residual stream of each arm's own fits, with FIX for
A0 and A1 and POOL (the single shared cell of experiment 133) for A2 and A3. POOL was chosen on
experiment 111's fits, not on T-all fits; this is a caveat.

**Rows.** `fold_a_daily_gap_v1`, 09:00 UTC `power_fcst_init_time`, leads 0.5 to 168 hours, rows
outside hand-filtered switching events. The scored year is 2025-07 to 2026-06; the tuning months
are the odd months and the replicate months the even months of 2024-07 to 2025-06, as in experiment
133. D-labels only.

**Breakouts.** Lead bands (under 24, 24 to 72, 72 to 168 hours), winter, cold nights, holidays and
Christmas, and tail calibration at the 0.01, 0.05, 0.95, and 0.99 quantiles, with the q95 coverage
on cold nights. Absolute skill is reported for BG2, climatology, and manual_heuristic beside the
contrasts. The curtailment cost saving against the manual heuristic follows once a node metric exists
for these substations; if not run, the write-up says so.

**Controls.** P1: the FIX arm of experiment 133 reproduces experiment 111's GAM-K (already passed in
experiment 133). P2: A3 with a zero-width window of one month should reproduce A2 within noise (a
placebo that a change of cadence alone moves nothing is not claimed; C4 tests it).

**Five triggers.** Stored: private files only. Serving path: not touched. Degradation rule: not
touched. More than one design defensible: yes (mean input, spread refit), pre-registered above. Callers
not nameable without searching: no, the scripts copy experiments 111, 114, and 133.

**Resources.** Central processing unit (CPU) only, at most 8 cores with one thread per worker, each
job under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`.

**Advice followed.** From experiment 125, do not rely on member spread for tail skill. From
experiment 133, test one shared noise setting before any per-substation fit.
