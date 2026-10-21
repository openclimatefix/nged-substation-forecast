---
id: "00157"
slug: differentiable-physics-joint-pv-capacity-known-answer
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00157-differentiable-physics-joint-pv-capacity-known-answer/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because the maintainer fixed the arms and the stop rule. Callers: this folder's scripts, which
  copy experiments 110 and 129 and call the reviewed `studies.dp_*` modules without editing them.
  Asset graph: not touched.
hypothesis: >-
  Letting the convex demand model (`studies.dp_gam`) fit the solar capacity as a smooth non-negative
  path across months, with a penalty on growth, finds more of a known solar farm's injected output
  than the two-stage capacity of experiment 129, which steps once per season-year and was recovered
  at 0.78 to 0.88.
prediction: >-
  Written before any fit. The joint fit's raw recovery stays between 0.78 and 0.92 at every Primary,
  because the shortfall in experiment 129 comes from the mismatch between a farm's output and the
  fixed fleet shape, and from demand terms that absorb part of the solar signal, and neither changes
  when the capacity is given monthly freedom. The joint fit reaches 0.95 to 1.05 at fewer than 4
  of the 5 decision Primaries, so the experiment stops.
decision_rule: >-
  The joint fit is worth taking to the forecast stage only if its raw recovery lies in 0.95 to 1.05
  at 4 of the 5 decision Primaries (D5, D10, D11, D14, D16; D9 is reported beside them), against
  the two-stage 0.78 to 0.88. If the joint fit does not clearly beat 0.88 (median over the five
  farms above 0.88 at 4 of 5 Primaries), stop and write that up. If it passes, plan the forecast
  arms and do not run them: the pin against the joint fit and the no-solar floor on
  `fold_a_daily_gap_v1`. There is no cost of porting to weigh at this stage, because no forecast is
  scored.
benchmark: >-
  The two-stage as-of arm: experiment 129's arm 2 (season-year steps, ridge 0.1), fitted on the same
  rows before the same origin, and the two-stage pin, which cannot see an injection
row_set_id: >-
  Experiment 129's training rows (the feature cache of experiment 106) before the origin 2025-07-01,
  at D5, D9, D10, D11, D14, and D16. No forecast is scored.
metric: >-
  Raw recovery: the capacity found with a real farm's output added at 50% of the Primary's pinned
  capacity, minus the capacity found without it, divided by the injected capacity, read as the mean
  of the fitted path over season-year 2024 (July 2024 to June 2025). Capacity unit: the 99th
  percentile of alternating-current output over daylight half-hours, in MW
code: ml-experiments/00157-differentiable-physics-joint-pv-capacity-known-answer/
production_status: not-for-production
adopted_in: none
---

# Experiment 157: does fitting the solar capacity jointly with the demand terms recover an injected farm?

Plan only; the write-up follows the runs.

## Plan

**Question.** [Experiment 129](../00129-differentiable-physics-pv-capacity-known-answer-test/result.md)
found a capacity per season-year from each Primary's own power and recovered an injected real
farm's output at 0.78 to 0.88. [Experiment 139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md)
found the as-of capacity no better than the pin, and the hindsight capacity equal to it. Net power
is linear in capacity, so the capacity can instead be fitted inside the GAM-K fit as a monthly path.
Does that recover the injection near 1.0? This experiment runs only the known-answer stage, as the
maintainer asked, because the expected forecast gain is under 0.05 points.

**Arms, all on the same rows.** Every arm trains on rows before the origin 2025-07-01 and has the
same demand terms, so that only the capacity differs.

| Arm | Capacity | Role |
|---|---|---|
| 1. Pin | Experiment 45's `frozen` vintage, held | Reference. It does not refit, so it recovers none of the injection |
| 2. Two-stage as-of | Experiment 129's arm 2: non-negative increments per season-year, ridge 0.1 on the cumulative capacity | Benchmark |
| 3. Joint | A non-negative level for each calendar month (the derated unit shape times the level), a penalty on the squared change of the level from month to month, and a near-zero ridge on the level | The arm under test |

**The joint fit's growth penalty is fixed in advance.** It is 0.1 times the mean diagonal of the
Gram matrix over the capacity columns, the share experiment 129's arm 4 used. Two values, 0.01 and
1.0, run as labelled sensitivity checks that cannot change the verdict. The shape parameters (tilt
35 degrees, south-facing, DC:AC 1.15, inverter clip at the AC capacity, 0.4% per kelvin
de-rating) stay as in `studies.dp_gam`. The Kalman filter acts on residuals only in the forecast
stage, so it is not involved here: this stage fits capacity and never forecasts.

**Injection.** Each of the five real farms of experiment 129 (labels A, B, C, D, F; anonymised) is
added to the Primary's power over all training rows at 50% of the Primary's pinned capacity, as a
farm's output divided by its 99th percentile. Rows where the farm's reading is missing or a
dropout are removed from both the injected and the matching base fit, so a difference of fits never
mixes two row sets. Recovery is read per Primary and farm, then summarised by the median over farms.
The base fit has no injection.

**Recovery read-out.** The raw recovery averages the fitted path over season-year 2024, the latest
complete season-year before the origin. The joint fit's last-month value, which is what a forecast
would hold, is reported beside it. The divisor that experiment 129 applied (the median recovery of
other farms) is not applied to the pre-registered gate, because the question is whether the joint
fit needs no divisor. The divided two-stage recovery is shown for context.

**Primaries and checks.** D5, D10, D11, D14, and D16 count for the decision. D9 is reported beside
them. The tables show every farm at every Primary, plus the unmodified base capacity of each arm
in MW, so that a reader can see the joint path's level against the two-stage steps. As a wiring
check, a fit with zero injection must reproduce arm 2's published capacity for the same cut to
three decimal places.

**Five triggers.** Stored: private files only. Serving path: not touched. Degradation rule: not
touched. More than one defensible design: no, because the maintainer fixed the arms and the stop
rule. Callers not nameable without searching: no.

**Compute.** CPU only, at most 4 cores, thread caps all 1, each job under `systemd-run --user
--scope -p MemoryMax=16G -p OOMPolicy=continue`. The first Primary is timed before the full grid.

**Reviews.** One Sonnet code review (leakage first: no row at or after the origin may train) and one
Sonnet prose review. No Opus review.

**Advice followed.** From experiment 129, count only season-years with at least 12,000 training
half-hours when reading a capacity, and use an as-of capacity and never a hindsight capacity in a
forecast test. Departs from none.
