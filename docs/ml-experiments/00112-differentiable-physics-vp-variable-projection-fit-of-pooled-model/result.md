---
id: "00112"
slug: differentiable-physics-vp-variable-projection-fit-of-pooled-model
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [differentiable-physics-demand-calendar-and-weather, differentiable-physics-long-history-and-drift]
status: finished
size: >-
  complex for the module (a new fit path in studies.dp_pooled, reused by any later pooled arm);
  medium for the experiment scripts, which reuse experiment 105's. See "Size triggers".
hypothesis: >-
  Part 3's pooled DP lost to the per-site DP because gradient descent did not reach the optimum,
  not because pooling hurts: solving every linear coefficient exactly at each step, with gradient
  steps only on the non-linear parameters, makes the pooled DP at least as good as the per-site DP
  on substations with 2 years of history.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  Experiment 105's gradient fits P, P-map, S, S-4w, S-3m, and the equivalence arm; the same Adam
  fit on VP's objective (P-map-same); Part 2's centred G-all; the per-site convex model on the same
  2-year window (G-2y); climatology; manual_heuristic
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103); MW sites scored 2025-07-01 to 2026-06-30, replicate
  2024-07-01 to 2025-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site
verdict: inconclusive
production_status: not-for-production
take_home_message: >-
  Solving the linear coefficients exactly did not make the pooled model forecast better: it reached
  a training objective 16,000 to 27,000 nats lower than the gradient fit, but with the interaction
  network on it scored worse on new data (5.49 against 3.27 points) and from two starts the scores
  differed by 1.6 points. With the network off, the exact fit is stable (both starts agree to 0.004
  points) and matches the gradient fit (3.30 against 3.27), so the optimiser was not what held the
  pooled model back. The known-answer test failed, because at realistic noise the model cannot pin
  down heating thresholds, slopes or capacities.
date: 2026-10-20
pr: 1396
advice:
  - >-
    Do not read a lower training objective as a better pooled model: with the interaction network
    on, a 16,000-nat gain scored 2.2 points worse (V1).
  - >-
    Drop the interaction network from the pooled demand model before comparing fitting methods:
    with it off, both variable-projection starts agree to 0.004 points and the fit matches the
    gradient fit (VP-P-nonet, exploratory).
  - >-
    Fit the equivalence configuration (one substation, no network, no shared shapes, capacity
    pinned to the convex path) by variable projection, not by Adam: it scores 0.45 points better
    than the gradient fit and 0.20 better than the convex model (L1).
  - >-
    Never fit the calendar and seasonal terms exactly to a window under three months, or onboard
    a substation by an exact fit of its own 4 weeks: the per-site VP fits scored 49 to 71 points.
  - >-
    Write any future parameter-recovery gate as a calibrated-interval test over at least 20
    synthetic substations, not as point tolerances: at noise sd 5.0 points no variant of the
    pooled model passes point tolerances, although recovery is exact without noise (gate).
  - >-
    Interpret no per-substation heating threshold, slope or capacity from a pooled or
    variable-projection fit at real noise levels; report them as weakly identified (gate, profile).
code: ml-experiments/00112-differentiable-physics-vp-variable-projection-fit-of-pooled-model/
---

# Differentiable physics, VP: fitting the pooled model's linear coefficients exactly

**Solving the linear coefficients exactly did not improve the pooled model, and the gap to a
per-substation model is not an optimiser problem.** Variable projection (VP) reached a training
objective 16,000 to 27,000 nats lower than the gradient fit on the identical objective, yet with
the interaction network on it forecast worse (5.49 against 3.27 points at the five decision
substations) and its two starts differed by 1.6 points. With the network off, the exact fit is
stable (both starts agree to 0.004 points) and matches the gradient fit (3.30 against 3.27).
The known-answer test failed on its pre-registered definition, so the verdict is capped at
inconclusive.

We asked whether the pooled model lost to a per-substation model in experiment 105 because gradient
descent stopped short of the optimum. Variable projection solves every coefficient that enters the
forecast linearly (shapes, calendar terms, heating slopes, capacities, levels) exactly at each step,
and moves only the remaining non-linear parameters by gradient steps. We fitted the pooled model
and a model per substation this way at three training cuts, scored 2025-07-01 to 2026-06-30 with
2024-07-01 to 2025-06-30 as a replicate, and compared with the gradient fit on one objective.
**It narrows the search:** exact optimisation helps a simple model (L1 below) and does nothing for
the pooled model, whose network absorbs structure the data cannot pin down. The gate failure, the
capacity profile and the network-off arm all point the same way. The gate failed (all three tests),
so the pooled results are labelled "gate failed", and the verdict cannot exceed `inconclusive`.

**Figure 1. The planned contrasts.** The pooled model and the per-substation model fitted by VP are
no better than the gradient fits, and the onboarding contrasts are dominated by a broken comparator.

![Figure 1: planned contrasts](figure_01_headline.svg)

**Figure 2. Training objective.** VP reaches a far lower objective than the gradient fit
(P-map-same), and its two starts differ by up to 3,000 nats.

![Figure 2: training objectives](figure_02_objectives.svg)

**Figure 3. Scores.** The exact fit with the network on scores worst; without the network it
matches the gradient fit, and none of the pooled fits beats the per-substation convex model.

![Figure 3: quantile scores](figure_03_scores.svg)

**Figure 4. Raw forecasts** in a winter week and a spring week, weeks fixed in advance.

![Figure 4: weeks](figure_04_weeks.svg)

**Figure 5. Capacity profile.** The free fits sit 1,400 to 8,600 nats above the lowest profile
point.

![Figure 5: capacity profile](figure_05_profile.svg)

**Figure 6. The failed gate.** Recovery of a synthetic fleet is exact without noise, and fails at
realistic noise for every variant.

![Figure 6: gate](figure_06_gate.svg)

## Results

- **Arms and fits.** VP-P: Part 3's pooled model (K = 2 shapes, network (64, 64), fitted capacity),
  MAP, fitted by VP from two starts (seed 0; seed 1 with thresholds at 17 C); the arm is the
  lower-objective start (start 0 at the 2024-07-01 and 2025-07-01 cuts, start 1 at 2023-07-01).
  VP-S: the same model per MW substation. P-map-same:
  experiment 105's Adam fit on VP's objective; P-map-polish: its non-linear parameters with the
  linear coefficients solved exactly. References: 105's P, P-map, S and equivalence arm,
  G-all (Part 2), G-2y (the convex model on the 2-year window), climatology, manual heuristic.
  Post hoc, exploratory: VP-P-nonet (network off). Row set, window, pool and masks as experiment 105.
  Every VP fit used 600 Adam warm-start steps, L-BFGS to 300 evaluations, and an Adam fallback of
  1,200 steps (VP-P) or 400 (others) when it missed the convergence test; no fit met that test, so
  none is exactly at its optimum. Schedule: one frozen fit per cut, no refits.
- **Scores (five decision sites, scored year; replicate year in brackets).** VP-P 5.49 (4.26);
  VP-P start 1 3.89 (3.64); VP-P-nonet 3.30 (3.21); VP-S 4.09 (3.70); P-map-same 3.27 (3.11);
  P-map-polish 3.29 (3.18); 105's P 3.42, P-map 3.46, S 3.04; G-2y 2.94 (3.02); G-all 3.07;
  climatology 4.07; manual heuristic 3.49.
- **V1, the fitting method on one objective (planned).** VP-P minus P-map-same: **+2.215 points
  (98.75% interval +0.398 to +5.041)**, 0 of 5 sites better; replicate year +1.154 (+0.527 to
  +1.849), 0 of 5 better. In nats, VP-P minus P-map-same is **-16,747** at the 2025-07-01 cut
  (172,257 against 189,005), -16,014 at 2024-07-01 and -27,257 at 2023-07-01; P-map-polish lowers
  P-map-same by 1,005 nats and changes the score by +0.013 (-0.001 to +0.025), so the linear
  coefficients are not what limits skill. Experiment 105's own P-map is 239,053 on this objective
  (a different objective, shown as a reference only).
- **C1v, pooling under VP (planned).** VP-P minus VP-S: +1.394 (-0.896 to +4.508), 0 of 5 better;
  replicate +0.561 (-0.095 to +1.289). Non-inferiority (upper end at most +0.10) fails.
- **C2v and C3v (planned, six substations).** VP-onb4w minus VP-S-4w -60.8 (-81.9 to -38.0); VP-short3m
  minus VP-S-3m -49.4 (-86.7 to -17.3); both 6 of 6 better. **These contrasts are not evidence
  for pooling:** the per-site VP fits on 4 weeks and 3 months score 70.6 and 48.4, because the exact
  solve fits the lightly penalised seasonal terms to a short window and extrapolates them. Against
  experiment 105's gradient arms: VP-onb4w minus P-onb4w +1.460 (+0.439 to +2.522); VP-short3m minus
  P-short3m -0.122 (-0.554 to +0.182).
- **Start spread.** The two VP-P starts differ by 1.599 points (limit 0.10). VP-P-nonet: 0.004.
- **Ladder (L1, exploratory).** The equivalence configuration fitted by VP scores **2.877**, against
  105's 3.326 (-0.449, -0.537 to -0.373, 5 of 5 better) and G-all's 3.072 (-0.195, -0.298 to
  -0.075). All five threshold starts reach the same objective to 1 nat at every site and cut.
  L1 minus s_equiv_all contains the optimiser and the change of objective (natural-scale slope
  prior, fixed noise scale) together, and the heating-form share of the 0.25 gap cannot be
  separated from it. L2 was dropped. D9 and D16 had a numerically singular replica fit and use
  the other sites' spread.
- **Capacity profile (labelled).** Each free fit's capacities agree across the three starts within
  3 points at every substation (largest 1.6 points for K = 2 and 1.3 for K = 4): **line 1 passes at
  both configurations.** Each free objective is 1,400 to 8,600 nats above the lowest profile point
  (tolerance 31 to 34 nats): **line 2 fails**; the profile minimum is at 0.85 times the convex
  capacity path, and the free capacities lie between 0.5 and 1.05 times it: **line 3 fails.** The
  three starts reach objectives up to 7,000 nats apart and none converged, so the network's
  landscape has many local optima. The pinned fits did not converge either.
- **Gate (failed, pre-registered definition).** Test 2: worst-case errors at noise sd 5.0 points
  were 5.1 to 5.7 C in threshold (limit 1), 70 to 76% in slope (15%), 39 to 44% in capacity (10%),
  2.0 to 2.5 points in growth increment (1.5), and 2.1 to 6.1 points of network output (0.1). Test 3:
  no start converged; spread 31 nats per 10,000 rows (limit 1). Test 4: recovered over injected
  capacity 0.77 to 0.90 (median 0.82; limit 0.9 to 1.1; experiment 105 had 0.78); other sites
  moved 1.1 to 4.8%.
- **Gate diagnostics (post hoc).** Continuing 1,500 more evaluations lowers the objective by 4,200
  nats and leaves the errors unchanged, so the failure is not under-training. Without noise,
  with the network off, recovery is exact (all errors 0.000), so the module is correct. At noise
  sd 1.0 the network-off fit recovers thresholds to 0.02 C, slopes to 1% and capacities to 1.3%; the
  network-on fit has slopes 17% off and a 0.40-point network output; hiding temperature from the
  network by day fixes the slope (0.8%) but not the output (0.31). At the pre-registered noise
  every variant fails: network off 2.7 C, 30%, 33%; temperature hidden by day 3.1 C, 37%, 37%.
  Generating and fitting the fleet without the day-of-year harmonics (the hypothesis that the
  seasonal terms absorb the seasonal cycle of temperature and sun) improves the network-off
  thresholds and slopes (1.7 C, 21%) but not the capacities (34%), and worsens the network-on fit
  (5.9 C, 79%), so the hypothesis is not confirmed. **Hypothesis, not a result:** the harmonics,
  the per-band profile and the calendar terms absorb most of the seasonal cycle, leaving only
  day-to-day weather anomalies to identify slope, threshold and capacity, so the effective sample
  is far smaller than the row count; identification should improve with more weather anomalies,
  that is, with more years of history, which could help interpretability even where scores do not
  move.
- **Tails (five sites, coverage at q01, q05, q95, q99; pinball at the same).** VP-P 0.023, 0.052,
  0.910, 0.983 (1.929, 2.548, 1.057, 0.272); VP-P-nonet 0.016, 0.047, 0.829, 0.922 (0.277, 0.735,
  0.996, 0.352); P-map-same 0.015, 0.043, 0.834, 0.920; G-2y 0.015, 0.043, 0.846, 0.940. VP-P's
  lower tail covers 2.3% of rows at q01 (nominal 1%) with a pinball loss of 1.929 against 0.198.

## Checks

- **Median minus mean forecast per site (scored year, points).** VP-P: D10 -1.11, D11 0.58, D14 0.27,
  D16 0.43, D5 0.01, D9 -0.27, D18 0.93. VP-P-nonet: D10 -1.80, D11 0.94, D14 0.21, D16 0.20, D5 0.32.
  P-map-same: D10 -1.72, D11 0.73. D10's gap of about -1 to -1.8 points appears in every arm,
  including S (-1.74), so it is not a VP effect.
- **Error of the mean forecast beside each contrast.** V1 +2.676 (+0.847 to +4.840); C1v +1.460
  (-0.665 to +3.950); C2v -64.9; C3v -53.1. The quantile-score and mean-error contrasts agree in sign.
- **Leave one site out.** V1 stays +2.1 to +2.3 with any site dropped (0 of 4 better); C1v stays
  +1.2 to +1.7. Dropping D10's unlabelled jump (2025-11-17 to 2025-12-14), the winter or the summer
  half, or masked rows changes no sign.
- **L0 control.** G-all scores 3.072 at the five sites, as in Part 2. P-map-polish is never above
  P-map-same on the VP objective, at all three cuts.
- **Audit triggers that fired.** Trigger 1: the gate failed. Trigger 2: V1 contradicts its
  prediction. Trigger 7: the result decides the pooled route. An Opus audit is requested, and
  its findings are triaged in the pull request.

## Predictions, held or failed

- **Held:** VP-P sits over 2,000 nats below P-map-same (by 16,747); the capacity-profile start
  agreement (line 1); P-map-polish does not exceed P-map-same; each VP fit took under 10 minutes
  of GPU time before its fallback (519 s at most) and peak memory was under 1 GB; VP-P's q01, q05
  and q99 coverage are within the predicted bands.
- **Failed:** V1 -0.15 to -0.45 at 5 of 5 sites (it is +2.215, 0 of 5); C1v within +/-0.10 (+1.394);
  start spread under 0.05 (1.599); C2v and C3v -0.5 to -2.0 and -0.2 to -1.0 (-60.8, -49.4, from a
  broken comparator); the capacity profile passes (lines 2 and 3 fail); VP-P's q95 coverage is
  0.910, 4 points below nominal (limit 3). The prediction that VP-P's capacities lie within 15% of
  G-2y's at 4 of 5 sites was not evaluated.
- **Not built:** VP-P-pin, the equivalence ladder's L2, batching of copies, and substation cards.
  Cards would show per-site thresholds, and none is interpretable at real noise levels (weakly
  identified at real noise levels (experiment 112); compare across substations with care).

## Caveats

- No VP fit met its convergence test within 300 L-BFGS evaluations and the Adam fallback, so the
  exact-fit comparison is between fits that are lower in objective but not at the optimum.
- ERA5 and CAMS flatter the solar term; the hand-filtered masks are hindsight labels; MVA
  `tan(phi)` and `q0` are zero in VP, unlike P-map's; the replicate year is in sample for K and the
  deviation scale; MAP only.
- The known-answer gate used the plan's noise (residuals of the per-site convex model, sd 5.0
  points). A gate for parameter recovery should test calibrated intervals (an 80% interval covering
  the truth 70 to 90% of the time over at least 20 synthetic substations, with the widths), not point
  tolerances.
- VP-S-4w and VP-S-3m are broken comparators (see C2v); the leave-one-out and short-history VP
  refits start cold with L-BFGS only. The 2023-07-01 and 2024-07-01 fits exist for the spread.
- GPU time used: 5.5 of the 10 hours cap, as tallied by `e00112_budget.py`.

## What the next experiment should answer

**Whether the pooled demand model is worth its network.** Run the pooled model without the network
(VP-P-nonet matches the gradient fit and its fits are unique) against the per-substation convex
model on more years of history, because identification should improve with more weather anomalies.
Do not fit a calendar or seasonal term exactly to a window under three months.
