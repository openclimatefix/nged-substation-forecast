---
id: "00112"
slug: differentiable-physics-vp-variable-projection-fit-of-pooled-model
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [differentiable-physics-demand-calendar-and-weather, differentiable-physics-long-history-and-drift]
status: planned
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
verdict: <supported, partly supported, refuted, or inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00112-differentiable-physics-vp-variable-projection-fit-of-pooled-model/
---

# Differentiable physics, VP: fitting the pooled model's linear coefficients exactly

**Status: planned.** This page is the plan, committed before any code runs. The write-up replaces it
when the experiment finishes.

**The theme's `plan.md`** gets a VP row in this PR, and the finished PR updates it.
**Earlier advice:** VP follows experiment 105's advice to fix the capacity fit with an optimiser
that reaches the profile minimum, and departs from experiment 106's pinned capacity in its decision
arm, because whether a free fit reaches the minimum is the question. **Data lessons** (such as a
threshold left unidentified above a site's highest temperature) go on
`docs/ml-experiments/data-processing.md` in the same PR.

## What VP asks, and the evidence that the optimiser is at fault

**VP asks whether the pooled DP matches a per-site model once its fit reaches the optimum.** Part
3's C1 (P minus S) is +0.385 points (98.33% interval +0.18 to +0.69, 5 of 5 sites worse, error of
the mean forecast +0.607). Three Part 3 results point at the optimiser, and none settles it:

- **Capacity profile.** With one multiplier on the convex capacity path, refitted over {0.5, 0.7,
  0.85, 1.0, 1.15, 1.3}, the loss is lowest at 1.0 (767,141 nats); the free fit sat 4,563 nats above
  it, and its capacity followed its start. The audit found the losses not like for like.
- **Equivalence check.** The per-site DP set up like the convex model (`s_equiv_all`: no network,
  no shared shapes, weak deviation penalty, capacity pinned to G's path, all years) scored 0.25
  worse than centred G-all (+0.12 to +0.41), 5 of 5 sites, mostly D10 (+0.90; others +0.05 to
  +0.12); its mean-forecast error is 0.60 worse. **That arm is not G's model:** its heating is one
  hinge at a learned threshold on a learned temperature mix, where G has one non-negative slope per
  smoothing and band at 15.5 °C. The +0.25 mixes optimiser, heating form, and other specification.
- **Ablations.** P-nonet minus P −0.03; P-map minus P +0.04.

**Variable projection (VP) separates the causes.** Given the non-linear parameters φ, every other
coefficient enters linearly, so its best values solve a bounded convex quadratic, which
`dp_gam.solve_bounded_quadratic` solves exactly. Gradient steps move only φ, on the objective with
the linear part already minimised.

## Which vehicle: its own framework PR and a small experiment

**Recommendation: a framework PR adding `fit_pooled_vp` to `studies.dp_pooled`, then a small
experiment on Part 3's rows.** Part 5's pooled branch is switched off and Part 5's module PR changes
`dp_gam` and `dp_operator`, which VP does not touch, so separate PRs are reviewed and merged in
parallel without holding up Part 5's convex results. The review is about 400 lines (a design
builder, an outer loop, one gradient formula), and later pooled arms switch with `fitter="vp"`.

## Size triggers

**Stored data:** private files only. **Serving path, degradation rules:** not touched. **More than
one defensible design:** yes (the linear set, the outer optimiser, the priors, the noise scale); one
of each is pre-registered. **Callers:** this experiment, N9, and the pooled arms of Parts 5 and 6.
The module PR goes off `main` with two Opus code reviews (the gradient formula and the training cut
first); the maintainer merges it.

## Module (`studies.dp_pooled`, tests under 5 minutes each)

| Linear θ (solved exactly, with bounds) | Non-linear φ (gradient steps) |
|---|---|
| the shared daily shapes; each site's level and deviation | the style loadings (softmax mix) |
| the holiday, tent, harmonic, and lighting coefficients | the heating threshold per site, and its population mean |
| the heating slope per site and band (≥ 0) | the thermal-inertia mix (softmax) |
| the capacity increments per site and season-year (≥ 0) | the interaction network's hidden layers |
| the network's last-layer weights; the population means of these families | |

Pooled drift is not built in `dp_pooled`, so no arm has drift.

- **`PooledDemandModel.linear_design(batch, phi)`** returns X(φ) and an offset with output
  X(φ)θ + offset(φ). Calendar, lighting, and network columns are centred per site-year with the
  training rows' means; the network's bias column, exactly zero after centring, is dropped. The Gram
  matrix is built site by site in chunks (shared columns plus about 720 per site); no dense X of all
  rows is formed (0.4 million rows by 3,800 columns would be about 12 GB).
- **Partial pooling as in N0:** the variables are each site's θ_s and the population mean μ, with
  the prior as a quadratic penalty on (θ_s − μ), so the bounds are boxes on θ_s, μ is free, and the
  problem is one convex quadratic. First and growth capacity increments keep separate families.
- **Priors (no retuning).** P-map's MAP population scales are fixed hyperparameters set before the
  softplus (level 5, threshold 2, slope 1, growth 1, first increment 8, deviation 1.0 from
  `tuned.json`). VP multiplies each by the softplus slope at P-map's population mean (0.985 for the
  slope, 0.946 for growth, 1.00 for the first increment), so no scale moves over 6%. K = 2.
- **The noise scale σ** is fixed within a round: round 1 from G-2y's training residuals at the same
  cut (never the stored scored-year G-2y), round 2 from round 1's residuals, then stop.
- **MVA rows** enter with Q = 0 (tan φ = q0 = 0), the series rule for Parts 2 to 4 and the only
  case where the operator is linear. P-map fitted tan φ from −0.006 to −1.21 and q0 from −0.95 to
  +10.17 points, which changes what MVA rows teach the shared terms; P-map-same isolates this.
- **The outer loop:** L-BFGS on φ (strong Wolfe, history 20, at most 300 evaluations including the
  line search's, stop below 1e-7 relative fall over 10 iterations). Each evaluation builds the Gram
  matrix, solves θ*(φ), evaluates the loss at (φ, θ*) with θ* constant, and back-propagates to φ,
  recomputing the centring means inside the autograd graph (as `_centred` does in training mode).
  Every linear column has a ridge or prior and the box does not depend on φ, so θ* is unique and
  the projected objective is continuously differentiable, active-set changes included (Danskin);
  the gradient is exact to the solver's tolerance. Only the second derivative jumps there, so
  L-BFGS skips curvature pairs with sᵀy ≤ 0. Adam (1,200 steps, exact solve each step) runs only if
  L-BFGS fails its convergence test, and both objectives are then reported.
- **`convex` mode** emits `dp_gam.design_matrix`'s columns for one unpooled site with no φ, so test
  1 checks the plumbing exactly; test 2 checks the DP design.
- **`objective="vp"`** gives `fit_pooled`'s Adam the same loss (Q = 0 frozen, natural-scale priors
  on the softplus outputs, σ fixed), for P-map-same. **One objective function** evaluates the
  deterministic MAP objective for any parameter set, VP's or Adam's.
- **Batching:** copies (onboarding sites, multipliers, starts) share one float64 GPU solve call.

**Tests:** (1) in `convex` mode VP equals `fit_gam` at every decision site, all years, Part 2's
tuned spec, within 1e-8 relative on the CPU and 1e-6 on the GPU; (2) for random φ and θ,
X(φ)θ + offset(φ) equals `PooledDemandModel.forward` within 1e-6 points, network on and off,
chunk by chunk; (3) with φ fixed, θ* matches a dense float64 bounded solve of a 3-site synthetic
batch within 1e-8, and no 5,000-step Adam run on θ reaches a lower objective; (4) the envelope
gradient matches central finite differences within 1e-4 relative at a φ with an active bound and
at one without, and to O(h) where the step crosses an active-set change (an increment or slope
leaving 0), each φ jittered off the hinge's kinks; (5) leakage asserts as in Parts 2 to 4: no
target, feature, capacity, centring mean, σ, or population mean from a row after the cut, an
onboarded site's rows inside its window, nothing scored on or after 2026-07-01.

## Sites, years, and rows

| Year | Cut (training ends) | Role |
|---|---|---|
| 2023-07-01 to 2024-06-30 | 2023-07-01 | replica rows for the replicate year's spread only |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate: every planned contrast; the capacity profile |
| 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict |

- **Pool, rows, and predictions as Part 3:** the MW Primaries D5, D9, D10, D11, D14, D16, and D18,
  plus the 8 MVA Primaries on dark rows, `default_rows == True`, 2 years before each cut, through
  `build_pooled_batch`; the 66 hand-filtered transfers masked ±24 hours (hindsight labels, a mild
  leak shared by every arm). Decision sites D5, D10, D11, D14, and D16 (4 of 5 needed); D9 beside
  them; D18 descriptive.
- **The replicate year is Part 3's tuning year,** in-sample for K and the deviation scale. P-map
  and S are refitted by gradient at the 2024-07-01 cut, since experiment 105 stored the scored year
  only.
- **Spread:** `dp_gam`'s centred replica spread (the arm refitted 12 months earlier by the same
  method); `dp_onboarding.cross_fitted_spread` for onboarding and short history. The replicate
  year's replicas are fits at the 2023-07-01 cut, trained from 2021-07-01 (the lockdown removes
  2021-07-01 to 07-19; D9 and D18 have no rows). Part 4 trained from 2019-09 on the same cache, and
  the scripts assert coverage from 2021-07-01.
- **Robustness rows, every contrast:** November to March against April to October; without D16;
  without D14; D10's unlabelled switching windows dropped (as Part 4); the all-rows score.

## Arms

**Every arm is screening tier, one frozen fit per cut.** VP-P and VP-S are fitted from two starts
(seed 0; seed 1 with the thresholds at 17 °C); the lower-objective start is the arm, and the score
spread between starts is reported beside every contrast that uses them.

| Arm | Model and fit | Role |
|---|---|---|
| VP-P | Part 3's P configuration (K = 2, network (64, 64), fitted capacity), MAP, by VP | decision arm |
| VP-S | the same model, one MW site at a time, by VP | per-site comparator |
| P-map-same | experiment 105's Adam fit (code, steps, schedule) on VP's objective, σ at VP's final round | V1's comparator |
| P-map, S, P | experiment 105's gradient fits (P-map and S refitted at the replicate cut) | references |
| P-map-polish | P-map-same's φ frozen, θ solved exactly | splits V1 (exploratory) |
| VP-P-nonet, VP-P-pin | VP-P without the network; with capacity pinned by Part 4's rule | exploratory |
| VP-onb4w, VP-S-4w | 4 weeks (2025-06-03 to 2025-06-30); shared modules frozen from a VP fit leaving the onboarded site out (B = 6, as Part 3); per-site VP | C2v |
| VP-short3m, VP-S-3m | 3 months (2025-04-01 to 2025-06-30): joint VP refit with the short site inside; per-site VP | C3v |
| G-2y | per-site convex model, 2-year window, no drift | convex anchor |

**VP-S's prior in full:** in a single-site MAP fit each learned population mean equals the site's
own value, so the active priors are the deviation and loadings centred on zero at scale 1.0, the
bands' slopes pulled to their common mean, and the threshold's 15.5 ± 2 °C hyperprior. The short
windows move back one year at the replicate cut.

**The equivalence ladder** (all years, per site, capacity pinned to G's path): **L0** is VP in
`convex` mode, G-all by construction, scored to confirm 3.072 points. **L1** is `s_equiv_all`'s
exact objective (its MAP prior with deviation scale 100, its penalties, its learned σ; none of VP's
priors or σ rounds), fitted by VP; its prior acts on the slopes before the softplus, so the slopes
and σ join φ, and L1 starts from a threshold grid of 12 to 20 °C every 1 °C, keeping the lowest
objective. **L2** is L1 with G's heating basis (one slope per smoothing and band at 15.5 °C). The
+0.25 is reported as three terms summing to it, per site: `s_equiv_all` minus L1 (the optimiser),
L1 minus L2 (the heating form), and L2 minus G-all (the remaining specification), with D10 also
shown without its unlabelled switching windows.

## Known-answer gate (blocking, before any scored or replicate fit)

**A failed blocking test stops scoring until it is fixed;** if it cannot be fixed inside the
experiment, the verdict is `inconclusive`. Every synthetic fit has the network on, because with it
off a test cannot see the route by which the network imitates solar.

1. **Module tests 1 to 5 pass.**
2. **A synthetic fleet:** eight sites on the real calendar and weather tensors of the MW Primaries,
   2 years each. The truth: K = 2 shapes with known loadings, a 14 °C threshold, a known inertia
   mix, non-zero shared holiday and harmonic terms, capacity paths with a 3 to 8 point growth
   increment in the last season-year, no network term, and residuals block-bootstrapped by week
   from G's training residuals at a different site. Pass if every threshold is within 1 °C, every
   slope within 15%, every season-year capacity within 10%, every growth increment within 1.5
   points, and the network's centred summer-midday output under 0.1 points.
3. **The outer loop converges** on the synthetic fleet from three starts (seed 0; seed 1; thresholds
   at 17 °C with uniform loadings): the projected gradient's norm falls below 1e-4 relative, the
   objectives agree within 1 nat per 10,000 rows, and each threshold agrees with a 0.1 °C grid
   profile within 0.2 °C.
4. **Injected farms** (experiment 39's harness), one farm at a time at D10, D11, D14, and D16,
   batched, night rows kept (`farm_norm` not null). Pass if the median recovered-to-injected ratio
   is in [0.9, 1.1], which Part 3's 0.78 would fail, and the other sites' capacities move under 5%.
   The shared modules' midday movement is reported against a second-seed base fit's. Only ratios
   appear: the farms are metered generators.

**Real-data thresholds are reported, never passed or failed.** P-map's include 48.3 °C and 25 to
27 °C around a population mean of 19.9 °C; above a site's highest temperature the hinge is always
on, the deviation absorbs a shift, and only the prior sets the threshold. Each site's threshold
profile is shown.

## The capacity profile re-run (pre-registered, labelled pass or fail)

**Does VP find the profile minimum from any start?** VP solves the capacity exactly, so a start
can matter only through φ. At Part 3's diagnostic configuration (K = 4, 2024-07-01 cut) and at
VP-P's (K = 2, 2025-07-01 cut), the six multipliers are refitted by VP as one batch, and free fits
run from three starts (default; seed 1 with thresholds at 17 °C; P-map's φ at that cut). **Pass,
every line:** the starts' capacities agree within 3 points at every MW site (the series rule); each
free objective is no higher than the lowest profile point plus 1 nat per 10,000 rows (about 40
nats); each free capacity lies where the profile is within that tolerance of its minimum. Every
loss goes through the one objective function; each site's VP capacity is shown beside G's, P's,
and the vintage's season-year mean (never its last week).

## Planned contrasts

Four planned contrasts, Bonferroni-corrected (98.75% intervals from `studies.bootstrap`, whole
calendar months of stored predictions resampled, paired across arms), each with site counts, every
arm's absolute score, the mean-forecast error, the replicate year's estimate and count, and the
robustness rows:

- **V1, the fitting method on one objective:** VP-P minus P-map-same.
- **C1v, pooling on all rows under VP:** VP-P minus VP-S, non-inferiority with a margin of 0.10
  points (Part 3 ruling 2), the interval's upper end printed against +0.10.
- **C2v, onboarding:** VP-onb4w minus VP-S-4w. **C3v, short history:** VP-short3m minus VP-S-3m.

**Exploratory at 95%:** VP-P minus P-map (method plus objective), P, S, and G-2y; VP-S minus S;
P-map-polish minus P-map-same; VP-P-nonet and VP-P-pin minus VP-P; the ladder; VP-onb4w minus
P-onb4w; contrasts against experiment 109's N in a follow-up note once stored. **Descriptive:** D9,
D18, coastal D5 and D9, bank holidays, the Christmas run, and days below 0 °C.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **Objective:** on VP's objective, VP-P sits over 2,000 nats below P-map-same at the same cut.
- **V1:** VP-P minus P-map-same is −0.15 to −0.45, 5 of 5 sites better, in both years;
  P-map-polish recovers over half of it. **VP-S minus S:** −0.05 to −0.30.
- **C1v:** within −0.10 to +0.10 in both years; dropping D16 moves it under 0.10. **Starts:** the
  score spread between VP-P's two starts is under 0.05.
- **Ladder:** the optimiser term is over half of the +0.25; D10's share in L1 is under +0.30.
- **C2v and C3v:** −0.5 to −2.0 and −0.2 to −1.0, 5 of 5 sites better each.
- **Capacity profile:** passes at both configurations; VP-P's capacities within 15% of G-2y's at
  4 of 5 decision sites, D11 included.
- **Tails:** VP-P's q01 and q99 coverage within 1.5 points of nominal, q05 and q95 within 3.
- **Run time:** each VP fit under 10 minutes; peak GPU memory under 10 GB.

## Decision rule

**The verdict reads C1v and V1 under Phase A labels with a replicate check.** C1v passes if, for
the lower-objective start, its estimate is at most +0.10, its interval does not lie wholly above
zero, and the replicate year's estimate is at most +0.10.

- **`supported`:** the gate passed; C1v passes; V1 is below zero in both years, with 4 of 5 sites
  better in the scored year; VP-S minus S is at most +0.05 in the scored year (so C1v is not bought
  with a weaker per-site fit); and the score spread between starts is at most 0.10.
- **`partly supported`:** the gate passed, and either V1 meets that bar but C1v fails, or V1 is
  below zero at 4 of 5 sites in the scored year but not in the replicate.
- **`refuted`:** V1 is at or above zero in the scored year and C1v's interval lies wholly above
  zero: the optimiser was not the cause.
- **`inconclusive`:** the start spread exceeds 0.10, a gate test cannot be fixed, or anything else.

**C2v and C3v are labelled in the bullets, with the capacity profile check beside them.**
**Handoffs (decided 2026-10-19), tied to different contrasts:** `fitter="vp"` becomes the
default for every pooled DP arm if the gate passes and V1 is below zero in both years at 4 of 5
sites, whatever C1v shows. If C1v is not refuted (its interval does not lie wholly above +0.10),
the pooled neural DP continues and N9 compares VP-P against N; otherwise pooling moves to N0's
convex route.

## Audit before merge: the seven triggers

**The "Checks" bullets answer each; trigger 7 fires by construction, so an Opus audit is planned.**
(1) Control: any gate line, L0 not scoring 3.072, or P-map-polish with a higher VP objective than
P-map-same. (2) Contradiction: C1v passing while VP-P loses to G-2y by over 0.2, or the objective
falling while the score worsens at most sites. (3) Mean forecast: each contrast's error of the
mean; median minus mean per site and arm. (4) Leave one site out for V1 and C1v to C3v, naming D10,
D14, and D16. (5) Nested arms: VP-P-nonet and VP-P-pin move little; VP-S does not lose to G-2y.
(6) Interval width against Part 3's C1. (7) N9 and Parts 5 and 6's pooled arms depend on it.

## Run order, compute, and time

**Order:** (1) the module PR, about 5 hours of agent time, measuring the Gram build first; (2) the
gate, about 20 minutes; (3) the capacity profile re-run; (4) VP-P and VP-S (two starts each),
P-map-same, and the gradient refits at the 2024-07-01 and 2025-07-01 cuts, plus every arm's
2023-07-01 replica; (5) the onboarding and short-history batches (B = 6) and replicas; (6) the
ladder; (7) report, cards, and write-up (about 3 hours), the numbers review, audit, prose sweep.

**Compute, re-estimated.** The RTX A6000's float64 rate is about 1/32 of float32, about 1.2
TFLOPS. Site-by-site blocks give about 4e11 flops per Gram build (about 0.4 s; dense would take
about 5 s), and each pivoting iteration adds a factorisation of about 2e10 flops. An evaluation
should take about 1 s, so a 300-evaluation fit up to 5 minutes. About 35 VP fits over three cuts
and about 8 Adam fits of 10 minutes come to about 4 hours of GPU, over the series' 2-hour target:
**the experiment asks for the GPU time before step 4.** One Gram matrix is about 115 MB, a
batch of 6 under 1 GB. **If the budget is cut, drop in this order:** VP-P-nonet, VP-P-pin, L2, the
K = 4 profile, then the exploratory arms' 2023-07-01 replicas. **Data paths:** as experiment 105's
scripts, plus its `predictions/` and `work/`, and experiment 104's `g_all_centred` and `tuned.json`.

## What the write-up must contain (beyond the series brief's standing rules)

The take-home says plainly whether Part 3's pooling loss came from the fitting method, how much an
exact fit gains for the pooled and the per-site DP, whether the pooled capacity lands in one place
from any start, and what that means for N9. Gloss "linear coefficients", "variable projection"
(solve exactly what can be solved exactly, and step the rest), and "pooled".

- **Figures:** the planned contrasts with intervals, counts, replicate year, and start spread;
  absolute scores per site; each fit's objective; the capacity profile with the free starts marked;
  the ladder's terms; the threshold profiles; the gate; a winter and a summer week per decision site
  (VP-P, P-map-same, VP-S); the tails (coverage and pinball at q01, q05, q95, q99, every arm).
- **Substation cards** (D-labels only, provisional, VP-P at the 2025-07-01 cut): the series brief's
  panels, capacity as VP-P's (with its start spread), G-2y's, and the vintage's season-year mean,
  plus one cross-substation figure.
- **Caveats:** ERA5 and CAMS flatter the solar term; the masks are hindsight; MVA tan φ and q0 are
  zero, unlike P-map's; the replicate is in-sample for K and the deviation scale; the priors are
  P-map's re-expressed; MAP only (with active bounds the posterior is a truncated Gaussian, so a
  Laplace posterior needs its own design). **Privacy:** D-labels with dates only; no generator
  data; G-a, G-b, and D15 out; injected farms as ratios only.

## Decisions recorded (2026-10-19)

P-map's re-expressed prior scales, no retuning; no Laplace posterior now; the extra fit is a second
L-BFGS start, with Adam only when L-BFGS fails its convergence test; the handoffs above; do not wait
for experiment 109's N. **Open questions:** none.
