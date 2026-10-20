---
id: "00120"
slug: differentiable-physics-sh-convex-daily-shape-library
part: SH
theme: differentiable-physics-pooling-across-substations
also_relevant_to:
  - differentiable-physics-demand-calendar-and-weather
  - hierarchy-and-pooling
status: planned
size: >-
  Complex for the new module studies.dp_shape_library (its own pull request, with tests and two
  reviews); medium for the experiment scripts. Stored: private files under data/ml-experiments/
  only. Serving path: not touched. Degradation rules: not touched. More than one design is
  defensible (shape sign, weight sharing across day types, how the orthogonality penalty enters,
  the starting shapes), and the plan pre-registers one of each. Callers: this experiment's scripts,
  and later pooled experiments.
hypothesis: >-
  A convex model in which each Primary's daily profile is a non-negative mix of K = 3 or 4 shared
  daily shapes per day type, plus a penalised deviation, forecasts Primary demand with 2 years of
  history at least 0.05 points better than experiment 109's shrinkage towards one fleet-mean
  profile. That needs the library to beat the per-site profile by about 0.02 points, because the
  fleet-mean model is 0.031 worse than it; the plan predicts a tie or a small gain, so `supported`
  is a long shot. The stability of the shapes and the shape mix of each substation are descriptive.
prediction: see "Predictions" below
decision_rule: see "Decision rule" below
benchmark: >-
  N (experiment 109, deviation penalty 3,000 rows), G (per-site convex), N-full, lagless XGBoost
  (stored sets), climatology, manual_heuristic
row_set_id: dp_mw_v1
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, at the
  day-ahead lead (day D+1 from a 09:00 UTC initialisation on day D); scored years 2025-07-01 to
  2026-06-30, replicate 2024-07-01 to 2025-06-30
verdict: planned
production_status: not-for-production
take_home_message: >-
  Planned, no result yet. The experiment asks whether Primary substations differ by kind and not
  only by size, by giving each substation a mix of a few shared daily shapes instead of one
  fleet-mean profile.
date: 2026-10-20
pr: 1410
adopted_in: none
advice:
  - >-
    Not yet written; the finished experiment writes it. The plan follows experiment 109's advice to
    repeat the known-answer gate before building on N and to rebuild partial-history capacity, and
    departs from it by not testing onboarding.
code: ml-experiments/00120-differentiable-physics-sh-convex-daily-shape-library/
---

# Differentiable physics, SH: a convex library of shared daily shapes

**This is the plan, committed before any model is fitted.** It records the question, the arms, the
known-answer gate, the contrasts, the predictions, and the decision rule, so that the commit order
shows they came first. The finished write-up replaces this text. One Sonnet reviewer read the plan
against the briefs and experiments 106, 109, 111, and 113; all 14 of its findings are applied.

## What the experiment asks, in plain words

**The question is whether the Primary substations differ by kind, not only by size.** A Primary
substation is the last electricity substation before the feeders that serve homes and businesses.
Its **daily profile** is the average shape of its demand through the day, separately for weekdays,
Saturdays, Sundays, and bank holidays. [Experiment
109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md) pulls every
substation's profile towards one fleet-mean profile, with a **penalty** whose size is counted in
rows of evidence. If the substations fall into a few kinds, one fleet mean wastes that penalty on
the difference between kinds. A **library** of K shared shapes lets each substation pull towards its
own mix of those shapes instead. [Experiment
105](../00105-differentiable-physics-3-pooled-and-per-site-demand-model/result.md) found that a
neural pooled model put most substations wholly on one of two shapes.

**The method is a constrained matrix factorisation fitted by alternating convex steps.** Writing the
substations' profiles as a table (one row per substation, one column per half-hour and day type),
the library factorises that table into non-negative weights times shared shapes. This is the setting
of [non-negative matrix factorisation](https://doi.org/10.1038/44565) fitted by [alternating
non-negativity-constrained least squares](https://doi.org/10.1137/07069239X), with two differences:
the shapes may be negative, because a non-negative shape is redundant with the free mean level of
each substation, and the weights sit inside the demand model of experiment 109 (holidays, heating,
lighting, and solar), which each step re-solves exactly. A mix of non-negative weights over shared
extreme profiles also resembles [archetypal
analysis](https://doi.org/10.1080/00401706.1994.10485840), and grouping electrical load patterns
into a few kinds is an established tool in [load-profile
clustering](https://doi.org/10.1016/j.energy.2011.12.031).

## What the score can show

**The headroom for the score is small.** With 2 years of history N scored 0.03 points worse than G
(99% interval +0.01 to +0.06), so a library that only closes that gap gains at most 0.03 points over
N. The decision rule lets a score tie be reported as such, and when C1 is `not supported` or `small
gain` the take-home leads with the stability and shape-mix readouts. Those readouts are weakly
identified (experiments 105 and 113), so they carry a null reference and intervals.

**How SH differs from experiment 105's two shared shapes.** Experiment 105 fitted shapes and
loadings by gradient descent inside a neural model with a network and a free PV capacity; it chose K
= 2 on the tuning year (3.197 against 3.255 for K = 4 and 3.207 for K = 6), with loadings near 0 or
1 that it called not identified one by one. It lost to per-site models at full history by 0.385
points, mostly from the neural fit (experiment 109). SH alternates exact convex solves of `dp_gam`
with pinned capacity, constrains weights to be non-negative, and adds an orthogonality penalty. It
is convex in each step but not jointly, so seeds measure the start's effect. SH compares with N;
105's scores are references only.

## The model

**Net power keeps `dp_gam`'s form; only the profile family changes.** The profile of substation `s`
(4 day types × 48 half-hours, 192 columns in `dp_gam`) is

    profile_s = m_s + sum_k w[s,k] * S_k + delta_s

- **Shapes `S_k`**, k = 1 to K, are shared 4 × 48 arrays (weekday, Saturday, Sunday, bank holiday),
  centred per hour band as in `dp_gam_pooled`'s `C_f` and scaled to unit norm over all 192 entries.
  Shapes are signed, because a non-negative shape would be redundant with the free band mean `m_s`.
- **Mixing weights `w[s,k] >= 0`**: one per substation and shape, shared across day types
  (per-day-type weights are the exploratory arm SH-wd). They are in points of effective capacity,
  carry the substation's size, and are not constrained to sum to one.
- **Band mean `m_s`** is free per substation, as in experiment 109, so size never pools.
- **Deviation `delta_s`** is centred like the shapes and ridge-penalised at `lambda_dev` rows of
  evidence on the same `g_f` scale as `dp_gam_pooled`, so it reads like experiment 109's 3,000.
- **Every other family** (holidays, Christmas tent, harmonics, heating, lighting) is pooled exactly
  as in N, so SH minus N isolates the profile. Capacity is pinned as in experiment 109; no drift, no
  network; 2-year window; lockdown and the 66 hand-filtered transfers out.

**Alternating steps** (module `studies.dp_shape_library`):

1. **Weights step.** Shapes fixed; one box-constrained quadratic per substation (`w >= 0`) alongside
   every other coefficient.
2. **Shapes step.** Weights fixed; shapes update one at a time (Gauss–Seidel), each with one joint
   ridge quadratic over all substations.
3. **Renormalise** each shape to unit norm, moving the scale into the weights. A sweep ends with a
   weights step. Repeat until a sweep lowers the objective by under 1e-6 per training row, or 50
   sweeps.

**Orthogonality penalty against duplicate shapes.** Two drifting-together shapes waste a library
slot and split their weights arbitrarily. The pairwise term `gamma * sum_{j<k} cos(S_j, S_k)^2` is
not convex, so shapes update one at a time (Gauss–Seidel) with penalty `gamma * kappa * || U_{-k}^T
S_k ||^2`, where `U_{-k}` is an orthonormal basis of the other K − 1 fixed shapes, `S_k` is the
unit-norm shape, and `kappa` is the data term's curvature along a unit shape (see "Implementation
choices"): a convex quadratic in `S_k`. The shape ridge (1e-3, fixed, not tuned) and this penalty
are both defined on unit-norm shapes, and the objective is evaluated after renormalising, so step 3
cannot change it. The penalty is a surrogate for the sum of squared pairwise cosines (equal when the
other shapes are orthogonal). Monotonic decrease is claimed per convex step, not across step 3.
`gamma` is tuned on the tuning year.

**Initialisation and seeds.** Seed 1: a rank-K singular value decomposition of the fitted G
profiles, weights projected to be non-negative. Seeds 2 to 5: random Dirichlet weights scaled to
each profile's size. The headline fit is the seed with the lowest **training** objective, and C1's
estimate is that seed's; the other seeds' scores and the pooled-seed interval are the spread.

**Pool: experiment 109's `POOL_SITES` (D5, D9, D10, D11, D14, D16, D18); no MVA dark-row
Primaries.** A site enters a cut with at least 1,344 usable rows and is partial under 28,000 rows in
its 2-year window. Expected: 5 sites at tuning (unverified), 6 at replicate (D9 partial), 7 at
scored (D18 partial). The script prints the pool at each cut and each fit's effective parameters; K
= 4 is near the pool size, so a K near it is nearly a per-site profile.

## Years, rows, and comparators

**Three years, each with its own frozen cut (Part 4, experiment 109):** | Year | Cut (training ends)
| Role | |---|---|---| | 2023-07-01 to 2024-06-30 | 2023-07-01 | tuning: K, `lambda_dev`, `gamma` |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate: every decision contrast, out of sample for SH's
choices | | 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict |

- **Sites.** The pool above. Decision sites D5, D10, D11, D14, and D16 (4 of 5 must improve). D9,
  D18 (G-c in anything published), and the rest are reported beside them. No onboarding or
  short-history arm runs.
- **Tier.** Screening: one frozen fit per cut and arm, no refits. Intervals resample whole calendar
  months of stored predictions (`studies.bootstrap`), paired across arms, seeds pooled.
- **Window.** 2 years; SH-all (all years) is exploratory.
- **Spread.** As `pooled_replica_residuals`: the replica fit 12 months before the cut (the replicate
  fit serves the scored cut, the tuning fit the replicate cut; one extra fit at 2022-07-01). Every
  arm uses the same recipe, so the spread does not decide a contrast.
- **Predictions written:** the standard scoring columns plus `arm`, `seed`, `cut`.

## Arms

| Arm | Profile | Notes |
|---|---|---|
| G, N | per-site free profile; fleet mean + deviation at 3,000 rows | stored from experiment 109, reproduced |
| N-full | complete pooling, free band mean | run at 2 years here |
| SH | K shapes + deviation, `gamma` > 0 | decision arm, K from the tuning year |
| SH-K1 | K = 1 | nests N when the weight is free: is more than one shape needed |
| SH-g0 | `gamma` = 0 | does the orthogonality penalty matter |
| SH-nodev | `lambda_dev` very large | a pure library: is the deviation needed |
| SH-wd, SH-all | weights per day type; all years | exploratory |

## Tuning (on 2023-24 only, by quantile score, at the 2023-07-01 cut)

- **K** in {2, 3, 4, 6}; the scope is 3 to 4, and K = 2 and 6 run on the tuning year only to show
  what K the data choose. The scored arm uses the better of K = 3 and 4; scores within 0.023 points
  tie, and a tie goes to the smaller K.
- **`lambda_dev`** in {300, 1,000, 3,000, 10,000} and **`gamma`** (multiples of `kappa`) in {0, 0.1,
  1, 10}, 16 pairs at K = 3 and 4 (32 configurations), then K = 2 and 6 at the chosen pair
  (exploratory, never scored later, counted in the compute); three seeds each.
- **Fairness.** N keeps 3,000, so the tuning year favours SH's extra knobs. The 2023-07-01 cut reads
  a 5-week-late vintage, a leak that reaches tuning choices only.

## Known-answer gate (blocking, after tuning, before any replicate-year or scored-year fit)

**A failed line stops scoring and caps the verdict at `inconclusive`.** The gate is a power check:
could SH, at this pool size and noise, recover a library that exists? Experiment 109's gate passed
on one of five reruns, so limits are fixed and a post hoc redefinition cannot lift the verdict
above `inconclusive`.

**Synthetic fleet.** Real calendar, weather, and pinned capacities from the tuning cut; the pool's
size; K* = 3 shapes from random sums of Gaussian bumps, half the fleets with truth shapes at cosine
0.3 to 0.6 to each other; a deviation of 10% of profile energy. **Weights:** the sparse prior
(Dirichlet 0.3, one site near each vertex, as experiment 105's mixes) must pass; the dense prior
(Dirichlet 1) is reported as identification-limited, since 5 to 7 sites with non-negative weights
admit a larger cone. **Noise:** both training and held-out noise are block-resampled from G's
residuals on rows before the cut; no tuning-year residual enters. **Pass rule:** 40 sparse-prior
fleets (two sets pooled); a line passes at 30 or more. At a true per-line rate of 0.9 that is above
0.99 per line and about 0.9 for all nine; per-line rates and Wilson 90% intervals are printed.

1. **Shapes:** with K = K*, each shape matched on **signed** cosine has cosine at least 0.9.
2. **Weights:** median relative error of matched weights at most 0.15.
3. **No duplicates:** with K = K* + 1 and tuned `gamma`, the largest pairwise |cosine| is under 0.5
   (|cosine| is right here); with `gamma` = 0 it is reported only.
4. **Deviation:** one site-specific evening bump absent from every shape leaves shape cosines at
   least 0.9 and `delta_s` holds at least 60% of the bump's energy.
5. **Null fleet:** independent random profiles; SH minus G on held-out rows is at least 0, and the
   signed cosine between two seeds' shapes is under 0.6. This is the null reference for readout 2.
6. **K choice:** the tuning rule picks K = 3 or 4 (30 of 40 needed; Wilson interval reported).
7. **Local optima:** of 5 starts, at least 4 reach a joint objective within 1% of the best.
8. **Module tests** pass.
9. **Score power:** on library fleets whose truth is sized so the oracle gain over N is 0.05
   points, SH minus N on held-out rows is at most −0.05. A `not supported` C1 can then be read as
   no gain and not as no power; if line 9 fails the verdict is capped at `inconclusive`.

## Module `studies.dp_shape_library` (own PR off `main`, tests under 5 minutes each)

It reuses `dp_gam` design matrices and `dp_gam_pooled`'s centring, scales, spreads, and
`with_effective_capacity`; no dependency. **Tests:** (1) K = 1, `w` fixed to 1, shape set to the
fleet mean reproduces `fit_pooled`; (2) the objective never rises across a convex step on unit-norm
shapes; (3) the weights step matches non-negative least squares to 1e-6; (4) the penalty is zero for
orthogonal shapes and rises for parallel ones; (5) changing rows after the cut leaves the fit
unchanged; (6) renormalising leaves predictions unchanged; (7) fits are seed-deterministic.

## Planned contrasts

**Three contrasts (98.33% `studies.bootstrap` intervals, month-resampled, paired, seeds pooled), at
day D+1, each with site counts, absolute scores, mean-forecast error, and the replicate estimate:**

- **C1, the headline:** SH minus N on all scored rows of the five decision sites. Superiority with a
  smallest gain worth porting of 0.05 points.
- **C2, against the per-site profile:** SH minus G. Non-inferiority, margin 0.05 points (about
  twice the noise reference below).
- **C3, is the library needed:** SH minus SH-K1. Descriptive unless its interval excludes zero.

**Noise reference.** Part 4's C4 interval between related convex arms was about ±0.023 points at
97.5%; the 0.05 margin is about twice it.

**Exploratory at 95%:** SH minus SH-g0, SH-nodev, N-full, and lagless XGBoost; SH-wd and SH-all
against SH; per-site gains; November–March against April–October; without D16; D10's two unlabelled
switching windows dropped (2025-11-17 to 2025-12-14, 2024-11-08 to 2024-12-01); the all-rows score.
**Descriptive:** D9, D18 (G-c), bank holidays, the Christmas run, days below 0 °C.

**Tails (every arm, all and by season):** coverage and pinball at q01, q05, q95, and q99. No
curtailment-cost metric: the series has no generator forecast.

## Readouts beyond the score

**Readout 1 is the score (C1 to C3). Readout 2, stability.** Match shapes between two fits with the
Hungarian algorithm on **signed** cosine (a sign flip is not a match; `S` and `−S` differ under `w
>= 0`), then report the cosine of each matched pair and the correlation of the matched weights.

- **Across seeds:** the 5 initialisations at each cut. Stable if the median matched cosine is at
  least 0.9, no matched cosine is under 0.7, and the median exceeds the null fleet's by at least
  0.3; unstable if the median is under 0.6.
- **Across years:** the fits at the three cuts. Consecutive cuts' windows overlap by a year, so the
  reading that counts is the disjoint pair (2023-07-01 against 2025-07-01). Stable if the median
  matched cosine is at least 0.8, partly stable between 0.6 and 0.8.
- **Null reference:** gate line 5's cosines (null fleet) are printed beside every real reading.

**Readout 3, shape mix.** Per substation: the weights `w[s,k]` as shares of the library's energy,
the deviation's size as a share of the profile, and whether the substation sits wholly on one shape
(a share above 0.9). **These are weakly identified** (experiment 105: loadings not identified one by
one; experiment 113: heating slopes about 38% wide), so every share carries a month-block interval
and the null-fleet value, heating slopes appear only with intervals, and the cards say "weights and
shapes are identified only up to the rotation the data fix". The shares-against-heating-slope
comparison is descriptive. If a shape takes more than 30% of its energy from D18, that shape is
plotted on the G-c figure only.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **K:** the tuning year chooses K = 3 or 4, within 0.023 points of K = 2 and K = 6.
- **C1:** −0.01 to −0.04 in the scored year, 3 or 4 of 5 sites better (a tie or small gain).
  **C2:** −0.02 to +0.03. **C3:** −0.01 to −0.04.
- **Stability:** across seeds, median matched cosine 0.9 or more; across the disjoint years 0.7 to
  0.9; the null fleet under 0.6.
- **Mix:** most substations put 70% or more of their weight on one shape, deviation under 20% of the
  profile; SH-g0 has a pair of shapes with |cosine| above 0.7 at K = 4 and SH does not.
- **Tails:** q95 and q99 coverage within 3 points of N's.

## Decision rule

**The verdict is C1's label, after the gate, with a replicate check.**

- **`supported`:** the gate passes; C1's estimate is at most −0.05 and its 98.33% interval lies
  wholly below zero; at least 4 of 5 decision sites improve; the replicate year's estimate is also
  below zero; the across-seed stability is not "unstable".
- **`small gain`:** the gate passes and C1's interval lies wholly below zero, but the estimate is
  above −0.05.
- **`partly supported`:** the gate passes, C1's estimate is at most −0.05, and one of the other
  `supported` conditions fails (the interval touches zero, fewer than 4 sites improve, the replicate
  reverses, or the shapes are unstable across seeds).
- **`not supported`:** the gate (line 9 included) passes and the interval includes zero and lies
  within ±0.05. This label departs from the series brief, which reserves it for Phase B.
- **`refuted`:** the gate passes and C1's lower limit is above +0.05.
- **`inconclusive`:** a failed gate line, or anything else (including an interval above zero but
  not beyond +0.05).

**Smallest gain worth porting: 0.05 points against N.** C2 and C3 are labelled in the bullets, not
in `verdict`; if C2 fails (SH loses to G by more than 0.05) the take-home says so first.

## Substation cards (provisional)

**Labelled "Provisional: fitted to past weather on a screening run".** Each card also has a "how it
does in a cold snap" panel (rule under "Implementation choices"). One card per decision substation
(D-labels, with dates) from the 2025-07-01 fit: shape mix as a stacked bar beside the deviation's
size; profile per day type (library and deviation drawn separately); pinned capacity; heating
response; holiday, Christmas, and lighting effects. **Example forecasts** on 4 weeks fixed by
calendar rule before any score is read: the first Monday on or after 15 July 2025, 15 October 2025,
15 January 2026, and 15 April 2026, each with median and 80% and 90% bands against observed, for SH
and N. One cross-substation figure compares shares, deviation size, capacity, and heating slope.

**D15 and D18 are generator data: label them G-c alone.** D15 is outside the pool; D18 is in it at
the scored cut and gets no shape of its own in any D-label figure. Any figure for D18, or the D15 +
D18 sum, uses G-c, normalised units, day counts and no dates, `aria=False`, and its own figure,
never beside a D-label, name, or identifier. The page may name interconnected D-label pairs such as
D5–D16, but no substation names, no other topology, no generator's public name with its series, and
no guess at D15's generator.

## Audit before merge: the seven triggers

1. **A control fails:** any gate line, or G and N not reproduced within 0.01.
2. **A contradiction:** SH beating G by more than 0.1; a mix against experiment 105's clusters
   without a tested explanation.
3. **Quantile score against mean error:** C1 or C2 opposite in sign to the mean-error contrast.
4. **Leave one out:** dropping any decision site (D16 first) flips C1 or C2.
5. **Nested arms:** SH-K1, SH-nodev, or SH-g0 beating SH by more than its paired half-width.
6. **Interval width:** C1 wider than 0.09 or narrower than 0.022 (twice and half experiment 109's
   C1 width of 0.044).
7. **Dependence:** cards or a later pooled experiment would rest on the shape mix; a positive
   verdict or stable shapes triggers an audit.

"Checks" bullets: median minus mean forecast per site and arm; mean-forecast error beside each
contrast; leave-one-out; which triggers fired.

## Run order and compute

**Measured: one SH fit of K = 3 takes about 30 seconds** (five Primaries, tuning cut, 6 threads, 50
sweeps, including the per-site fit that supplies the starting shapes). The estimate for the whole
experiment on 6 cores is therefore **about 3 hours**, against the briefs' 4 to 20 hours.

| Job | Fits | Estimate at 6 cores |
|---|---|---|
| Tuning: 32 configurations (K = 3 and 4, 16 pairs of `lambda_dev` and `gamma`) × 3 seeds, then K = 2 and 6 at the chosen pair × 3 seeds | 102 | about 1 hour |
| Known-answer gate: about 15 fits per fleet × 40 fleets | about 600 | about 1 to 1.5 hours |
| Main fits: 2 cuts × (SH, SH-K1, SH-g0, SH-nodev, SH-wd) × 5 seeds, SH-all, N, G, N-full, 3 replica fits, and the replica fit at 2022-07-01 | about 70 | about 0.5 hour |

**Cut order if the total passes 8 hours:** the gate to one set of 20 fleets (threshold 15), then
SH-all, SH-wd, seeds 4 and 5, and the K = 2 and 6 fits. The gate, C1 to C3, and the cards stay.

**Review order (maintainer ruling).** One Sonnet review of the module ran before the tuning and the
gate; one Opus review runs after the weekly quota resets. No replicate-year or scored-year fit runs,
and the module's pull request does not merge, before the Opus review lands. Tuning and the gate may
run on the Sonnet-reviewed branch, and are redone if the Opus review finds a leak.

## Write-up

Title "Differentiable physics, SH: a convex library of shared daily shapes". Short, with "Checks".
Caveats: screening tier on one year; the late vintage at the tuning cut and the replica fit; the
tuning year favouring SH's extra knobs; ERA5 and CAMS inputs flatter all arms equally; a biconvex
fit may sit at a local optimum; a K near the pool size is nearly a per-site profile; the spread
copies experiment 109's single replica year (q95 and q99 covered 83% and 93%), a known weak point.

## Implementation choices fixed before any fit

**These choices settle details the briefs left open. Each is recorded before a score exists.**

- **The orthogonality weight is a multiple of the shape's own data curvature.** A bare weight in
  {0, 0.1, 1, 10} is negligible beside a data term of order 10,000 to 100,000, so the grid would
  not change the fit. The module scales the penalty by `kappa`, the data term's curvature along a
  unit shape at the first weights step (`g_profile` times the sum over substations of training
  rows times the mean squared weight). The grid {0, 0.1, 1, 10} keeps its values and now means 0,
  0.1, 1, and 10 times that curvature. `kappa` is fixed for the rest of the fit.
- **The stopping rule is a drop in the objective of under 1e-6 per training row** (squared
  points), or 50 sweeps. A relative drop is useless here because the objective is of order 10^8
  and negative. The first timing fit (K = 3, five Primaries, tuning cut) hit the 50-sweep cap with
  a drop of 1.7 per sweep; letting it run to 94 sweeps lowered the objective by a further 27, about
  0.00015 squared points per row. The cap stays at 50 and the table of fits reports how many
  converged.
- **Starting shapes.** Seed 1 is the singular value decomposition of the per-site profiles, and
  seeds 2 to 5 are least-squares shapes for random Dirichlet weights. In the module's unit tests
  the singular value start reached the same objective as the Dirichlet start (within 0.00002
  relative) but a rotated cone of shapes (cosines 0.85 and 0.98 against the truth, against 0.98 and
  0.98), which is the identification limit the gate's sparse and dense priors are built to
  separate.
- **The tuning spread is each configuration's own.** Every tuning fit builds its spread from its own
  replica fit at 2022-07-01, the recipe every arm uses later, so tuning costs two fits per
  configuration and seed.
- **Gate line 9 sizes the truth so the oracle gain over N is 0.10 points, and passes at SH minus N
  of −0.05 or lower.** The briefs' wording (oracle gain 0.05, pass at −0.05) could pass only for a
  fit that loses nothing to estimation, which no fit with noise achieves. The new rule asks that SH
  recovers half of a gain of twice the smallest gain worth porting. The oracle is the true
  coefficients; the gain is N's held-out score minus the oracle's, and the script rescales the
  library energy until the gain is within 0.01 of 0.10. The scores in this line use a Gaussian
  spread fitted to each arm's own training residuals, because a gate fleet has no replica year.
- **Winter is reported beside the annual score (maintainer priority, 2026-10-20).** Winter is valid
  times from November to March, with the evening peak (16:00 to 20:00 UTC, which is local time in
  winter) and cold snaps inside it. A **cold-snap day** is a day whose mean 6-hour-smoothed ERA5
  temperature at the site is below 0 °C; the rule is fixed here before any score is read, and a
  panel with fewer than 5 such site-days in the scored year says "too few days". Every winter
  contrast against N and G has a paired month-resampled interval, and a winter loss to N is a
  reason to hold SH back whatever the annual mean does; the verdict says so. The out-of-sample
  evidence holds two independent winters (2024-25 in the replicate year, 2025-26 in the scored
  year); the tuning winter (2023-24) chose the settings and does not count.
- **Documents not carried over.** The briefs name `docs/ml-experiments/research-plan.md`, which
  the documentation restructure deleted; the theme's `plan.md` row is the only plan entry.

## References

- [Lee and Seung (1999)](https://doi.org/10.1038/44565), non-negative matrix factorisation.
- [Kim and Park (2008)](https://doi.org/10.1137/07069239X), alternating non-negativity-constrained
  least squares with active-set (principal-pivoting) solves, the solver family `dp_gam` uses.
- [Cutler and Breiman (1994)](https://doi.org/10.1080/00401706.1994.10485840), archetypal analysis.
- [Chicco (2012)](https://doi.org/10.1016/j.energy.2011.12.031), clustering methods for electrical
  load patterns.
- Earlier experiments:
  [105](../00105-differentiable-physics-3-pooled-and-per-site-demand-model/result.md),
  [106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md),
  [109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md),
  [113](../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md).
