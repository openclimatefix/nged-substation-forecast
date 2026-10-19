---
id: "00110"
slug: differentiable-physics-5-embedded-generation-and-capacity
part: 5
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - differentiable-physics-demand-calendar-and-weather
  - differentiable-physics-pooling-across-substations
  - solar-disaggregation
status: planned
size: >-
  complex for the module changes (capacity-identification options and a smooth clip in the convex
  model, the convex model fitted through the MVA meter, an unmetered generator node, all in
  packages/studies and reused by Part 6); medium for the experiment scripts. See "Size triggers".
hypothesis: >-
  The convex differentiable-physics model can represent the generation side of a Primary substation
  without making its demand forecast worse: an MVA meter through an observation layer, a flat
  unmetered generator found where one exists and not invented where none does, and an export-limited
  PV clip; and its PV capacity stays the same when the routes by which demand terms can imitate
  solar are closed.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  The convex model with pinned capacity and Part 4's recent-level filter (the base arm B, Part 4's
  stored A2, or A3 under the node); experiment 45's frozen vintages for capacity; lagless XGBoost
  with its default start, a zero start, and experiment 64's solar offset with each start, with and
  without the filter, on MW and MVA substations; climatology; manual_heuristic; lagged XGBoost as a
  descriptive reference at leads of 24.5 to 48 hours
row_set_id: >-
  dp_mw_v1 and dp_mva_dark_v1 (experiment 103), and dp_mva_sunlit_v1 (new); scored 2025-07-01 to
  2026-06-30, replicate 2024-07-01 to 2025-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site, at the
  day-ahead lead (day D+1 from a 09:00 UTC initialisation on day D); |S| in points at MVA sites
verdict: <supported, partly supported, refuted, or inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00110-differentiable-physics-5-embedded-generation-and-capacity/
---

# Differentiable physics, part 5: embedded generation, the MVA meter, and identifying PV capacity

**Status: planned.** This page is the plan, committed before any code runs. The write-up replaces it
when the experiment finishes. Parts 3 and 4 of the series (experiments 105 and 106) set the
starting point, and each choice that still depends on them is a stated branch, settled before any
scored fit and recorded in `decisions.json` in the experiment's private data folder.

## What Part 5 asks, and the bar it starts from

**Part 5 asks four questions, each answerable on its own:**

1. **Capacity identification.** Does the fitted PV capacity stay put when the routes by which
   demand terms can imitate solar are closed, and does it land near the capacity estimates that
   were available in real time (the "vintages" of experiment 45)?
2. **The MVA observation layer.** Does reading the 8 MVA Primaries through
   P = D − PV, Q = tanφ·D + q0, |S| = √(P² + Q² + ε) forecast |S| at least as well as reading |S|
   as P, and does a learned Q beat NGED's convention Q = 0?
3. **The unmetered generator node.** Does a flat, non-negative, piecewise-constant generator recover
   an injected generator, and stay off on real demand substations?
4. **The export clip.** Does the smooth clip recover an injected clip, stay open where Part 1 found
   none, and help at D5, the one decision site Part 1 flags (clipping in 4 of 7 summers)?

**The bar is the convex model, not the pooled neural model.** In Part 3's scored year the pooled
neural model lost to the per-site neural model (+0.385 points; verdict capped at `inconclusive` by a
failed known-answer gate), while the convex model with every year of history scored 2.726 and, with
Part 4's filter and pinned capacity, 2.29 (2.90 without the filter). Part 3 also found that the
pooled model's fitted capacity follows its starting value, but that its profile has a minimum near
the convex capacity: the data identify capacity to about 15%. The "near the vintage" band and the
precision floor below are read with that figure in mind. Every decision contrast gets a replicate
year, a season split, and a row without D16.

## Which model and which base arm Part 5 runs on

**The default is the convex per-site model `studies.dp_gam` with pinned capacity and Part 4's
filter.**

- **The base arm B is Part 4's A2** (all years from each site's first usable date, lockdown out, no
  drift, pinned capacity, the Kalman filter), at Part 4's three cuts, reused as stored. If Part 4's
  contrast C4 fails (drift still needed), B is A3 (drift D1 plus filter). If Part 4 hands the
  exponentially weighted mean to Part 6, B uses it.
- **The node's base is B-D1, Part 4's A3** (D1 plus filter), whatever B is. A non-negative step path
  on a base without drift absorbs any fall in demand (D10's fell from 55 to 43 points), so the node
  needs a drift term beside it to be read as a generator.
- **Every arm is scored twice:** after the base's filter (hyperparameters by marginal likelihood on
  that arm's own tuning-year residuals), which is the forecast that would be used, and without it,
  which is the structural reading. Planned contrasts are read filtered, with the unfiltered contrast
  beside each, because the filter absorbs part of what a node or clip explains. The known-answer
  gates run without the filter.
- **The pooled branch is switched off.** Part 3's gate failed, so the pooled questions go to the
  partially pooled convex model (N0) and to a variable-projection fit of the pooled model, planned
  separately. The manager may switch the branch on before the scored fits; if so, it adds the
  layer in `dp_pooled` and capacity arms on three starts, on the GPU.

## Size triggers

- **Stored data:** private files only, and one new row set (`dp_mva_sunlit_v1`).
- **Serving path:** not touched.
- **Degradation rules:** not touched.
- **Several defensible designs:** yes, and this page pre-registers one of each.
- **Callers:** Parts 5 and 6 of the series, and N9. The module pull request goes to `main` first,
  with two Opus reviews (leakage and known-answer truths first), and merges before any real score.

## Module changes (`studies.dp_gam`, `studies.dp_operator`; each test under 5 minutes)

- **(i) Orthogonal demand terms.** A quadratic penalty λ·Σ cov(term, u)², taken separately for the
  harmonics block and the heating block, over daylight rows, within each calendar quarter ×
  half-hour cell, where u is the derated unit PV shape. Covariance is linear in θ, so the fit stays
  convex. Lighting is left out: its ramps are a function of elevation and anti-correlated with u by
  construction, so the penalty would only delete lighting. Its midday movement is reported. λ is in
  units of the Gram matrix's mean diagonal over the penalty matrix's mean diagonal.
- **(ii) Demand terms blind to clear-sky output.** The day-of-year harmonics get one coefficient
  shared by all hour bands (no day-of-year × time-of-day term), and the heating hinge uses only the
  72 and 168 hour smoothed temperatures, whose day-to-day link to sunshine is weak.
- **(iii) Capacity from irradiance anomalies only.** The solar term becomes C·(u − ū) + A[month,
  half-hour], with ū u's month-by-half-hour mean over the training window and A ≥ 0 a free solar
  climatology per site under a ridge (A is aliased with the profile). A cannot grow with capacity,
  so K-iii trains on the last two season-years only. Forecasts use both terms.
- **Fitted capacity** in each K arm follows Part 2's behaviour. K-base is A2 with fitted capacity.
  The convex solver `solve_bounded_quadratic` takes no start, so the K arms need no start check.
- **The meter fit** (`fit_gam_through_meter`). The loss (|S| − √(P² + Q² + ε))² is not convex: a
  mirror basin exists wherever P can change sign, and tanφ·D(θ) is bilinear. The fit is
  Gauss-Newton: linearise √ at the current P and Q and solve each step with the existing
  `solve_bounded_quadratic`, which keeps the bounds exact. With learned Q, θ steps alternate with
  (tanφ, q0) steps. The fit is multi-start from four states (capacity 0.5, 1, and 2 times N's, and
  G-dark's demand with no solar), keeps the lowest loss, and reports the spread across starts.
  `qmode` is `zero` (tanφ = q0 = 0, NGED's convention) or `learned`. tanφ ≥ 0 (lagging load), which
  removes the (tanφ, q0) → (−tanφ, −q0) mirror. The power-factor prior is centred on 0.97 with 95%
  of its mass on 0.95 to 0.99. q0 is free in sign (cable capacitance makes it negative) with scale 2
  points. MVA capacity is fitted, because MVA sites have no vintage.
- **The unmetered generator node (flat type, behind a typed interface).** Net power is D − PV −
  G(t) with G(t) ≥ 0 built from daily positive and negative step parts under an L1 penalty, so an
  outage is a pair of steps. The level is aliased with the profile: it is identified only as the
  deepest outage under G ≥ 0, so where no full outage occurs it is unrecoverable and only the
  steps are read. A fit with G(t) below zero fails loudly.
- **The smooth clip.** PV = P_max − softplus(β(P_max − m·C·u))/β, with β = 2 per point, one P_max
  per site, and a capacity multiplier m. For given (m, P_max) the clipped PV is a fixed column, so
  both are profiled on a 2-D grid (m from 0.7 to 1.3 in steps of 0.05; P_max from 40 to 100% of
  m·C in 5% steps), with one convex fit each. Profiling m stops the clip absorbing error in the
  pinned capacity (D5's convex-to-vintage ratio is 0.85, close to a 70 to 90% "clip").
- **Tests.**
    1. With λ = 0, route switches off, A = 0, no node, m = 1, and P_max infinite, every option
       reproduces B's fit exactly.
    2. The meter fit with qmode `zero` on P > 0 everywhere matches the plain convex fit to 1e-5.
    3. The apparent operator's gradient is finite at P = Q = 0.
    4. The leakage asserts of Parts 2 and 4 hold, including that ū, A, the clip grid, tanφ, and q0
       use training rows only.
    5. The node on a synthetic series with no generator stays under 0.5 points.
  The module pull request also times one site's multi-start meter fit.

## Years, sites, and rows

**Part 4's three cuts, so that every choice is made on a year that is never scored:**

| Year | Cut | Role |
|---|---|---|
| 2023-07-01 to 2024-06-30 | 2023-07-01 | tuning: λ, the node's penalties, the XGBoost bar, C4's arm |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate, and the first capacity cut |
| 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict, and the second capacity cut |

- **Sites.** MW: D5, D9, D10, D11, D14, D16, D18 (decision count over D5, D10, D11, D14, D16; D9
  beside it; D18 descriptive). MVA: D2, D3, D4, D8, D12, D19, D21, D22.
- **Rows and windows.** As Part 4's A2 (all years, lockdown out, `default_rows == True`, 24-hour
  switching masks, none before 2022-03). `dp_mva_sunlit_v1` is the MVA rows with CAMS GHI of 1 W/m²
  or more under the same rules. Pinned capacity follows `dp_pinned_capacity` exactly, including
  D14's held value and the disclosed leak at the tuning cut.
- **Leads:** 09:00 UTC daily, 7 days ahead; the headline is D+1.
- **Predictions written:** `time_series_id`, `valid_time`, `power_fcst_init_time`, all 11 levels
  (`q01`, `q05` to `q95`, `q99`), `actual`, `lead_hours`, `arm`, `seed`, and `filtered`. Meter arms
  add `p_mw` and `q_mvar`, and node arms add `generator`.
- **Spread:** Part 4's post-filter replica spread when filtered, and the centred replica spread if
  not.
- **Robustness rows for every contrast:** November to March against April to October; without D16;
  all rows beside the masked headline; and summer midday (April to September, 10:00 to 14:00).

## Arms

**Every arm is screening tier: one frozen fit per cut, on the CPU.**

**Capacity identification (all three cuts; the label reads the 2024-07-01 and 2025-07-01 cuts):**

| Arm | What closes the route | Window |
|---|---|---|
| K-base | nothing (A2 with fitted capacity) | all years |
| K-i | (i) per-term covariance penalty in quarter × half-hour cells | all years |
| K-ii | (ii) no day-of-year × time-of-day term, slow temperature only | all years |
| K-iii | (iii) anomalies only | last two season-years |
| K-ii+iii | (ii) and (iii) together (exploratory) | last two season-years |

Each arm gets 20 month-block bootstrap weightings in the batched Gram matrix, at the cost of one
fit.

**The MVA layer (all three cuts):**

| Arm | MVA rows | Meter | Role |
|---|---|---|---|
| N | dark and sunlit | none: \|S\| read as P (experiment 79's naive fit) | C2's comparator |
| L0 | dark and sunlit | Q = 0 | the decision arm: NGED's convention |
| LQ | dark and sunlit | learned Q | tests the Q = 0 assumption |
| G-dark | dark only | demand alone, Parts 2 to 4's rule | the no-solar reference |

L0 and N differ only in reading |P| against P, so L0 minus N isolates the meter. L0 minus G-dark
measures training a solar term on sunlit MVA rows, not the meter.

**The generator node and the clip:**

- **B-node:** B-D1 with the node at every MW site, where the node must stay off.
- **B-clip:** B with the clip at D5 only. **B-clip-all:** the clip at every MW site, where the clip
  must stay open at each site Part 1 did not flag.
- **B-gen:** B-D1 with the node at every MW site and the clip at D5, the full generation-side model.
- **The real generator case (hindsight only).** G-c is the sum of two adjacent series, usable from
  2025-03-01: a Primary with an embedded generator, summed with its neighbour. The node is fitted
  over 2025-03 to 2026-06, and the node's level is compared with the mean dark-row export of the
  series that carries the generator, which is mostly the generator (no temperature slope). The fit
  covers the scored months, so it is labelled hindsight and enters no forecast score. There is no
  forecast arm: 4 months before the 2025-07-01 cut cannot identify heating or a season.

**XGBoost bars** (CPU, lagless, experiment 87's Primary hyperparameters without the three power
lags, 11 quantiles, Part 2's ERA5 and CAMS inputs, on B's window). Part 4's X-best, X-bestz,
X-bestamq, X-bestam, and its C6 bar (the tuning-year winner of X-best and X-bestamq) are reused as
stored, with X-best+K (the filter on XGBoost's residuals) beside it. **X-mva** is X-best on |S| at
each MVA site (no offset, because there is no vintage), filtered and unfiltered. Lagged XGBoost at
24.5 to 48 hours is descriptive. BG2 does not enter Part 5, because it is Part 6's bar. References
are climatology and `manual_heuristic`.

**Tuning (on 2023-24, fitted at the 2023-07-01 cut):**

- **λ for K-i:** the smallest value on {0.1, 1, 10, 100} at which every penalised term's |corr| with
  u falls under 0.05 in every daylight cell. If none does, λ = 100 and the residual correlations
  are reported; the arm is still judged.
- **The node's step and level penalties** (3 × 3 grid) on an injection that gate test 4 never uses:
  flat generators of 6 and 15 points with one step and one 3-day outage, injected at D5 and D9 on
  other dates. The chosen values are the largest at which that injection is recovered.
- **C4's arm:** among the K arms, the one with the best tuning-year score, recorded before any later
  fit.

## Known-answer gate (blocking, after tuning, before any replicate-year or scored-year fit)

**A failed blocking test stops scoring of the component it tests until it is fixed.** If it cannot
be fixed inside the part, that component counts as failed, and the verdict follows "Decision rule".

1. **Injected PV from a mismatched plant.** Subtract synthetic PV at extra capacities of 10, 30, and
   60 points from each MW decision site's net power, built from real CAMS irradiance through a plant
   unlike `dp_features.FLEET`: an east-west mix (azimuth 90° and 270°, tilt 15°), DC:AC 1.3 with
   inverter clipping, no temperature derating; and a second plant facing south-west at tilt 25°,
   otherwise the same. PV with the model's own shape lies in the capacity column's span, and any
   linear fit would recover it exactly, so only a mismatched plant tests anything. A K arm passes if
   the fitted capacity rises by the injected AC capacity within 20% at every site, level, and plant.
2. **Injected farms** (experiment 39's harness, at D10, D11, D14, and D16, keeping night rows at the
   injected site). A K arm passes if the median ratio of recovered to injected capacity lies in
   [0.75, 1.25]. For K-iii the ratio is also reported divided by its K-base ratio.
3. **The meter recovers sign, capacity, and power factor.** At the MW decision sites, inject a farm
   at experiment 79's level 1.0 (the window's median absolute midday power) so that reverse flow is
   common. Build Q = tanφ·D with power factor 0.97 from the demand component of the pinned B fit
   (not from the reading, which carries rooftop PV), and replace each reading by √(P² + Q²). Fit L0
   and LQ from the multi-start. The test passes if LQ's balanced accuracy of flow direction is at
   least 0.78 (experiment 79's 0.779), each site's capacity is within 15% of LQ fitted to the signed
   series, and the median recovered power factor is within 0.01 of 0.97. A variant at experiment
   79's level 0.5 (little reverse flow, the real sites' regime) is reported, not blocking.
4. **The node recovers a generator.** Into D10, D11, D14, and D16 (one fit each, on B-D1), inject a
   flat generator of 10 points, with one +5-point step on a known date, a 7-day outage, and a 1-day
   outage. The test passes if the step is placed within 7 days and sized within 30%, the 7-day
   outage is found, and the level, read at that outage, is within 20%. The 1-day outage is reported
   only.
5. **The clip recovers an injected clip.** At D10, D11, D14, and D16, inject a farm and clip the
   total PV (pinned rooftop C·u plus the farm) at 60% of that total's summer peak, so that the truth
   has the model's form. Fit the 2-D profile. The test passes if P_max is within 10% of the injected
   clip at 3 or more of 4 sites.
6. **Module tests 1 to 5** pass.

Injected farms are metered generators, so only ratios appear in any output.

## The capacity rule (pre-registered; labels, not contrasts)

**A K arm (other than K-iii) is `identified` at a cut if, at 4 or more of the 5 decision sites,
both of these hold for the last trained season-year:**

- **Precise:** the 90% bootstrap range (the 2nd to the 19th of the 20 ordered values) is at most
  max(3 points, 10% of the capacity).
- **Near the vintage:** the capacity lies within ±20% of the vintage's season-year mean (experiment
  40's capacity uncertainty). D14 is judged against `dp_pinned_capacity`'s held value and flagged.
  Part 3's profile found capacity identified to about 15%, so a ±20% band is already loose.

**The label needs both capacity cuts and gate tests 1 and 2.** Otherwise the arm is `not
identified`, with the failing clause named per site. The bootstrap measures sampling noise, not
route bias, so precision alone proves nothing, and the closed-route arms carry the bias test.
**`robust`** holds if at least two non-base K arms are `identified` and both agree with K-base
within the precision floor at 4 or more of 5 sites. **K-iii is judged by its ratio to K-base:** it is
`consistent` if, at 4 or more of 5 sites, the bootstrap range of K-iii / K-base overlaps [0.65,
0.87], the slope with which CAMS tracks a real farm's anomaly (experiment 63). A ratio in range
supports K-base, and a ratio near 1 or below 0.65 does not.

## Planned contrasts

Five planned contrasts, Bonferroni-corrected (99% intervals from `studies.bootstrap`, resampling
whole calendar months of stored predictions, paired across arms), at day D+1. Each is reported with
site counts, every arm's absolute score, the mean-forecast error, the replicate year's estimate and
count, the unfiltered contrast, and the robustness rows. **The margin is 0.10 points.** One year's
screening interval is about ±0.4 points, so non-inferiority is decided on the point estimate, as in
Part 3's screening ruling. The interval is reported, and an interval wholly above zero means
`refuted`.

- **C1, the generation side does not harm the MW forecast:** B-gen minus B-D1 at the MW decision
  sites. It passes if the point estimate is at most +0.10 in both years. The arms are nested and
  the node and clip are sparse, so a pass is weak evidence on its own, and the components'
  real-data clauses ("Decision rule") carry the rest.
- **C2, the meter:** L0 minus N on |S| at the 8 MVA sites, all rows, with sunlit and dark rows
  reported apart. It passes if the point estimate is at most +0.10 in both years. The expected
  result is near zero, because the trial MVA sites barely bounce.
- **C3, learned Q against Q = 0:** LQ minus L0 on |S| at the MVA sites (two-sided).
- **C4, fitted against pinned capacity:** C4's arm (chosen on the tuning year) minus B at the MW
  decision sites, at most +0.10 in both years. It runs only if that arm is `identified`.
- **C5, the DP against lagless XGBoost:** B minus Part 4's C6 bar, unfiltered against unfiltered,
  with B minus X-best+K beside it. It is labelled and is not part of the verdict.

**C3 and C4 cannot be decided at this noise level** (predicted within ±0.05 against a ±0.4
interval). Each is reported descriptively, with no verdict word, wherever its interval spans both
−0.10 and +0.10, and a choice handed on from either is labelled provisional.

**Exploratory at 95%:** L0 minus G-dark (a solar term trained on sunlit MVA rows), with its dark-row
part as a control that should be zero; each K arm minus B; K-ii+iii; per-site power factor and q0
with bootstrap intervals; B-node minus B-D1 and B-clip minus B per site; and L0 minus X-mva.
**Descriptive only:** D9; D18; coastal D5 and D9; bank holidays, the Christmas run, and days below
0 °C; L0's MW against experiment 79's fit, which includes the scored months.

## Predictions

Written before any fit. Each line can fail, and none overrides the decision rule.

- **K-base:** precise at 5 of 5 sites; within ±20% of the vintage at 3 or 4 (Part 2's convex
  capacity runs 4 to 21% below the vintage); `identified` at one cut or both.
- **K-i and K-ii:** within the floor of K-base at 4 of 5 sites; at least one `identified`, so
  `robust` is reachable but not certain. K-ii's unfiltered score is 0.05 to 0.20 worse than B's.
- **K-iii:** ratio to K-base 0.65 to 0.87 at 4 of 5 sites (`consistent`); bootstrap range wider than
  K-base's at every site.
- **Gate test 1:** K-base recovers 0.85 to 1.05 of the injected AC capacity; the east-west shape is
  the harder case for K-iii.
- **Gate test 3:** LQ's balanced accuracy 0.78 to 0.85; power factor 0.96 to 0.98; at level 0.5 the
  power factor's interval spans most of 0.95 to 0.99.
- **Real power factors:** recovered (bootstrap interval narrower than 0.95 to 0.99) at 2 or fewer of
  8 MVA sites, since on dark rows secφ folds into the demand scale.
- **Node:** with D1 beside it, under 1 point at 4 or more of 5 decision sites; D10 is the likeliest
  false positive (its demand fell from 55 to 43 points, and it has a jump in November 2024). On the
  hindsight sum, the node's level lies within 25% of the dark-row reference.
- **Clip:** the profiled m at D5 is 0.80 to 0.95; P_max sits at the grid top at every site Part 1
  does not flag; B-clip minus B at D5 on summer midday rows is −0.05 to −0.3 unfiltered.
- **C1:** within ±0.05 filtered and ±0.10 unfiltered, in both years. **C2:** within ±0.03. **C3:**
  within ±0.05. **C5:** −0.2 to −0.6 unfiltered; B minus X-best+K −0.1 to −0.4.
- **Tails:** B-gen's q05 and q95 coverage within 3 points of nominal, and q01 and q99 within 1.5.
- **Run time:** the default plan in under 2 hours of wall clock on 8 CPU cores.

## Decision rule

**The verdict answers whether the convex model can represent the generation side without making the
demand forecast worse.** Capacity gets its own labels (`identified`, `robust`, `consistent`), stated
in the take-home message and the bullets, never in the `verdict` field.

**A component passes when its gate and its real-data clause both hold:**

- **Layer:** gate test 3 passes and C2 passes.
- **Node:** gate test 4 passes; on real data the node stays under 1 point at 4 or more of 5 decision
  sites in both years, and the hindsight sum's level is within 25% of the dark-row reference.
- **Clip:** gate test 5 passes, and on real data B-clip-all's P_max stays at the grid top at every
  MW site that Part 1 does not flag.

**The verdict:**

- **`supported`:** all three components pass and C1 passes.
- **`partly supported`:** C1 passes and one or two components pass.
- **`refuted`:** C1's interval lies wholly above zero in the scored year, so the generation side
  makes the MW forecast worse.
- **`inconclusive`:** anything else, including any gate that cannot be fixed inside the part.

**What is handed on, each needing the same sign in both years:**

- **Capacity to Part 6 and N9:** if K-base is `identified` and `robust`, Part 6 may run fitted
  convex capacity as a variant beside pinned capacity. Otherwise capacity stays pinned, and the
  failing clause names what to try next.
- **The layer:** L0 goes to N9 if the layer passes. LQ replaces L0 only if C3 lies below −0.10 in
  both years.
- **The node:** it stays behind its switch, with drift beside it, and is used only where Part 1's
  screen flags a generator, if the node passes.
- **The clip:** it is used at D5 only if the clip passes and B-clip minus B at D5 is below zero in
  both years, filtered.

## Run order, compute, and time

**Order:** (1) the module pull request (route options, clip profile, meter fit, node, tests 1 to 5,
timing one meter fit); (2) the branch rule, in `decisions.json`; (3) tuning; (4) the gate; (5) the K
arms; (6) N, L0, LQ, G-dark, B-node, B-clip, B-clip-all, and B-gen at the later cuts, each fit
reused as the next cut's replica; (7) the filter on every arm; (8) the hindsight sum; (9) X-mva,
alongside.

**Compute.** The K arms with 20 weightings, the clip grid (13 × 13 values), and the node grid take
seconds to minutes per site. The meter fit is the risk: 8 sites × 3 cuts × 2 Q modes × 4 starts,
plus gate test 3. At about a minute per start, that takes about 3 hours on 1 core and under 30
minutes across 8. X-mva takes about 30 minutes on 4 cores. The default plan needs no GPU; a GPU of
up to 10 GB is used for the meter fit only if the timing in the module pull request calls for it.
The cut order if time is short is K-ii+iii, then the meter starts beyond two (the two nearest N's
capacity), then the node grid's middle values.

## What the write-up will contain

- **Take-home message:** whether the data can pin down how much rooftop solar sits behind each
  substation, and whether that holds when the model's demand terms are stopped from imitating solar;
  whether modelling the MVA meter changes the forecast at MVA substations; and whether the model
  finds hidden generators only where they exist.
- **Figures:** a sunny and a cloudy week at two MVA sites, |S| against N, L0, and G-dark with L0's
  `p_mw`; the headline chart of C1 to C5 with intervals, site counts, and the replicate year beside
  each; every arm's absolute score per site, filtered and unfiltered; capacity per site per K arm
  with bootstrap ranges against the vintage's ±20% band; K-iii / K-base against [0.65, 0.87]; the
  demand terms' midday movement by term; the gate; power factor per MVA site with intervals; D5's
  summer-midday power against irradiance with and without the clip; the tails (coverage and pinball
  at q01, q05, q95, and q99, every arm); and the score by month since the cut (MW and MVA sites
  only).
- **Substation cards** (D-labels only, labelled "Provisional: fitted to past weather on a screening
  run"), from B-gen's and L0's fits at the 2025-07-01 cut: for each demand substation, capacity per
  season-year shown as K-base with its bootstrap range, each `identified` arm, and the pinned
  vintage; the filter's level and anomaly over the scored year; at MVA sites, power factor and q0
  with intervals (marked "not recovered" where the interval spans 0.95 to 0.99) and the modelled MW
  profile; at D5, the clip level and m; and the node's level at every demand site. One
  cross-substation figure compares capacity, power factor, and heating slope.
- **Checks:** the seven audit triggers, the median forecast minus the mean forecast per site and
  arm, and the mean-forecast error beside each contrast.
    1. A failed control: any gate test; the node or clip switching on at a site that should be off;
       a dark-row gap between L0 and G-dark.
    2. A result against its prediction or an earlier finding without a tested explanation (above
       all, an `identified` capacity outside experiment 40's ±20%).
    3. The quantile-score contrast and the median forecast's mean absolute error disagreeing in
       direction, for C1 to C5, filtered or unfiltered.
    4. Leaving one site out flipping a contrast's sign or verdict, or a capacity label.
    5. A nested arm beating its parent beyond the paired interval's half-width (B-D1 beating B-gen
       at a site the node and clip leave unchanged; N beating L0; K-ii beating K-base).
    6. A headline interval more than twice or under half as wide as Part 4's on the same rows.
    7. A later design depending on an unreplicated number. This fires by construction, because Part 6
       and N9 take the capacity choice and the layer from this result. The audit is planned.
- **Caveats:** ERA5 and CAMS flatter the solar term, and CAMS's anomaly attenuation lowers K-iii; the
  filter's residuals use past weather up to the information cut; the vintage is itself uncertain by
  ±20%; MVA sites have no capacity truth, and their power factor is weakly identified; the real
  generator case is a hindsight fit; and the trial MVA sites barely bounce, so bounce rests on the
  known-answer test.

## Privacy

- **Substations appear by D-label only.** No figure, table, or bullet names a Primary, and no
  output copies a name from experiment 45's vintage file or experiment 79's write-up.
- **G-c is generator data.** The real generator case and every series built from it (the sum, the
  node's output on it, the dark-row reference) is plotted under the neutral label G-c, normalised by
  its own capacity, with day counts from an unstated origin, `aria=False`, and kept out of the
  score-by-month figure. Published text describes G-c as "a Primary with an embedded generator,
  summed with its neighbour", and never links G-c to a D-label.
- **Injected farms appear as ratios only.**
