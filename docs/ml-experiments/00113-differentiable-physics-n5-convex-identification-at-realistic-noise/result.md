---
id: "00113"
slug: differentiable-physics-n5-convex-identification-at-realistic-noise
title: "Differentiable physics, N5: does the convex model identify heating slopes and PV capacity at realistic noise?"
theme: differentiable-physics-embedded-generation
also_relevant_to: [differentiable-physics-pooling-across-substations, differentiable-physics-demand-calendar-and-weather, solar-disaggregation]
status: planned
size: >-
  medium: experiment scripts only, no module change (every variant fit is a column subset of one
  Gram matrix per substation, solved with studies.dp_gam's solver). See "Size triggers".
hypothesis: >-
  On synthetic substations built from the real calendar and weather, with noise as large and as
  autocorrelated as the convex model's real residuals, the convex model's 80% month-block bootstrap
  intervals cover the truth 70 to 90% of the time, separately for each hour band's heating slope
  and for the latest season-year's PV capacity. Labelled separately, not in the verdict: the
  day-of-year harmonics and the drift term make those intervals wide, by absorbing the seasonal
  cycles of temperature and clear-sky PV.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  The posterior interval and the sandwich variance of the penalised fit; an unstratified
  month-block bootstrap; experiment 112's pooled VP gate errors (different noise, see below);
  experiment 45's frozen vintages and Part 5's K-base bootstrap ranges on real data
row_set_id: >-
  Synthetic targets on the training rows of dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and
  D16, cut 2024-07-01; real dp_mw_v1 at the 2025-07-01 cut, replicate 2024-07-01
metric: >-
  Coverage of 80% intervals over 100 synthetic substations, with site-clustered bootstrap intervals,
  for each hour band's total heating slope and the latest season-year's PV capacity; median width
  in points and as a share of the truth; variance ratios and effective sample size
production_status: not-for-production
adopted_in: none
code: ml-experiments/00113-differentiable-physics-n5-convex-identification-at-realistic-noise/
---

# Differentiable physics, N5: does the convex model identify heating slopes and PV capacity at realistic noise?

**Theme: embedded generation, because N5 feeds Part 5's capacity question** (fitted or pinned, and
what interval a card may print).

## What N5 asks, and why now

**N5 asks whether the convex per-site model can tell how much heating and how much rooftop PV sits
in a substation's demand at real noise levels.** Experiment 112's gate (noise sd 5.01 points, week
blocks from any season of a donor's residuals) saw the pooled VP fit miss thresholds by up to
5.7 °C, band slopes by 40 to 76%, and growth by up to 2.5 points (most capacities within 10%, the
worst 42 to 44% off); network off, it failed at sd 5.0, was exact with no noise, and fine at sd 1.0
(numbers from the design session; 112's write-up is quoted for them).

**The mechanism to test.** The convex model carries two day-of-year harmonics per hour band, a
monthly drift path per band (D1), and a daily profile per day type. Together these terms can
reproduce the seasonal cycles of temperature and of clear-sky PV, which leaves day-to-day weather
anomalies to identify the slopes and the capacity. Removing columns always narrows a nested fit,
so the question is only how much: the label below is pre-registered as a size.

**Weak identification is acceptable if the interval says so.** **Size triggers:** private files
only; serving path untouched; one design of several pre-registered; no caller in `packages/`.

## Synthetic substations

**One hundred synthetic substations: 20 draws on each decision site's real rows** (`default_rows`,
all years to the 2024-07-01 cut, lockdown out, masks applied), with its real calendar, weather, PV
shape, and `dp_gam.design_matrix`; only the target is synthetic. For a linear fit on a fixed design
the estimation error and the bootstrap width depend only on the noise, not on θ_truth, so draws at
one site are independent trials only as far as their noise is. The noise pool below makes two draws
share a month's noise rarely (about one month in 24), and coverage is judged with site-clustered
intervals, whose design effect the write-up quotes. For scale: a correct 80% interval lands in 70
to 90% about 95% of the time over 50 independent trials, but only 67% of the time over 15.

**Generator F:** target = X·θ_truth + noise, X the full convex design with D1. θ_truth starts from
Part 2's tuned all-years D1 fit at that site and cut:

- **Heating:** each band's total slope (the sum over the 6, 24, 72, and 168 hour smoothings: the
  response to a sustained cold spell) is the real band total times U(0.7, 1.3), floored at 2 points
  per 10 °C, split across smoothings as (0.2, 0.1, 0.3, 0.4).
- **Capacity:** first season-year U(5, 25) points, each later increment U(1, 5) points.
- **Truth T-HD:** the real fit's harmonics and its D1 drift path (a path the tuned penalty already
  supports). **Truth T-0:** harmonics and drift zero.
- Profile, holidays, tent, and lighting at the real fit's values.

**The truth is in the model's form but not free of its penalty.** The ridge and the drift penalty
shrink every fit deterministically. The script computes that bias analytically for each functional,
b = −vᵀ(G + P)⁻¹Pθ_truth, and coverage is scored twice: of the truth (primary), and of the
noise-free fit's value, truth + b (the interval method's own coverage). Bias over width is reported.

**Noise, drawn independently per synthetic substation:**

- **iid5:** Gaussian, sd 5.0 points. **Controls:** noise-free, and iid sd 1.0 (error only), one
  draw per site.
- **Real (the decision noise):** each target month takes a whole month of real convex training
  residuals (from each donor's all-years D1 fit at the same cut), drawn uniformly from all four other
  decision sites and every year other than the target's in the same calendar month (about 10 to 12
  options), with a random sign per month block, aligned by day and half-hour; missing slots are
  zero, and their share is reported. The calendar month keeps the season's noise level; the other
  year stops the donor's weather lining up with the target's. This differs from 112's noise (week
  blocks from any season), which the write-up says wherever N5 is read against 112. Reported: the
  noise's sd and autocorrelation at lags of 1, 48, and 336 half-hours; and the bias from seasonal
  misfit in the donor residuals, measured by fitting G-full to the noise alone, averaged over draws.

## Fits

**Every fit is a column subset of one design, so each site needs one Gram matrix per window.** Per
site and window, `dp_gam.accumulate_gram` builds `X^T W X` with the full layout for 201 weightings
(unweighted, 100 for I1, 100 for I1u; about 400 MB at about 500 columns); each target then needs only
`X^T W y`. A fit drops the rows and columns of the terms it omits and calls
`dp_gam.solve_bounded_quadratic` with `dp_gam.penalty_matrix`'s penalties. A pinned fit drops the
capacity columns and shifts the right-hand side by the Gram block times the pinned increments, which
equals adding the pinned PV to net power (the capacity columns carry a minus sign).

| Arm | Harmonics | D1 | Capacity | Run on |
|---|---|---|---|---|
| G-full | yes | yes, Part 2's `drift_step` 3,000 | fitted | T-HD and T-0 |
| G-full-30 | yes | yes, Part 4's `drift_step` 30 | fitted | T-HD and T-0 |
| G-full-pin | yes | yes, 3,000 | pinned to the truth | T-HD |
| G-full-pin80 | yes | yes, 3,000 | pinned to 0.8 × truth (inside experiment 40's ±20%) | T-HD (exploratory) |
| G-0 | no | no | fitted | T-0 (the diagnostic); T-HD (misspecified, exploratory) |
| G-0-pin | no | no | pinned to the truth | T-0 |
| G-H, G-D | one of the two | | fitted | T-0 (which term absorbs more, exploratory) |

**The decision cell is G-full on T-HD with the real noise; the iid5 cell is G-full on T-HD with
iid5.** The diagnostic pair is G-0 against G-full (and G-full-30) on T-0 with the same draws. G-full
on the 2-year window before the cut (112's window) runs beside the decision cell. **Generator V
(112's) is dropped:** its heating, relu(threshold − mixed temperature), has no band total slope to
score, and thresholds belong to N4.

## Intervals and scored functionals

**Three 80% interval methods:**

- **I1 (decision):** a month-block bootstrap of 100 weightings, stratified by calendar quarter, so
  every resample keeps every season. **Pre-registered inflation:** its half-widths about the
  bootstrap median are multiplied by √(m/(m − 1)), with m the months per quarter in the window for
  band slopes (about 13 on all years, 6 on 2 years) and m = 6 for the latest capacity, which rests
  on its season-year's six summer months. The uninflated interval is reported beside it, with each
  functional's supporting-month count and the share of resamples holding under half of them.
- **I1u:** the unstratified `dp_gam.month_block_weights`, 100 weightings.
- **I2 (posterior):** Gaussian, from A⁻¹σ̂², A the penalised Gram matrix.

**Scored:** each band's total heating slope (400 intervals per cell) and the latest season-year's
capacity (100). The cut falls on a season-year boundary, so that season-year is complete.
Reported: every season-year's capacity, the latest growth increment, and the share of each band's
slope on the 6 and 24 hour smoothings; single smoothing slopes are never scored (nearly collinear).
**Per functional and cell:** coverage of the truth and of truth + b, with a site-clustered bootstrap
interval (sites resampled, then draws); median signed relative error; root mean square relative
error; median width, in points and relative; coverage per real site. A family is `precise` if its
median relative width is at most 20% (capacity) or 30% (band slope), otherwise `weak`.

**Active bounds:** the near-collinear heating columns hit non-negativity in many bootstrap fits, so
the decision cell is also solved unbounded; reported: the share of fits with an active heating
bound, and the unbounded interval's coverage.

## Variance ratios and effective sample size

- **Absorption ratio (the label's quantity):** r = vᵀA₀⁻¹v / vᵀA⁻¹v, the exact variance of
  c = vᵀθ in the smaller fit over the full fit (G-0, harmonics only removed, drift only removed),
  at both drift penalties. Under iid noise √r predicts the diagnostic pair's width ratio; the Monte
  Carlo widths check it.
- **Collinearity share ρ (descriptive):** (vᵀv)² / (vᵀAv · vᵀA⁻¹v). For a sum of coefficients ρ
  depends on how the other directions are parameterised, so it carries no threshold.
- **Autocorrelation factor τ:** I1's variance over the sandwich variance A⁻¹GA⁻¹σ̂², the frequentist
  variance of a penalised fit under independent noise (G the unpenalised Gram matrix).
- **Effective sample n_eff = n · ρ / τ,** in rows and site-days, on the synthetic and real designs.
- **Spectral check:** the share of the partialled column's variance at periods under 10 days.

## Real data: five decision sites

**Bootstrap intervals for the cards.** At the 2025-07-01 cut (replicate 2024-07-01), all years,
lockdown out, fitted capacity, with and without D1: I1, I1u, and I2 for each band's slope and each
season-year's capacity at D5, D10, D11, D14, and D16, beside the vintage's season-year mean,
`dp_pinned_capacity`'s value (D14 flagged), Part 5's K-base range where stored, and Part 3's pooled
fit; r, τ, and n_eff on the real design. **Card labels, per family:** where that family's gate
passes, "80% month-block bootstrap; calibrated for noise on in-family synthetic data (N5); excludes
PV-shape bias", with N5's `precise` or `weak`; otherwise no convex interval. A pooled DP estimate
carries "weakly identified at real noise levels (experiment 112)".

## Known-answer controls (blocking, before any noisy cell is read)

**A failed control stops the noisy cells; if it cannot be fixed, the verdict is `inconclusive`.**
(1) The G-full and no-drift subsets reproduce `dp_gam.fit_gam` with the matching `GamSpec` on each
real target to 1e-8 relative; G-0 and the pinned shift match a dense float64 bounded least-squares
fit of the same columns on the adjusted target (net power plus the pinned PV) to 1e-8.
`PinnedCapacity` is weekly and cannot reproduce season-year steps, so it is not the reference.
(2) Generator F with no noise: every fit containing the truth's columns returns truth + b for every
scored functional to 1e-6. (3) G-0 on T-0 with iid5 on 500 draws (one solve each): I2's coverage
within 77 to 83% for both families; I1's on the cell's 100 draws within 70 to 90%. (4) Every
stratified resample keeps each quarter's month count. (5) No training row after its cut; nothing
on or after 2026-07-01.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **Decision cell (I1):** coverage of the truth 70 to 90% for both families; band slopes `weak`
  (median relative width 30 to 80%); latest capacity median relative width 10 to 25%; |b| under a
  quarter of the median width; the noise-only bias under 5% of the truth.
- **I2 at real noise:** coverage under 60% for both families; τ 3 to 30. **iid5:** I1 and I2 both
  70 to 90%, widths 30 to 60% of the real-noise widths.
- **Absorption (Part 2's penalty):** r (G-0 over G-full) at most 0.36 for each band slope and at
  most 0.56 for the latest capacity; harmonics give the smaller r for the capacity, drift for the
  night-band slope; at Part 4's penalty r is smaller still for every functional.
- **Diagnostic pair (T-0, same draws):** the Monte Carlo width ratio within 0.1 of √r under iid5;
  G-0 calibrated under both noises.
- **Spectral and n_eff:** over 70% of each partialled column's variance at periods under 10 days in
  G-full, under 50% in G-0; n_eff for each band slope under 5% of the rows in G-full.
- **Bounds:** an active heating bound in over 30% of the decision cell's bootstrap fits; unbounded
  coverage within 5 points of bounded.
- **Other arms:** G-0 on T-HD covers band slopes under 50%; G-full-pin80 moves them under 10%
  from G-full-pin; 2-year widths 1.3 to 2 times all years'.
- **Real data:** I1's capacity interval no wider than max(3 points, 20%) and overlapping the
  vintage's ±20% band at 4 of 5 sites; the night-band slope excludes zero at 5 of 5; I1 2 to 5
  times as wide as I2.

## Decision rule

**Each family (band slopes, latest capacity) gets its own gate label from I1's coverage of the
truth in the decision cell and in the iid5 cell:** `calibrated` if both are within 70 to 90%;
`overconfident` if either is under 70%; `too wide` if either is over 90% and neither under 70%.
The point estimate decides; the site-clustered interval is reported beside it.

- **`supported`:** both families `calibrated`.
- **`partly supported`:** one family `calibrated`, the other `too wide`.
- **`refuted`:** either family `overconfident` (its intervals overstate what the data identify).
- **`inconclusive`:** a control cannot be fixed, or both families `too wide`.

**Absorption label, never in the verdict, stated at each drift penalty:** `absorption large` if r
is at most 0.36 for every band slope and at most 0.56 for the latest capacity (a 40% and 25%
narrower interval), and the Monte Carlo pair agrees within 0.1 of √r; `absorption small` if r is
over 0.72 everywhere (under 15% narrower); otherwise `partial`. The write-up says the label holds
at the penalty it names.

**Handed on, per family:** a `calibrated` family's I1 intervals go on the cards, and Part 5's
capacity rule may use the capacity's I1 for its 20-weighting range. A `calibrated`, `precise`
capacity overlapping the vintage band at 4 of 5 real sites is evidence for Part 5's fitted-capacity
variant, without changing Part 5's rule. If absorption is large, the next pooled test removes
per-band harmonics and drift from the shared modules (nearest planned arm: Part 5's route (ii)),
not a new optimiser.

## Audit before merge: the seven triggers

**N5 scores parameters, not forecasts, so triggers 3 and 6 are adapted; trigger 7 fires by
construction and an Opus audit is planned.** (1) Control: any of the five. (2) Contradiction: the
convex model `calibrated` and `precise` where 112 failed with no diagnostic saying why, or a real
capacity outside experiment 40's ±20% of the vintage. (3) Bias against coverage: coverage in range
with median bias over half the median width, or coverage failing with bias near zero. (4) Leaving
one real site out moves a family's site-clustered interval across 70 or 90%, or changes the
absorption label. (5) Nested arms: G-full beating the true smaller G-0 on T-0 beyond Monte Carlo
error; a pinned-truth arm recovering slopes worse than G-full. (6) Real I1 capacity widths over
twice or under half Part 5's K-base ranges. (7) The cards, Part 5's capacity reading, and the next
pooled test depend on it.

## Run order, compute, and time

**Order:** (1) scripts and controls 1, 4, and 5 (about 3 hours of agent time); (2) Gram matrices
(5 sites × 2 windows, 5 sites × 2 cuts) and the analytic b, r, ρ, and sandwich terms; (3) controls
2 and 3; (4) **time one full cell** (100 targets × 201 bounded solves; the solver handles one system
at a time, so a 1-hour grid estimate may be 2 to 5 times short); (5) the grid; (6) real data; (7)
write-up (about 3 hours), numbers review, audit, prose sweep. CPU only, at most 6 cores.

**If the timed cell projects the grid over 4 hours, cut in this order:** I1u outside the decision
cell, G-H and G-D, G-full-pin80, G-0 on T-HD, the 2-year window, then 10 draws per site (the noise
pool unchanged). **Data:** Part 2's `tuned.json`, Part 4's `tuned.json` (`drift_step` 30),
`studies.dp_pinned_capacity`, experiment 45's vintages, Part 5's and Part 3's stored capacities.

## What the write-up must contain (beyond the series brief's standing rules)

The take-home says plainly whether the convex model's intervals for heating and rooftop PV can be
trusted, family by family, how wide they are at real noise, how much the seasonal terms widen them
and at which drift penalty, and what that means for 112's failure. Gloss "identified", "calibrated
interval", and "effective sample size". No forecast is scored, so there is no quantile score, tail
table, or curtailment number; say so.

- **Figures:** coverage per family and cell with clustered intervals (first); recovered against
  true values; width, G-full against G-0, beside √r; r by term at both penalties; n_eff; noise
  autocorrelation; real intervals per site beside the vintage band, K-base, and the pooled fit.
- **Cards** ("Provisional: fitted to past weather on a screening run"): band heating and capacity
  with I1 and the per-family label, the pooled fit with 112's line, the vintage; one fleet figure.
- **Caveats:** ERA5 and CAMS flatter the solar term; in-family truths cannot show PV-shape bias;
  donors are other decision sites; the inflation is an approximation; N5's noise is not 112's.
- **Privacy:** D-labels only; synthetic substations as site and draw (D10-s3); never a Primary's
  name (experiment 45's file carries names); G-a, G-b, and D15 out.

**Decisions (2026-10-20):** 20 draws per site with the widened noise pool and site-clustered
intervals; the stratified bootstrap with its inflation decides; the card wording is split by
family; generator V is dropped; the 2-year window runs beside the decision cell only.
