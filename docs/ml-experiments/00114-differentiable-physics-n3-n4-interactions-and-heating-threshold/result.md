---
id: "00114"
slug: differentiable-physics-n3-n4-interactions-and-heating-threshold
part: n/a
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to: [differentiable-physics-forecast-engine, lagged-power-and-error-correction]
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/ only.
  Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible (which interaction
  terms, how to choose a threshold), and this plan pre-registers one of each. Callers: this
  folder's scripts only; no change to packages/.
hypothesis: >-
  Two labelled claims about the convex model of Primary demand in the form Part 6 made the DP's
  candidate (GAM-K: 24 months, no drift, pinned capacity, Kalman filter), on past weather. N3:
  interaction terms (heating by day-type group, heating on eight sub-band blocks, lighting by
  day-type group), each shrunk towards the base by a difference penalty, lower the day-ahead
  quantile score after the filter. N4: a per-site heating threshold, chosen by profiling the
  training loss, is recoverable on synthetic substations at real noise, is identified away from
  15.5 C at one or more real substations under the per-site printing rule, and lowers the score
  after the filter.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  B0 and B0-K (GAM-K's convex model on past weather, without and with the filter), refitted here;
  Part 4's stored A2, WK, A1, lagless XGBoost with and without the filter, climatology,
  manual_heuristic
row_set_id: >-
  dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and D16 (D9 descriptive); tuned 2023-07 to
  2024-06, replicate 2024-07 to 2025-06, scored 2025-07 to 2026-06
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1; tail coverage
  and pinball at q01, q05, q95, and q99; for the gates, power, coverage, and width over 100
  synthetic substations
production_status: not-for-production
code: ml-experiments/00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/
---

# Differentiable physics, N3 and N4: interaction terms and a per-site heating threshold

**Status: planned.** This page is the pre-registered plan. The results replace it in a later
commit of the same pull request.

## What the two questions are, and what earlier experiments already say

**The design session's list also held a third question, the orientation mix of the PV panels (N13),
and
the plan drops it.** Experiment 73 found the east-south-west shares unidentified on a noisy known
answer (median share error 0.200 against a limit of 0.15). Experiment 77 found the fitted mix moved
experiment 64's arm `am` (lagged XGBoost started from the solar term) by −0.001 points (98.33%
interval −0.018 to +0.014), on April-to-September daylight rows at 5 Primaries. The series brief
keeps the orientation mix fixed.

**The base is GAM-K's convex model, read after the filter, because GAM-K is the DP's production
candidate.** Experiment 111 made GAM-K (24-month window, no drift, pinned capacity, ridge 0.1, the
Kalman filter) a `candidate` at 2.800 points against BG2's 2.780 on ENS. A term that helps only the
unfiltered model is no use to GAM-K, so the decision contrasts are read after the filter, with the
unfiltered contrasts printed beside them. **Every contrast here is a past-weather ceiling:** the
heating terms are read at the ERA5 temperature of the target half-hour, which no live forecast has.
Experiment 111's calibration shifted ENS temperature by −1.15 to +2.05 K, and its filter gain on
ENS was about two-thirds of Part 4's, so a gain here need not transfer. Only the ENS confirmation in
"Handed on" tests transfer.

**N3 asks a direct question about the convex terms, because Part 3's interaction network added
nothing measurable.** Experiment 105 found P-nonet minus P at −0.03 points (−0.14 to +0.07), so
there is no network gain to "recover"; that comparison stays descriptive. **Round 2 of the design
debate points at the shape of the day, more for slow shape than for day type.** GAM-K's next-day
half-hourly error is 4.3 to 4.5 points against 2.77 for its daily mean. A 14-day residual-profile
correction cut the median error from 4.32 to 4.19, most in December to February and in the morning
and evening, but a same-weekday profile did no better (4.21). That weakens T1 and T3, the day-type
terms, and leaves T2, the sub-band heating shape, the likelier source of any gain. The filter's
persistent level per band cannot follow a weekday-to-weekend alternation, so the filter should
absorb part of a T2 gain and little of a T1 or T3 gain.

**N5 makes N4 hard, and the gate says so before any score is read.** Experiment 113 found a band
heating slope weak at real noise (80% interval about 38% of the slope wide) on the all-years design,
because the day-of-year harmonics absorb about 80% of its variance. B0 has about two-fifths of those
rows, and a threshold rests on the same mild-weather days, so its profile is likely to be flat.

## Base model, years, sites, and rows

**B0** is `dp_gam` with `drift="none"`, `capacity="pinned"` (experiment 45's `frozen` vintage via
`studies.dp_pinned_capacity`), ridge 0.1, `TrainingWindow(months=24)`, lockdown out, as
`e00111_fits.py` builds GAM-K's model. **B0-K** is B0 plus `dp_assimilation.run_kalman` on its
residuals, under the 3.5-hour cut and with no switching mask. Each arm's four filter
hyperparameters per site are fitted by `fit_filter_hyperparameters` on that arm's out-of-sample
tuning-year residuals, with Part 4's prior, as Part 4 did.

| Year | Cut | Role |
|---|---|---|
| 2023-07-01 to 2024-06-30 | 2023-07-01 | tuning: N3's λ, every arm's filter hyperparameters |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate: every decision contrast, out of sample for both choices |
| 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict |

- **Screening tier:** one frozen fit per arm and cut, no refits, one seed (the fit is
  deterministic). Each cut's spread comes from the replica fit 12 months earlier
  (`dp_gam.replica_residuals`), so fits run at 2022-07-01 to 2025-07-01. The 2022-07-01 replica's
  24-month window loses its first 12 months to the lockdown cut. Nothing scores a valid time on or
  after 2026-07-01.
- **Sites:** D5, D10, D11, D14, and D16 decide (4 of 5 needed); D9 beside them in the scored year
  only; MVA sites, D15, and D18 not fitted; no onboarding or short-history arms.
- **Masks and leads:** the hand-filtered switching list as experiment 101 filters it, padded 24
  hours, out of training and scoring (all-rows score beside); 09:00 UTC origins, day D+1.
- **Predictions:** the keys, `q01`, `q05` to `q95`, `q99`, `actual`, `lead_hours`, `arm`, `seed`,
  and `level` and `anomaly` for filtered arms; quantiles from `dp_gam.ResidualSpread` on each arm's
  replica, run through the same filter (first 14 days dropped) for a filtered arm.

## Arms

**Every arm keeps B0's columns and adds columns of its own, solved by
`dp_gam.solve_bounded_quadratic` on per-month Gram matrices** (the pattern of `e00113_design.py`,
copied, not imported). Each arm runs unfiltered and filtered (suffix `-K`).

**N3, interaction terms, nested in B0 by a difference penalty.** `dp_gam`'s ridge is not scaled by
the row count (`penalty_matrix`), so on 35,000 rows it is 10⁻⁵ to 10⁻³ of the data term and cannot
shrink anything. Each interaction instead splits one of B0's columns into two and adds
λ·(n_rows/SITE_YEAR_ROWS)·(β_a − β_b)², with each split column carrying half of B0's ridge, so that
as λ → ∞ the arm is B0 exactly. The penalty is quadratic, so the box constraints still hold.

| Arm | Split | Constraint |
|---|---|---|
| T1 | each heating column into weekday and (Saturday, Sunday, bank holiday) | non-negative |
| T2 | each heating band into its two halves (local 00–03, 03–06; 06–08:30, 08:30–11; 11–13:30, 13:30–16; 16–20, 20–24) | non-negative |
| T3 | each lighting ramp into weekday and (Saturday, Sunday, bank holiday) | free |
| T-all (decision arm) | T1, T2, and T3 together, each pair of split columns tied to its parent | as above |

- **λ** comes from {1, 10², 10⁴}, chosen per arm (one value for all sites) by the filtered
  tuning-year score; the new columns' effective degrees of freedom are printed at each λ.
- **Filter-fit control:** T-all-K-b0 runs T-all through the filter with B0-K's hyperparameters. If
  C1 moves by more than its half-width between T-all-K and T-all-K-b0, the gain sits in the filter
  fit, not the terms.

**N4, the heating threshold.** The hinge `relu(θ − T_k)` at θ from 12 to 19 °C in 0.5 °C steps (15
values), one θ shared by the four smoothings and all bands of a site. All 15 hinge blocks sit in one
augmented design, so one pass over the rows builds every Gram matrix.

- **H-site:** at each cut, each site's θ is the arg-min of the unpenalised training loss (the
  weighted squared error) over the grid; the penalised loss is printed beside it to show the two
  agree. The choice uses only that cut's training rows, so the gate tests exactly this rule.
- **Profile:** at each cut, the loss per θ in effective nats (the loss difference over 2σ̂²τ, with
  B0's τ re-measured in gate 3), and the I1 month-block bootstrap distribution of the arg-min.
- **Per-site printing rule:** a site's θ is printed only if its I1 80% arg-min interval lies wholly
  outside [13.5, 17.5] °C and 15.5 sits at least 2 effective nats above the profile minimum;
  otherwise "not identified at this site". An arg-min at a grid edge on a flat profile is not a
  departure from 15.5.
- **H-site is a decision arm only if gate 3 passes.** Otherwise C2 is a descriptive row with no
  label, and no card shows a per-site threshold.

**Capacity check (descriptive, scored cut only):** B0-fc and T-all-fc, the same models with fitted
capacity, with 50 I1 weightings each. T3's free lighting ramps follow solar elevation and could
imitate PV. A move is flagged only if the 80% interval of the paired difference T-all-fc minus B0-fc
lies wholly beyond ±2 points, since N5's capacity interval alone is 2 to 4 points wide.

**Stored references (Part 4, scored / replicate year):** A2 2.275 / 2.284; WK 2.267 / 2.260; A1
2.895 / 3.303; lagless XGBoost 3.068 / 3.473, filtered 2.496 / 2.432; climatology 4.068 / 3.910;
manual heuristic 3.488 / 3.839. Part 6's GAM-K on ENS (2.800) is context only.

## Known-answer tests and gates

**Order: test 1, then tuning (λ and filter hyperparameters, hashed into `decisions.json`), then
gates 2 and 3, all before any replicate-year or scored-year row is read.** A failed test 1 or gate 2
stops that question's scored reading; if it cannot be fixed, the label is `inconclusive`. Gate 3
can fail by design: its failure sets `not identified`.

1. **Reproduction.** The augmented code with no extra columns reproduces `dp_gam.fit_gam` for B0 to
   1e-8 relative in the parameters; each N3 arm at λ = 10¹⁰ reproduces B0's fitted values to 1e-5
   points (amended from 1e-6 before any fit ran: development runs of the penalised solve showed a
   float64 conditioning floor of 0.8e-6 to 2.6e-6 points at every penalty weight, with no scored
   outcome read); θ = 15.5 reproduces B0 exactly; B0-K's filter on stored B0 residuals reproduces
   `run_kalman` called directly; no training row after its cut.
2. **N3 power check (synthetic, unfiltered, can fail).** The truth is B0's real fit at the
   2024-07-01 cut plus truth I, weekend heating and lighting scaled by a common amplitude. The
   amplitude is set on 5 pilot draws so that the true-parameter forecast beats B0's fit by 0.03
   points on the held-out synthetic year, the decision threshold. Noise follows N5's real-noise rule
   (`e00113_noise.py`, copied): whole donor months from the other decision sites' B0 training
   residuals at that cut, on the training rows and on the held-out year alike, so no 2024-25
   residual enters. Each draw is scored with B0's spread rule, on a synthetic replica fitted 12
   months earlier. 20 draws per site at the tuned λ, every λ reported. **Pass** if (a) on truth I,
   T-all beats B0 in at least 80% of the 100 draws, with the paired mean and its interval over draws
   and the per-site pass count printed, and (b) on truth 0 (no interaction), T-all is no worse than
   B0 by more than 0.01 points on average. The gate tests the unfiltered model only.
3. **N4 recovery (synthetic, can fail).** The same generator with the hinge at a true θ from
   U(12.5, 18.5) per draw and slopes from B0's fit times U(0.7, 1.3); 20 draws per site; H-site's
   rule and 50 I1 weightings per draw; the interval of a draw is [lo − 0.25, hi + 0.25] °C over its
   bootstrap arg-mins. **Gate (`identified`)** if all hold: (a) the interval covers the truth in 70
   to 90% of the 100 draws; (b) its median width is at most 3 °C; (c) among draws whose truth lies
   2 °C or more from 15.5, the point arg-min is within 1 °C of the truth in at least 60%. Otherwise
   `not identified`. A pass shows the rule recovers a threshold on average, not that any real site's
   profile identifies its own; the per-site printing rule decides that. **Placebo:** with the truth
   at 15.5 (20 draws), the arg-min lies within 1 °C of 15.5 in at least 70%. **τ:** B0's bootstrap
   variance over sandwich variance at θ = 15.5, from the same weightings.

## Planned contrasts

**Two planned contrasts, Bonferroni-corrected: 97.5% intervals from `studies.bootstrap`, resampling
whole calendar months of stored predictions, paired across arms, at day D+1, after the filter.**
Each is reported with the count of sites moving each way, every arm's absolute score, the
mean-forecast error contrast, the replicate year's estimate and count, November to March against
April to October, the four hour bands, and leaving one site out.

- **C1 (N3):** T-all-K minus B0-K. Beside it: T-all minus B0, and T-all-K-b0 minus B0-K.
  Exploratory at 95%: T1-K, T2-K, and T3-K each minus B0-K.
- **C2 (N4):** H-site-K minus B0-K, decision only if gate 3 passes. Beside it: H-site minus B0.

**The 0.03-point threshold.** Convex columns cost almost nothing to carry into GAM-K, so the
threshold is there to justify an ENS confirmation run, not a porting cost. The sister contrast on
these rows is Part 4's C4 (A2 minus A3, two filtered convex arms, scored year, D+1): −0.015 (−0.053
to +0.003) at 99.17%, about ±0.023 at 97.5%, so 0.03 is about 1.3 half-widths.

**Tails, every arm:** coverage and pinball at q01, q05, q95, and q99 beside the headline. A gain
that moves q95 or q99 coverage more than 1 point further from nominal is flagged; the flag changes
no label, and the take-home states any `gain` and its flag together. No curtailment cost: these are
demand Primaries, and `studies.node_curtailment` scores a node and its generators.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **C1:** T-all-K minus B0-K between −0.06 and +0.01 in both years. T2-K carries most of any gain,
  in November to March and the morning and evening bands, and its unfiltered gain is at least twice
  its filtered gain. T1-K and T3-K are each within ±0.02, filtered or not. Gate 2 passes. No
  capacity move is flagged.
- **N4 gate:** `not identified`, failing (b) with a median width over 3 °C. B0's τ at 15.5 is 10 to
  40. No real site passes the per-site printing rule, the profiles are within 2 effective nats of
  their minimum over at least 3 °C at 4 of 5 sites, and the arg-min sits at a grid edge at 1 or
  more sites.
- **C2:** H-site-K minus B0-K between −0.04 and +0.03; replicate same sign at 3 of 5 sites at most.
- **B0-K** within 0.06 points of Part 4's WK (2.267) in the scored year; run under 1.5 hours.

## Decision rule

**Each question gets its own label, in the skill's words; the verdict counts the labels.**

- **N3 (C1) and N4 (C2, if gated in):** `gain` if the estimate is at most −0.03 points, the 97.5%
  interval lies wholly below zero, the replicate year's estimate is below zero, and at least 4 of 5
  sites improve in the scored year. `small gain` if the interval lies wholly below zero and the
  estimate is between −0.03 and zero. `not replicated` if the interval and estimate meet `gain` but
  the replicate sign or the site count fails. `worse` if the interval lies wholly above zero; `tie`
  otherwise. The replicate condition is required because λ and every filter's hyperparameters are
  chosen on the tuning year.
- **N4 with a failed gate:** label `not identified`; C2 is printed without a label.
- **Verdict:** `supported` if both questions get `gain`; `partly supported` if one does; `small
  gain` if neither does and one gets `small gain`; `refuted` if none of those and a decision
  contrast is `worse`; `inconclusive` otherwise, including a blocking test that cannot be fixed.
  `small gain` follows the skill's `candidate` flag; `not replicated`, `tie`, and `not identified`
  count as not passing.
- **Handed on:** a `gain` arm becomes a candidate for GAM-K's base only after a confirmation run on
  ENS with Part 6's monthly refits, which needs its own plan and approval. A `not identified` gate
  keeps the threshold at 15.5 °C. A C2 gain validates no site's θ. Combining T-all with W1's
  shape state is a later follow-up, not part of this plan.

## Audit before merge: the seven triggers

**Run the `ml-experiment` skill's seven triggers on the results, and add the "Checks" bullets.**
Trigger 7 fires on any `gain`, so an Opus audit is planned for that case.

1. **A control fails:** test 1, gate 2, or the N4 placebo. Gate 3 failing is a pre-registered
   outcome, not a failed control, unless its placebo also fails.
2. **Contradiction:** C1 gains more than 0.10 points after the filter (Part 3's network gave −0.03);
   gate 3 passes, or a real site passes the printing rule, although N5 found slopes weak; B0-K
   misses WK by more than 0.10.
3. **Score and mean-forecast error disagree in sign** for C1 or C2.
4. **Leaving one site out** flips the sign of C1 or C2, or changes a label.
5. **A nested arm wins, or a null control moves:** B0 (the λ → ∞ limit) beats T-all on the tuning
   year by more than T-all's paired half-width; θ = 15.5 moves the score at all; T-all-K-b0 moves C1
   by more than its half-width.
6. **Interval width:** a decision contrast's 97.5% interval more than twice or under half the width
   of Part 4's C4 at the same level (about ±0.023).
7. **A later design depends on it:** a `gain` would change GAM-K's base, and the result is one year
   plus a replicate, at the screening tier.

**Checks bullets:** median minus mean per site (after the filter); mean-forecast error per contrast.

## Substation cards

**Section label: "Provisional: fitted to past weather on a screening run".** One card per decision
substation (D5, D10, D11, D14, D16; D9 marked short-history), from T-all-K and B0-K at the
2025-07-01 cut, with the series brief's panels:

- **Daily profile** per day type; **holiday and Christmas effects**; **lighting** by day-type group
  (T-all).
- **Heating:** the N4 profile curve (effective nats against θ, 12 to 19 °C) with a mark at 15.5, the
  gate 3 label, and the site's θ only under the printing rule; B0's band slopes with the label
  "weak", and their I1 interval only if gate 3's τ is within a factor of 2 of N5's 27; T-all's
  weekday and weekend slopes per sub-band block; the thermal-inertia mix across the four smoothings.
- **Solar:** the pinned vintage capacity, with T-all-fc's and B0-fc's latest capacity beside it and
  no interval.
- **Filter:** B0-K's and T-all-K's level and transient state over the scored year.
- **One cross-substation figure:** profile minima and flat ranges, weekend-to-weekday heating ratio,
  and the capacity move.

## Run order, compute, and time

**Order:** scripts and test 1; Gram matrices (5 sites × 4 cuts, about 500 columns); tuning, hashed
into `decisions.json`; gates 2 and 3; fits, spreads, filters; scoring and write-up. A bounded solve
of about 500 columns takes roughly 0.05 to
0.1 seconds, and `solve_bounded_quadratic` batches a leading dimension of weightings. A synthetic
draw changes only `X^T W y`, so the Gram matrices are built once per site and cut. Real fits: 5
sites × 4 cuts × (B0 + 15 θ + 4 N3 arms × 3 λ + 2 capacity fits) is about 600 solves, plus 50
weightings × 15 θ × 5 sites for the profiles and 500 for the capacity check: a few minutes. Gate 3:
120 draws × 15 θ × 51 weightings is about 92,000 solves, about 30 minutes; gate 2: about 1,500
solves. Gram building about 10 minutes; filters seconds per arm and site; scoring minutes. **In all
about 1 hour of wall clock, budgeted at 2 hours.** **If the grid projects over 2 hours, cut in this
order:** gate 3's weightings from 50 to 30, T1 to T3 to the tuned λ only, the capacity check.

## What the write-up must contain (beyond the series brief's standing rules)

The take-home says whether interaction terms or a per-site threshold improve GAM-K's convex model
after its filter, by how much, whether a threshold can be read off a fit, that every number is a
past-weather ceiling, and why N13 was dropped.

- **Figures:** the headline chart of C1 and C2 with intervals, site counts, the replicate, and the
  unfiltered contrast (first); absolute scores per site with the references; C1 by month and band;
  threshold profiles; gate 3's recovered against true θ; gate 2's power; the tails table.
- **Privacy:** D-labels only, with dates. D15, D18, and their sum are generator data, labelled G-c
  with day counts, never dates or a D-label; none is fitted, so G-c should appear nowhere. Never
  print experiment 45's Primary names, nor a generator's public name beside its series.
- **Caveats:** past weather; one scored year plus a replicate, frozen fits; in-family truths cannot
  show a wrong PV shape; hindsight masks. **Leaks (Part 4's list; none reaches a scored row):** the
  2022-07-01 replica, which
  sets the tuning-year spread that selects λ and the filters, reads the 2023-08-07 capacity vintage,
  13 months late; the 2023-07-01 cut has the 5-week vintage leak; D9's 2024-07-01 fit (its
  scored-year replica) reads the 2025-04-14 vintage; the replicate-year filter spread is in sample
  for the filter hyperparameters.
