---
id: "00109"
slug: differentiable-physics-n0-partially-pooled-convex-model
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [differentiable-physics-demand-calendar-and-weather, hierarchy-and-pooling]
status: planned
size: >-
  complex for the module (a new joint convex fit across substations in packages/studies, reused by
  later pooled experiments); medium for the experiment scripts. See "Size triggers".
hypothesis: >-
  A convex model whose per-substation coefficients are a fleet mean plus a ridge-penalised
  per-substation deviation forecasts Primary demand as well as the per-substation convex model when
  both have 2 years of history, and clearly better when the substation has 4 weeks or 3 months.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  The per-substation convex model G on the same rows and capacity; complete pooling; lagless
  XGBoost on the same ERA5 and CAMS inputs and window, plain and with experiment 64's solar offset;
  experiment 105's stored pooled DP (P) and per-site DP (S); Part 2's G-all+D1; climatology;
  manual_heuristic; lagged XGBoost as a descriptive reference at leads of 24.5 to 48 hours
row_set_id: >-
  dp_mw_v1 (experiment 103); MW sites scored 2025-07-01 to 2026-06-30, replicate 2024-07-01 to
  2025-06-30
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored site
verdict: <supported, partly supported, refuted, or inconclusive>
production_status: not-for-production
take_home_message: <written after the results>
date: <date>
pr: <pr>
advice: <written after the results>
code: ml-experiments/00109-differentiable-physics-n0-partially-pooled-convex-model/
---

# Differentiable physics, N0: a partially pooled convex model across substations

**Status: planned.** This page is the plan, committed before any code runs. The write-up replaces it
when the experiment finishes.

## What N0 asks, and why the convex route

**N0 asks whether pooling works when the optimiser cannot get in the way.** Experiment 105's neural
pooled model (P) scored 0.385 points worse than its per-site neural model (S) on all rows (98.3%
interval +0.18 to +0.69, 5 of 5 sites worse), and S scored about 0.3 points worse than Part 2's
convex per-site model. Both results involve an optimisation problem: the pooled capacity's flat ridge
against the interaction network, and a neural fit losing to a convex one. A box-constrained penalised
least-squares fit has one optimum, and a pinned capacity removes the ridge.

**The model is Part 2's convex model, fitted jointly across substations.** For each substation `s` and
each pooled coefficient family `f`, the coefficient is `theta[s, f] = mu[f] + delta[s, f]`: a fleet
mean plus a deviation under a ridge penalty of weight `lambda`. At `lambda = 0` the fit is the
per-site convex model G exactly. Complete pooling ties the families to one value. The objective stays
a convex quadratic, so every `lambda` has one optimum. N0 reuses Part 3's contrasts C1 to C3, adds
partial against complete pooling for a new site, and uses Part 4's three frozen years.

## Size triggers

- **Stored data:** private files under `data/ml-experiments/00109-*/` only.
- **Serving path and degradation rules:** not touched.
- **More than one defensible design:** yes (pooled families, per-family scaling, short-history
  capacity and scale, `lambda` tuning). The plan pre-registers one of each.
- **Callers:** N0's scripts, and later pooled experiments.
- **Review:** the module gets its own pull request off `main` and two Opus code reviews, leakage
  first (the cut, the held-out site's rows, the scaling, every capacity read), before any scored fit.

## Module (`studies.dp_gam_pooled`)

**The joint fit is built from the per-site Gram matrices of `dp_gam.accumulate_gram`** (one pass over
each site's rows) plus the deviation penalty's coupling. `solve_bounded_quadratic` solves it over
`[mu, theta_1, ..., theta_S]`, batched over the `lambda` grid, in float64 on the CPU.

- **Parameterisation.** The fit carries `theta_s` and `mu` and penalises
  `lambda * sum_s sum_f g_f * ||C_f (theta[s, f] - mu[f])||^2`, so sign constraints stay box
  constraints (heating `theta_s >= 0`, fitted capacity increments `>= 0`). `mu` is unconstrained
  (ridge 1e-6). The base penalties act on each `theta_s` as in `dp_gam`. No column is added to
  `dp_gam`'s design, so `lambda = 0` is G to machine precision.
- **Pooled families:** the profile, holiday periods, Christmas tent, harmonics, heating, and
  lighting. `C_f` is the identity, except for the profile, where it removes each hour band's mean.
  The profile's shape pools and each site's band means stay free, so size never pools. The penalty is
  positive semi-definite, so the fit stays convex. Capacity and the band means are not pooled.
- **Complete pooling (N-full)** is exact tying: one `theta` per pooled family, band means free,
  built as its own design and never as a huge `lambda`, which would push the condition number near
  1e13.
- **Per-family scale `g_f`:** the mean diagonal of family `f`'s per-row Gram matrix over the fit's own
  training rows, so `lambda` reads as rows of evidence. A site with far more rows than `lambda` keeps
  its own coefficients, and one with far fewer is shrunk to the fleet mean.
- **Forecast and capacity.** Each `theta_s` becomes a `GamFit` for `dp_gam.predict_gam`. One fit mixes
  capacity modes per site: pinned, or fitted from that site's rows before the cut.
- **Spread.** Full-history sites use the centred replica spread of `dp_gam`, from the joint fit 12
  months earlier. A short-history arm's spread for site `s` uses only the residuals of each other
  decision site `t`, from the replica fits in which `t` was the short site with the same weeks of
  history, as `dp_onboarding.cross_fitted_spread` enforces. G-4w and G-3m use the same rule.
- **Tests (each under 5 minutes):**
    1. `lambda = 0` equals `fit_gam` to 1e-8.
    2. With no bound active, the joint solution equals a dense NumPy generalised ridge to 1e-8.
    3. The tied fit equals a separately built complete-pooling fit, and `lambda = 1e6` approaches it
       (gap under 1% of `lambda = 1e4`'s).
    4. A held-out site's readings outside its window change nothing.
    5. Leakage asserts as in Parts 2 to 4, plus `g_f` from training rows only, no late vintage read,
       and no scale from unseen rows.
    6. A 7-site, 9-`lambda` fit takes under 2 minutes on 6 cores.

## Years, sites, and rows

**Three years, each with its own frozen cut, as in Part 4:**

| Year | Cut (training ends) | Role |
|---|---|---|
| 2023-07-01 to 2024-06-30 | 2023-07-01 | tuning: `lambda`, the short-history arms' ridge, the XGBoost bar |
| 2024-07-01 to 2025-06-30 | 2024-07-01 | replicate: every decision contrast, out of sample for `lambda` |
| 2025-07-01 to 2026-06-30 | 2025-07-01 | scored: the verdict |

The tuning year's replica fit (2022-07-01 cut) holds only about 11.5 months of usable rows, because
the lockdown runs to 2021-07-19. The write-up therefore also reports `lambda` tuned on the median's
mean absolute error, which needs no spread.

- **Pool:** the MW Primaries D5, D9, D10, D11, D14, D16, and D18 with `default_rows == True`
  (lockdown out, the 66 hand-filtered transfers masked ±24 hours; the masks are hindsight labels, a
  mild leak shared by every arm). D9 and D18 enter each cut only with the rows they hold.
- **Window:** 2 years before each cut, no drift, for every arm, so pooling is measured apart from
  years and drift.
- **Decision sites:** D5, D10, D11, D14, and D16 (4 of 5 needed). D9 is reported beside them. D18 is
  descriptive, as the one real short-history site.
- **Predictions written:** `time_series_id`, `valid_time`, `power_fcst_init_time` (09:00 UTC on the
  previous day), `q01`, `q05` to `q95`, `q99`, `actual`, `lead_hours`, `arm`, `seed`, and for the
  short-history arms `held_out` and `weeks_of_history`. The convex model has no lags, so its forecast
  does not depend on lead; any pairing with a lead-banded arm (lagged XGBoost) is on valid time.
- **Robustness rows, for every contrast:** November to March against April to October; without D16;
  without D14; D10 with its two unlabelled switching windows dropped from scoring (2025-11-17 to
  2025-12-14 scored, 2024-11-08 to 2024-12-01 replicate); the all-rows score beside the masked
  headline.

**Capacity is pinned by Part 4's rule, except where the vintage is late.** `studies.dp_pinned_capacity`
at each cut (`frozen` vintage, 13-week median held after the cut, D14's unreadable weeks held). A late
vintage would not stay at its own site: in the pooled fit it would shift that site's `theta_s`, hence
`mu`, hence every decision site's forecast, a leak in the pooled model's favour. So wherever the rule
falls back to a vintage fitted after the cut (more than `EARLY_VINTAGE_MAX_GAP_DAYS` late), the site's
capacity is fitted in the joint fit from its rows before the cut, unpooled, in N and G alike. The
cases, all disclosed in the write-up: D9 at the 2024-07-01 cut (9.5 months late); D18 at the
2025-07-01 cut (vintages after the scored year) and any earlier cut where it has rows; every site at
the 2022-07-01 tuning-replica cut (13 months late). The 2023-07-01 cut uses the earliest vintage as it
is, a disclosed 5-week leak, as Part 4 does.

**Effective capacity, the scale of "points",** is one constant per series in experiment 103's
`sites.parquet`, built from the whole history. For a short-history arm the held-out site's scale is
rebuilt by the same definition from the rows the arm sees, in N and G alike. If that cannot be done,
the write-up discloses the hindsight scale, which favours N.

## Arms

**Every arm is screening tier (one frozen fit per cut, one seed).** All rows (C1):

| Arm | Model | Notes |
|---|---|---|
| G | per-site convex, `lambda = 0` | the comparator, fitted with `fit_gam` |
| N | partially pooled, `lambda = lambda*` | the decision arm |
| N-full | complete pooling, exact tying, band means free | exploratory |
| N-2λ | separate `lambda` for heating and for the other families, 3 × 3 grid | exploratory |
| N-ml | `lambda` by marginal likelihood on the training rows | descriptive; autocorrelated residuals should make this `lambda` too small |

**Short history (C2, C3, and C5), one held-out site at a time.** For each of D5, D9, D10, D11, D14,
and D16, the site keeps only its last 4 weeks (2025-06-03 to 2025-06-30 at the scored cut, the same
calendar weeks before the other cuts) or its last 3 months (April to June), and the other sites keep
their 2 years. The convex joint fit is exact and cheap, so N0 refits the whole pool with the short
site inside. This does not transfer to a frozen-module onboarding, and the write-up says so.

| Arm | Held-out site's history | Model |
|---|---|---|
| N-4w, G-4w, N-full-4w | 4 weeks | partial pooling at `lambda*`, per-site, complete pooling |
| N-3m, G-3m | 3 months | partial pooling at `lambda*`, per-site |
| N-{2, 8, 26}w, G-{2, 8, 26}w | 2, 8, 26 weeks | the onboarding curve (exploratory) |
| N-4w-pin, G-4w-pin | 4 weeks | the vintage capacity, an oracle leak (exploratory) |
| N-4w-cappool | 4 weeks | capacity shrunk towards the median of the other sites' pinned capacities at the cut (exploratory) |
| N-win4w, G-win4w | 4 weeks ending 2025-01-31 | winter onboarding: one extra cut at 2025-02-01, scored to 2026-01-31, other sites 2 years before it, `lambda*` unchanged (exploratory) |
| X-4w, X-3m | 4 weeks, 3 months | per-site lagless XGBoost (descriptive; experiment 105's `x_4w`, `x_3m` for the scored year) |
| X-pool-4w | 4 weeks | one site-agnostic lagless XGBoost on every other site plus the held-out site's 4 weeks, in points (descriptive) |

**The short-history site's capacity is fitted from its own short window, in both arms,** because its
vintage was fitted on its whole history. The `-pin` pair measures that leak. N-4w-cappool tests a
fleet prior on capacity, and no decision rests on it. The winter cut tests transfer of the solar term.

**G-4w and G-3m get their own ridge,** chosen on the tuning year from {0.1, 1, 10, 100}, so the
per-site comparator is not handicapped by a ridge set for 2 years of rows.

**XGBoost bar (CPU, lagless, experiment 87's Primary hyperparameters without the three power lags,
11 quantiles, Part 2's ERA5 and CAMS inputs, 2-year window).** X-2y (default start) and X-2yamq
(quantile start plus experiment 64's solar offset, built as in `e00104_xgb.py`, each site's weight
reported). The C4 bar is their tuning-year winner, recorded before any scored number. Experiment
105's `x_2y` and `x_2yamq` cover the scored year; the 2024-07-01 and 2023-07-01 fits are new. Part
4's X-best (12 months) is reported beside the bar.

**References:** experiment 105's stored P and S (scored year only); Part 2's G-all+D1;
`baseline_forecasters.climatology`; `manual_heuristic`; lagged XGBoost at 24.5 to 48 hours (Part 3's
`l_2y_d1`, scored year only, paired on valid time).

**Tuning `lambda` on 2023-24, fitted at the 2023-07-01 cut,** over `lambda` in {0, 100, 300, 1,000,
3,000, 10,000, 30,000, 100,000, 1,000,000} rows, each with its own replica spread. `lambda*`
minimises the mean of the 4-week and 3-month ratios, subject to N's all-rows score being within +0.02
points of G's, at D5, D10, D11, D14, and D16. Each ratio's denominator is N at `lambda = 0` with the
base ridge on the same rows, not G-4w or G-3m with their tuned ridge. The write-up reports each
score's own best `lambda` and the `lambda` tuned on the median's mean absolute error. Disagreement
signals misspecification.

## Known-answer gate (blocking, before any replicate-year or scored-year fit)

**A failed blocking test stops scoring until it is fixed.** If it cannot be fixed inside the
experiment, the verdict is `inconclusive`.

1. **A synthetic fleet with a known fleet mean and known deviations.** Eight synthetic sites on the
   real calendar and weather tensors of the MW Primaries, 2 years each, pinned synthetic capacity.
   The truth: `mu` is G's fleet-average fit at the 2025-07-01 cut, and `delta[s, f]` is normal with
   family scale `tau_f` set to G's between-site spread. The noise is whole weeks block-bootstrapped
   from G's training-window residuals at a different site, never replica or scored-year residuals, so
   it keeps the multi-day persistence that makes `lambda` hard to tune. A synthetic tuning year picks
   `lambda` by the real rule. Every line must pass:
    - **(a) Fleet mean:** the heating slopes of the fitted `mu` are within 10% of the mean of the 8
      sites' true `theta_s` in every band, and the profile shape is within 0.5 points root mean
      square.
    - **(b) Shrinkage on 4 June weeks:** the slope of the fitted deviations on the true deviations
      over heating is within 0.15 of the closed-form shrinkage factor (true `tau` and noise
      variance), and N-4w's heating root mean square error across sites is no worse than 1.2 times
      the oracle posterior mean's.
    - **(c) Full history keeps its own coefficients:** with 2 years, that slope is at least 0.85.
    - **(d) The tuning rule responds to the fleet:** with every `tau_f` ten times larger, the tuned
      `lambda` falls at least tenfold and N's all-rows score is within 0.02 points of G's; with every
      `tau_f` a tenth, the tuned `lambda` rises at least tenfold.
    - **Discrimination checks, reported (they pass by construction):** G-4w's heating error, and N-4w
      against G-4w with every `tau_f` a tenth.
2. **Capacity from a short window:** N-4w and N-3m recover the synthetic held-out site's capacity
   within 25% at 6 or more of 8 sites; G's ratios are reported. Synthetic solar output comes from
   CAMS, with no farm data.
3. **Module tests 1 to 6** pass.
4. **Reported, not blocking:** the real `lambda*` against the closed-form `lambda` from the real
   fleet's spread and residual variance; each real site's short-window capacity against its vintage,
   as ratios.

## Planned contrasts

Five planned contrasts, Bonferroni-corrected (99% intervals from `studies.bootstrap`, resampling
whole calendar months of the stored predictions, paired across arms), each with site counts, every
arm's absolute score, the mean-forecast error beside the quantile score, the replicate year's
estimate and count, and the robustness rows:

- **C1, pooling on all rows:** N minus G, non-inferiority with a margin of 0.10 points.
- **C2, onboarding on 4 weeks:** N-4w minus G-4w.
- **C3, short history of 3 months:** N-3m minus G-3m.
- **C4, against lagless XGBoost:** N minus the C4 bar.
- **C5, partial against complete pooling for a new site:** N-4w minus N-full-4w. C2 is nearly
  certain; C5 says whether `lambda` matters at all for onboarding.

**Exploratory at 95%:** N-full minus N; N-2λ and N-ml; N minus experiment 105's P and S (how much of
Part 3's loss the convex route recovers); N minus Part 2's G-all+D1 and minus Part 4's X-best; the
onboarding curve; the `-pin` pairs; N-4w-cappool minus N-4w; the winter cut; N-4w minus X-pool-4w; N
minus lagged XGBoost at 24.5 to 48 hours. **Descriptive (no interval where scarce):** D9; D18 under N
and G; bank holidays, the Christmas run, and days below 0 °C; the coastal sites D5 and D9.

## Predictions

Written before any fit; each line can fail, and none overrides the decision rule.

- **`lambda*`:** between 1,000 and 30,000 rows, so 4 weeks (1,344 rows) shrinks mostly to the fleet and
  2 years (about 35,000 rows) mostly keeps its own.
- **C1:** N minus G within −0.10 to +0.05 in both years, interval about ±0.05; D16 moves it by less
  than 0.05.
- **C2:** N-4w minus G-4w is −0.5 to −2.0, interval wholly below zero, 5 of 5 sites better in both
  years, and over two-thirds of the gain falls in November to March (4 June weeks hold no heating).
- **C3:** N-3m minus G-3m is −0.2 to −1.0, interval below zero, 4 of 5 sites better in both years.
- **C4:** N minus the bar is −0.1 to −0.4.
- **C5:** N-4w minus N-full-4w is −0.05 to −0.3.
- **Against Part 3:** N beats experiment 105's P by 0.3 to 0.6 points and S by 0.0 to 0.3.
- **Complete pooling on all rows:** N-full minus N is +0.2 to +0.8.
- **Onboarding:** N's gain over G halves by 26 weeks; N-win4w gains less than C2.
- **Tails:** N's q01 and q99 coverage within 1.5 points of nominal and q05 and q95 within 3; N-4w's
  upper-tail coverage nearer nominal than G-4w's in November to February.
- **Run time:** gate under 15 minutes; DP arms and tuning under 1 hour; XGBoost under 1.5 hours.

## Decision rule

**The verdict applies the series' pooling criterion under Phase A labels with a replicate check:**
pooling succeeds if C2 or C3 is `supported` and C1 is not `refuted`.

- **C2 or C3 `supported`:** estimate below zero in the scored year, at least 4 of 5 decision sites
  better, the replicate year's estimate also below zero, and the gate passed.
- **C1 uses the series' confirmation-tier reading, a deliberate departure from the screening rule.**
  N and G are nested convex fits with no seed noise, so the paired interval is about ±0.02 to ±0.05,
  and the screening rule would refute a +0.03 difference that sits well inside the margin. C1 passes
  if the upper end of its 99% interval is at most +0.10 in the scored year and the replicate year's
  estimate is at most +0.10. C1 is `refuted` only if the lower end exceeds +0.10 in the scored year.
- **`supported`:** C2 or C3 supported, and C1 passes.
- **`partly supported`:** C1 not refuted, and C2 or C3 below zero in the scored year but short of
  `supported` (sites or replicate), or supported while C1 fails to pass.
- **`refuted`:** C1 refuted, or C2 and C3 both with intervals wholly above zero.
- **`inconclusive`:** anything else, or a gate test that cannot be fixed inside the experiment.

C4 and C5 are labelled in the bullets, not in the `verdict` field.

**The smallest gain worth porting** is zero for C1 (a convex fit shared across sites adds no serving
cost over a per-site fit) and any interval wholly below zero for C2 and C3.

## Audit before merge: the seven triggers

The write-up's "Checks" bullets answer each trigger, and the session tells the manager if any fires.

1. **Control:** any blocking gate line, or N-full-4w failing to beat G-4w.
2. **Contradiction:** especially N failing C1 (pooling in points hurts even in convex form).
3. **Mean forecast:** each contrast's mean absolute error of the mean beside it; median minus mean per
   site and arm.
4. **Leave one site out** for C1 to C5, with D14 and D16 named.
5. **Nested arms:** N at `lambda*` must not lose to both G and N-full on the tuning year; the `-pin`
   pairs and N-ml are controls that should move little.
6. **Interval width** against experiment 105's C1 (+0.18 to +0.69) and Part 4's C4. A narrow C1
   interval is expected for nested convex fits, and the write-up says so.
7. **A later design depends on it:** a pooled convex model as the start of later pooled experiments.
   Trigger 7 fires by construction if a later experiment adopts N, so an Opus audit is planned.

## Run order, compute, and time

The module pull request comes first, with two Opus reviews, and merges before any real score. Then the
gate, tuning, every convex arm at the 2024-07-01 and 2025-07-01 cuts (each replica being the matching
fit one cut earlier) and the winter cut, XGBoost alongside, and the write-up. All work runs on the
CPU, with at most 6 cores (`nthread=6`; check `uptime` first, because other sessions run). The joint
system has about 2,400 columns, and dense float64 solves take seconds.

**Data paths:** Part 4's, plus experiment 105's `p`, `s`, `x_2y`, `x_2yamq`, `x_4w`, `x_3m`, and
`l_2y_d1`, and experiment 104's `g_all_d1`.

## What the write-up must contain

The take-home says in plain words whether one convex model shared across substations matches a model
per substation, how much it gains with 4 weeks or 3 months of history (both years, both seasons),
whether partial beats complete pooling for a new site, and how much of Part 3's loss came from the
neural fit. It glosses "pooled", "fleet mean", and "shrinkage".

- **Figures:** C1 to C5 with intervals, counts, and the replicate year; absolute scores per site; the
  onboarding curve; the `lambda` curves; the gate; a winter and a summer week per decision site
  (N-4w, G-4w); the fleet-mean heating curve with each site's; the tails at q01, q05, q95, q99.
- **Substation cards** (D-labels only, provisional, from N at the 2025-07-01 cut): each coefficient as
  fleet mean, site value, and how far `lambda` pulled it; capacity as the pinned vintage beside Part
  2's fit and the short-window fit; one cross-substation figure.
- **Caveats:** ERA5 and CAMS are estimates of past weather and flatter the solar term; the main
  4-week window is June, so C2 tests heating transfer more than solar; the masks are hindsight; the
  5-week vintage leak at the 2023-07-01 cut and the late-vintage cases fitted instead; the
  effective-capacity scale, if it could not be rebuilt; the joint refit is not a frozen-module
  onboarding.
- **Privacy:** D-labels with dates only; no generator data anywhere; G-a, G-b, and D15 out; the gate
  is synthetic.
