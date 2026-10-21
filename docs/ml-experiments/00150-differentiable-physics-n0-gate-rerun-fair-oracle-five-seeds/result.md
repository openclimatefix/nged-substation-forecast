---
id: "00150"
slug: differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [hierarchy-and-pooling]
status: done
size: >-
  medium. Stored: private files under
  data/ml-experiments/00150-differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds/ (a copy of
  experiment 109's feature cache, synthetic fleets, and tuning grids); no contract or production
  table. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design
  per question, fixed in this plan before any run (a fair oracle and a count of seeds). Callers: this
  experiment's scripts only; the gate code is copied from experiment 109 and the pooled fit it calls
  lives in packages/studies/ and is not changed.
hypothesis: >-
  When the reference estimator estimates every coefficient from the same 4 weeks of rows (so that no
  coefficient is fixed at its true value), the partially pooled convex model of experiment 109
  recovers the injected truth about as well as that reference on most of 5 random seeds, and its
  tuning rule picks a deviation penalty within one grid step of the best one on most of the 5 seeds.
prediction: >-
  The pooled fit passes the shrinkage line (b) on 4 or 5 of 5 seeds against a fair oracle, because
  experiment 109's oracle was better only through knowing the true capacity. The tuning rule is
  within one grid step of the best penalty on 3 or 4 of 5 seeds: the real grid has a minimum at
  3,000 rows and the synthetic grids had it at 300 to 1,000, which suggests the rule is sensitive to
  noise.
decision_rule: >-
  The gate is passed when, on at least 4 of the 5 seeds, lines (a), (b), (c), and (e) all pass, and
  both controls fail line (b) on at least 4 of the 5 seeds (so that line (b) can detect both
  over-shrinkage and under-shrinkage). The tuning rule is stable when line (d) passes on at least 4
  of the 5 seeds. Verdict `supported` if the gate is passed and the tuning rule is stable; `partly
  supported` if exactly one of the two holds; `not supported` if neither holds; `inconclusive` if a
  control passes line (b) on 2 or more seeds, because then the line cannot tell a good fit from a
  bad one. Smallest result worth acting on: none beyond these counts, because the gate decides
  whether later experiments may start from the pooled model.
benchmark: >-
  A fair oracle: the posterior mean of a Gaussian model that knows the fleet mean of every pooled
  coefficient, each family's true spread, and the noise variance, and estimates every coefficient
  including the capacity from the same rows. Controls: the pooled fit with no pooling (penalty 0)
  and with complete pooling (penalty 1,000,000 rows). Experiment 109's five gate runs are the
  earlier record.
row_set_id: >-
  not applicable (synthetic fleets on the real calendar and weather of experiment 109's MW
  Primaries; no scored forecast rows)
metric: >-
  Gate lines (a) to (e) as defined under "Pre-registered lines", counted over seeds
verdict: inconclusive
production_status: not-for-production
adopted_in: none
date: 2026-10-21
pr: 1452
take_home_message: >-
  The gate rerun could not tell a good pooled fit from a bad one in the pre-registered form: with 4
  June weeks the heating signal is so weak that complete pooling passed the shrinkage line on 2 of 5
  seeds. The pooled fit's other results were good: it recovered capacity within 25% at 8 of 8
  substations on every seed, the tuning rule landed within one grid step of the best penalty on 5 of
  5 seeds, and its heating error was smaller than the fair oracle's on 5 of 5 seeds. The next step
  is a gate whose shrinkage line measures error against the truth, not a slope against an oracle
  that treats the noise as independent.
advice:
  - >-
    Do not use a slope against an oracle with independent noise as a shrinkage test for the pooled
    model: with 4 weeks of data the fair oracle's heating error was 2.4 to 4.1 against 1.9 to 2.5
    for the pooled fit in January, so the oracle is the worse estimator (post hoc January rerun).
  - >-
    Hold 4 winter weeks, not 4 June weeks, when a gate tests heating shrinkage: in June complete
    pooling passed line (b) on 2 of 5 seeds, in January it failed on 5 of 5 (line (b) controls).
  - >-
    Start later experiments from the pooled convex model N with its tuning rule: capacity from 4
    weeks was within 25% at 8 of 8 substations on 5 of 5 seeds, and the rule was within one grid
    step of the best penalty on 5 of 5 seeds (lines (e) and (d)).
  - >-
    Keep the 0.85 bar for line (c) only if the gate is rerun with it justified: the full-history
    slope was 0.82 to 0.94 and passed on 2 of 5 seeds, so the line as set sits at the noise level
    (line (c)).
code: ml-experiments/00150-differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds/
---

# Differentiable physics, N0: rerunning experiment 109's known-answer gate with a fair oracle and five seeds

**The rerun could not say whether the shrinkage of the pooled model is right, because the test line
that measures shrinkage could not tell a good fit from a bad one; the rest of the pooled model's
results were good.** With 4 June weeks of data a substation's heating signal is so weak that
complete pooling (no per-substation deviation at all) passed the shrinkage line on 2 of 5 seeds. The
pooled fit recovered capacity within 25% at 8 of 8 substations on every seed, the tuning rule chose a
penalty within one grid step of the best on 5 of 5 seeds, and the pooled fit's heating error was
smaller than the fair oracle's on 5 of 5 seeds. The verdict by the pre-registered rule is
`inconclusive`. The next gate should measure the heating error against the injected truth and hold
winter weeks.

**This experiment asks whether the partially pooled convex model of
[experiment 109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md) finds
the answer when we know it.** The model gives each substation fleet-shared coefficients plus a
penalised deviation. Experiment 109's known-answer test (the "gate") built 8 synthetic substations
with injected coefficients, and failed under its pre-registered definition. Four redefinitions
followed on four new seeds, so a change of definition and a change of seed were confounded. Two
faults made that gate unfair. The reference estimator (the "oracle") was given each substation's true
capacity, which the pooled fit had to estimate from the same 4 weeks, and the tuning rule was checked
on one seed per definition. Here the oracle estimates every coefficient, the capacity included, and
the gate runs 5 seeds under one definition. The gate code is copied from
[`e00109_gate.py`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/ml-experiments/00109-differentiable-physics-n0-partially-pooled-convex-model/e00109_gate.py).

![The pooled fit has a smaller heating error than the fair oracle on all 5 seeds](figure_01_heating_error.svg)

![In January the pooled fit shrinks heating deviations harder than the oracle](figure_02_shrinkage_slope.svg)

![The tuning rule is within one grid step of the best penalty on 5 of 5 seeds](figure_03_tuning.svg)

## Results

Every number below is printed in [`report.txt`](report.txt).

| Line (pre-registered) | Seeds passed | Detail |
|---|---|---|
| (a) fleet mean | 4 of 5 | Seed 1 missed by 0.02 points (profile 0.518 against 0.5). |
| (b) shrinkage against the fair oracle | 2 of 5 | Passes on seeds 4 and 5, where the oracle's slope is near 0. |
| controls for (b) both fail | 3 of 5 | Complete pooling passed on seeds 4 and 5; no pooling never passed. |
| (c) full history keeps its own coefficients | 2 of 5 | Slopes 0.82 to 0.94 against the 0.85 bar. |
| (d) tuning within one grid step | 5 of 5 | Exactly the best penalty on 3 of 5 seeds. |
| (e) capacity from a short window | 5 of 5 | 8 of 8 within 25% at 4 weeks and at 3 months on every seed. |

- **Verdict rule as pre-registered.** The gate is not passed (line (b) passed on 2 of 5 seeds, below
  the 4 needed). The tuning rule is stable. Complete pooling passed line (b) on 2 seeds, so the
  rule's `inconclusive` clause applies.
- **The oracle is the worse heating estimator in January and in June.** The oracle's heating error
  was 2.35 to 3.45 against 2.01 to 2.38 for the pooled fit in June, and 2.30 to 4.06 against 1.87 to
  2.46 in January. The oracle treats the noise as independent, but the noise keeps multi-day
  persistence, so the oracle trusts four weeks of evidence too much.
- **Line (b) fails by design in June.** The slope half asks the pooled fit to match the oracle's
  slope. The oracle's slope was 0.35, 0.43, 0.42, -0.10, and 0.08 on the 5 seeds, so a fit that
  keeps nothing (slope 0) is within 0.15 of the oracle on seeds 4 and 5.
- **Capacity is estimated well.** The fair oracle's fitted capacity was 0.90 to 1.20 of the truth
  (report section "Fair oracle capacity ratios").
- **Tuning.** The rule picked 1,000 rows on 4 seeds and 300 on seed 4. The best penalty on the
  noise-free truth was 1,000, 300, 1,000, 100, and 1,000.

## Post hoc: line (b) with 4 January weeks

**This rerun is labelled post hoc and changes no pre-registered verdict.** It holds the 4 weeks
before 2025-02-01 on the same 5 fleets, with the seed's tuned penalty.

- **The controls failed on 5 of 5 seeds**, so with winter weeks line (b) can detect both no pooling
  (error 4.3 to 8.4) and complete pooling (error 2.1 to 2.6).
- **The pooled fit failed line (b) on 5 of 5 seeds, on the slope half only.** Its slope was 0.20 to
  0.50 against the oracle's 0.63 to 1.06, while its error (1.87 to 2.46) was below the oracle's (2.30
  to 4.06). The pooled fit shrinks harder than the oracle and is more accurate, which fits the
  oracle's independent-noise assumption being wrong.

## Checks

- **Median forecast minus mean forecast, per site.** Not applicable: the gate scores coefficients and
  noise-free truth, not forecast spreads.
- **Error of the mean forecast beside each contrast.** The tuning check scores the mean forecast
  against the noise-free truth (line (d)); the quantile-score rule gave the same penalty as the
  truth-based rule on 3 of 5 seeds. Tuning on the median's absolute error instead picked 1,000, 300,
  300, 300, and 1,000, which agrees with the quantile-score rule on 3 of 5 seeds.
- **Audit triggers that fired.** Trigger 1 (a known-answer test fails), trigger 2 (the result
  contradicts the prediction that line (b) passes on 4 or 5 seeds; explained above), and trigger 7
  (the pooled experiments listed in the theme's plan depend on this gate). Triggers 3 to 6 do not
  apply, because the gate has no per-site forecast contrast. A Sonnet audit ran in place of an Opus
  audit, as the brief required; its findings and their triage are under "Review".

## Pre-registered lines

- **Synthetic fleet.** As in experiment 109: eight synthetic substations on the real calendar and
  weather of the MW Primaries, a known fleet mean and known deviations, noise taken as whole weeks of
  residuals from a different substation. Each of seeds 20261101 to 20261105 builds a new fleet.
- **(a) Fleet mean.** The fitted heating slopes (summed over the four smoothing windows) are within
  10% of the true fleet mean in every hour band, and the centred profile is within 0.5 points root
  mean square. Unchanged.
- **(b) Shrinkage on 4 June weeks, against the fair oracle.** The fair oracle is the posterior mean
  of the Gaussian model with the realised fleet mean, the true spread of each family, and the true
  noise variance, and a flat prior on the capacity. It estimates every coefficient jointly from the
  held-out substation's 4 weeks. Line (b) passes when both hold: the slope of the pooled fit's
  deviation on the true deviation, measured on the heating contribution to the prediction, is within
  0.15 of the oracle's slope, and the heating root mean square error across the eight substations is
  at most 1.2 times the oracle's.
- **Controls for (b).** The same line, with the pooled fit's penalty set to 0 rows (no pooling) and
  to 1,000,000 rows (complete pooling). Both must fail line (b); if either passes, the line cannot
  detect a bad fit.
- **(c) Full history keeps its own coefficients.** With 2 years of rows the slope of the fitted
  deviation on the true deviation is at least 0.85. Unchanged.
- **(d) Tuning rule picks the best penalty within one grid step.** The rule is experiment 109's
  (the penalty minimising the mean of the 4-week and 3-month score ratios against no pooling, subject
  to the all-rows score within 0.02 points). The best penalty is found by applying the same rule to
  the root mean square error of the mean forecast against the noise-free injected truth, on the
  same rows. Line (d) passes when the rule's penalty is within one step of the best on
  `common.LAMBDA_GRID` (nine values from 0 to 1,000,000). Experiment 109's line (d), which
  multiplied every spread by 10 or by 0.1, is dropped, and the two extra fleets that it needed are
  not built.
- **(e) Capacity from a short window.** Unchanged: the 4-week and 3-month pooled fits recover the
  held-out substation's capacity within 25% at 6 or more of 8 substations.
- **Reported, not blocking.** The heating-penalty sweep of experiment 109 (now including the two
  control values), the fair oracle's capacity ratio, and the error of each pooled family against the
  oracle.

## Predictions against outcome

- **Predicted: line (b) passes on 4 or 5 of 5 seeds.** It passed on 2. The prediction assumed the
  oracle's capacity advantage was the only fault; the oracle's independent-noise assumption was a
  second one.
- **Predicted: the tuning rule within one grid step on 3 or 4 of 5 seeds.** It was within one step on
  5 of 5.

## Caveats

- **The pooled fit is given the true capacity of the seven full-history substations**, because the
  synthetic vintages equal the truth (inherited from experiment 109). Lines (a), (c), and the tuning
  feasibility rule run with capacity known; lines (b) and (e) estimate the held-out substation's
  capacity.
- **The oracle's prior mean is the mean of all 8 true coefficient vectors**, including the held-out
  substation's own, which slightly flatters the oracle and biases against passing line (b).
- **The 5 fleets share the real calendar, weather, and noise weeks**, and differ only in the
  injected coefficients, so they are not independent replicates of the noise.
- **Schedule.** One refit per fit at the 2025-07-01 cut (2025-02-01 for the post hoc rerun), so the
  contrasts do not depend on a retraining schedule.
- **Torch.** `e00150_tune.py` imports torch inside its worker function and is on the `ty` torch
  override list.

## What to try next

**Replace line (b) by the error of the pooled fit against the injected truth, held on 4 winter
weeks, with the complete- and no-pooling errors as controls.** The pooled fit's heating error
(1.87 to 2.46) beat complete pooling (2.1 to 2.6) on 4 of 5 seeds and no pooling (4.3 to 8.4) on 5 of
5 in January, which such a line would test directly.
