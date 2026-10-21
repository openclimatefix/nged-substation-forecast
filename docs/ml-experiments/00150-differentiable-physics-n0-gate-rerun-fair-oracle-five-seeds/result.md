---
id: "00150"
slug: differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds
theme: differentiable-physics-pooling-across-substations
also_relevant_to: [hierarchy-and-pooling]
status: planned
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
---

# Differentiable physics, N0: rerunning experiment 109's known-answer gate with a fair oracle and five seeds

**This experiment asks whether the partially pooled convex model of
[experiment 109](../00109-differentiable-physics-n0-partially-pooled-convex-model/result.md) finds
the answer when we know it.** The model gives each substation a fleet-shared set of coefficients plus a
penalised deviation. Experiment 109's known-answer test (the "gate") built eight synthetic
substations with injected coefficients and checked the fit against them. It failed under its
pre-registered definition, and four redefinitions followed on four new seeds, so a change of
definition and a change of seed were confounded.

**Two faults made that gate unfair, and this experiment removes both.** First, the reference
estimator (the "oracle") knew each substation's true capacity while the pooled fit had to estimate it
from the same 4 weeks, so the oracle beat the pooled fit on heating error by construction. The new
oracle estimates every coefficient, the capacity included. Second, the tuning rule was checked on
one seed per definition. The new gate runs 5 seeds under one definition and counts passes.

The method follows experiment 109's gate (see its page, and
[`e00109_gate.py`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/ml-experiments/00109-differentiable-physics-n0-partially-pooled-convex-model/e00109_gate.py)),
which uses the pooled fit of `studies.dp_gam_pooled` unchanged. Nothing here is tuned on real
substations' scored years.

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
  the root mean square error of the median forecast against the noise-free injected truth, on the
  same rows. Line (d) passes when the rule's penalty is within one step of the best on
  `common.LAMBDA_GRID` (nine values from 0 to 1,000,000). Experiment 109's line (d), which
  multiplied every spread by 10 or by 0.1, is dropped, and the two extra fleets that it needed are
  not built.
- **(e) Capacity from a short window.** Unchanged: the 4-week and 3-month pooled fits recover the
  held-out substation's capacity within 25% at 6 or more of 8 substations.
- **Reported, not blocking.** The heating-penalty sweep of experiment 109 (now including the two
  control values), the fair oracle's capacity ratio, and the error of each pooled family against the
  oracle.

## Runs

- **Compute.** CPU only, at most 8 cores with one thread each, inside `systemd-run --user --scope`
  with a 16 GB memory cap. Each seed builds one fleet, runs one tuning grid (9 penalties by 3
  histories, plus the per-site comparator's ridges), and runs the gate fits. Experiment 109's gate
  took 7 to 22 minutes per seed with three fleets; one fleet per seed should take under 10.
- **Schedule.** One refit at the 2025-07-01 cut per fit, as in experiment 109, so the result does not
  depend on a retraining schedule.
- **Anonymisation.** The fleets are synthetic and carry no generator data.
- **Torch.** The scripts that fit the pooled model import torch inside the worker function (as
  experiment 109's did); they are added to the `ty` torch override list.

## Predictions against outcome

To be filled in when the experiment finishes.
