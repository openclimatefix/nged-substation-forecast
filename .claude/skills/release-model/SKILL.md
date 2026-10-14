---
name: release-model
description: >-
  Load before choosing which ml-experiment winners go into the next numbered model release of the
  Flexpectation production forecast, before writing a release plan, and before registering a
  `model-release-N` cross-validation run. The routine: list the candidate experiments, test the
  chosen winners together in one combination experiment, plan and implement the port, run the full
  cross-validation gate, release, and watch the first runs. A change to production that no
  experiment decided is an ordinary issue; use `plan-issue` for it.
---

# Cutting a model release

**A model release is a numbered, trained XGBoost model that the maintainer ships to the live
service.** `docs/live_service/model-releases.md` defines the number, the log, and the deploy steps.
This skill covers the work that comes before the deploy: choosing the experiments, proving that
they work together, and porting them.

**Each step ends in one output, and the next step does not start without it.** Steps 2, 5, and 6
stop for the maintainer's decision. Every other step runs autonomously, except that the maintainer
picks the candidates in step 0.

**Only one session runs the fits of steps 1 and 4 at a time.** Ask the ML experiment manager for a
resource slot (the file `data/ml-experiments/SLOTS.md` lists the slots), and check the CPU and GPU
load before a fit.

## Step 0: write the agenda

**List every experiment now in the "Candidates for the next release" section of
`docs/ml-experiments/research-log.md`.** An adopted experiment leaves that section, so the section
holds exactly the experiments not yet released, including winners that an earlier release deferred.
For each candidate, give the verdict, the headline number with its interval, and the production
setting it would change. Mark candidates that change the same setting, because only one of them can
be ported.

**Exclude a candidate that needs a design production does not have yet.** The staged design of
experiments 00001 and 00002 is an example. Such a candidate goes through `plan-issue` as its own
build, and it joins a later release.

The maintainer picks the candidates for the release. The output is that list, written into the
planned `result.md` of the combination experiment in step 1.

## Step 1: test the winners together

**Run one ml-experiment that asks whether the chosen winners together beat the current production
recipe, and whether any pair of winners interferes.** Size the experiment and write its planned
`result.md` first, as the `ml-experiment` skill says. Experiments test one idea against a fixed
benchmark, and ideas that win alone can cancel each other out. The arms are:

- the current production recipe;
- all chosen winners together;
- "all winners minus one", once for each winner. An arm that beats "all winners together" shows that
  the missing winner interferes.

**Build the training and scoring frames in the shape production uses,** so that the result predicts
what cross-validation will show. Report the continuous ranked probability score (CRPS), the mean
absolute error (MAE), and the coverage of the prediction intervals per lead band. Split every score
by series type: Primary substations, bulk supply points (BSP), and grid supply points (GSP). Primary
substations are the headline, because they are the series that the network operator, National Grid
Electricity Distribution (NGED), most wants forecast.

**A winner that fails in combination is dropped from the release, or deferred.** The release plan
lists each dropped winner with its reason, and the maintainer reviews that list when approving the
plan. The combination experiment sets its own `production_status` to `not-for-production`, because
the experiment is a diagnostic, so it never appears as a candidate. The output is the final release
set and the combined effect size.

## Step 2: write the release plan

**Create one tracking issue for the release and load `plan-issue` against it.** Size the release
with the five triggers that `plan-issue` lists, and write one line for each. The plan:

- splits the port into small PRs by area: features, training and loss, calibration, and serving;
- orders the PRs by dependency;
- lists each PR's tests;
- names every PR that changes a data contract or the serving path, because the maintainer approves
  those before any code is written;
- records the dropped winners from step 1;
- states the gate tolerances of step 4 and says whether the release retrains every series, which
  decides whether the release needs to be a release of its own.

**The maintainer approves the plan, including the tolerances.** The output is an approved plan and
one sub-issue per PR.

## Step 3: implement the PRs

**Load `implement-issue` and implement each PR in the plan.** Re-implement each winner in the
reviewed packages. Never copy an experiment's script into `src/` or `packages/`, because experiment
code is unreviewed and production never imports it. No PR in this step bumps the model release
number. The output is the merged PRs.

## Step 4: run the full cross-validation gate

**Register the cross-validation run `model-release-N`, with `model_release: N` in its
`config_overrides`, and compare the result with the previous release and the baselines
(`manual_heuristic` and `climatology`).** N is the next unused release number. The first release is
N = 1, and it is compared with a cross-validation run of the current recipe. The gate has four
parts:

- the run reproduces the combination experiment's predictions on its row set, within the tolerance
  in the plan;
- the release is no worse than the previous release on any series type, beyond the tolerance in the
  plan;
- the release meets the combination experiment's predicted improvement;
- the diagnostics are acceptable: calibration and coverage by lead band, behaviour when an input is
  missing, and the series that got worse.

**A failed gate has two routes.** A mismatch in the reproduction check is a porting bug, so the
fix is a new PR under step 3. A regression or a missed target goes to the maintainer in step 5.
Primary substations are the headline, as in step 1. Ask the maintainer before spending more than the
£30 threshold that the `ml-experiment` skill sets. The output is the gate result in the tracking
issue.

## Step 5: the maintainer decides

**The maintainer reads the gate result and chooses to release or to return to step 1.** The output
is that decision, recorded in the tracking issue.

## Step 6: release

**The maintainer promotes and deploys the release, and one PR then records it.** The order is the one
in `docs/live_service/model-releases.md`: promote the champion fold run, deploy it, and check that
the `promoted_model` asset reports `model_release: N`. The PR that records the release adds the row
to the log with the promoted run's identifier and image tag and the rollback pairing, and the
maintainer tags the commit `model-N`. The first release has no earlier release to roll back to.
The same PR sets `production_status: adopted` and `adopted_in: N` in each incorporated experiment's
`result.md`, moves those experiments in the research log, and updates the flag of each winner that
step 1 dropped. The output is the deployed release.

## Step 7: watch the first runs

**Check the live forecasts and the degradation records of the new release for its first two days of
runs.** The live-service guide `docs/live_service/operations.md` says where to read them. Note any
degradation record, or any forecast that differs from the combination experiment's predictions, in
the tracking issue, which then closes.
