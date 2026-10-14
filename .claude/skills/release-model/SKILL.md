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

**Each step ends in one output, and the next step does not start without it.** Steps 0, 2, and 5
stop for the maintainer's decision. Every other step runs autonomously. The maintainer sets the
three open decisions listed at the end before the step that uses each one.

## Step 0: write the agenda

**List the experiments flagged `candidate` in `docs/ml-experiments/research-log.md`.** For the first
release, take every candidate. For a later release, take the candidates added after the date in the
previous log row of `docs/live_service/model-releases.md`. For each candidate, give the verdict, the
headline number with its interval, the production setting it would change, and whether it needs a
new design before it can be ported. Mark candidates that change the same setting, because only one
of them can be ported.

The maintainer picks the candidates for the release. The output is that list, which the release plan
records.

## Step 1: test the winners together

**Run one ml-experiment that asks whether the chosen winners together beat the current production
recipe, and whether any pair of winners interferes.** Size it and write its planned `result.md`
first, as the `ml-experiment` skill says. Experiments test one idea against a fixed benchmark, and
ideas that win alone can cancel each other out. The arms are:

- the current production recipe;
- all chosen winners together;
- "all winners minus one", once for each winner. An arm that beats "all winners together" shows that
  the missing winner interferes.

Build the training and scoring frames in the shape production uses, so that the result predicts what
cross-validation will show. Report the continuous ranked probability score (CRPS), the mean absolute
error (MAE), and the coverage of the prediction intervals per lead band. Split every score by
series type: Primary substations, bulk supply points (BSP), and grid supply points (GSP). Primary
substations are the headline, because they are the series NGED forecasts most.

A winner that fails in combination is dropped from the release, or deferred. The release plan lists
each dropped winner with its reason, and the maintainer reviews that list when approving the plan.
The release PR sets the dropped experiment's `production_status` to `candidate` again or to
`not-for-production`. The output is the final release set and the combined effect size.

## Step 2: write the release plan

**Load `plan-issue` and size the release.** A release changes what is stored and runs through the
serving path, so the release usually sizes as complex. The plan:

- splits the port into small PRs by area: features, training and loss, calibration, and serving;
- orders the PRs by dependency;
- lists each PR's tests;
- names every PR that changes a data contract or the serving path, because the maintainer approves
  those before any code is written;
- records the dropped winners from step 1.

The output is an approved plan and a tracking issue with one sub-issue per PR.

## Step 3: implement the PRs

**Load `implement-issue` and implement each PR in the plan.** Re-implement each winner in the
reviewed packages. Never copy an experiment's script into `src/` or `packages/`, because experiment
code is unreviewed and production never imports it. No PR in this step bumps the model release
number. The output is the merged PRs.

## Step 4: run the full cross-validation gate

**Register the cross-validation run `model-release-N`, with `model_release: N` in its
`config_overrides`, and compare the result with the previous release and the baselines
(`manual_heuristic` and `climatology`).** The gate has four parts:

- the run reproduces the combination experiment's predictions on its row set, within a stated
  tolerance. A mismatch means that the port differs from what was tested, so the cause is fixed
  first;
- the release is no worse than the previous release on any series type, beyond a stated tolerance;
- the release meets the combination experiment's predicted improvement;
- the diagnostics are acceptable: calibration and coverage by lead band, behaviour when an input is
  missing, and the series that got worse.

Primary substations are the headline. Ask the maintainer before spending more than the £30 that the
`ml-experiment` skill sets as its threshold. The output is the gate result in the tracking issue.

## Step 5: the maintainer decides

**The maintainer reads the gate result and chooses to release or to return to step 1.** The output
is that decision, recorded in the tracking issue.

## Step 6: release

**One PR records the release, and the maintainer then deploys it.** The PR adds the row to the log
in `docs/live_service/model-releases.md`, with the rollback pairing, sets `production_status:
adopted` and `adopted_in: N` in each incorporated experiment's `result.md`, and moves those
experiments in the research log. The same PR updates the flag of each winner that step 1 dropped.
After that PR merges, the maintainer promotes the champion fold run, deploys it, and tags the commit
`model-N`, in the order the release-log page gives. The output is the deployed release.

## Step 7: watch the first runs

**Check the live forecasts and the degradation records of the new release for its first runs.** The
live-service guide `docs/live_service/operations.md` says where to read them. Note anything
unexpected in the tracking issue, which then closes.

## Decisions that belong to the maintainer

- **The gate tolerances** of step 4.
- **Whether a change that retrains every series needs a release of its own.**
- **The cadence.** The proposal is a release every one to two weeks when enough candidates have
  accumulated.
