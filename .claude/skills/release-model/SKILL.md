---
name: release-model
description: >-
  The routine for cutting a new numbered model release of the Flexpectation production forecast:
  collect the ml-experiment winners flagged as candidates, test them together in one combination
  experiment, plan and implement the port, check that the ported pipeline reproduces the
  experiment, run the full cross-validation gate, then release and watch. Load before choosing
  which experiment winners go into the next release, before writing a release plan, and before
  registering a `model-release-N` run. A single change to production that no experiment decided is
  an ordinary issue; use `plan-issue` for it.
---

# Cutting a model release

A model release is a numbered, trained XGBoost model that the maintainer ships to the live service.
`docs/live_service/model-releases.md` defines the
number, the log, and the final deploy steps. This skill covers the work that comes before them:
choosing what goes in, proving that the combination works, and porting it. The routine runs every
one to two weeks, when at least two experiments are flagged `candidate`.

**Each step ends in one output, and the next step does not start without it.** Steps 0, 2, and 5
stop for the maintainer's decision. Everything else runs autonomously.

## Step 0: write the agenda

**List the experiments flagged `candidate` in `docs/ml-experiments/research-log.md` since the last
release.** For each, give the verdict, the headline number with its interval, the production
setting it would change, and whether it needs a new design before it can be ported (an experiment
that needs the staged design is not portable into the single-stage model). Mark pairs that change
the same setting, because only one of a pair can be ported.

The maintainer picks which candidates go into the release. The output is that list, written into
the release plan.

## Step 1: test the winners together

**Run one ml-experiment, under the `ml-experiment` skill, that asks whether the chosen winners
together beat the current production recipe, and whether any pair interferes.** Experiments test one
idea against a fixed benchmark, and ideas that win alone can cancel out. The arms are:

- the current production recipe;
- all chosen winners together;
- "all winners minus one", once for each winner. An arm that beats "all winners together" means that
  the missing winner interferes.

Build the experiment's training and scoring frames in the shape production uses, so that the result
predicts what cross-validation will show. Pre-register the planned contrasts. Report CRPS, MAE, and
coverage per lead band, split by Primary substation, BSP, and GSP, with Primary as the headline.
Report each series type's absolute score and not only the contrasts.

A winner that fails in combination is dropped from the release or deferred, and its research-log
flag changes. The output is the final release set and the combined effect size.

## Step 2: write the release plan

**Load `plan-issue` and size the release.** A release changes what gets stored and runs through the
serving path, so it usually sizes as complex. The plan:

- splits the port into small PRs by area: features, training and loss, calibration, and serving;
- orders the PRs by dependency;
- lists, for each PR, its tests;
- names every PR that changes a data contract or the serving path, because those need the
  maintainer's approval before any code is written.

The output is an approved plan and a tracking issue with one sub-issue per PR.

## Step 3: implement the PRs

**Load `implement-issue` and implement each PR in the plan.** Re-implement each winner in the
reviewed packages. Never promote an experiment's script into `src/` or `packages/`, because
experiment code is unreviewed and production never imports it. None of these PRs bumps the model
release number. The output is the merged PRs.

## Step 4: check that the port reproduces the experiment

**Run the ported production recipe on the row set and frames of step 1, and compare its predictions
with the combination experiment's.** The two must agree within a stated tolerance. A mismatch means
that the port differs from what was tested, and the cause is fixed before step 5. The output is a
short statement of agreement, with the tolerance, in the tracking issue.

## Step 5: run the full cross-validation gate

**Register the cross-validation run `model-release-N` and compare it with the previous release and
the baselines (`manual_heuristic` and `climatology`).** The run's `config_overrides` set
`model_release: N`. The gate has three parts:

- the release is not worse than the previous release on any series type, beyond the tolerance the
  maintainer set;
- the release meets the combination experiment's prediction, within the tolerance of step 4;
- the diagnostics are acceptable: calibration and coverage by lead band, behaviour when an input is
  missing, and the series that got worse.

Primary substations are the headline. The maintainer reviews the gate result, and the output is a
decision to release or to go back to step 1. Do not spend more than the £30 cap in the
`ml-experiment` skill without asking.

## Step 6: release

**One PR cuts the release.** It follows the steps in
`docs/live_service/model-releases.md`: promote the
champion fold run, tag the commit `model-N`, and add the row to the log, including the rollback
pairing. The same PR sets `production_status: adopted` and `adopted_in: N` in each incorporated
experiment's `result.md`, and moves those experiments in the research log. The output is the
release.

## Step 7: watch the first runs

**Check the live metrics and the degradation records for the first runs of the new release.** Name
any experiment whose winner step 1 dropped, and give the reason, in the research log. The output is
a note in the tracking issue, and the tracking issue closes.

## Decisions that belong to the maintainer

Three decisions are open until the maintainer sets them, and this skill does not guess them:

- **The gate tolerances** of steps 4 and 5.
- **Whether a change that retrains every series** needs a release of its own.
- **The cadence.** The default is every one to two weeks when at least two candidates have
  accumulated.
