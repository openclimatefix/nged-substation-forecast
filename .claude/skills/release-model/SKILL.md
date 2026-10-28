---
name: release-model
description: >-
  Load before choosing what goes into the next numbered model release of the Flexpectation
  production forecast, before writing a release plan, and before registering a `model-release-N`
  cross-validation run. The routine: read the research-default log since the last release, test the
  current research default against the production recipe in one experiment, plan and implement the
  port, run the full cross-validation gate, release, and watch the first runs. A change to production
  that no experiment decided is an ordinary issue; use `plan-issue` for it.
---

# Cutting a model release

**A model release is a numbered, trained XGBoost model that the maintainer ships to the live
service.** `docs/live_service/model-releases.md` defines the number, the log, and the deploy steps.
This skill covers the work that comes before the deploy: choosing the experiments, proving that
they work together, and porting them.

**Each step ends in one output, and the next step does not start without it.** Steps 2, 5, and 6
stop for the maintainer's decision. Every other step runs autonomously, except that the maintainer
picks the candidates in step 0.

**Only one session runs the fits of steps 1 and 4 at a time.** Ask the ML experiment manager (the
`ml-experiment-manager` skill of the `nged-ml-research` repository) for a resource slot (the file
`data/ml-experiments/SLOTS.md` lists the slots), and check the CPU and GPU load before a fit. Both
repositories share the `data/ml-experiments/` folder, so the research-default experiment's
predictions sit there, and the step 4 gate compares against them.

## Step 0: write the agenda

**List every adoption in [the research-default
log](https://openclimatefix.github.io/nged-ml-research/ml-experiments/research-default-log/) since
the last release.** The log holds each family's versions of the research default, and each row names
the adopting experiment, its gains with intervals, and the stored forecasts. The release starts from
the family's current research default (for example, Primary demand research default 3) and not from
a list of separate winners. Also list the experiments in the "Candidates for the next release" table
of [the ML experiments overview](https://openclimatefix.github.io/nged-ml-research/ml-experiments/)
that no log row covers (`production_status: candidate`), because a candidate that was never adopted
into the research default is a separate proposal. For each item, give the verdict, the headline
number with its interval, and the production setting it would change. Mark items that change the
same setting, because only one of them can be ported. Step 1 tests the research default against the
production recipe, with each component left out in turn.

**For the first release, a single-family candidate is the default.** The first production update is
a relatively simple model implemented as one model family, for example XGBoost only, while blends
stay of interest. Take a blend into the first release only when its experiment's single-family table
shows an added value over the best single family that the maintainer judges worth the extra
complexity.

**Exclude a candidate that needs a design production does not have yet.** The staged design of
experiments 00001 and 00002 of `nged-ml-research` is an example. Such a candidate goes through
`plan-issue` as its own build, and it joins a later release.

The maintainer picks the candidates for the release. The output is that list, written into the
planned `result.md` of the research-default experiment in step 1.

## Step 1: test the current research default against the production recipe

**Run one ml-experiment (in the `nged-ml-research` repository) that asks whether the family's
current research default, with leave-one-out arms, beats the production recipe.** Size the
experiment and write its planned `result.md` first, as the `ml-experiment` skill of
`nged-ml-research` says. The adoptions in the log each beat the previous version of the research
default, and none was measured against the recipe production uses now, so the experiment re-measures
the whole chain against that recipe.

The arms are the current production recipe, the current research default, and, for each component
adopted since the last release, "the research default minus that component". **The one deciding
contrast is the research default against production.** The leave-one-out arms are diagnostics: they
estimate each component's marginal effect in context, and they cannot say which pair of components
interferes. With k components the experiment has k + 1 contrasts, so run at least three seeds,
include the `climatology` row and the calendar-memorisation control, and expect little power on 12
validation months. Drop or defer a component only when its marginal interval lies wholly on the
harmful side, because a marginal effect smaller than the effect the component showed alone is
usually absence of evidence. A candidate in the agenda that the log does not cover joins as one more
arm, and where two candidates change the same setting, add one arm per candidate and compare them
head to head.

**Build the training and scoring frames in the shape production uses,** so that the result predicts
what cross-validation will show. Report the continuous ranked probability score (CRPS), the mean
absolute error (MAE), and the coverage of the prediction intervals per lead band. Split every score
by series type: Primary substations, bulk supply points (BSP), grid supply points (GSP), solar
farms, and wind farms. Primary substations are the headline for the substations. The headline for
each winner is the series type that the winner was selected on, because the candidates so far are
solar and wind results measured on 6 solar farms and 3 wind farms.

**The validation year chose the adopted components, so it overstates their combined effect.** Treat the
validation-year effect as an upper bound. Score the final release set against the production recipe
once more on months that no candidate and no earlier release was selected on: the untouched holdout
from 2026-07-01 onward. The holdout is a check on the sign of the effect, not an estimate of its
size, while it is short and holds no winter. Gate part 3 in step 4 uses it. Record in the
release-log row which holdout months the decision read, and do not reuse a holdout month for
selection: a month may confirm one release only. When the holdout is spent, roll it forward to the
newest unseen months.

**A component that fails against production is dropped from the release, or deferred.** The release plan
lists each dropped component with its reason, and the maintainer reviews that list when approving the
plan. The research-default experiment sets its own `production_status` to `not-for-production`, because
the experiment is a diagnostic, so it never appears as a candidate. The experiment lives in
`nged-ml-research`, so a companion pull request there sets the field. The output is the final release
set and the combined effect size.

## Step 2: write the release plan

**Create one tracking issue for the release and load `plan-issue` against it.** Size the release
with the five triggers that `plan-issue` lists, and write one line for each. The plan:

- splits the port into small PRs by area: features, training and loss, calibration, and serving;
- orders the PRs by dependency;
- lists each PR's tests;
- names every PR that changes a data contract or the serving path, because the maintainer approves
  those before any code is written;
- records the dropped components from step 1;
- states the gate tolerances of step 4, the numeric coverage bounds, the rule for beating
  `climatology`, and the rollback trigger of step 7, and says whether the release retrains every
  series, which decides whether the release needs to be a release of its own.

**The maintainer approves the plan, including the tolerances.** The output is an approved plan and
one sub-issue per PR.

## Step 3: implement the PRs

**Load `implement-issue` and implement each PR in the plan.** Re-implement each winner in the
reviewed packages. Never copy an experiment's script into `src/` or `packages/`, because experiment
code is unreviewed and production never imports it. No PR in this step bumps the model release
number. The output is the merged PRs.

## Step 4: run the full cross-validation gate

**Register the cross-validation run `model-release-N`, with `model_release: N` in its
`config_overrides`, and re-run the previous release on the same folds.** N is the next unused
release number. The first release is N = 1, and its comparison is a cross-validation run of the
current recipe. Compare both with the baselines (`manual_heuristic` and `climatology`); the release
must beat `climatology` by the rule in the plan. The gate has four parts:

- **Feature parity and score agreement.** The release's feature frame equals the combination
  experiment's on a shared sample of rows, checked deterministically. The aggregate score difference
  between the cross-validation run and the experiment lies within the seed-to-seed spread measured
  in step 1, with production's seed as one of the seeds. Predictions from two fits with different
  seeds do not match row by row, so no tolerance applies to individual rows.
- **Non-inferiority on Primary substations.** The upper bound of the month-bootstrap interval of
  (release minus previous release) in CRPS is below the tolerance in the plan. BSP, GSP, solar
  farms, and wind farms are reported per series, as descriptions with no pass or fail, because each
  type has too few series for an interval.
- **The holdout agrees.** On the holdout months of step 1, the effect of the release over the
  previous release has the predicted sign, and its interval excludes harm.
- **The diagnostics are acceptable.** The plan states numeric coverage bounds per lead band. The
  degraded path (a missing NWP run) is scored for widened bands and for coverage, and the series
  that got worse are listed.

**A failed gate has two routes.** A failed feature-parity check is a porting bug, so the fix is a
new PR under step 3. A regression or a missed target goes to the maintainer in step 5. Primary
substations are the headline, as in step 1. Ask the maintainer before spending more than the £30
threshold that the `ml-experiment` skill of `nged-ml-research` sets. The output is the gate result
in the tracking issue.

## Step 5: the maintainer decides

**The maintainer reads the gate result and chooses to release or to return to step 1.** The output
is that decision, recorded in the tracking issue.

## Step 6: release

**The maintainer promotes and deploys the release, and one PR then records it.** The order is the
one in `docs/live_service/model-releases.md`: promote the champion fold run, deploy it, and check
that the `promoted_model` asset reports `model_release: N`. The PR that records the release adds the
row to the log with the promoted run's identifier and image tag and the rollback pairing, and the
maintainer tags the commit `model-N`. The first release has no earlier release to roll back to. The
release PR links a companion pull request in `nged-ml-research`. That pull request sets
`production_status: adopted` and `adopted_in: N` in each incorporated experiment's `result.md`,
which removes them from the candidates table, and updates the flag of each winner that step 1
dropped. The output is the deployed release.

## Step 7: watch the first runs

**For at least two weeks, compare the live scores of the new release with the previous release's
scores on the same days, and check the degradation records.** Use a shadow run of the previous
release where one is available. The plan states the rollback trigger in advance, and
`docs/live_service/operations.md` gives the rollback recipe and says where to read the live
forecasts and the degradation records. Note any degradation record, or any forecast that differs
from the research-default experiment's predictions, in the tracking issue, which then closes.
