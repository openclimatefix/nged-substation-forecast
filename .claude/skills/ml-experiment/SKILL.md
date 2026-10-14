---
name: ml-experiment
description: >-
  How to run a small, focused ML experiment that tests one hypothesis about improving the
  Flexpectation production forecast, and keep its result where a later experiment can find it, as a
  one-screen `result.md` and an immutable set of predictions. Load before planning, running, or
  writing up an experiment under `ml-experiments/`, and before an overnight session. A question that
  needs a long report, several scientific questions, or a new dataset as its main subject is a
  study; use the `study` skill for it.
---

# Running an ml-experiment

An ml-experiment tests **one hypothesis**, such as "adding ERA5 pre-training lowers the PV
forecast's CRPS", and ends in a verdict: supported, refuted, or inconclusive. The write-up is about
a screen long. An experiment is free to use its own metric or its own probabilistic output, so it
often cannot post to the leaderboard. A **study** is the other kind of work: a long report that
answers several scientific questions, often on a new dataset, run with the `study` skill.

**"Experiment" means only an ml-experiment.** A pass of the production pipeline over the
cross-validation folds is a cross-validation run, and `docs/cross-validation/` describes it. Adding
an improvement to the production pipeline after an experiment supports it is updating the pipeline,
and the cross-validation run that follows checks the update. That run is not a second experiment.

**Work autonomously by default.** Plan, run, review, and write up without waiting for the
maintainer. When the maintainer asks to review the plan first, push the plan, say so, and wait for
a reply. Two decisions always belong to the maintainer: **spending money** (a download or compute
that would cost more than £30, whatever the mode), and **accepting a data licence** in someone's
name.

## Step 0: write the hypothesis, then size the experiment

**State the hypothesis before any result exists.** It has three parts, written into the front
matter of `result.md`: the claim, a prediction that could fail (a direction and a size of effect),
and the decision rule (which interval, against which benchmark, would count as support).

**Size the experiment with the five `plan-issue` triggers, and write one line for each:** what gets
stored, whether the experiment touches the production serving path, whether it touches a degradation
rule, whether more than one design would defensibly test the hypothesis, and whether it spans code
whose callers you could not name without searching.

- **Simple** is a permutation of settings the pipeline already has (input-feature sets, a
  hyperparameter, a loss), with no new data and no new method. Write the plan section, implement,
  run, and write up. One Sonnet reviewer reads the code for leakage and for the fit stopping at the
  fold's `train_end`, and one Sonnet reviewer checks the write-up against the printed report.
- **Anything else** gets the full process: Sonnet writes the plan, Opus reviews the plan, the
  maintainer reviews the plan only if asked, Sonnet implements, Opus reviews the code, Sonnet runs
  the code and writes up, Sonnet reviews the write-up.

Reuse `implement-issue` by reference for the mechanics: the worktree, the green-before-push set, one
commit per review, and the triage rule (verify each finding, record each rejected one with its
reason). An experiment merges on its own under "Merging", below, so it does not stop for human
review as `implement-issue` does.

**A download follows the `data-download` and `data-validation` skills.**

## Where the pieces live

- **The id and the slug.** Call `studies.experiment_store.reserve_experiment(slug=...)`. It returns
  the next id, written as five digits, and creates `data/ml-experiments/<id>-<slug>/`. Make the slug
  descriptive.
  **Write the id without leading zeros in the title of a pull request or an issue**, as in
  "Experiment 6: which input features work best", never "Experiment 00006". The five digits belong
  to folder and file names only.
- **The write-up:** `docs/ml-experiments/<id>-<slug>/result.md`, with its figures beside it. Add it
  to the navigation in `mkdocs.yml`, under "ML experiments" and titled with the hypothesis, in the
  commit that first adds the page. When two experiments finish together, the two navigation lines
  conflict on rebase; keep both.
- **The research log:** `docs/ml-experiments/research-log.md` lists every experiment in the order it
  was started, with its question, headline result, date, and links, and it lists the planned
  experiments. Keep it current in the same PR that changes an experiment's state, as the next
  section says.
- **The code:** `ml-experiments/<id>-<slug>/`. Name every script with the id in a valid module form,
  such as `e00012_fit.py` and `e00012_report.py`, because no two scripts under `studies/` or
  `ml-experiments/` may share a basename. A script imports its own folder, `studies.*`, and the
  other reviewed packages, and nothing in `src/` or any other package imports it
  (`packages/studies/tests/test_study_boundaries.py` enforces this). An experiment has no tests of
  its own scripts; the machinery it calls is tested in `packages/studies/`.
- **The predictions:** `data/ml-experiments/<id>-<slug>/predictions/<name>/`, written with
  `experiment_store.write_predictions`. The folder is private and never committed.

**`result.md` is the plan first and the write-up second.** Commit it with status `planned` before
the code runs, so the commit order proves the hypothesis came first. Its front matter holds the
hypothesis, the prediction, the decision rule, the status, the size, the benchmark, the row set id,
the metric, and, when finished, the verdict (`supported`, `refuted`, `inconclusive`, `invalid`, or
`failed`), the best arm with its headline number, and `advice`. A finished experiment
also has `production_status` (`candidate`, `adopted`, `superseded`, or `not-for-production`) and
`adopted_in` (the model release number, or `none`). Every result has the same front matter, so a later
session finds results with `grep` over `docs/ml-experiments/*/result.md`.

**`advice` is a list of one-line instructions for future experiments, and a result is not finished
without it.** Each item says what to do, where it applies, and which contrast or section supports it,
for example "Put recent power into the stage 2 calibrator before building a stage 1b (C1)". Write
the items from the write-up's lessons and headline contrasts, and drop any item that no number or
statement in the write-up supports. Before planning a new experiment, read the earlier advice with
`awk '/^advice:/{f=1; print FILENAME} /^[a-z_]+:/&&!/^advice:/{f=0} f' docs/ml-experiments/*/result.md`,
and say in the plan which items it follows and which it departs from. The research log's "What the
experiments taught us so far" section is the curated roll-up of the `advice` lists.

## How to score

**Pre-register the planned contrasts, and score them with the existing machinery.**

- **Benchmark.** To beat an earlier experiment, read its predictions with
  `experiment_store.read_predictions(experiment_id=..., name=..., row_set_id=...)`. The function
  raises when the set covers a different row set, because scores on different rows are not
  comparable. The first shared row set is `fold_a_tuesdays_v1`, the stage 1 pilot's: 53 runs, all
  on Tuesdays, and the 9 generator series. State a different `row_set_id` in the plan, with the
  reason, when the experiment needs different rows. No experiment scores a valid time on or after
  2026-07-01, which is kept for a finalist.
- **Interval.** Build the loss frame with `experiment_store.losses_for_bootstrap`, which pairs rows
  on the run and the valid time together and refuses arms with different seeds. Read each planned
  contrast through `studies.bootstrap.bootstrap_difference_at_level`, at the level a Bonferroni
  correction sets across the experiment's planned contrasts, with `n_resamples` raised so that each
  tail holds enough draws. A contrast counts as a gain by the `study` skill's rule: the interval
  over repeated seeds excludes zero.
- **A climatology row.** Include a month-by-half-hour climatology from
  `baseline_forecasters.climatology.ClimatologyForecaster` (local month, half-hour, and weekend flag,
  smoothed over neighbouring cells), trained on the fold's training window. The climatology has no
  seed, so repeat its rows once per seed of the arms it is compared with.
- **No calendar memorisation.** A model must not be able to memorise the calendar position. The
  stage 1 pilot found a control whose error changed by 1.1 to 3.4 points when its hour and
  day-of-year columns were removed. Use no calendar columns, or include a control without them. Group
  any inner early-stopping or cross-validation split by valid time, because repeated ensemble runs
  of one valid time would otherwise sit on both sides of the split.
- **The leaderboard.** An experiment whose output has `PowerForecast`'s columns and covers the
  reference experiment's row keys may post through `score_study.py`, as the `study` skill
  describes.

**An experiment that finds leakage is `invalid`.** Change the verdict, say so in the first line of
`result.md`, and leave the predictions, because another experiment may have read them.

## Keep the research log current

**Update `docs/ml-experiments/research-log.md` in the PR that commits the `planned` `result.md`,
and again in the PR that adds the finished write-up.** The first PR adds a row under "Planned
experiments" with the id, the question in one line, and the state. The finished PR moves that row
into the table at the top and adds a short entry: the question, the verdict in one sentence, the
headline number with its interval, any finding that changes how later experiments are run, links
to `result.md` and the PR, and one figure. An idea the maintainer has agreed to run but that has no
folder yet goes in the planned table marked "Proposed". An experiment that ends `invalid` or
`failed` keeps its entry, and the entry says why. Write the entry for a reader who has not opened
`result.md`.

## When a winner reaches production

**Flag each finished experiment for the next model release.** Set `production_status` in
`result.md` to `candidate` when the verdict is `supported` or `partly supported` and the result is
worth considering for the next model release, and to `not-for-production` for a diagnostic, a
refuted idea, or a stage that failed its test. The log's "Candidates for the next release" section
is the agenda for the maintainer's release review.

**The pull request that ports a winner into the production pipeline records why.** In the same
change, set `production_status` to `adopted` and `adopted_in` to the model release number in the
experiment's `result.md`, move its row in the research log, and add a row to
`docs/live_service/model-releases.md` listing the experiments the release incorporates. A later
experiment that overturns an adopted result sets `production_status` to `superseded` on the old
experiment and names the new one in the log.

## The write-up

**Keep `result.md` to about a screen:** the verdict in one sentence, the winner with its interval,
each arm's absolute score, 2 to 4 figures (load the `dataviz` skill), the failures, and the lessons
for later experiments, which also become `advice`. **A published page never names a generator, gives
its identifier, or shows its time series beside either.** Use the anonymised labels the `study`
skill describes. Print every number the page quotes from a committed script, as the `study` skill
requires. An experiment that plots feature importance follows the recipe in the `study` skill's
"Design rules that keep a comparison fair".

## Running a night's experiments

The maintainer and Claude agree the next night's experiments by day, each as a committed `result.md`
with status `planned`. The overnight session:

- **Runs the planned experiments in order.** It may add unplanned experiments up to the budget the
  agreed plan sets (for example, one that combines the night's winners), and labels each post hoc.
- **Records a failed fit** in that experiment's `result.md` with verdict `failed` and the exception
  text, and continues with the next experiment.
- **Stops between experiments, never mid-fit,** when the night's budget (hours or experiments) is
  spent.
- **Ends with a morning report as its final message:** one line per experiment with its verdict and
  link, what failed, and the hypotheses it proposes for the next plan.

## Merging

**An experiment's PR may be merged without waiting for the maintainer when all three hold:** the
diff touches only `ml-experiments/`, `docs/ml-experiments/`, and the experiment's line in
`mkdocs.yml`; the experiment has had the reviews its size calls for, with the findings triaged, and
the research log is updated; and the checks pass (`ci` green, no closing keyword in the PR body or
commit messages, no `data/` or `.env` file, the branch contains `main`, a merge commit and not a
squash). A change to `packages/studies/` stops at a reviewed PR, as `implement-issue` does.
