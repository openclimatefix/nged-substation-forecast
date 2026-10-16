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

**State the smallest gain worth the cost of porting the change, and set it by that cost.** A change
that costs nothing in production, such as a different column, training start, or loss, is worth
porting at any gain whose interval excludes zero, so its threshold is zero. A change that adds a
model, a data source, or a pipeline step needs a threshold that pays for the added complexity. Write
the threshold into the decision rule, with the reason. Do not copy a threshold from an earlier
experiment, because a paired contrast on the same rows often resolves gains much finer than the
whole-sample figures the research plan quotes.

**Size the experiment with the five `plan-issue` triggers, and write one line for each:** what gets
stored, whether the experiment touches the production serving path, whether it touches a degradation
rule, whether more than one design would defensibly test the hypothesis, and whether it spans code
whose callers you could not name without searching.

- **Simple** is a permutation of settings the pipeline already has (input-feature sets, a
  hyperparameter, a loss), with no new data and no new method. Write the plan section, implement,
  run, and write up. One Sonnet reviewer reads the code for leakage and for the fit stopping at the
  fold's `train_end`, one Sonnet reviewer checks the write-up against the printed report, and one
  Sonnet reviewer runs the prose review that "The write-up" describes.
- **Anything else** gets the full process: Sonnet writes the plan, Opus reviews the plan, the
  maintainer reviews the plan only if asked, Sonnet implements, Opus reviews the code, Sonnet runs
  the code and writes up, Sonnet reviews the write-up against the printed report, and a second
  Sonnet reviewer runs the prose review that "The write-up" describes.

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
- **The research log and plan:** `docs/ml-experiments/research-log.md` lists every finished
  experiment in the order it was started, with its question, headline result, date, and links.
  `docs/ml-experiments/research-plan.md` lists the planned and proposed experiments, sorted by
  priority. Keep both current in the same PR that changes an experiment's state, as the next
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
the metric, and, when finished, the verdict (`supported`, `small gain`, `refuted`, `inconclusive`,
`invalid`, or `failed`), the best arm with its headline number, the lessons learned, and `advice`
(below). A finished experiment also has `production_status` (`candidate`, `adopted`, `superseded`, or
`not-for-production`) and `adopted_in` (the model release number, or `none`). Every result has the
same front matter, so a later session finds results with `grep` over
`docs/ml-experiments/*/result.md`.

**`advice` is a YAML list of one-line imperatives for future experiments, and a result is not
finished without it.** Each item says what to do and where it applies, and ends with a pointer to the
evidence, such as a planned contrast (C1) or a section of the write-up. For example: "Put recent
power in the stage 2 calibrator, not in stage 1b, for Primary substations (C1)." Write each item as
a folded YAML scalar (`- >-`), wrapped at 100 characters, so that it makes sense without the page
around it. Write the items from the write-up's lessons and headline contrasts, and drop any item
that no number or statement in the write-up supports. The field is separate from the `lessons`
summary and the page's lessons section, which explain the reasoning, and from `production_status`,
which says whether the winner is ready for production. Before planning a new experiment, read the
earlier advice with
`awk '/^advice:/{f=1; print FILENAME} /^[a-z_]+:/&&!/^advice:/{f=0} f' docs/ml-experiments/*/result.md`,
which prints a whole list however long it is, and say in the plan which items the plan follows and
which it departs from. The research log's "What the experiments taught us so far" section is the
curated roll-up of the `advice` lists.

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

## Use the compute the slot gives you

**Run independent work in parallel, up to the slot's budget of cores and GPU memory.** Series,
seeds, arms, folds, and refits that do not read each other's output can run side by side. A job that
fits many small models, such as one per series or one tiny update per day, cannot keep a GPU or many
cores busy one model at a time. Experiment 34 ran its daily warm-start fits for 16 series one after
another, at about 2.7 of its 12 cores and almost no GPU, for 3.3 hours. Running 4 to 8 series at a
time would have taken under an hour.

- **Check utilisation in the first minutes of a long run.** Look at `uptime`, `nvidia-smi`, and the
  job's own CPU use. If the job uses well under its quota and will run for more than about 30
  minutes, split it into parallel workers before letting it run on.
- **Stay inside the slot.** The worker count times the threads per worker stays at or below the
  slot's cores, and the workers' GPU memory adds up to no more than the slot's share. Ask the
  manager before exceeding either.
- **Keep the results independent of the schedule.** Fix the seed of each job, write each job's output
  to its own file, and write it atomically, so that a run in parallel gives the numbers a run in
  series would, and a crashed worker is rerun without touching the others.

## Keep the research log current

**Update `docs/ml-experiments/research-plan.md` in the PR that commits the `planned` `result.md`,
and `docs/ml-experiments/research-log.md` in the PR that adds the finished write-up.** The first PR
updates the experiment's row in the table of `research-plan.md` with its id, the question in one
line, the priority, what it waits for, and the state. The finished PR deletes that row and its
section from `research-plan.md`, adds the experiment to the table at the top of `research-log.md`,
and adds a short entry: the question, the verdict in one sentence, the
headline number with its interval, any finding that changes how later experiments are run, links
to `result.md` and the PR, and one figure. An idea the maintainer has agreed to run but that has no
folder yet goes in the table of `research-plan.md` marked "Proposed". An experiment that ends
`invalid` or `failed` keeps its entry, and the entry says why. Write the entry for a reader who has
not opened `result.md`.

**The first column of each table in `research-log.md` states the topic, and the position in a series
when the experiment belongs to one.** Write it as the id, a dash, a few words naming the topic, and
the part number for an experiment in a series, such as "00037 - Switching event detector, part 1".
Name every experiment in a series with the same topic words, so a reader can find the series by eye.

## Exploratory series

**When the question is exploratory, run a series of small experiments, each designed from the
result of the one before.** A question is exploratory when no single hypothesis yet states what to
test, such as "can switching events be found in the telemetry at all?" or "can embedded solar be
separated from substation demand?". A series replaces one large pre-registered experiment, which
would spend its whole budget before anything is learned. Every experiment in the series still has
its own `result.md`, hypothesis, prediction, decision rule, and stop rule, and the series is one
topic in the research log.

The loop has two steps, repeated:

1. **Opus plans one experiment, chosen to learn as much as possible about how to proceed.** Make the
   first experiment fast to run. It may be data analysis with no model training, for example a look
   at whether the events in a log are visible in the telemetry at all. It produces plenty of plots
   of the data (load the `dataviz` skill, and show the raw data before any model output). A first
   Sonnet agent implements the code, a second Sonnet agent reviews the code, a third Sonnet agent
   runs the experiment, and a fourth Sonnet agent gathers the results into the write-up. Each agent
   works from the files and never starts a sub-agent of its own.
2. **Opus reads the results and decides what comes next.** The decision rule depends on how many
   experiments the series has run and on what they found:
    - **The results are positive.** Opus designs the next experiment, which builds on the finding,
      and the loop repeats.
    - **The results are inconclusive or negative, and the series has run only one experiment.** Do
      not stop yet. Opus designs a second experiment that attacks the problem from a different angle,
      such as a different signal, a different data source, or a different way of scoring.
    - **The results are still negative after at least two experiments.** Stop the series. The last
      write-up says what was tried, what the data showed, and what would have to change for another
      attempt to be worth running.

Each write-up in a series ends with a short "what the next experiment should answer" paragraph,
written after Opus has read the results. The experiment's own session does not start that next
experiment. The maintainer decides when each step runs, and may stop the series at any step.

## Small gains and stacking

**`small gain` is the verdict for an interval that excludes zero in the predicted direction but falls
under the stated threshold.** It is not `inconclusive`, because the direction is established. The
write-up says how large the gain is and what the change costs, and the experiment is flagged
`candidate`.

**Gains from different experiments may not add, so test them together.** Two changes can act on the
same error. When several `candidate` and `small gain` results exist for one target (solar and wind,
or Primary demand), run one stacking experiment that combines all of them against the production
recipe. It reports the combined gain, and the loss in score when each component is dropped in turn.

## When a winner reaches production

**Flag each finished experiment for the next model release.** Set `production_status` in
`result.md` to `candidate` when the verdict is `supported`, `small gain`, or `partly supported` and
the result is worth considering for the next model release, and to `not-for-production` for a
diagnostic, a refuted idea, or a stage that failed its test. The log's "Candidates for the next
release" section is the agenda for the maintainer's release review.

**The pull request that cuts a model release records which experiments it adopted.** In that pull
request, set `production_status` to `adopted` and `adopted_in` to the model release number in each
incorporated experiment's `result.md`, move the experiment's row in the research log, and add a row
to `docs/live_service/model-releases.md` listing the experiments the release incorporates. The pull
requests that port the winners earlier do not set `adopted_in`, because the release number does not
exist until the release is cut. A later experiment that overturns an adopted result sets
`production_status` to `superseded` on the old experiment and names the new one in the log. The
`release-model` skill describes the whole routine.

## The write-up

**Open `result.md` with a few plain sentences that say what question the experiment asked, how it
went about answering the question, and what it concluded.** Write them for a reader who has not seen
the experiment, the plan, or the research log. The opening uses no jargon: it names what was tested
in ordinary words, says what each number is a number of, and leaves the arm names, the check names,
the abbreviations, and the decision rule to later sections. Test every phrase by asking "what is
this a share of?", "what was put into what?", and "what does this label stand for?". A sentence such
as "injected solar-farm output is recovered to within 25%, but the real series pass the four checks
at 6 of 8 Primaries" fails all three: it does not say what the output was injected into, what the
25% is a share of, or what the four checks test. Write the same finding as, for example: "We asked
whether the power a substation draws can be split into ordinary demand and the output of rooftop
solar panels. We added the recorded output of a real solar farm to a substation's demand, and
checked how much of it a fitting method could find again. The method found about 77% to 85% of it.
At 2 of the 8 substations, other checks suggested the answer could not be trusted." The verdict word
and its numbers follow in the next paragraph, with each term defined where it first appears.

**Keep the rest of `result.md` to about a screen:** the verdict in one sentence, the winner with its
interval, each arm's absolute score, 2 to 4 figures (load the `dataviz` skill), the failures, and
the lessons for later experiments, which also become `advice`. **A published page never names a
generator, gives its identifier, or shows its time series beside either.** Use the anonymised labels
the `study` skill describes. Print every number the page quotes from a committed script, as the
`study` skill requires. An experiment that plots feature importance follows the recipe in the
`study` skill's "Design rules that keep a comparison fair".

## Running a night's experiments

When several experiment sessions run at once, the `ml-experiment-manager` skill covers the queue, the
compute slots, and the merging.

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
squash). **Before any merge, a fresh Sonnet agent reviews the prose of `result.md` and the research-log
entry.** The agent receives the files and no account of the reasoning behind them, loads the
`prose-review` skill, and reads the opening paragraph first. It reports every term, abbreviation, and
number in the opening that a reader who has not seen the experiment could not parse, and then checks
the rest of the page against the prose rules in `CLAUDE.md`. The experiment's session triages the
findings, applies the valid ones, and records each rejected one with its reason. A change to
`packages/studies/` stops at a reviewed PR, as `implement-issue` does.
