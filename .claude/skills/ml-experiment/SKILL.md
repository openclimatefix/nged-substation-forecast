---
name: ml-experiment
description: >-
  How to run a small, focused ML experiment that tests one hypothesis about improving the
  Flexpectation production forecast, and keep its result where a later experiment can find it, as a
  one-screen `result.md` and an immutable set of predictions. Load before planning, running, or
  writing up an experiment under `ml-experiments/`, before adopting a change into a family's research
  default, and before an overnight session. A question that
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
whole-sample figures a theme's Research plan page quotes.

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

Either size then runs the triggers in "Audit before merge", below, and gets an audit if one fires.

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
- **The write-up:** `docs/ml-experiments/<id>-<slug>/result.md`, with its figures beside it. The
  page is not in the navigation, because the generated Experiments table of its theme links to it,
  so do not edit `mkdocs.yml`.
- **The themes:** every experiment belongs to one theme, a folder under
  `docs/ml-experiments/themes/` with a Summary (`index.md`), an Experiments page, and a Research
  plan page (`plan.md`). The Experiments table and the planned-experiments table are generated from
  the `result.md` front matter when the documentation is built, so the experiment's front matter is
  the only entry to write.
- **The code:** `ml-experiments/<id>-<slug>/`. Name every script with the id in a valid module form,
  such as `e00012_fit.py` and `e00012_report.py`, because no two scripts under `studies/` or
  `ml-experiments/` may share a basename. A script may import its own folder, the folder of an
  earlier experiment or study, `studies.*`, and the other reviewed packages, and nothing in `src/`
  or any other package imports it (`packages/studies/tests/test_study_boundaries.py` enforces
  this). Run a script that imports from another folder with that folder on `PYTHONPATH`, because
  only the script's own folder is on `sys.path` otherwise. An example is
  `PYTHONPATH=ml-experiments/<earlier-folder> uv run --no-sync python <script>`. An experiment has
  no tests of its own scripts; the machinery it calls is tested in `packages/studies/`.
- **Reuse of an earlier experiment's code:** when a script imports another experiment's module,
  name that experiment's folder in the `result.md` bullets. Never edit an earlier experiment's
  script after its PR merged, except to fix a bug, and record that fix in the later experiment's PR
  and `result.md`. The earlier result is immutable, so a fix that would change its numbers is
  made only together with a correction to the earlier `result.md`; otherwise copy the code into the
  new folder.
- **The predictions:** `data/ml-experiments/<id>-<slug>/predictions/<name>/`, written with
  `experiment_store.write_predictions`. The folder is private and never committed.

**`result.md` is the plan first and the write-up second.** Commit it with status `planned` before
the code runs, so the commit order proves the hypothesis came first. Push that commit to the
experiment's own branch, and do not open a pull request for it. Its front matter holds the
`id`, `slug`, `theme`, `also_relevant_to`, `status`, `size`, `hypothesis`, `prediction`,
`decision_rule`, `benchmark`, `row_set_id`, and `metric`, and, when finished, the `verdict`
(`supported`, `partly supported`, `small gain`, `not supported`, `refuted`, `inconclusive`,
`invalid`, or `failed`), `best_arm` with its headline number, `headline`, `lessons`,
`take_home_message`, and `advice` (below). A finished experiment also has `production_status`
(`candidate`, `adopted`, `superseded`, or `not-for-production`) and `adopted_in` (the model release
number, or `none`). `kind` is `change` (the experiment asks whether a research default plus x
beats the research default) or `other`, and it is required from id 320 up. `promoted_to` is always
optional, needs `kind: change`, and names the version of the research default that the
experiment created, such as "Primary demand research default 2". The field name keeps the word
"promoted" although the rule avoids it. Every result has the same front matter, so a later session
finds results with `grep` over `docs/ml-experiments/*/result.md`.

**`theme` names exactly one theme folder under `docs/ml-experiments/themes/`, and `also_relevant_to`
lists the other themes the result bears on (an empty list if none).** The experiment appears in the
Experiments table of its own theme as "primary" and of each listed theme as "also relevant". An
experiment that fits no existing theme takes the closest theme. A new theme should cover several
related experiments, and is a folder under `themes/` with `index.md`, `experiments.md`, and
`plan.md`, plus an entry in the `mkdocs.yml` navigation.

**`take_home_message` is two or three plain sentences saying what the result means for the forecast,
and it is required for every experiment with an id of 103 or more.** Write it for a reader who has not
opened the page, with no jargon, and put the effect on the forecast first. It is shown in the
Experiments table of each theme. `uv run python scripts/mkdocs_hooks/experiment_tables.py --check`
validates the whole front matter, and `... --theme <theme>` prints a theme's Experiments table.

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
which it departs from. The overview's "What the experiments taught us so far" section is the
curated roll-up of the `advice` lists.

A plan also begins by reading `docs/ml-experiments/background/index.md`, and then
`docs/ml-experiments/data-processing.md`, which lists the processing that the data needs before a
model trains or scores on it.

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

## Keep the theme's Research plan page current

**Update the owning theme's Research plan page (`docs/ml-experiments/themes/<theme>/plan.md`) in the
first commit that adds the `planned` `result.md`, and again in the pull request that adds the
finished write-up.** An experiment opens one pull request, when its result is ready, and the plan
commit is the first commit in that pull request's history. No pull request is opened at plan time. A
change to `packages/studies/` that the experiment needs is the exception: open its pull request as
soon as the module is ready for review, because the module needs its own review. The planning commit
updates the experiment's row in the priority table at the top of the page with its id, the question
in one line, the priority, what it waits for, and the state, and commits the `result.md` with
`status: planned`, which the page's generated planned-experiments table then lists. The finished
commit deletes that row and the experiment's proposal section from the page, sets the final
`status`, `verdict`, and `production_status` in the front matter, and writes `take_home_message`.
Nothing else is edited: the Experiments table of the theme, and the candidates table on the
overview, are generated from the front matter. An idea the maintainer has agreed to run but that has
no folder yet goes in the priority table marked "Proposed". An experiment that ends `invalid` or
`failed` keeps its `result.md`, and its `take_home_message` says why.

**Name every experiment in a series with the same topic words in its slug, so a reader can find the
series by eye.** Write the slug as a few words naming the topic and the part, such as
`switching-event-detector-part-1`.

## Exploratory series

**When the question is exploratory, run a series of small experiments, each designed from the
result of the one before.** A question is exploratory when no single hypothesis yet states what to
test, such as "can switching events be found in the telemetry at all?" or "can embedded solar be
separated from substation demand?". A series replaces one large pre-registered experiment, which
would spend its whole budget before anything is learned. Every experiment in the series still has
its own `result.md`, hypothesis, prediction, decision rule, and stop rule, and the series shares one
topic in the slugs of its experiments.

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

**When a series of more than three experiments finishes or pauses, write one summary report that
tells the whole story.** A series is a set of experiments sharing a topic in their slugs. The report
is the Summary page of the series' theme, `docs/ml-experiments/themes/<theme>/index.md`. The report
refits nothing: it collates figures and numbers from the existing write-ups into one page. Its three
parts, in this order:

- **What the data looks like.** Plots of the raw data and of the labelled examples, with one plain
  sentence each, taken from the series' first experiments.
- **What worked.** The findings that changed the best guess or the next step, each with its headline
  number, interval, and the figure that shows it, and the one-line take-home first.
- **What was tried.** A table of every experiment in the series with its id, the question in one
  line, the verdict, and the headline number, so a reader can see the refuted ideas without opening
  them.

Keep the report short, link every experiment to its `result.md`, and use the figures already
committed. A Sonnet agent writes it from the write-ups and one Sonnet agent reviews it against them.

## Adopting a change into the research default

**Each target family has a research default, and a change to it is adopted one change at a time.**
`docs/ml-experiments/research-default.md` defines the families, the headline metric, the tolerances,
the minimum gain by tier, the five adoption conditions, and the cadence, and
`docs/ml-experiments/research-default-log.md` lists each family's versions. Read the rule page before
planning any experiment that could improve a family's forecast. Do not write a metric, a tolerance,
or a minimum gain into the plan for a `change` experiment, because the family definition owns them.

**Naming checklist for a `change` experiment.**

- Set `kind: change` in the front matter, and `kind: other` for a diagnostic, an audit, a data check,
  or a detection or decision study.
- Cite the version by its full name in the plan, in every `result.md`, and in the pull-request body,
  for example "default plus x versus Primary demand research default 1". Never write "default 3" or
  "v3" alone.
- Name the two arms "default" and "default plus x", and report the research default's absolute score
  against BG2, `manual_heuristic`, and `climatology`.
- Say whether the change is tier 1 (acts on stored forecasts) or tier 2 (acts inside a fitted stage,
  with the default's code imported from the folder of the experiment that holds it), and which
  complexity tier sets the minimum gain.
- State the family's lead range in the headline caveat. NGED care about the 1 day to 10 day horizon,
  so a shorter range, such as 12 to 36 hours for metered solar farms, is a limit on the result.
- If the experiment evaluates a blend, report each constituent model family alone beside the blend.
  The single-family table comes first, and the page states the blend's added value over the best
  single family, with an interval.
- If the five conditions hold, set `promoted_to` to the new version's name, and add a row to the log
  in the same pull request. If they do not, the verdict stays and the result is a candidate.

**`small gain` is the verdict for an interval that excludes zero in the predicted direction but falls
under the stated threshold.** It is not `inconclusive`, because the direction is established. The
write-up says how large the gain is and what the change costs, and the experiment is flagged
`candidate`. A change that wins on the gain condition but is not adopted is also a "candidate" in
the research-default rule's sense, which is separate from `production_status: candidate` and from
`production_status: adopted`, which mean production. The rule's rebase section says how a candidate
is retested against the current research default, and a candidate is not stacked with other
candidates in a combination experiment.

## When a winner reaches production

**Flag each finished experiment for the next model release.** Set `production_status` in
`result.md` to `candidate` when the verdict is `supported`, `small gain`, or `partly supported` and
the result is worth considering for the next model release, and to `not-for-production` for a
diagnostic, a refuted idea, or a stage that failed its test. The overview's "Candidates for the
next release" table is the agenda for the maintainer's release review.

**The pull request that cuts a model release records which experiments it adopted.** In that pull
request, set `production_status` to `adopted` and `adopted_in` to the model release number in each
incorporated experiment's `result.md`, and add a row to `docs/live_service/model-releases.md`
listing the experiments the release incorporates. The pull requests that port the winners earlier do
not set `adopted_in`, because the release number does not exist until the release is cut. A later
experiment that overturns an adopted result sets `production_status` to `superseded` on the old
experiment and names the new one in its `take_home_message`. The `release-model` skill describes the
whole routine.

## The write-up

**Lead with the single most useful take-home message of the experiment, in plain language, in the
write-up and in every report to the maintainer.** The take-home message says what the experiment
taught us, and it matters most when the hypothesis failed. A bare verdict word such as "refuted"
tells the reader what did not work and nothing about what to do next. State what the result rules
out, what it narrows the search to, and what the next experiment should therefore try. For example,
write "A second weather forecast adds almost nothing to the solar forecast, so the limit is the
quality of the sunshine forecast itself, and a source of a different kind is the next thing to try",
and give the verdict word after that sentence. A refuted hypothesis still shrinks the space of ideas
worth testing, and the take-home message is where the experiment says by how much. Keep the message
honest: do not turn an inconclusive result into a finding, and say plainly when the experiment could
not tell.

**Open `result.md` with a few plain sentences that say what question the experiment asked, how it
went about answering the question, and what it concluded.** Write them for a reader who has not seen
the experiment or the plan. The opening uses no jargon: it names what was tested
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

**Cite the work an experiment builds on, generously, in the plan and in `result.md`.** Whenever an
arm, a method, a metric, or a design choice comes from someone else's paper, software, or
published result, name the authors and link the work where it is first used, following the
citation rule in `CLAUDE.md`. Credit colleagues inside the project the same way: link the earlier
experiment, study, or docs page whose idea the experiment takes forward. An experiment that
re-implements a published method, such as a Kalman filter on the coefficients of a fitted additive
model, says so in its opening sentences, so a reader never mistakes the idea for ours. When in
doubt, cite.

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
- **Ends with a morning report as its final message:** one line per experiment that opens with its
  take-home message, then its verdict and link, what failed, and the hypotheses it proposes for the
  next plan.

**Add or fix at most five bullets in `docs/ml-experiments/background/` in the same PR as the
write-up.** The [background index](../../../docs/ml-experiments/background/index.md#update-rule)
owns the full rule, including the evidence tag every bullet carries.

**Record every new finding about data processing in `docs/ml-experiments/data-processing.md`, in the
same PR as the write-up.** A finding about data processing is a step that the data needs before a
model trains or scores on it: a period to cut or mask, a series to exclude, a unit, sign, or
timestamp convention, a product that cannot be mixed with another, or a check that failed silently.
Write it as an instruction with a bold lead, followed by where it applies (experiments, production,
or both), the evidence with a link to the experiment, the series and period it covers, and the
confidence. Update an existing entry instead of adding a second one, and mark the entry provisional
until the experiment has had its reviews. A result that contradicts an entry changes the entry in the
same PR. Keep a rule about how to run an experiment, such as a control arm or a scoring rule, in this
skill and not on that page.

## Audit before merge

**A review that checks the printed numbers against the stored predictions cannot catch a method that
is wrong, so a result that looks wrong in any of the ways below gets an audit before it merges.**
Experiment 104 showed why. Its numbers review confirmed every printed number. An audit afterwards
(`data/ml-experiments/briefs/e104_audit.md`) found that the residual spread shifted each forecast by
the previous year's bias, which reversed two of the headline conclusions. Three failed controls had
been explained away instead of investigated.

**An audit is triggered when any of these holds:**

1. **A control fails:** a positive control, a known-answer test, or a placebo.
2. **A result contradicts its own prediction or an earlier experiment's finding,** and the
   experiment's session has not tested an explanation and recorded it in the write-up. Experiment
   87's solar offset lowered the score by 0.010 points against a predicted 0.10 to 0.15, although
   experiment 64 had found a gain of 0.152.
3. **The quantile-score contrast and the same contrast on the error of the mean or median forecast
   point in different directions.**
4. **Leaving one site out flips the sign or the verdict of a planned contrast.** Experiment 104's C1
   was −0.20 with D14 and +0.13 without it. A site scoring more than twice the median across sites
   is a prompt to run the leave-one-out check, not a trigger, because with five sites that ratio is
   noisy.
5. **A nested arm beats the arm that contains it,** such as an arm with the same model given less
   input, **or a control that should change nothing moves the score by more than its own paired
   interval's half-width.**
6. **A headline interval is more than twice as wide, or less than half as wide,** as the same
   contrast's interval in a sister experiment on the same rows.
7. **A later experiment's design depends on this result's headline number or sign, and the result has
   not been replicated.** The ordinary "what the next experiment should answer" handoff of an
   exploratory series does not count on its own.

**Who runs it, and when.** The manager, or the experiment's own session when no manager is running,
starts a fresh Opus agent with the audit brief. The agent that wrote the experiment never audits it.
The audit runs after the code and write-up reviews and before the prose review. Opus is justified
here even when credits are short, because a wrong method poisons every experiment built on it. The
audit brief differs from a review brief: the agent re-derives the headline contrasts from the raw
stored predictions, by its own code, and tries to break them. It checks each method's assumptions,
such as how quantiles are built, how a base margin enters, and how a held value is extrapolated. It
does not re-check the printed arithmetic. The audit report goes in the private
`data/ml-experiments/briefs/` folder, and the PR triages each finding.

**A shared method is audited once, where it lives.** A method reused from `packages/studies/`, such
as a spread model or a base-margin helper, counts as checked only once its own tests or an audit
cover the property in question. An experiment that is first to depend on an unchecked property
audits it there. The fix lands in its own reviewed PR to `packages/studies/` first, as "Merging"
requires. The experiment either waits for that PR or records the unchecked property as a caveat.

**Every write-up carries a short "Checks" list of bullets after its figures, triggered or not:**

- **the median forecast minus the mean forecast, per site,** which exposes a spread that shifts the
  forecast;
- **the error of the mean or median forecast beside every quantile-score contrast,** which exposes
  trigger 3;
- **which triggers fired,** with a link to the audit's findings and how each was triaged.

## Merging

**An experiment's PR may be merged without waiting for the maintainer when all four hold:** the diff
touches only `ml-experiments/` and `docs/ml-experiments/`; the experiment has had the reviews its
size calls for, with the findings triaged, and its front matter passes `experiment_tables.py
--check`; any audit a trigger in "Audit before merge" calls for has run, with its findings triaged
in the PR; and the checks pass (`ci` green, no closing keyword in the PR body or commit messages, no
`data/` or `.env` file, the branch contains `main` or everything that landed on `main` since the
branch was cut sits under `docs/` or `ml-experiments/` and touches no file the PR changes, a merge
commit and not a squash).

**Run every CI step locally before each push,** including the `pre-commit` hooks on every changed
file (the SVG end-of-file hook needs `svgo --final-newline`), `ty check`, `pydoclint`, and
`experiment_tables.py --check`. A failure found by CI costs a ten-minute rerun.

**Before any merge, a fresh Sonnet agent reviews the prose of `result.md` and its
`take_home_message`.** The agent receives the files and no account of the reasoning behind them,
loads the `prose-review` skill, and reads the opening paragraph first. It reports every term,
abbreviation, and number in the opening that a reader who has not seen the experiment could not
parse, and then checks the rest of the page against the prose rules in `CLAUDE.md`. The experiment's
session triages the findings, applies the valid ones, and records each rejected one with its reason.
A change to `packages/studies/` stops at a reviewed PR, as `implement-issue` does.
