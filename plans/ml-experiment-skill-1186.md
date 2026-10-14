# Plan: the `ml-experiment` skill and the store it writes to (#1186)

**The problem.** Phase 1 of agentic ML research needs the agent to run many small experiments
overnight, each aimed at improving the Flexpectation production forecast. The `study` skill is the
wrong tool for them. A study is a long report that answers several scientific questions, with two
Opus scientific-validity reviews, persona reviews, and a charted page. An experiment that answers
one question, such as "does adding ERA5 pre-training improve the PV forecast?", needs a page of about
a screen. The repository also has nowhere to keep an experiment's predictions so that a later
experiment can reuse them as the benchmark to beat, and no rule for what an experiment may score on
(an experiment with its own metric or its own probabilistic output cannot post to the leaderboard).

**The solution.** A new skill, `ml-experiment`, that sizes each experiment and runs the matching
reviews. A store with three parts: code in `ml-experiments/<id>-<slug>/`, one short write-up in
`docs/ml-experiments/<id>-<slug>/result.md`, and predictions in
`data/ml-experiments/<id>-<slug>/predictions/` with a manifest. A small tested package,
`packages/ml_experiments/`, owns the store's mechanics: reserving an experiment id, reading and
writing a manifest, the shared benchmark row set, and a default scorer. The `study` skill is left
alone, apart from the boundary sentence that [issue #1168 (Make the study skill say when a study
posts to the leaderboard)](https://github.com/openclimatefix/nged-substation-forecast/issues/1168)
adds.

## Verdict, size, and departures

- **Verdict: worth implementing.** The maintainer has asked for it. `grep` finds no store for
  predictions outside `power_forecasts`, whose contract (`PowerForecast`) cannot hold 19-level
  quantile outputs or reordered ensemble members, and the contract is not changed without the
  maintainer's say-so.
- **Size: complex.**
    - What gets stored: fires. A new store of predictions and manifests under `data/`, and a new
      docs tree.
    - Production serving path: does not fire.
    - Degradation rule: does not fire.
    - More than one defensible design: fires. The layout, the manifest, the row set, whether to
      reuse `power_forecasts`, and where the code lives each have an alternative.
    - Callers not nameable without searching: does not fire. The package is new, and no existing
      code calls it.
- **Reviews: both plan reviews, the simplicity review (Sonnet) and the correctness review (Opus).**
  The package diff later gets the diff reviews that `implement-issue` prescribes for production
  code, with the mutation pass, because the scorer and the row-set check are logic a wrong study
  would trust.
- **Delivery in two pull requests.** The package first (code, tests, boundary test), then the skill
  and the docs, because the skill names the package's functions and a skill that names a function
  that does not exist yet misleads the first session that loads it.

## Solution

### Walk through

#### The happy path

To run an experiment, the session first sizes the work (step 0 of the skill; see "Sizing and
reviews" below). It then calls `ml_experiments.store.reserve_experiment(slug="input-features-for-pv")`,
which creates `data/ml-experiments/<id>-<slug>/` with `Path.mkdir(exist_ok=False)` and returns the
experiment's id. The id is the next free integer, found by listing `data/ml-experiments/`. The
creation is the reservation: `data/` is one folder that every worktree shares, so two parallel
sessions cannot take the same id, and the loser retries with the next integer. The id is a
five-digit, zero-padded integer, which orders the experiments by the order in which they were
started.

The session writes `docs/ml-experiments/<id>-<slug>/result.md` as the **plan first**. The file
holds YAML front matter (the question, the status `planned`, the size, the benchmark it will beat,
the row set, the metric) and the body sections the plan needs: the arms, the planned contrasts with
their interval and decision rule, the budget for unplanned arms, and the stopping rule. The plan is
committed before the code runs, so the commit order is the proof that the contrasts were stated
first. The same file later becomes the write-up, so the experiment has one document and not two.

The session writes the experiment's code in `ml-experiments/<id>-<slug>/`. A fit script writes each
arm's predictions with `ml_experiments.store.write_predictions(experiment_id=..., name=...,
frame=..., manifest=...)`. The function writes a parquet file and a `manifest.json` beside it and
refuses to overwrite an existing set, because a set another experiment may be using as its
benchmark must not change underneath it. A re-run writes the next version, `name-v2`. The manifest
names the kind of output (`quantiles`, `members`, or `stage1`), the column list and dtypes, the row
set id, the fold, the training end date, the code's git commit, the seed, the scorer, and the
`consumes` list (the experiment id, set name, and content hash of every earlier predictions set the
run read).

A later experiment that wants a benchmark calls `ml_experiments.store.read_predictions(
experiment_id=..., name=...)`. The function reads the manifest, checks that the set's row set id
equals the new experiment's, and returns the frame. A mismatch raises, because two scores on
different rows are not a comparison. The new experiment scores the benchmark and every arm with one
scorer call, `ml_experiments.scoring.paired_contrasts(...)`, which returns each arm's absolute score
and each planned contrast's month-level paired interval with the Holm adjustment across the
experiment's planned contrasts. It always includes a month-by-half-hour climatology row, and it
checks that repeated ensemble runs of one valid time cannot sit on both sides of a training and
validation split (the pilot found a control that memorised the calendar position).

The session writes the figures (2 to 4, following the `dataviz` skill) and completes `result.md`:
the answer in one sentence, the winner with its interval, the absolute scores, the figures, and the
failures. The front matter's status becomes `done`, with the best arm and its headline number. A
script, `ml_experiments.index.build_index`, rebuilds `docs/ml-experiments/index.md` from every
front matter, one line per experiment, so a human and a Claude session can both scan the results.

#### Sizing and reviews

Step 0 of the skill sizes the experiment, and the size sets the reviews. The sizing reuses the
`plan-issue` triggers where they apply, and the review mechanics reuse `implement-issue` by
reference (the worktree setup, the green-before-push set, one commit per review, and the triage
rule). The skill does not reuse `implement-issue` as a whole, because that skill ends with a ready
PR and a stop for human review, and an experiment merges on its own under the conditions the `study`
skill sets.

- **Grind.** A permutation of settings the pipeline already has (input-feature sets, hyperparameters,
  a loss), no new data and no new method. There is no plan review. The session writes the plan
  section, implements, runs, and writes up. One Sonnet reviewer checks the code for leakage and
  that the fit stopped at the fold's `train_end` (the reviewer reads the code, never runs it), and
  one Sonnet reviewer checks the write-up against the printed report.
- **Standard.** A new method or one new input the repository already holds. The maintainer's seven
  steps: Sonnet writes the plan, Opus reviews the plan, Sonnet implements, Opus reviews the code,
  Sonnet runs and writes up, Sonnet reviews the write-up.
- **New data.** Everything in Standard, plus a download. The download follows the `data-download`
  and `data-validation` skills, and **a download that would cost more than £30 stops for the
  maintainer's approval** whatever the mode.
- **Fully autonomous mode is the default.** The session does not wait for the maintainer's review
  of the plan. When the maintainer asks, the session pushes the plan, says so, and stops until the
  maintainer replies. The skill says to ask for this only for a Standard or New-data experiment.

#### The overnight procedure

The maintainer and Claude agree the next night's experiments by day, each as a planned `result.md`
already committed. The overnight session runs them in order. It may add unplanned experiments up to
the budget the agreed plan sets (for example, one that combines the night's winners), and each is
labelled post hoc. An arm that the morning report would call a gain is re-run with more seeds, and
the report says "gain" only when the experiment's paired interval over the repeated seeds excludes
zero. The session writes one short morning report, `docs/ml-experiments/morning-reports/<date>.md`:
one line per experiment with its one-sentence answer and link, what failed, and the ideas it
proposes for the next plan. If a fit fails, the session records the failure in that experiment's
`result.md` and continues with the next experiment. When the night's budget (hours or experiments)
is spent, the session stops between experiments, never mid-fit, and writes the report.

### The main branches off the happy path

- **An experiment that cannot use the shared row set** (it needs a different fold or different rows)
  names its own row set id and says why in the plan. It may not claim a contrast against a benchmark
  that has a different row set id.
- **An experiment that qualifies for the leaderboard** (its output is `PowerForecast`-shaped and
  covers the reference experiment's row keys) may post through `scripts/forecasting/score_study.py`
  as the `study` skill describes. This is optional and not part of the default flow.
- **An experiment that consumes an earlier experiment's intermediate artifact** (for example
  forward-chained out-of-fold stage 1 predictions) lists it in the manifest's `consumes`. The
  artifact is immutable and versioned, so the dependency is stable.
- **A finalist** (an experiment whose winner might reach production) is scored on fold B and on the
  untouched holdout from July 2026 onward, once, by a separate step. Small experiments never touch
  the holdout.

### The main error-handling paths

- **`reserve_experiment` finds the folder exists.** It retries with the next integer. It never
  reuses an id.
- **`write_predictions` finds the set exists.** It raises `FileExistsError`. The session writes
  `name-v2`.
- **`read_predictions` finds a row set mismatch or a missing manifest.** It raises with both ids
  named. The session fixes the experiment's row set, or drops the contrast.
- **A fit fails overnight.** The failure goes into the experiment's `result.md` (status `failed`,
  exception text) and the night continues.
- **A reviewer finds leakage.** The experiment's status becomes `invalid`, its predictions stay
  (another experiment may have read them) and its manifest gains a `superseded_by` or `invalid`
  note, and the result page says so at the top.

## What changes, file by file

**Pull request 1, the package.**

- **`packages/ml_experiments/` (new)** with a README that states what the package owns and what it
  does not (it owns the store's mechanics and the default scorer; `studies` owns study machinery;
  `ml_core` owns the production scorer).
    - `store.py`: `reserve_experiment`, `write_predictions`, `read_predictions`, and the manifest
      model (a Pydantic model, since a manifest is configuration and not a data table).
    - `row_sets.py`: the shared benchmark row sets, each named by id and defined as init times,
      leads, members, series, and the fold's validation window, with the July 2026 holdout excluded
      from every row set. The first id wraps the stage 1 pilot's definition (see "Risks", item 3).
    - `scoring.py`: `paired_contrasts` (month-level paired t-interval, Holm adjustment, absolute
      scores, the climatology row) and the ensemble-leakage check. Thresholds are arguments.
    - `index.py`: `build_index`.
- **`packages/ml_experiments/tests/`**, each test asserting what fails on `main` today (the module
  does not exist): two `reserve_experiment` calls in two threads never return the same id;
  `write_predictions` refuses to overwrite; `read_predictions` raises on a row set mismatch;
  `paired_contrasts` on synthetic data with a known effect returns an interval that excludes zero
  and, on the null, includes it; Holm adjustment on three contrasts matches a hand-computed value;
  the leakage check flags a split that puts two runs of one valid time on both sides.
- **`packages/studies/tests/test_study_boundaries.py`** (or a sibling test in the new package): the
  same import rules for `ml-experiments/` as for `studies/`: an experiment script imports its own
  folder, `ml_experiments.*`, `studies.*`, and the other reviewed packages. Nothing in `src/` or a
  package other than these two imports `ml_experiments` or an experiment script.
- **`pyproject.toml` and `uv.lock`**: the workspace member.

**Pull request 2, the skill and the docs.**

- **`.claude/skills/ml-experiment/SKILL.md` (new).** Sections: when to use it and when to use
  `study`; step 0, sizing; the seven steps; where the pieces live; the store; the default scorer and
  what an experiment may change; the overnight procedure; anonymisation and the £30 rule.
- **`CLAUDE.md`**: a row in the skills table, the new folders in the Architecture section, and the
  import rules extended to `ml-experiments/`. Sweep every other summary that lists the skills.
- **`docs/ml-experiments/index.md` (new, generated)** and `mkdocs.yml`, with the index page in the
  navigation and each result page reached through the index. `docs/documentation-guide.md` gains a
  row for `ml-experiments/`.
- **`docs/roadmap/auto-research.md`**: rewrite the phase 1 section in the present tense for both
  kinds of work, remove the stale lines about the scorer that the merged
  [PR #1171 (Remove the scorer checks that only stop deliberate cheating)](https://github.com/openclimatefix/nged-substation-forecast/pull/1171)
  left behind, and leave phase 2 alone.
- **`docs/ml_experimentation/index.md`**: one cross-link, because the names are close
  (`ml_experimentation` describes running the production pipeline's cross-validation; `ml-experiments`
  holds the agent's small experiments).

## Design-philosophy check

- **R&D, fail fast.** Every function here runs in research, never in production: the store refuses
  to overwrite, the reader raises on a mismatch, and a failed fit is named in the result page.
- **Nothing in `src/` or the Dagster asset graph changes**, and no Patito contract changes.
  Experiments have their own output schema, because `PowerForecast` cannot hold their outputs.
- **Anonymisation.** `docs/` is published, so an experiment's `result.md` and figures never carry a
  generator's name, identifier, or time series. Predictions stay under `data/`, which is private.
- **No principle is traded away.** The store trades a single predictions table for several
  per-experiment sets. It buys schema freedom, immutability, and the ability to reuse a benchmark,
  and it gives up one place to query every forecast.

## Tests

See the package's tests above. The skill and the docs have no tests of their own; `mkdocs build
--strict`, the docs-link checker, and `pymarkdown` check them. A dry run of the skill, on one real
Grind experiment (a feature-set permutation chosen by the maintainer), is the end-to-end test, run
after pull request 2.

## Docs to update

`CLAUDE.md`, `docs/documentation-guide.md`, `docs/roadmap/auto-research.md`,
`docs/ml_experimentation/index.md`, `mkdocs.yml`, and the new docs tree. This issue does not
complete a roadmap item, so there is no ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check
uv run pytest
uv run pytest --run-studies packages/studies    # the boundary test lives there
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict    # and read the rendered index and one result page
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files
```

## Risks and open questions

1. **Where the code lives.** I recommend `ml-experiments/<id>-<slug>/` for each experiment's scripts
   (beside `studies/`) and a new package `packages/ml_experiments/` for the shared mechanics.
   The alternative is to put the mechanics in `packages/studies/`. That avoids a package but names
   the code wrongly and ties the two kinds of work together. **Decision for the maintainer.**
2. **May an experiment import `studies.*`?** I recommend yes, so that `bootstrap`, the arm runner and
   the checkpointed fits are not duplicated. This makes `ml_experiments` and `studies` both leaves
   that experiments may import, and neither imports the other.
3. **The shared row set.** The stage 1 pilot scores Quick wins' fold A unchanged: ENS runs at 00:00 UTC
   every 7 days from 2025-07-01 to 2026-06-30 (53 runs, all Tuesdays), forecast origin 09:00 UTC, all
   51 members, valid times to about 14.6 days, the 9 generator series (6 solar, 3 wind), errors
   divided by each series' effective capacity. The stride is `SAMPLE_STRIDE_DAYS = 7` in
   `studies/quick_wins/quick_wins_inputs.py`, and the pilot's scored cases are the member-0 rows
   with a finite observed power. A fixed weekday suits generators but not demand, which depends on
   weekday. The planning session for PR #1183 recommends a stride of 5 or 6 days. **Decision for the
   maintainer:** (a) adopt the pilot's Tuesday row set as `fold_a_tuesdays_v1` for the first
   generator experiments, which needs no rebuild and lets them reuse the pilot's frames, and add a
   rotating-weekday row set before the first demand experiment; or (b) rebuild the frames now with a
   stride of 6 days. I recommend (a). The row set module records the definition by id either way, so
   the later set is one more id and not a migration.
4. **Default decision rule.** The machinery is the default (contrasts and interval pre-registered, a
   month-level paired t-interval, Holm across the experiment's planned contrasts, absolute scores
   beside every contrast, a failures section). The thresholds belong to each experiment, because
   they are tied to CRPS on a 12-month fold. Fold B and the holdout apply to finalists only.
5. **Outputs differ between experiments.** The planning session for PR #1183 expects a long-format
   quantile table (series, init time, valid time, lead, product, quantile level, value), a separate
   members table, and a stage 1 table. The manifest's `kind` carries this. The package ships the
   manifest and the reader, and leaves the table schemas to the first experiment that needs each.
6. **The index is generated, and nothing enforces that it is current.** I recommend a test that
   fails when `build_index` would change the committed file.
7. **A hundred result pages in `docs/` make the site slow to build and noisy to search.** I
   recommend keeping only the index in the navigation, which `mkdocs --strict` accepts, and
   revisiting if the count passes a few hundred.
8. **The `ml-experiments` and `ml_experimentation` names are close.** I recommend the maintainer's
   chosen `ml-experiments` and the cross-link above.
9. **The experiment merges on its own** under the `study` skill's three conditions, adapted: the
   diff touches only `ml-experiments/`, `docs/ml-experiments/`, and the experiment's tests; every
   line of code has had the review its size calls for; the checks pass. A change to
   `packages/ml_experiments/` stops at a reviewed PR.
10. **The plan is the first half of `result.md`.** An alternative is a separate plan file per
    experiment. A single file is simpler, and the git history shows the plan as first committed.
