# Plan: the `ml-experiment` skill and the store it writes to (#1186)

**The problem.** Phase 1 of agentic ML research needs the agent to run many small experiments
overnight, each aimed at improving the Flexpectation production forecast. The `study` skill is the
wrong tool for them. A study is a long report that answers several scientific questions, with two
Opus scientific-validity reviews, persona reviews, and a charted page. An experiment that answers
one question, such as "does adding ERA5 pre-training improve the PV forecast?", needs a page of about
a screen. The repository also has nowhere to keep an experiment's predictions so that a later
experiment can reuse them as the benchmark to beat, and no rule for what an experiment may score on.
An experiment with its own metric or its own probabilistic output cannot post to the leaderboard.

**The solution.** A new skill, `ml-experiment`, that sizes each experiment and runs the matching
reviews, plus a store with three parts: code in `ml-experiments/<id>-<slug>/`, one short write-up in
`docs/ml-experiments/<id>-<slug>/result.md`, and predictions in
`data/ml-experiments/<id>-<slug>/predictions/` with a small manifest. One new module,
`packages/studies/src/studies/experiment_store.py`, owns the store's mechanics, and the interval
and Holm adjustment live beside the existing month bootstrap. The `study` skill is left alone,
apart from the boundary sentence that
[issue #1168 (Make the study skill say when a study posts to the leaderboard)](https://github.com/openclimatefix/nged-substation-forecast/issues/1168)
adds.

## Verdict, size, and departures

- **Verdict: worth implementing.** The maintainer has asked for it. No store exists for predictions
  outside `power_forecasts`, whose `PowerForecast` contract cannot hold 19-level quantile outputs or
  reordered ensemble members, and a contract change needs the maintainer's say-so.
- **Size: complex.**
    - What gets stored: fires. A new store of predictions and manifests under `data/`, and a new
      docs tree.
    - Production serving path: does not fire.
    - Degradation rule: does not fire.
    - More than one defensible design: fires. The layout, the manifest, and where the code lives
      each have an alternative.
    - Callers not nameable without searching: does not fire. The module is new, and nothing calls
      it yet.
- **Reviews: both plan reviews, the simplicity review (Sonnet) and the correctness review (Opus).**
  The diff later gets the reviews `implement-issue` prescribes for the module's code, including the
  mutation pass on `experiment_store.py` and `holm_adjust`.
- **Delivery: one pull request.** The simplicity review found that a new package was the only
  reason for two pull requests.
- **Departures from the issue body:** the issue lists "the overnight procedure" and the roadmap
  rewrite; this plan keeps both but shrinks the roadmap edit to the lines that become false. The
  issue does not mention the morning-report file, the generated index, or a new package, and the
  plan drops all three (see "Findings considered and rejected").

## Solution

### Walk through

#### The happy path

To run an experiment, the session first sizes the work (step 0 of the skill; see "Sizing and
reviews"). It then calls `studies.experiment_store.reserve_experiment(slug="input-features-for-pv")`,
which creates `data/ml-experiments/<id>-<slug>/` with `Path.mkdir(exist_ok=False)` and returns the
experiment's id. The id is the next free integer, found by listing `data/ml-experiments/`. The
creation is the reservation: `data/` is one folder that every worktree shares, so two sessions
cannot take the same id, and the loser retries with the next integer. The id is a five-digit,
zero-padded integer, which orders experiments by the order in which they were started.

The session writes `docs/ml-experiments/<id>-<slug>/result.md` as the **plan first**. The file
holds YAML front matter (the question, the status `planned`, the size, the benchmark it will beat,
the row set id, the metric) and the sections the plan needs: the arms, the planned contrasts with
their interval and decision rule, the budget for unplanned arms, and the stopping rule. The plan is
committed before the code runs, so the commit order is the proof that the contrasts were stated
first. The same file later becomes the write-up, so the experiment has one document and not two.

The session writes the experiment's code in `ml-experiments/<id>-<slug>/`. A fit script writes each
arm's predictions with `experiment_store.write_predictions(experiment_id=..., name=..., frame=...,
kind=..., row_set_id=...)`. The function writes a parquet file and a `manifest.json` beside it. The
manifest holds three keys: `kind` (`quantiles`, `members`, or `stage1`), `row_set_id`, and the
code's git commit. The function raises `FileExistsError` when the set exists, because another
experiment may be using the set as its benchmark. A re-run is saved under a new name, such as
`name-v2`, which is a naming habit and needs no code.

A later experiment that wants a benchmark calls `experiment_store.read_predictions(
experiment_id=..., name=..., row_set_id=...)`. The function reads the manifest and raises when the
set's `row_set_id` differs from the one passed, because two scores on different rows are not a
comparison. The experiment scores the benchmark and every arm with the existing
`studies.bootstrap.bootstrap_difference` (the month-resampled paired interval that the `study`
skill already requires), and adjusts the planned contrasts' p-values with the new
`studies.bootstrap.holm_adjust`. The skill tells every experiment to include a month-by-half-hour
climatology row (from `studies.baselines`) and to check that repeated ensemble runs of one valid
time cannot sit on both sides of a training and validation split, because the stage 1 pilot found a
control that memorised the calendar position. The leakage check is a short function that the first
ensemble experiment writes, with a test, in `ml-experiments/<id>-<slug>/` and then promotes into
`studies` when a second experiment needs it.

The session writes 2 to 4 figures (following the `dataviz` skill) and completes `result.md`: the
answer in one sentence, the winner with its interval, the absolute scores, the figures, and the
failures. The front matter's status becomes `done`, with the best arm and its headline number.
Because every result has the same front matter, Claude finds results with `grep` over
`docs/ml-experiments/*/result.md`, and a human reads the folder listing, whose integer prefixes
order the experiments.

#### Sizing and reviews

Step 0 of the skill sizes the experiment with the `plan-issue` triggers (what gets stored, whether
it touches production, more than one defensible design). Review mechanics reuse `implement-issue`
by reference: the worktree setup, the green-before-push set, one commit per review, and the triage
rule. The skill does not reuse `implement-issue` as a whole, because that skill ends with a ready PR
and a stop for human review, and an experiment merges on its own under the conditions the `study`
skill sets.

- **Simple** (a permutation of settings the pipeline already has, such as input-feature sets, a
  hyperparameter, or a loss; no new data and no new method). The session writes the plan section,
  implements, runs, and writes up. One Sonnet reviewer reads the code for leakage and for the fit
  stopping at the fold's `train_end`, and one Sonnet reviewer checks the write-up against the printed
  report.
- **Otherwise,** the maintainer's seven steps: Sonnet writes the plan, Opus reviews the plan, Sonnet
  implements, Opus reviews the code, Sonnet runs and writes up, Sonnet reviews the write-up.
- **A download** follows the `data-download` and `data-validation` skills, and **a download that
  would cost more than £30 stops for the maintainer's approval**, in every mode.
- **Fully autonomous mode is the default.** The session does not wait for the maintainer to review
  the plan. When the maintainer asks, the session pushes the plan, says so, and waits for a reply.

#### The overnight procedure

The maintainer and Claude agree the next night's experiments by day, each as a planned `result.md`
already committed. The overnight session runs them in order, and may add unplanned experiments up to
the budget the agreed plan sets (for example, one that combines the night's winners), each labelled
post hoc. A contrast counts as a gain by the `study` skill's rule, a month-and-seed resampled
interval over repeated seeds. If a fit fails, the session records the failure in that experiment's
`result.md` (status `failed`, with the exception text) and continues with the next experiment. When
the night's budget (hours or experiments) is spent, the session stops between experiments, never
mid-fit. The session's final message is the morning report: one line per experiment with its
one-sentence answer and link, what failed, and the ideas it proposes for the next plan. Each
`result.md` is the durable record.

### The main branches off the happy path

- **An experiment that needs a different row set** names its own `row_set_id` and says why in the
  plan. It cannot claim a contrast against a benchmark with another `row_set_id`.
- **An experiment that qualifies for the leaderboard** (its output has `PowerForecast`'s columns and
  covers the reference experiment's row keys) may post through `scripts/forecasting/score_study.py`
  as the `study` skill describes. Posting is optional and outside the default flow.
- **A finalist** (an experiment whose winner might reach production) is scored once on the holdout
  from July 2026 onward by a separate step that the maintainer defines. Small experiments never
  touch the holdout.

### The main error-handling paths

- **`reserve_experiment` finds the folder exists.** It retries with the next integer and never
  reuses an id.
- **`write_predictions` finds the set exists.** It raises `FileExistsError`. The session saves the
  new fit as `name-v2`.
- **`read_predictions` finds a different `row_set_id` or no manifest.** It raises with both ids
  named. The session fixes the experiment's row set or drops the contrast.
- **A fit fails overnight.** The failure goes into that experiment's `result.md`, and the night
  continues.
- **A reviewer finds leakage.** The experiment's status becomes `invalid`, and the first line of
  `result.md` says so. The predictions stay, because another experiment may have read them.

## What changes, file by file

- **`packages/studies/src/studies/experiment_store.py` (new)**: `reserve_experiment`,
  `write_predictions`, `read_predictions`, and the shared first row set id, `fold_a_tuesdays_v1`,
  documented as the stage 1 pilot's definition (see "Risks", item 3). The July 2026 holdout is
  excluded by the row set's definition. `read_predictions` asserts that no row falls on or after
  2026-07-01.
- **`packages/studies/src/studies/bootstrap.py`**: add `holm_adjust`.
- **`packages/studies/tests/test_experiment_store.py` (new)** and a case in the bootstrap tests. Each
  test asserts what fails on `main` today (the functions do not exist): two sequential
  `reserve_experiment` calls return different ids, and a call after a manual `mkdir` of the next id
  skips it; `write_predictions` raises on the second write of one name; `read_predictions` raises on
  a `row_set_id` mismatch and on a frame holding a row from 2026-07-01 onward; `holm_adjust` on
  three p-values matches a hand-computed list.
- **`packages/studies/tests/test_study_boundaries.py`**: extend the existing rules to
  `ml-experiments/`. A script there imports its own folder, `studies.*`, and the other reviewed
  packages, and nothing in `src/` or another package imports `ml-experiments/`. This test always
  runs.
- **`.claude/skills/ml-experiment/SKILL.md` (new).** When to use it and when to use `study`; step
  0, sizing; the seven steps; where the pieces live; the store; the default contrasts and the
  climatology and leakage rules; the overnight procedure; anonymisation and the £30 rule. It links
  to the `study` skill for seeds, interval rules, and optional leaderboard posting, instead of
  restating them.
- **`CLAUDE.md`**: a row in the skills table, and one clause in the "Import rules for studies"
  paragraph that `ml-experiments/` follows the same rules. Sweep every other summary of the skills.
- **`docs/ml-experiments/README.md` (new)**: three sentences on the folder layout and the front
  matter, with no table of experiments. The front matter is the index.
- **`docs/documentation-guide.md`**: one row for `ml-experiments/`.
- **`docs/roadmap/auto-research.md`**: point phase 1 at the `ml-experiment` skill, and rewrite the
  scorer sentences at lines 207 to 210 that the merged
  [PR #1171 (Remove the scorer checks that only stop deliberate cheating)](https://github.com/openclimatefix/nged-substation-forecast/pull/1171)
  left false. Phase 2 stays.

Not changed: `mkdocs.yml` navigation (result pages are reached by search and the folder listing;
`mkdocs build --strict` is run to confirm that unnavigated pages do not fail the build),
`docs/ml_experimentation/` (the skill's description states the distinction), and `scripts/`.

## Design-philosophy check

- **R&D, fail fast.** Every function runs in research and never in production. The store refuses to
  overwrite, the reader raises on a mismatch, and a failed fit is named in its result page.
- **Nothing in `src/` or the Dagster asset graph changes**, and no Patito contract changes.
- **Anonymisation.** `docs/` is published, so `result.md` and its figures never carry a generator's
  name, identifier, or time series. Predictions stay under `data/`, which is private.
- **No principle is traded away.** Several per-experiment prediction sets replace one predictions
  table. That buys schema freedom, immutability, and benchmark reuse, and gives up one place to
  query every forecast.

## Tests

See the module's tests above. The skill and the docs have no tests of their own; `mkdocs build
--strict`, the docs-link checker, and `pymarkdown` check them. A dry run of the skill on one real
simple experiment, a feature-set permutation chosen by the maintainer, is the end-to-end test, run
after the pull request merges.

## Docs to update

`CLAUDE.md`, `docs/documentation-guide.md`, `docs/roadmap/auto-research.md`, and the new
`docs/ml-experiments/README.md`. This issue does not complete a roadmap item, so there is no
ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run --all-packages ty check
uv run pytest
uv run pytest --run-studies packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict    # and confirm result pages outside the nav do not fail the build
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files
```

## Risks and open questions

1. **The code lives in `packages/studies/` for now.** The maintainer asked for `ml-experiment` in the
   directory names. `experiment_store.py` is one module, and `git mv` into a `packages/ml_experiments/`
   package is cheap once it passes about 200 lines or a second experiment kind needs its own helpers.
   **Decision for the maintainer:** accept that, or create the package now. I recommend accepting.
2. **An experiment's scripts may import `studies.*`.** The boundary rules for `ml-experiments/` are
   the same as for `studies/`.
3. **The shared row set.** The stage 1 pilot scores Quick wins' fold A unchanged: ENS runs at 00:00
   UTC every 7 days from 2025-07-01 to 2026-06-30 (53 runs, all Tuesdays), forecast origin 09:00
   UTC, all 51 members, valid times to about 14.6 days, the 9 generator series (6 solar, 3 wind),
   errors divided by each series' effective capacity. The stride is `SAMPLE_STRIDE_DAYS = 7` in
   `studies/quick_wins/quick_wins_inputs.py`, and the scored cases are the member-0 rows with a
   finite observed power. A fixed weekday suits generators but not demand, which depends on
   weekday. The planning session for PR #1183 recommends a stride of 5 or 6 days. **Decision for the
   maintainer:** (a) adopt the pilot's Tuesday row set as `fold_a_tuesdays_v1` for the first
   generator experiments, which needs no rebuild and lets them reuse the pilot's frames, and add a
   rotating-weekday row set before the first demand experiment; or (b) rebuild the frames now with a
   stride of 6 days. I recommend (a).
4. **Default decision rule.** The machinery is the default: contrasts and interval pre-registered,
   the month-resampled paired interval, Holm across the experiment's planned contrasts, absolute
   scores beside every contrast, and a failures section. The thresholds (for example the 0.15
   CRPS-point margin that the staged-forecast plan uses) belong to each experiment, because they are
   tied to CRPS on a 12-month fold. Fold B and the holdout apply to finalists only.
5. **Outputs differ between experiments.** The planning session for PR #1183 expects a long-format
   quantile table (series, init time, valid time, lead, product, quantile level, value), a separate
   members table, and a stage 1 table. The manifest's `kind` carries this, and the table schemas
   are left to the first experiment that needs each.
6. **No generated index.** A hand-kept table of experiments would conflict whenever two sessions
   finish together, and a generator needs a freshness test. `grep` over the front matter does the
   job for the first 30 experiments or so; revisit then.
7. **The `ml-experiments` and `ml_experimentation` names are close.** I recommend the maintainer's
   chosen `ml-experiments`, and the skill's description states the distinction.
8. **An experiment merges on its own** under the `study` skill's three conditions, adapted: the diff
   touches only `ml-experiments/`, `docs/ml-experiments/`, and the experiment's own tests; the
   experiment has had the review its size calls for; the checks pass. A change to
   `packages/studies/` stops at a reviewed PR.

## Findings considered and rejected

Findings of the simplicity review (Sonnet) that the plan takes: no new package, no `row_sets.py`
registry, no generated index, a three-key manifest without `consumes`, two sizes, no morning-report
file, no finalist procedure, no roadmap rewrite beyond the false lines, no cross-link in
`docs/ml_experimentation/`, one pull request, reuse of `studies.bootstrap`, and no threaded race
test. Rejected:

- **Replace the generated index with a hand-kept table** in `docs/ml-experiments/README.md`:
  rejected, because two sessions finishing together would conflict on that file. The plan drops the
  index and relies on the front matter.
- **Drop the holdout assertion** in `read_predictions`: rejected, because it is one line that makes
  the holdout rule enforceable, and the row-set definition alone would not stop a frame built
  elsewhere.
