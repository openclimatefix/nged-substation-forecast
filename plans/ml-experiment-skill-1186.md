# Plan: the `ml-experiment` skill, framed around testing one hypothesis, and the store it writes to (#1186)

**The problem.** Phase 1 of agentic ML research needs the agent to run many small experiments
overnight, each aimed at improving the Flexpectation production forecast. The `study` skill is the
wrong tool for them. A study is a long report that answers several scientific questions, with two
Opus scientific-validity reviews, persona reviews, and a charted page. An experiment that tests one
idea, such as "adding ERA5 pre-training lowers the PV forecast's CRPS", needs a page of about a
screen. The repository also has nowhere to keep an experiment's predictions so that a later
experiment can reuse them as the benchmark to beat, and no rule for what an experiment may score on.
An experiment with its own metric or its own probabilistic output cannot post to the leaderboard.

**The solution.** A new skill, `ml-experiment`, in which **every experiment tests one stated
hypothesis**. The hypothesis is a falsifiable claim with a prediction and a decision rule, written
before the run, and the experiment ends in a verdict: supported, refuted, or inconclusive. The skill
sizes each experiment and runs the matching reviews. A store with three parts holds the work: code in
`ml-experiments/<id>-<slug>/`, one short write-up in `docs/ml-experiments/<id>-<slug>/result.md`
(the hypothesis record), and predictions in `data/ml-experiments/<id>-<slug>/predictions/` with a
small manifest. One new module, `packages/studies/src/studies/experiment_store.py`, owns the store's
mechanics. The `study` skill is left alone, apart from the boundary sentence that
[issue #1168 (Make the study skill say when a study posts to the leaderboard)](https://github.com/openclimatefix/nged-substation-forecast/issues/1168)
adds.

**Why hypotheses.** The roadmap's phase 2 already plans a "hypothesis store": one Markdown file per
idea, holding the hypothesis, each implementation and its score, a status, and lessons learned
([issue #1034 (Set up the auto-research hypothesis store)](https://github.com/openclimatefix/nged-substation-forecast/issues/1034)).
An ml-experiment's `result.md` has the same fields. Framing phase 1 around hypotheses means that the
experiments phase 1 produces are already the records phase 2's research lead will read, and the two
phases share one format. Pre-registering the hypothesis also gives the experiment its decision rule
and its stopping point.

## Verdict, size, and departures

- **Verdict: worth implementing.** The maintainer has asked for it. No store exists for predictions
  outside `power_forecasts`, whose `PowerForecast` contract cannot hold the 19-level calibrated
  quantile table, a per-product table, or stage 1 outputs. (The final 51 ECC-coupled members would
  fit `PowerForecast`, because `ensemble_member` already has a per-forecaster meaning; that is the
  staged-forecast plan's decision.)
- **Size: complex.**
    - What gets stored: fires. A new store of predictions and manifests under `data/`, and a new
      docs tree.
    - Production serving path: does not fire.
    - Degradation rule: does not fire.
    - More than one defensible design: fires. The layout, the manifest, and where the code lives
      each have an alternative.
    - Callers not nameable without searching: does not fire. The module is new, and nothing calls
      it yet.
- **Reviews: both plan reviews, the simplicity review (Sonnet) and the correctness review (Opus),
  have run.** The diff later gets the reviews `implement-issue` prescribes, with the mutation pass on
  `experiment_store.py`.
- **Delivery: one pull request.**
- **Departures from the issue body:** the plan adds the hypothesis framing, which the maintainer
  proposed after the issue was written, and shrinks the roadmap edit to the lines that become
  false. The morning-report file, the generated index, and a new package, which the first draft had,
  are dropped.

## Solution

### Walk through

#### The happy path

To run an experiment, the session first writes down a **hypothesis** and sizes the work (step 0 of
the skill; see "Sizing and reviews"). The hypothesis has a claim ("adding ERA5 pre-training lowers
the PV CRPS"), a prediction that could fail (a size and direction of effect), and a decision rule
(what interval, against which benchmark, would count as support).

The session calls `studies.experiment_store.reserve_experiment(slug="era5-pretraining-for-pv")`.
The function validates the slug against `^[a-z0-9]+(-[a-z0-9]+)*$`, then finds the next id as the
maximum five-digit prefix over three places plus one: the folders of `data/ml-experiments/`, the
reservation markers in `data/ml-experiments/.ids/`, and the folders of `docs/ml-experiments/` in the
main checkout (so a lost data folder cannot cause an id to be reused). It reserves the **bare id**
by creating `data/ml-experiments/.ids/<id>` with `Path.mkdir(exist_ok=False)`, which is atomic on a
local filesystem, and retries with the next integer if the marker exists. Reserving the bare id, not
the id plus the slug, is what stops two sessions that choose different slugs from taking the same
id. The function then creates `data/ml-experiments/<id>-<slug>/` and returns the id. `data/` is one
folder that every worktree shares (`studies.sources.REPO_DATA_DIR` resolves linked worktrees to the
main checkout), so the reservation holds across sessions. The id is a five-digit, zero-padded
integer, so experiments sort in the order they were started. The root folder is a constant in
`studies.sources`, so tests monkeypatch it and never write into the real data folder.

The session writes `docs/ml-experiments/<id>-<slug>/result.md` as the **plan first**. The file's
YAML front matter holds: the hypothesis, the prediction, the decision rule, the status `planned`,
the size, the benchmark it will beat, the row set id, the metric, and, once finished, the **verdict**
(`supported`, `refuted`, `inconclusive`, `invalid`, or `failed`), the best arm with its headline
number, and the lessons learned. The body holds the arms, the planned contrasts, the budget for
unplanned arms, and the stopping rule. The plan is committed before the code runs, so the commit
order is the proof that the hypothesis was stated first. The same file later becomes the write-up,
so the experiment has one document.

The experiment's code lives in `ml-experiments/<id>-<slug>/`. Every script basename is unique across
`studies/` and `ml-experiments/`, so the skill prefixes each with the id in a valid module form:
`e00012_fit.py`, `e00012_report.py`. A fit script writes each arm's predictions with
`experiment_store.write_predictions(experiment_id=..., name=..., frame=..., kind=...,
row_set_id=...)`. The function requires a `valid_time` column, writes the parquet file and a
`manifest.json` into a temporary folder, and renames the folder into place with the manifest written
last, so a crash cannot leave a set that can be neither read nor rewritten. The manifest holds four
keys: `kind` (`quantiles`, `members`, or `stage1`), `row_set_id`, the code's git `commit`, and
`dirty` (true if the working tree had uncommitted changes). The function raises `FileExistsError`
when the set exists, because another experiment may be using the set as its benchmark. A re-run is
saved as `name-v2`, which is a naming habit and needs no code.

A later experiment that wants a benchmark calls `experiment_store.read_predictions(
experiment_id=..., name=..., row_set_id=...)`. The function reads the manifest and raises
`ValueError` when the set's `row_set_id` differs from the one passed, when the manifest is missing,
or when any `valid_time` falls on or after 2026-07-01 (the holdout; see "Risks", item 3). It raises
and does not `assert`, because `python -O` strips assertions.

The experiment scores with the machinery that exists. `experiment_store.losses_for_bootstrap(...)`
turns two prediction sets and the observed power into the frame `studies.bootstrap` expects. It
builds the composite key that `paired_differences` pairs on (run and valid time, because a valid time
sits in 2 or 3 runs), takes the month label from the run's init time so that a run stays in one
block, and raises unless the two sets hold the same seeds. `studies.bootstrap.bootstrap_difference_at_level`
then gives each planned contrast's month-resampled paired interval, at a level the Bonferroni
correction sets across the experiment's planned contrasts, with `n_resamples` raised so that each
tail holds enough draws. The climatology benchmark row comes from
`baseline_forecasters.climatology.ClimatologyForecaster`, trained on the fold's training window and
giving 51 quantiles per valid time. The skill states one leakage rule: **a model must not be able to
memorise the calendar position**. The stage 1 pilot found a control whose error changed by 1.1 to
3.4 points when hour and day-of-year columns were removed. So an experiment either uses no calendar
columns, or includes a control without them, and any inner early-stopping or cross-validation split
must group by valid time, because repeated ensemble runs of one valid time would otherwise sit on
both sides of the split.

The session writes 2 to 4 figures (following the `dataviz` skill) and completes `result.md`: the
verdict in one sentence, the winner with its interval, the absolute scores, the figures, the
failures, and the lessons. The status becomes `done`. Because every result has the same front
matter, Claude finds results with `grep` over `docs/ml-experiments/*/result.md`, and a human reads
the folder listing, whose integer prefixes order the experiments.

#### Sizing and reviews

Step 0 of the skill sizes the experiment with all five `plan-issue` triggers: what gets stored,
whether it touches the production serving path, whether it touches a degradation rule, whether more
than one design would defensibly satisfy the hypothesis, and whether the experiment spans code whose
callers cannot be named without searching. Review mechanics reuse `implement-issue` by reference:
the worktree setup, the green-before-push set, one commit per review, and the triage rule. The skill
does not reuse `implement-issue` as a whole, because that skill ends with a ready PR and a stop for
human review, and an experiment merges on its own under the conditions the `study` skill sets.

- **Simple** (a permutation of settings the pipeline already has, such as input-feature sets, a
  hyperparameter, or a loss; no new data and no new method). The session writes the plan section,
  implements, runs, and writes up. One Sonnet reviewer reads the code for leakage and for the fit
  stopping at the fold's `train_end`, and one Sonnet reviewer checks the write-up against the printed
  report.
- **Otherwise,** the maintainer's process: Sonnet writes the plan, Opus reviews the plan, optionally
  the maintainer reviews the plan, Sonnet implements, Opus reviews the code, Sonnet runs and writes
  up, Sonnet reviews the write-up.
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
`result.md` (verdict `failed`, with the exception text) and continues with the next experiment. When
the night's budget (hours or experiments) is spent, the session stops between experiments, never
mid-fit. The session's final message is the morning report: one line per experiment with its verdict
and link, what failed, and the hypotheses it proposes for the next plan. Each `result.md` is the
durable record.

### The main branches off the happy path

- **An experiment that needs a different row set** names its own `row_set_id` and says why in the
  plan. It cannot claim a contrast against a benchmark with another `row_set_id`.
- **An experiment that qualifies for the leaderboard** (its output has `PowerForecast`'s columns and
  covers the reference experiment's row keys) may post through `scripts/forecasting/score_study.py`
  as the `study` skill describes. Posting is optional and outside the default flow.
- **A finalist** (an experiment whose winner might reach production) is scored once on the holdout
  by a separate step that the maintainer defines. Small experiments never touch the holdout.

### The main error-handling paths

- **`reserve_experiment` finds the marker exists.** It retries with the next integer, and never
  reuses an id.
- **`write_predictions` finds the set exists.** It raises `FileExistsError`. The session saves the
  new fit as `name-v2`.
- **`read_predictions` finds a different `row_set_id`, no manifest, or a holdout row.** It raises
  with the details named. The session fixes the experiment's row set or drops the contrast.
- **`losses_for_bootstrap` finds different seeds.** It raises, and the session re-runs the benchmark
  with matching seeds or uses a one-seed contrast and says so.
- **A fit fails overnight.** The failure goes into that experiment's `result.md`, and the night
  continues.
- **A reviewer finds leakage.** The verdict becomes `invalid`, and the first line of `result.md` says
  so. The predictions stay, because another experiment may have read them.

## What changes, file by file

- **`packages/studies/src/studies/experiment_store.py` (new)**: `reserve_experiment`,
  `write_predictions`, `read_predictions`, `losses_for_bootstrap`, and the first row set id,
  `fold_a_tuesdays_v1` (the stage 1 pilot's definition; see "Risks", item 3).
- **`packages/studies/src/studies/sources.py`**: a constant for `data/ml-experiments/`, with a
  docstring saying why it sits beside and not under `data/studies/`.
- **`packages/studies/tests/test_experiment_store.py` (new).** Each test asserts what fails on `main`
  today (the functions do not exist):
    - `reserve_experiment` skips an id whose marker exists, even when the existing folder has a
      different slug, and skips an id present only in `docs/ml-experiments/`; it rejects a slug
      containing `/` or `..`.
    - `write_predictions` raises on the second write of one name, and leaves no readable set when
      the write is interrupted before the manifest (simulated by monkeypatching the manifest write
      to raise).
    - `read_predictions` raises on a `row_set_id` mismatch, on a missing manifest, and on a
      `valid_time` of 2026-07-01 or later.
    - `losses_for_bootstrap` gives one row per run and valid time (a valid time in three runs gives
      three rows, not a many-to-many blow-up), and raises on unequal seed sets.
- **`packages/studies/tests/test_study_boundaries.py`**: run every existing rule over the union of
  `studies/` and `ml-experiments/`, so that an experiment script importing a study script by bare
  name, and a basename shared between the two trees, are both caught. Add the experiment scripts'
  basenames to the forbidden set for production imports, and scan `ml-experiments/` in
  `test_no_study_names_the_raw_power_table`. Add `tmp_path` tests with a synthetic offender for each
  extended rule, because the real-tree assertions pass on `main`.
- **`ml-experiments/README.md` (new)**: so the folder exists for the test and states the layout in
  three sentences.
- **`pyproject.toml`**: add `ml-experiments/<id>-<slug>` folders to pytest's `pythonpath` and ty's
  `extra-paths` as `studies/` does, if experiment scripts are to be imported by tests; the plan's
  default is no tests of experiment scripts, as for studies, so the change is a glob only.
- **`.github/workflows/studies_tests.yml`**: add `ml-experiments/**` and `docs/ml-experiments/**` to
  the trigger paths.
- **`.claude/skills/ml-experiment/SKILL.md` (new).** When to use it and when to use `study`; the
  hypothesis; step 0, sizing; the review process; where the pieces live and the `e<id>_` basename
  rule; the store; the default contrasts, the climatology benchmark, the calendar-memorisation rule,
  and the interval mapping; the overnight procedure; anonymisation (a published page never names or
  numbers a generator beside a time series, and uses anonymised labels, as `CLAUDE.md` says) and the
  £30 rule. It links to the `study` skill for seeds, interval rules, and optional leaderboard
  posting, and does not restate them.
- **`CLAUDE.md`**: a row in the skills table, and one clause in the "Import rules for studies"
  paragraph that `ml-experiments/` follows the same rules. Sweep every other summary of the skills.
- **`packages/studies/README.md`, `studies/README.md`, `docs/architecture/testing.md`**: add
  `experiment_store` to the module list, and mention `ml-experiments/` where the boundary and path
  rules are described.
- **`docs/ml-experiments/README.md` (new)**: three sentences on the layout and the front matter,
  with no table of experiments. The front matter is the index.
- **`docs/documentation-guide.md`**: one row for `ml-experiments/`.
- **`docs/roadmap/auto-research.md`**: rewrite line 214 ("Overnight, a Claude session runs the plan
  with the `study` skill") and lines 219 to 224 ("Each study will post its scores to the
  leaderboard"), and add a sentence that phase 1's experiments are hypothesis records in the same
  format that phase 2's store will use. Phase 2 stays.

Not changed: `mkdocs.yml` navigation (result pages are reached by search and the folder listing;
`mkdocs build --strict` is run to confirm that unnavigated pages do not fail the build),
`docs/ml_experimentation/`, and `scripts/`.

## Design-philosophy check

- **R&D, fail fast.** Every function runs in research and never in production. The store refuses to
  overwrite, the reader raises on a mismatch, and a failed fit is named in its result page.
- **Nothing in `src/` or the Dagster asset graph changes**, and no Patito contract changes.
- **Anonymisation.** `docs/` is published, so `result.md` and its figures carry no generator's name
  or identifier. Predictions stay under `data/`, which is private.
- **Hypotheses.** An experiment's hypothesis is local to the experiment and carries no `H` or `T`
  label. A verdict that bears on a labelled test in `engineering-hypotheses.md` links to that test.
- **No principle is traded away.** Several per-experiment prediction sets replace one predictions
  table. That buys schema freedom, immutability, and benchmark reuse, and gives up one place to
  query every forecast.

## Tests

See the module's tests above. The skill and the docs have no tests of their own; `mkdocs build
--strict`, the docs-link checker, and `pymarkdown` check them. A dry run of the skill on one real
simple experiment, a feature-set permutation chosen by the maintainer, is the end-to-end test, run
after the pull request merges.

## Docs to update

The files listed above. This issue does not complete a roadmap item, so there is no ship-time
triage.

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
   directory names. `experiment_store.py` is one module, and `git mv` into a
   `packages/ml_experiments/` package is cheap once it passes about 200 lines. **Decision for the
   maintainer:** accept that, or create the package now. I recommend accepting.
2. **Should the phase 2 hypothesis store reuse this format?** The roadmap plans a separate store
   under `studies/auto_research/` on a branch in `researcher`'s clone, with aggregate scores only.
   One idea there has several implementations, and an ml-experiment has one hypothesis and several
   arms. I recommend that issue #1034 adopt the `result.md` front matter, so that phase 2 reads
   phase 1's experiments directly. This plan does not edit #1034 or the phase 2 text.
   **Decision for the maintainer.**
3. **The shared row set and the holdout.** The stage 1 pilot scores Quick wins' fold A unchanged:
   ENS runs at 00:00 UTC every 7 days from 2025-07-01 to 2026-06-30 (53 runs, all Tuesdays),
   forecast origin 09:00 UTC, all 51 members, valid times to about 14.6 days, the 9 generator series
   (6 solar, 3 wind; Quick wins itself scores 31 series), errors divided by each series' effective
   capacity. The stride is `SAMPLE_STRIDE_DAYS = 7` in `studies/quick_wins/quick_wins_inputs.py`. A
   fixed weekday suits generators but not demand, which depends on weekday. The planning session for
   PR #1183 recommends a stride of 5 or 6 days. **Decision for the maintainer:** (a) adopt the
   pilot's Tuesday row set as `fold_a_tuesdays_v1` for the first generator experiments, and add a
   rotating-weekday row set before the first demand experiment; or (b) rebuild the frames now with a
   stride of 6 days. I recommend (a). The holdout date, 2026-07-01, comes from the staged-forecast
   plan in the unmerged PR #1183, and `conf/cv/default.yaml` defines no holdout, so the date is a
   constant that the maintainer confirms.
4. **Default decision rule.** The machinery is the default: hypothesis, prediction and decision rule
   written first, the month-resampled paired interval, a Bonferroni level across the planned
   contrasts, absolute scores beside every contrast, and a failures section. The staged-forecast
   plan's Holm-adjusted t-interval is a different method, and Holm needs p-values that
   `studies.bootstrap` does not return. I recommend one method for every experiment, the existing
   bootstrap with Bonferroni, and the thresholds (such as a 0.15 CRPS-point margin) set by each
   experiment. Fold B and the holdout apply to finalists only.
5. **Outputs differ between experiments.** The staged-forecast plan's outputs that do not fit
   `PowerForecast` are a 19-level calibrated quantile table, a per-product table with a product
   column, and stage 1 outputs (9 levels, per product, member, series, valid time). The manifest's
   `kind` carries this, and the table schemas are left to the first experiment that needs each.
6. **No generated index.** A hand-kept table would conflict whenever two sessions finish together,
   and a generator needs a freshness test. `grep` over the front matter does the job for the first
   30 experiments or so; revisit then.
7. **The `ml-experiments` and `ml_experimentation` names are close.** I recommend the maintainer's
   chosen `ml-experiments`, and the skill's description states the distinction.
8. **An experiment merges on its own** under the `study` skill's three conditions, adapted: the diff
   touches only `ml-experiments/` and `docs/ml-experiments/`; the experiment has had the review its
   size calls for; the checks pass. An experiment has no tests of its own scripts, as for studies. A
   change to `packages/studies/` stops at a reviewed PR.

## Findings considered and rejected

Findings of the simplicity review (Sonnet) that the plan takes: no new package, no row-set registry,
no generated index, a four-key manifest without `consumes`, two sizes, no morning-report file, no
finalist procedure, no roadmap rewrite beyond the false lines, no cross-link in
`docs/ml_experimentation/`, one pull request, and reuse of `studies.bootstrap`. Findings of the
correctness review (Opus) that the plan takes: reserve the bare id and take the maximum over the
data and docs folders, Bonferroni through `bootstrap_difference_at_level` instead of a new Holm
function, `ClimatologyForecaster` for the climatology row, an adapter for the bootstrap's pairing
key, a required `valid_time` column and a raise for the holdout check, the boundary-test rewrite
over the union of trees, workflow trigger paths, the basename rule, the roadmap lines, a testable
data root, slug validation, an atomic write, a `dirty` flag, and the extra doc pages. Rejected:

- **Replace the generated index with a hand-kept table:** rejected, because two sessions finishing
  together would conflict on that file.
- **Drop the threaded race test** (simplicity review) is kept dropped, but the correctness review
  showed the reservation bug was in the name, not the threading, so the plan tests a different slug
  at the same id instead.
- **Record the manifest's git commit only when clean** (correctness review): rejected in favour of
  a `dirty` flag, because an overnight run on uncommitted code is common and should not block.
