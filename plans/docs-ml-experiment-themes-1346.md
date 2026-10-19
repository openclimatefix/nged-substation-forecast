# Plan: restructure the ML experiment docs around themes (#1346)

## Problem

**The "ML experiments" nav lists about 100 experiment write-ups in no obvious order, and the list will reach thousands.** Humans read only the four series reports. The write-ups are mostly read by Claude Code, which today finds them through `research-log.md` (2435 lines) and `research-plan.md` (323 lines), two global pages that cannot scale.

## Solution in brief

**Group every experiment under one theme, list only the themes in the nav, and build each theme's table of experiments from the `result.md` front matter at MkDocs build time.** A theme is three pages: Summary (the answer, headline plots, best design so far, what to implement in production), Experiments (a generated table of experiments tried), and Research plan (planned experiments). The experiment folders stay flat, named by id, so a reclassified experiment breaks no link. The global research log and research plan are deleted.

## Verdict, size and departures

Verdict: worth doing as described, with the departures below.

Size: **complex**, though no production code changes. The five triggers:

- **What gets stored:** no Patito model, Delta table, or asset changes. The front matter of 98 `result.md` files gains fields.
- **Production serving path:** not touched.
- **Degradation rule:** none touched. The new pre-commit check is a docs check, not a Dagster asset check.
- **More than one defensible design:** yes. Where the theme lives (front matter versus folder), whether tables are generated or hand-written, and whether the check runs at build time or commit time all admit alternatives.
- **Callers not nameable without searching:** yes. About 100 pages, the `ml-experiment`, `ml-experiment-manager`, `study`, and `release-model` skills, and the inbound links cite the pages being moved.

Reviews bought: both plan reviews (simplicity, then correctness), and in `implement-issue` the cut-it-down review, the correctness review, and no mutation pass beyond the generator script's tests.

Departures from the schema agreed in chat (issue #1346 lists the agreed version):

- **Reuse existing fields instead of adding `prod`, `pr`, and `date`.** Every `result.md` already carries `production_status` (`candidate` or `not-for-production`) and 33 of 98 carry `pull_request`. Adding `prod` and `pr` would duplicate them. `date` is dropped because the id orders experiments by start.
- **Keep the existing `verdict` words in this PR.** The 98 files use `inconclusive` (39), `refuted` (23), `supported` (14), `partly supported` (14), `small gain` (6), and `not supported` (1). They state what happened to the pre-registered hypothesis. The four agreed words (`helps`, `no clear effect`, `hurts`, `inconclusive`) state the effect on the forecast. The recommendation is to keep the existing words, because renaming them rewrites 98 verdicts and the decision rules that cite them, and to let `take_home_message` carry the plain-language effect. This is open question 1.
- **`take_home_message` is new.** The existing `headline` (69 of 98) and `lessons` (92 of 98) fields are multi-sentence prose with different purposes. `take_home_message` is two or three plain sentences, written once per experiment and shown in the generated table.
- **No per-theme research log.** The Experiments table, sorted by id descending, replaces it.

## Solution: walk through

### The happy path

To add a finished experiment, the author writes `docs/ml-experiments/<id>-<slug>/result.md` with the front matter fields `id`, `slug`, `theme`, `also_relevant_to`, `hypothesis`, `verdict`, `production_status`, `take_home_message`, and the rest of today's fields. Nothing else is edited. At build time the MkDocs hook `scripts/mkdocs_hooks/experiment_tables.py` reads every `result.md` front matter. In `on_page_markdown` it replaces the line `<!-- experiments-table -->` in `themes/<theme>/experiments.md` with a table of every experiment whose `theme` is that theme or whose `also_relevant_to` includes it, one row each, sorted by id descending. A row links to the experiment's `result.md` with the id and slug as link text, then shows the hypothesis, the verdict, `production_status`, `take_home_message`, and a column that says "primary" or "also relevant". The line `<!-- planned-table -->` in `plan.md` is replaced the same way with the experiments whose `status` is `planned`. The line `<!-- candidates-table -->` in the Overview is replaced with every experiment whose `production_status` is `candidate`.

Because the tables are not committed, concurrent experiment PRs never conflict on a table. Claude reads a theme's experiments with `uv run python scripts/mkdocs_hooks/experiment_tables.py --theme <theme>`, which prints the same table as markdown, and finds experiments by any front matter field with `grep`.

The same module has a `--check` mode that validates every front matter block and exits non-zero on a missing `theme`, a `theme` with no folder, a `theme` listed in `also_relevant_to` that has no folder, an unknown `verdict`, an unknown `production_status` (allowed: `candidate`, `not-for-production`, `adopted`), or a missing `take_home_message` on an experiment whose integer id is above 102. `verdict` and `production_status` are required when `status` is `done` or `finished`, and not required when `status` is `planned`. `mkdocs build --strict` runs the same validation in `on_config`, so a bad experiment fails the build whether or not a theme page renders it, and a validate-only pre-commit hook `check-experiment-front-matter` runs `--check` on any change under `docs/ml-experiments/`.

### The main branches

- **A planned experiment** (`status: planned`, which the `ml-experiment` skill commits before running) appears in the theme's planned table and not in the Experiments table. No `result.md` has this status today, so the planned table is empty on the real tree until the next plan PR. An idea with no `result.md` yet is a hand-written row in `plan.md` outside the placeholder line.
- **A new theme** needs a folder under `docs/ml-experiments/themes/` with `index.md`, `experiments.md`, and `plan.md`, plus a nav entry. A theme needs at least three related experiments.
- **A group of themes in the nav** (differentiable physics) is a nav grouping in `mkdocs.yml` only. Each sub-theme is a flat folder and slug such as `differentiable-physics-network`, created with its first experiment.
- **Cross-links:** an experiment with `also_relevant_to: [b]` appears in theme b's table with the "also relevant" marker.

### The main error-handling paths

All of this is docs tooling that runs on a developer machine and in CI. It raises (fail fast) because a wrong table misleads the reader, and no production data path imports it. A malformed front matter block names the file and the field in the error. A placeholder line the hook cannot find in a theme page is an error, so a theme page cannot silently show no table.

## What changes, file by file

- **`scripts/mkdocs_hooks/experiment_tables.py`** (new; outside `docs/` so MkDocs does not publish it): front matter reader (PyYAML, already installed with MkDocs), table builder, an `on_config` validation, an `on_page_markdown` hook, an `on_serve` watch on the experiment folders, a `--theme` print mode, and a `--check` mode. About 120 lines. `tests/test_experiment_tables.py` imports it through a `pythonpath` entry in `[tool.pytest.ini_options]`, mirrored for `ty` in `pyproject.toml` as the comments there describe.
- **`tests/test_experiment_tables.py`** (new): see Tests.
- **`mkdocs.yml`**: `hooks: [scripts/mkdocs_hooks/experiment_tables.py]`.
- **`.pre-commit-config.yaml`**: one local hook, `check-experiment-front-matter`, running `--check`, triggered by changes under `docs/ml-experiments/`.
- **`docs/ml-experiments/*/result.md`** (98 files): add `theme` and `also_relevant_to`, mechanically from the theme list below, which a Sonnet pass checks against each body. No `take_home_message` is backfilled in this PR.
- **`docs/ml-experiments/themes/<theme>/{index,experiments,plan}.md`** (new, 12 themes): the Summary page is the converted series report where one exists (four themes) and a short stub saying "Summary in progress" with the generated table otherwise. The solar disaggregation theme is finished end to end in this PR as the template. The other three series reports are converted in follow-up PRs, one per theme, so they can run in parallel.
- **`docs/ml-experiments/series/`**: moved to `themes/` (the four reports become the Summary pages).
- **`docs/ml-experiments/research-log.md`, `research-plan.md`**: deleted. Their "candidates for the next release" table moves to `docs/ml-experiments/README.md` as a generated table of every experiment with `production_status: candidate`. The "what would change in production" text moves into each theme's Summary.
- **`docs/ml-experiments/README.md`**: rewritten as the Overview: how themes work, the front matter fields, the list of themes, and the candidates table.
- **`mkdocs.yml`** (nav): the nav lists the Overview and each theme (three pages each), alphabetical, with the 98 experiment pages removed from it, and new `redirect_maps` entries for the moved series pages and the deleted log and plan pages.
- **`.claude/skills/ml-experiment/SKILL.md`, `ml-experiment-manager/SKILL.md`, `study/SKILL.md`, `release-model/SKILL.md`**: replace every instruction to update the research log or plan with the front matter and generator workflow. The `ml-experiment` skill states the new front matter fields and the take-home message rule.
- **`data/ml-experiments/briefs/common.md`** (private, not in git): updated by the ML manager after merge.
- **Inbound links to the deleted pages and the moved series pages**, all of which fail `mkdocs build --strict` if left: `docs/roadmap/index.md:51`; `docs/roadmap/staged-probabilistic-forecast.md` (lines 4, 5, 410, 580, 591, 592); `docs/live_service/model-releases.md:34`; `docs/documentation-guide.md` (lines 16, 20, 96); and the `result.md` files of experiments 7, 9, 12, 14, 16, 20, 23, 26, and 29, which link `../research-plan.md`, some with anchors. Each is retargeted to the owning theme's Research plan page. `CLAUDE.md:436` ("the research plan in `docs/ml-experiments/` owns their queue") is rewritten to name each theme's Research plan page. Links to experiment pages are unchanged because the folders do not move.
- **Moved series reports:** a report moves from `series/<name>.md` to `themes/<theme>/index.md`, one directory deeper, so every relative link and image in it gains one `../`. The strict build checks this.
- **Content with no home yet in the plan:** the research plan's 286 lines of "Proposals in detail" (the P-numbered proposals that result pages cite) are split by theme into each theme's `plan.md` as hand-written sections, with each theme's priority table at the top. The research log's "What the experiments taught us so far" section moves to the Overview until each theme's Summary absorbs its part.
- **The skills' full edit list:** `ml-experiment/SKILL.md` lines 43, 79 to 81, 115, 170 to 197, 224, 225, 262, 285, and 329 to 331; `ml-experiment-manager/SKILL.md` lines 15, 23, 64, 117, 173, 174; `release-model/SKILL.md` lines 30 and 150. The `study` skill has no reference and is not edited.

## Theme list (first assignment; a Sonnet pass checks each against the `result.md`)

Alphabetical in the nav. Experiment numbers are first-pass. Each id has exactly one primary theme: 44 is primary in "Switching labels in the forecast" with `also_relevant_to: [switching-detectors]`, and 64 is primary in "Solar disaggregation". A script expands the ranges and asserts that every folder id appears exactly once, because a hand-expanded range already put 64 in two themes.

- **Calibration and uncertainty:** 2, 16, 20, 35, 80, 97, 99 (and 100 when its write-up lands)
- **Curtailment cost:** 83, 84
- **Demand inputs:** 6, 10, 11, 15, 25, 27, 28, 30, 31, 74, 94
- **Differentiable physics:** a nav group only. Sub-themes (demand: calendar and weather; pooling across substations; long history and drift; embedded generation; forecast engine; network, reserved) get a flat folder and slug when each has its first experiment.
- **Generator forecasting (solar and wind farms):** 12, 23, 26, 29, 52, 54, 57, 72
- **Hierarchy and pooling:** 24, 32, 36
- **Lagged power and error correction:** 5, 8, 18, 21
- **Production reference and best guess:** 13, 19, 22, 33, 56, 87, 102
- **Retraining and training history:** 17, 34, 48, 49, 81, 85, 92, 93, 95
- **Solar disaggregation:** 39, 40, 45, 46, 51, 64, 73, 76, 77, 78, 79
- **Switching detectors:** 37, 38, 41, 43, 44, 47, 55, 58 to 63, 65 to 71, 82, 86, 98, 101
- **Switching labels in the forecast:** 44, 75, 88 to 91, 96
- **Weather products and blending:** 1, 3, 7, 9, 14, 50, 53

## Design-philosophy check

This is R&D and docs tooling, outside the production path, so the fail-fast rule applies and the inherent-stability rules do not. No asset check is added. No hypothesis from `engineering-hypotheses.md` is delivered. The change trades away nothing in `design-principles.md`.

## Tests

- `test_table_rows_and_order`: three fixture `result.md` files; assert the rows, order (id descending), verdict column, link targets, and the primary or also-relevant marker. On `main` the hook does not exist, so the test fails.
- `test_older_experiment_without_take_home_message_uses_headline`: a fixture id 50 with `headline` and no `take_home_message` shows the first sentence of `headline`.
- `test_check_fails_on_bad_front_matter`: one fixture each for a missing `theme`, a `theme` with no folder, an unknown `verdict`, and a missing `take_home_message` at id 103; each exits non-zero and names the file.
- `test_theme_print_mode_prints_the_table`, `test_candidates_table_lists_only_candidates`, `test_sort_is_numeric_on_string_ids` ("00099" before "00100"), `test_finished_status_is_valid`, `test_planned_result_needs_no_verdict`, and `test_primary_theme_is_not_repeated_as_also_relevant`.
- `test_missing_placeholder_is_an_error`: a theme page without its placeholder line raises.
- The strict MkDocs build in CI runs the validation over the real tree.

## Docs to update

Every page that names the research log, the research plan, or the series reports: the four skills above, `docs/documentation-guide.md` (check its "which place do I use?" table), `docs/ml-experiments/README.md`, and any `docs/roadmap/` page found by grep. The `CLAUDE.md` skills table row for `ml-experiment` is swept for stale wording.

## Verification commands

```bash
uv run pytest tests/test_experiment_tables.py
uv run python scripts/mkdocs_hooks/experiment_tables.py --check
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run python scripts/lint/check_docs_links.py
uv run mkdocs build --strict
```

Read the rendered nav in `site/` to confirm that the themes appear, that the experiment pages do not, and that the redirects exist. Also run every other CI step locally, including `pydoclint`.

## Sequence and the docs freeze

The docs freeze stays on until this PR merges. Plan PRs #1343 and #1345 each add their finished `result.md` afterwards, with the new front matter. The differentiable-physics session writes its parts with the final schema after merge.

1. This PR: the missing `verdict` and `production_status` for experiment 45 (read from its body; it has neither today), hook, tests, pre-commit check, front matter backfill, nav, redirects, the solar disaggregation theme finished, the other three series reports moved as Summary pages unchanged, stubs for the remaining themes, skills.
2. Follow-up PRs, in parallel, one per theme: convert the Summary to the "Does it work? / Best design so far" structure, with a prose review each.

## Risks and open questions

0. **Global queue order.** Today one research plan gives one ordered queue. After this change each theme's Research plan page holds its own priority table, and the ML experiment manager merges the themes into one order in its private brief. Is that acceptable?

1. **Verdict words.** Keep the existing six words (recommended), or migrate to the four agreed words? Migration rewrites 98 verdicts and the decision rules that cite them.
2. **Older take-home messages** are blank or taken from `headline` until each theme's follow-up PR writes them.
3. **Pages outside the nav** produce only INFO log lines in MkDocs. If `mkdocs build --strict` objects, the `not_in_nav` option (MkDocs 1.5 and later) is the fix.
4. **Edit collisions.** Any experiment PR that edits `docs/ml-experiments/` conflicts with the backfill. The freeze covers that, and the follow-up PRs rebase on `main` first.
5. **A separate git repository** was deferred (see the ML manager's reasons in #1346). This structure makes a later split cheaper, because a theme is already a self-contained folder.

## Review 1 (simplicity) triage

Accepted:

- Build the tables at MkDocs build time with a hook instead of committing generated tables. Committed tables would make every concurrent experiment PR conflict on the same file, and the `--theme` print mode keeps the tables readable to Claude.
- Require `take_home_message` only above id 102, with a `headline` fallback, so the backfill stops blocking this PR.
- Flat theme slugs and a nav-only group for differentiable physics, and no `Unsorted` theme.
- One table per theme with a primary or also-relevant column, replacing the second generated block.
- Drop the repository-level test, because the strict build covers it.
- `not_in_nav` only if the strict build objects.

Rejected:

- One page per stub theme instead of three. The maintainer asked for the same pages in every theme, and the three files per stub theme are generated by copy and hold one placeholder line each.

## Review 2 (correctness) triage

Accepted, and fixed above:

- The deleted pages and the moved series reports break links under the strict build; the inbound-link list, the one-level-deeper link fix, and homes for the proposals and the lessons section are now in the plan.
- `--check` would fail on experiment 45 (no `verdict`, no `production_status`) and on `production_status: adopted`; the rules now cover `finished`, `planned`, and `adopted`, and the plan fills in 45.
- Ids 44 and 64 were each in two themes; every id now has one primary theme, with a script asserting it.
- Validation runs in `on_config`, so it does not depend on a page rendering.
- The hook moves to `scripts/mkdocs_hooks/` so MkDocs does not publish it, with `on_serve` watching the experiment folders and a stated `pythonpath` for tests.
- The tests gain the cases the reviewer listed.

Rejected:

- A pytest that runs `--check` over the real tree. The strict build in CI already runs the same validation in `on_config`.
