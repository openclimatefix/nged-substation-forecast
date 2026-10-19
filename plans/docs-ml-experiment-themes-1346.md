# Plan: restructure the ML experiment docs around themes (#1346)

## Problem

**The "ML experiments" nav lists about 100 experiment write-ups in no obvious order, and the list will reach thousands.** Humans read only the four series reports. The write-ups are mostly read by Claude Code, which today finds them through `research-log.md` (2435 lines) and `research-plan.md` (323 lines), two global pages that cannot scale.

## Solution in brief

**Group every experiment under one theme, list only the themes in the nav, and generate each theme's table of experiments from the `result.md` front matter.** A theme is three pages: Summary (the answer, headline plots, best design so far, what to implement in production), Experiments (a generated table of experiments tried), and Research plan (planned experiments). The experiment folders stay flat, named by id, so a reclassified experiment breaks no link. The global research log and research plan are deleted.

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

To add a finished experiment, the author writes `docs/ml-experiments/<id>-<slug>/result.md` with the front matter fields `id`, `slug`, `theme`, `also_relevant_to`, `hypothesis`, `verdict`, `production_status`, `take_home_message`, and the rest of today's fields. The author runs `uv run python scripts/lint/generate_experiment_tables.py --write`. The script reads every `result.md` front matter, groups by `theme`, and rewrites the block between `<!-- experiments-table:start -->` and `<!-- experiments-table:end -->` in `docs/ml-experiments/themes/<theme>/experiments.md`, and the planned-experiments block in `plan.md`. A row links to the experiment's `result.md` with the id and slug as link text, then shows the hypothesis, the verdict, `production_status`, and `take_home_message`. Rows are sorted by id descending. A second block lists the experiments in `also_relevant_to` for that theme, so the cross-links are generated too.

The pre-commit hook `check-experiment-front-matter` runs `generate_experiment_tables.py --check`. It fails when a `result.md` lacks a `theme`, names a theme with no folder, uses an unknown `verdict` or `production_status`, or when a generated block differs from what `--write` would produce. The check runs at commit time rather than build time, so the tables are in the committed files that Claude reads without running MkDocs.

### The main branches

- **A planned experiment** (`status: planned`) is listed in the theme's `plan.md` generated block, not in the Experiments table. An idea with no `result.md` yet is a hand-written row in `plan.md` outside the generated block.
- **A new theme** needs a folder under `docs/ml-experiments/themes/` with `index.md`, `experiments.md`, and `plan.md`, plus a nav entry. `Unsorted` is the inbox for an experiment that fits no theme. A theme needs at least three related experiments.
- **A two-level theme** (differentiable physics) is a folder `themes/differentiable-physics/<subtheme>/`. The `theme` field then holds `differentiable-physics/<subtheme>`, and the script resolves the path.
- **Cross-links:** `also_relevant_to` lists other themes. The generated "Also relevant" block in each theme's `experiments.md` lists those experiments.

### The main error-handling paths

All of this is docs tooling that runs on a developer machine and in CI. It raises (fail fast) because a wrong table misleads the reader, and no production data path imports it. A malformed front matter block names the file and the field in the error. The script never rewrites text outside its marker blocks, and it fails if a marker is missing.

## What changes, file by file

- **`scripts/lint/generate_experiment_tables.py`** (new): front matter reader (PyYAML, already a dependency through MkDocs), table generator, `--write` and `--check`. About 150 lines.
- **`tests/test_generate_experiment_tables.py`** (new): see Tests.
- **`.pre-commit-config.yaml`**: one local hook, `check-experiment-front-matter`, triggered by changes under `docs/ml-experiments/`.
- **`docs/ml-experiments/*/result.md`** (98 files): add `theme`, `also_relevant_to`, and `take_home_message`. A Sonnet sub-agent per theme writes these from the existing `headline`, `lessons`, and body. The assignment follows the theme list below.
- **`docs/ml-experiments/themes/<theme>/{index,experiments,plan}.md`** (new, about 12 themes): the Summary page is the converted series report where one exists (four themes) and a short stub saying "Summary in progress" with the generated table otherwise. The solar disaggregation theme is finished end to end in this PR as the template. The other three series reports are converted in follow-up PRs, one per theme, so they can run in parallel.
- **`docs/ml-experiments/series/`**: moved to `themes/` (the four reports become the Summary pages).
- **`docs/ml-experiments/research-log.md`, `research-plan.md`**: deleted. Their "candidates for the next release" table moves to `docs/ml-experiments/README.md` as a generated table of every experiment with `production_status: candidate`. The "what would change in production" text moves into each theme's Summary.
- **`docs/ml-experiments/README.md`**: rewritten as the Overview: how themes work, the front matter fields, the list of themes, and the candidates table.
- **`mkdocs.yml`**: the nav lists the Overview and each theme (three pages each), alphabetical, with `not_in_nav` for the 98 experiment pages (MkDocs 1.5 and later), and new `redirect_maps` entries for the moved series pages and the deleted log and plan pages.
- **`.claude/skills/ml-experiment/SKILL.md`, `ml-experiment-manager/SKILL.md`, `study/SKILL.md`, `release-model/SKILL.md`**: replace every instruction to update the research log or plan with the front matter and generator workflow. The `ml-experiment` skill states the new front matter fields and the take-home message rule.
- **`data/ml-experiments/briefs/common.md`** (private, not in git): updated by the ML manager after merge.
- **Inbound links:** a grep of `docs/`, READMEs, skills, and `ml-experiments/` scripts for `research-log`, `research-plan`, and `series/` finds every reference. Links to experiment pages are unchanged because the folders do not move.

## Theme list (first assignment; a Sonnet pass checks each against the `result.md`)

Alphabetical in the nav. Experiment numbers are first-pass.

- **Calibration and uncertainty:** 2, 16, 20, 35, 80, 97, 99, 100
- **Curtailment cost:** 83, 84
- **Demand inputs:** 6, 10, 11, 15, 25, 27, 28, 30, 31, 74, 94
- **Differentiable physics** (two-level, sub-themes agreed with the differentiable-physics session): Demand: calendar and weather; Pooling and long history; Embedded generation; Forecast engine; Network (reserved). No experiments yet.
- **Generator forecasting (solar and wind farms):** 12, 23, 26, 29, 52, 54, 57, 72
- **Hierarchy and pooling:** 24, 32, 36
- **Lagged power and error correction:** 5, 8, 18, 21
- **Production reference and best guess:** 13, 19, 22, 33, 56, 87, 102
- **Retraining and training history:** 17, 34, 48, 49, 81, 85, 92, 93, 95
- **Solar disaggregation:** 39, 40, 45, 46, 51, 64, 73, 76, 77, 78, 79
- **Switching detectors:** 37, 38, 41, 43, 44, 47, 55, 58 to 71, 82, 86, 98, 101
- **Switching labels in the forecast:** 44, 75, 88 to 91, 96
- **Weather products and blending:** 1, 3, 7, 9, 14, 50, 53
- **Unsorted:** any experiment that fits none.

## Design-philosophy check

This is R&D and docs tooling, outside the production path, so the fail-fast rule applies and the inherent-stability rules do not. No asset check is added. No hypothesis from `engineering-hypotheses.md` is delivered. The change trades away nothing in `design-principles.md`.

## Tests

- `test_generated_table_matches_front_matter`: build three fixture `result.md` files in a temp directory, run `--write`, and assert the table rows, their order (id descending), the verdict column, and the link targets. On `main` the script does not exist, so the test fails.
- `test_check_fails_on_stale_table`: edit a fixture's `verdict` after `--write` and assert `--check` exits non-zero and names the file.
- `test_check_fails_on_missing_or_unknown_theme`: a fixture without `theme`, and one naming a theme with no folder, each exit non-zero.
- `test_also_relevant_to_lists_experiment_under_second_theme`: a fixture with `also_relevant_to: [b]` appears in theme b's "Also relevant" block and not in its main table.
- `test_two_level_theme_path_resolves`: `theme: differentiable-physics/network` writes to the nested folder.
- `test_text_outside_markers_is_untouched`: hand-written prose around the markers is byte-identical after `--write`.
- A repository-level test runs `--check` over the real `docs/ml-experiments/` tree, so CI catches an experiment merged without a theme.

## Docs to update

Every page that names the research log, the research plan, or the series reports: the four skills above, `docs/documentation-guide.md` (check its "which place do I use?" table), `docs/ml-experiments/README.md`, and any `docs/roadmap/` page found by grep. The `CLAUDE.md` skills table row for `ml-experiment` is swept for stale wording.

## Verification commands

```bash
uv run pytest scripts/lint tests/test_generate_experiment_tables.py
uv run python scripts/lint/generate_experiment_tables.py --check
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run python scripts/lint/check_docs_links.py
uv run mkdocs build --strict
```

Read the rendered nav in `site/` to confirm that the themes appear, that the experiment pages do not, and that the redirects exist. Also run every other CI step locally, including `pydoclint`.

## Sequence and the docs freeze

The docs freeze stays on until this PR merges. Plan PRs #1343 and #1345 each add their finished `result.md` afterwards, with the new front matter. The differentiable-physics session writes its parts with the final schema after merge.

1. This PR: script, tests, hook, front matter backfill, nav, redirects, the solar disaggregation theme finished, the other three series reports moved as Summary pages unchanged, stubs for the remaining themes, skills.
2. Follow-up PRs, in parallel, one per theme: convert the Summary to the "Does it work? / Best design so far" structure, with a prose review each.

## Risks and open questions

1. **Verdict words.** Keep the existing six words (recommended), or migrate to the four agreed words? Migration rewrites 98 verdicts and the decision rules that cite them.
2. **Rewriting 98 take-home messages is the main cost.** Sonnet drafts them from the existing `headline` and `lessons`, and a sampled review checks about 10 of them against the bodies. The rest are spot-checked in the follow-up PRs.
3. **The `not_in_nav` option** needs MkDocs 1.5 or later. The first step checks `uv run mkdocs --version`. If the installed version is older, the fallback is to leave the pages out of the nav and accept the "not included in nav" log lines at INFO level.
4. **Edit collisions.** Any experiment PR that edits `docs/ml-experiments/` conflicts with the backfill. The freeze covers that, and the follow-up PRs rebase on `main` first.
5. **A separate git repository** was deferred (see the ML manager's reasons in #1346). This structure makes a later split cheaper, because a theme is already a self-contained folder.
