# Organise scripts/ into subdirectories by job (#763)

**Problem:** `scripts/` holds eleven files (ten `.py`/`.sh` scripts plus `markdown_wrap.py`) in one
flat directory, mixing three unrelated jobs — CI/lint tooling (both the automated pre-commit/CI
gates and the manual docs-reflow helpers that fix what they flag), AWS deployment, and
forecasting/data maintenance — with nothing marking which is which.

**Solution:** move the eleven files into three subdirectories by role — `scripts/lint/` (every
markdown/docstring/notebook quality tool, automated and manual alike), `scripts/deploy/` (AWS image
build/push), `scripts/forecasting/` (baseline export/training/maintenance) — and update every
reference: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `Dockerfile`, `pyproject.toml`,
two `docs/` pages, four `SKILL.md` files, two skill scripts that import `markdown_wrap`, four
`packages/` files, two `tests/` files that locate a script by path, and the deploy scripts'
cross-references to each other. No logic changes; every moved file is untouched except for import
paths and self-referential usage comments/docstrings.

## Verdict, size and departures

**Verdict: worth doing, roughly as described.** The issue's premise (a flat, unmarked directory
mixing automated gates, deployment, and forecasting tooling) is accurate — confirmed by reading
every file in `scripts/` and grepping every reference to it across the repo.

**No departure from the issue body.** The issue's own text guesses three categories (CI/lint
tooling, AWS deployment, forecasting/data) and asks the plan to check whether the docs-reflow
helpers (`markdown_wrap.py`, `reflow_docs.py`, `reflow_python_prose.py`, added by #690/#764) belong
with the lint tooling or in a category of their own. An earlier draft of this plan gave them their
own `scripts/docs/` category; a simplicity review (recorded under "Plan review" below) found that
split not earning a fourth top-level directory for three files a side, and that `scripts/docs/`
reads as documentation rather than doc-tooling next to the top-level `docs/`. This plan now follows
the issue's own three-category guess.

**Size: Medium.**

- **What gets stored:** unaffected. No Patito model, Delta table or asset changes.
- **Production serving path:** unaffected. Nothing in `scripts/` sits on it —
  `run_baseline_experiment.py`/`export_baseline_forecasts.py` drive CV/backtesting jobs
  (`nged_substation_forecast.defs.cv_assets`/`.jobs`), not the production `defs/assets.py`/
  `checks.py` path, and `rewrite_nwp_row_groups.py` is a one-off maintenance script with no
  `@asset` decorator and no schedule.
- **Degradation rule:** unaffected. No change to `inherent-stability.md` behaviour.
- **More than one defensible design:** yes — the category boundaries and names are a genuine
  judgement call, which is exactly why the issue asked for a plan rather than specifying the split
  itself.
- **Callers you could not name without searching:** no — every reference is enumerated below via an
  exhaustive grep of the whole repo (workflows, pre-commit config, `Dockerfile`, `pyproject.toml`,
  `docs/`, `.claude/skills/`, `packages/`, `tests/`, and the scripts themselves), not a sample.

One trigger fires (design choice), so this is Medium, not Complex. Given the low mechanical risk —
every file move is a `git mv` plus a string substitution, and the verification set below can check
every reference exhaustively — this plan already had **one plan review** (simplicity, recorded
below), and I am recommending **one diff review** (correctness-and-cut-it-down, in
`implement-issue`) to catch any reference even this revised grep missed. No correctness plan review
(there is no ambiguity about current behaviour to get wrong — every script's current behaviour is
unchanged) and no mutation-testing diff review (there is no new behaviour to mutate; the change is
structural).

## Plan review

**Simplicity review (sub-agent, no visibility into the reasoning above):** found the original
four-category split (separating `scripts/lint/` from a `scripts/docs/`) not earning its complexity,
and `scripts/docs/` a confusable name next to top-level `docs/`. Recommended collapsing to the
issue's own three-category guess. **Accepted** — this plan now uses three categories; see "What
changes" below.

It also re-ran the reference grep independently and found two gaps this plan's first draft missed:

- **Four `packages/` files** carry a stale-path reference in a comment or docstring, missed because
  the first grep pass never searched `packages/`: `packages/ml_core/tests/test_production_helpers.py:257`,
  `packages/ml_core/src/ml_core/production_helpers.py:219`,
  `packages/dashboard/tests/test_view_forecasts.py:9`, `packages/notebooks/view_baseline_export.py:1`.
  **Accepted** — confirmed by re-running the grep myself; added to "References updated" below.
- **The two deploy scripts reference each other** by old path in their own header comments
  (`build_and_verify_image.sh:10,137`, `push_and_deploy_image.sh:6,11,63`), which the first draft's
  "self-referential usage docstrings" list didn't cover since it only considered a script
  referencing *itself*, not its sibling. **Accepted** — confirmed by re-reading both files; added
  below.

One review suggestion was **not** adopted: dropping the subdirectory move entirely in favour of a
`scripts/README.md` categorising the flat directory. The reviewer itself concluded this contradicts
the issue's explicit ask for subdirectories and isn't worth proposing — recorded here for
completeness, not because it was a close call.

## What changes, file by file

### New directory structure

```text
scripts/
  lint/
    check_docs_links.py
    check_marimo_notebooks.py
    lint_docstring_markdown.py
    markdown_wrap.py
    reflow_docs.py
    reflow_python_prose.py
  deploy/
    build_and_verify_image.sh
    push_and_deploy_image.sh
  forecasting/
    export_baseline_forecasts.py
    run_baseline_experiment.py
    rewrite_nwp_row_groups.py
```

**Why all six lint/docs-quality files share one `scripts/lint/`:** `check_docs_links.py`,
`check_marimo_notebooks.py` and `lint_docstring_markdown.py` are automated gates, wired into
`.pre-commit-config.yaml` (and `check_docs_links.py` into `.github/workflows/ci.yml` too).
`markdown_wrap.py`, `reflow_docs.py` and `reflow_python_prose.py` are manual helpers a person runs
by hand (or a skill script imports directly) after a gate flags something to fix. That
automated-versus-manual distinction is real, but a reader who wants to know which scripts run
themselves already has `.pre-commit-config.yaml` as the authoritative source for that — one grep
away — so splitting the directory on the same distinction buys little for the cost of a fourth
top-level category holding three files a side. One `scripts/lint/` for every markdown/docstring/
notebook quality tool, automated or manual, is the simpler answer and is what this plan uses.

### Files moved (`git mv`, no content change beyond the reference updates below)

| From | To |
|---|---|
| `scripts/check_docs_links.py` | `scripts/lint/check_docs_links.py` |
| `scripts/check_marimo_notebooks.py` | `scripts/lint/check_marimo_notebooks.py` |
| `scripts/lint_docstring_markdown.py` | `scripts/lint/lint_docstring_markdown.py` |
| `scripts/markdown_wrap.py` | `scripts/lint/markdown_wrap.py` |
| `scripts/reflow_docs.py` | `scripts/lint/reflow_docs.py` |
| `scripts/reflow_python_prose.py` | `scripts/lint/reflow_python_prose.py` |
| `scripts/build_and_verify_image.sh` | `scripts/deploy/build_and_verify_image.sh` |
| `scripts/push_and_deploy_image.sh` | `scripts/deploy/push_and_deploy_image.sh` |
| `scripts/export_baseline_forecasts.py` | `scripts/forecasting/export_baseline_forecasts.py` |
| `scripts/run_baseline_experiment.py` | `scripts/forecasting/run_baseline_experiment.py` |
| `scripts/rewrite_nwp_row_groups.py` | `scripts/forecasting/rewrite_nwp_row_groups.py` |

**Unaffected by the move, checked explicitly:** `reflow_docs.py` and `reflow_python_prose.py` import
`markdown_wrap` with a bare `import markdown_wrap` / `from markdown_wrap import ...` — this works
because Python adds a running script's own directory to `sys.path[0]`, and all three stay siblings
in `scripts/lint/` after the move, so this import needs no change.
`lint_docstring_markdown.py`'s `PYMARKDOWN_CONFIG = ".pymarkdown-docstrings.json"` is resolved
relative to the *current working directory* (pre-commit and CI both run from the repo root), not
relative to the script, so this also needs no change.

### References updated (every occurrence found by grepping the whole repo, not a sample)

- **`.github/workflows/ci.yml:63`** — `scripts/check_docs_links.py` →
  `scripts/lint/check_docs_links.py`.
- **`.pre-commit-config.yaml:53,79,120`** — `scripts/check_marimo_notebooks.py`,
  `scripts/lint_docstring_markdown.py`, `scripts/check_docs_links.py` → their `scripts/lint/` paths.
- **`docs/live_service/aws.md`** (lines 181, 215, 245, 456, 525, 1261) — every
  `scripts/build_and_verify_image.sh`/`scripts/push_and_deploy_image.sh` reference →
  `scripts/deploy/...`.
- **`docs/architecture/testing.md`** (lines 106, 261, 374) — `scripts/run_baseline_experiment.py` →
  `scripts/forecasting/...`; `scripts/check_docs_links.py` and `scripts/check_marimo_notebooks.py` →
  `scripts/lint/...`.
- **`Dockerfile:10`** (a comment, not a `COPY` — the image never includes `scripts/`) —
  `scripts/build_and_verify_image.sh` → `scripts/deploy/build_and_verify_image.sh`.
- **`pyproject.toml:457`** (a comment) — `scripts/check_docs_links.py` →
  `scripts/lint/check_docs_links.py`.
- **`.claude/skills/mkdocs-authoring/SKILL.md:16`** — `scripts/lint_docstring_markdown.py` →
  `scripts/lint/lint_docstring_markdown.py`.
- **`.claude/skills/marimo-notebooks/SKILL.md:57`** — `scripts/check_marimo_notebooks.py` →
  `scripts/lint/check_marimo_notebooks.py`.
- **`.claude/skills/prose-review/SKILL.md`** — every reference to the *root* `scripts/check_docs_links.py`
  (lines 570, 573, 703) → `scripts/lint/check_docs_links.py`. This skill's own
  `.claude/skills/prose-review/scripts/*.py` files (`apply_findings.py`, `find_duplication.py`,
  `check_structure.py`, `check_render_loss.py`, `check_information_loss.py`) are a *different*,
  unaffected `scripts/` directory (the skill's own) and are not touched.
- **`.claude/skills/prose-review/scripts/apply_findings.py`** — `from scripts.markdown_wrap import
  WIDTH` → `from scripts.lint.markdown_wrap import WIDTH` (the `sys.path.insert` line already
  inserts the repo root and treats `scripts/` as an implicit namespace package, so nesting one more
  level works the same way).
- **`.claude/skills/literature-review/scripts/reflow_paragraphs.py`** — same import-path update as
  above.
- **`packages/ml_core/tests/test_production_helpers.py:257`** — `` ``scripts/build_and_verify_image.sh`` ``
  → `` ``scripts/deploy/build_and_verify_image.sh`` `` in its docstring.
- **`packages/ml_core/src/ml_core/production_helpers.py:219`** — the comment naming
  `scripts/build_and_verify_image.sh` → `scripts/deploy/build_and_verify_image.sh`.
- **`packages/dashboard/tests/test_view_forecasts.py:9`** — `scripts/check_marimo_notebooks.py` →
  `scripts/lint/check_marimo_notebooks.py` in its module docstring.
- **`packages/notebooks/view_baseline_export.py:1`** — `` ``scripts/export_baseline_forecasts.py`` ``
  → `` ``scripts/forecasting/export_baseline_forecasts.py`` `` in its module docstring.
- **Self-referential and cross-referential usage comments/docstrings** — each moved script's own
  `Usage::`/module-docstring example invocation updates to its new path:
  `scripts/lint/reflow_docs.py`, `scripts/lint/reflow_python_prose.py`,
  `scripts/forecasting/export_baseline_forecasts.py`, `scripts/forecasting/run_baseline_experiment.py`
  (two occurrences — its own usage line and the "Now run:" pointer to `export_baseline_forecasts.py`),
  `scripts/forecasting/rewrite_nwp_row_groups.py` (two occurrences). The two deploy scripts also
  reference *each other*, not just themselves: `scripts/deploy/build_and_verify_image.sh` (its own
  usage line at the old `:10`, plus the "Push + deploy it with scripts/push_and_deploy_image.sh"
  pointer at the old `:137`) and `scripts/deploy/push_and_deploy_image.sh` (its own usage lines at
  the old `:6,11`, plus the "run scripts/build_and_verify_image.sh" pointer at the old `:63`).
- **`tests/test_check_docs_links.py:26`** — `SCRIPT_PATH` → `REPO_ROOT / "scripts" / "lint" /
  "check_docs_links.py"`.
- **`tests/test_marimo_notebooks.py:23`** — `CHECKER_PATH` → `REPO_ROOT / "scripts" / "lint" /
  "check_marimo_notebooks.py"`; its module docstring's `scripts/check_marimo_notebooks.py` mention
  (line 3) updates too.
- **`tests/test_rewrite_nwp_row_groups.py:23`** — `SCRIPT_PATH` → `REPO_ROOT / "scripts" /
  "forecasting" / "rewrite_nwp_row_groups.py"`.

**Checked, no change needed:** `CLAUDE.md`'s Commands section has no `scripts/` reference (the
`pymarkdown scan` command it documents doesn't invoke anything under `scripts/`).
`README.md`/`docs/documentation-guide.md` have no `scripts/` reference.
`.github/workflows/docs.yml`, `nightly_network_tests.yml`, `contribution-bot.yml` have none either.
No `pyproject.toml` `ruff`/`ty` per-file-ignore glob is keyed to a `scripts/*.py` path, so the move
doesn't silently drop or misapply a lint exemption.

## Design-philosophy check

Pure structural reorganisation of developer/CI tooling — no production asset, degradation path, or
stored data is touched. `docs/design-philosophy/inherent-stability.md` and
`engineering-hypotheses.md` don't apply.

## Tests

No new test *behaviour* — this issue moves files, it doesn't change what any script does. The three
existing tests that locate a script by constructing its path (`test_check_docs_links.py`,
`test_marimo_notebooks.py`, `test_rewrite_nwp_row_groups.py`) get their `SCRIPT_PATH`/`CHECKER_PATH`
constant updated to the new location; each already fails today if that path is wrong (a missing or
misnamed script fails at `subprocess`/`importlib` load time), so running the full suite after the
move is itself the regression check — no new test is needed to prove the move didn't break anything
that wasn't already covered.

## Docs to update

Covered above under "References updated": `docs/live_service/aws.md`,
`docs/architecture/testing.md`, four `SKILL.md` files, and four `packages/` docstrings/comments. No
roadmap page or "Implementation details" section references this issue — it isn't a roadmap item, so
there's no ship-time triage beyond deleting this plan file into the PR body per the standard rule.

## Verification commands

- `uv run ruff check .`
- `uv run ruff format .`
- `uv run ty check`
- `uv run pytest` (exercises the three path-updated tests directly, plus the full suite as a
  regression check that nothing else broke)
- `uv run pre-commit run --all-files` — runs every hook (`check-marimo-notebooks`,
  `lint-docstring-markdown`, `check-docs-links`, and the pymarkdown/ruff hooks) against the new
  paths, which is the most direct proof the `.pre-commit-config.yaml` path updates are correct
- `uv run mkdocs build --strict` — the two touched `docs/` pages still build clean
- `grep -rn "scripts/check_docs_links\.py\|scripts/check_marimo_notebooks\.py\|scripts/lint_docstring_markdown\.py\|scripts/markdown_wrap\.py\|scripts/reflow_docs\.py\|scripts/reflow_python_prose\.py\|scripts/build_and_verify_image\.sh\|scripts/push_and_deploy_image\.sh\|scripts/export_baseline_forecasts\.py\|scripts/run_baseline_experiment\.py\|scripts/rewrite_nwp_row_groups\.py" --include='*' .` after the move, restricted to matches with no `/lint/`, `/deploy/` or `/forecasting/` segment — should return nothing outside `.claude/skills/prose-review/scripts/` (a different, unaffected directory) and this plan file itself

## Risks and open questions

- **Does one `scripts/lint/` for both automated gates and manual reflow helpers lose useful
  signal?** A reader can no longer tell "is this invoked automatically" from directory placement
  alone. Accepted trade: `.pre-commit-config.yaml` is one grep away and is the authoritative answer
  to that question regardless of directory structure, so the directory name doesn't need to carry
  it too. Flagging this once more for the human reviewer in case the automated/manual distinction is
  valued more than this plan assumes.
