# Organise scripts/ into subdirectories by job (#763)

**Problem:** `scripts/` holds eleven files (ten `.py`/`.sh` scripts plus `markdown_wrap.py`) in one
flat directory, mixing three unrelated jobs — automated CI/pre-commit checks, manual docs-reflow
helpers, AWS deployment, and forecasting/data maintenance — with nothing marking which is which.

**Solution:** move the eleven files into four subdirectories by role — `scripts/lint/` (automated
pre-commit/CI gates), `scripts/docs/` (manual markdown-reflow helpers), `scripts/deploy/` (AWS
image build/push), `scripts/forecasting/` (baseline export/training/maintenance) — and update every
reference: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `Dockerfile`, `pyproject.toml`,
two `docs/` pages, four `SKILL.md` files, two skill scripts that import `markdown_wrap`, and two
`tests/` files that locate a script by path. No logic changes; every moved file is untouched except
for import paths and self-referential usage docstrings.

## Verdict, size and departures

**Verdict: worth doing, roughly as described.** The issue's premise (a flat, unmarked directory
mixing automated gates, deployment, and forecasting tooling) is accurate — confirmed by reading
every file in `scripts/` and grepping every reference to it across the repo.

**Departure from the issue body:** the issue's own text guesses three categories (CI/lint tooling,
AWS deployment, forecasting/data) and explicitly asks the plan to check whether the docs-reflow
helpers (`markdown_wrap.py`, `reflow_docs.py`, `reflow_python_prose.py`, added by #690/#764) belong
with the lint tooling or in a category of their own. This plan puts them in their own category,
`scripts/docs/`, separate from `scripts/lint/` — see "What changes" below for why.

**Size: Medium.**

- **What gets stored:** unaffected. No Patito model, Delta table or asset changes.
- **Production serving path:** unaffected. Nothing in `scripts/` sits on it —
  `run_baseline_experiment.py`/`export_baseline_forecasts.py` drive CV/backtesting jobs
  (`nged_substation_forecast.defs.cv_assets`/`.jobs`), not the production `defs/assets.py`/
  `checks.py` path, and `rewrite_nwp_row_groups.py` is a one-off maintenance script with no
  `@asset` decorator and no schedule.
- **Degradation rule:** unaffected. No change to `inherent-stability.md` behaviour.
- **More than one defensible design:** yes — the category boundaries and names (three vs. four
  categories, where the docs-reflow helpers sit) are a genuine judgement call, which is exactly why
  the issue asked for a plan rather than specifying the split itself.
- **Callers you could not name without searching:** no — every reference is enumerated below via an
  exhaustive grep of the whole repo (workflows, pre-commit config, `Dockerfile`, `pyproject.toml`,
  `docs/`, `.claude/skills/`, `tests/`), not a sample.

One trigger fires (design choice), so this is Medium, not Complex. Given the low mechanical risk —
every file move is a `git mv` plus a string substitution, and the verification set below can check
every reference exhaustively — I am running **one plan review** (simplicity, step 5) to sanity-check
the four-category split against a simpler alternative, and recommending **one diff review**
(correctness-and-cut-it-down, in `implement-issue`) to catch any reference this plan's grep missed.
No correctness plan review (there is no ambiguity about current behaviour to get wrong — every
script's current behaviour is unchanged) and no mutation-testing diff review (there is no new
behaviour to mutate; the change is structural).

## What changes, file by file

### New directory structure

```text
scripts/
  lint/
    check_docs_links.py
    check_marimo_notebooks.py
    lint_docstring_markdown.py
  docs/
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

**Why `scripts/docs/` is separate from `scripts/lint/`, not merged into it or split further:**
`check_docs_links.py`, `check_marimo_notebooks.py` and `lint_docstring_markdown.py` share a role —
each is wired into `.pre-commit-config.yaml` (and `check_docs_links.py` into
`.github/workflows/ci.yml` too) as an automated gate that fails a commit or a CI run.
`markdown_wrap.py`, `reflow_docs.py` and `reflow_python_prose.py` share a different role — none is
invoked automatically anywhere; a person runs one by hand (or a skill script imports
`markdown_wrap.py` directly) after a lint failure, to fix the prose the gate flagged. Grouping by
that automated-versus-manual distinction is more useful to a reader than grouping by topic
(markdown), because `check_marimo_notebooks.py` is not about markdown at all (it checks marimo cell
`refs`/`defs`), so a topic-based `scripts/markdown/` directory would either misplace it or leave it
alone in `scripts/lint/` by itself.

### Files moved (`git mv`, no content change beyond the reference updates below)

| From | To |
|---|---|
| `scripts/check_docs_links.py` | `scripts/lint/check_docs_links.py` |
| `scripts/check_marimo_notebooks.py` | `scripts/lint/check_marimo_notebooks.py` |
| `scripts/lint_docstring_markdown.py` | `scripts/lint/lint_docstring_markdown.py` |
| `scripts/markdown_wrap.py` | `scripts/docs/markdown_wrap.py` |
| `scripts/reflow_docs.py` | `scripts/docs/reflow_docs.py` |
| `scripts/reflow_python_prose.py` | `scripts/docs/reflow_python_prose.py` |
| `scripts/build_and_verify_image.sh` | `scripts/deploy/build_and_verify_image.sh` |
| `scripts/push_and_deploy_image.sh` | `scripts/deploy/push_and_deploy_image.sh` |
| `scripts/export_baseline_forecasts.py` | `scripts/forecasting/export_baseline_forecasts.py` |
| `scripts/run_baseline_experiment.py` | `scripts/forecasting/run_baseline_experiment.py` |
| `scripts/rewrite_nwp_row_groups.py` | `scripts/forecasting/rewrite_nwp_row_groups.py` |

**Unaffected by the move, checked explicitly:** `reflow_docs.py` and `reflow_python_prose.py` import
`markdown_wrap` with a bare `import markdown_wrap` / `from markdown_wrap import ...` — this works
because Python adds a running script's own directory to `sys.path[0]`, and all three stay siblings
in `scripts/docs/` after the move, so this import needs no change.
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
  WIDTH` → `from scripts.docs.markdown_wrap import WIDTH` (the `sys.path.insert` line already
  inserts the repo root and treats `scripts/` as an implicit namespace package, so nesting one more
  level works the same way).
- **`.claude/skills/literature-review/scripts/reflow_paragraphs.py`** — same import-path update as
  above.
- **Self-referential usage docstrings** — each moved script's own `Usage::`/module-docstring example
  invocation updates to its new path: `scripts/docs/reflow_docs.py`,
  `scripts/docs/reflow_python_prose.py`, `scripts/forecasting/export_baseline_forecasts.py`,
  `scripts/forecasting/run_baseline_experiment.py` (two occurrences — its own usage line and the
  "Now run:" pointer to `export_baseline_forecasts.py`), `scripts/forecasting/
  rewrite_nwp_row_groups.py` (two occurrences).
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
`docs/architecture/testing.md`, and four `SKILL.md` files. No roadmap page or "Implementation
details" section references this issue — it isn't a roadmap item, so there's no ship-time triage
beyond deleting this plan file into the PR body per the standard rule.

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
- `grep -rn "scripts/check_docs_links\.py\|scripts/check_marimo_notebooks\.py\|scripts/lint_docstring_markdown\.py\|scripts/markdown_wrap\.py\|scripts/reflow_docs\.py\|scripts/reflow_python_prose\.py\|scripts/build_and_verify_image\.sh\|scripts/push_and_deploy_image\.sh\|scripts/export_baseline_forecasts\.py\|scripts/run_baseline_experiment\.py\|scripts/rewrite_nwp_row_groups\.py" --include='*' .` after the move, restricted to matches with no `/lint/`, `/docs/`, `/deploy/` or `/forecasting/` segment — should return nothing outside `.claude/skills/prose-review/scripts/` (a different, unaffected directory) and this plan file itself

## Risks and open questions

- **Is `scripts/docs/` too easily confused with the top-level `docs/` directory?** A reader
  skimming `scripts/` sees `docs/` as a subdirectory name and might expect it to hold documentation
  rather than docs-*tooling*. Alternatives considered: `scripts/reflow/` (accurate, but undersells
  that `markdown_wrap.py` is a shared library, not a reflow script itself) and `scripts/prose/`
  (reads oddly next to `check_docs_links.py`'s absence from the same folder). Recommendation: keep
  `scripts/docs/` — it directly answers the issue's own question ("does the docs-reflow tooling
  belong with lint tooling or its own category") in a name that says what the files are *for*, and
  the confusion risk is low given `scripts/docs/markdown_wrap.py` is never referenced without its
  `scripts/` prefix. Flagging this as the one naming choice most worth a second opinion in the
  simplicity review.
- **Should `scripts/lint/` and `scripts/docs/` simply be one `scripts/markdown/` (or
  `scripts/quality/`) directory instead of two?** Considered and rejected above (`check_marimo_notebooks.py`
  isn't about markdown, and automated-vs-manual is a more load-bearing distinction for a reader
  deciding whether a script runs itself or needs to be run by hand) — but this is exactly the kind
  of call the simplicity review should attack directly, since a 3-category split (matching the
  issue's own initial guess) is a real alternative.
