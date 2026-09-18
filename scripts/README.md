# `scripts`

Standalone programs that are run directly rather than materialised: the repository's own linters,
the two deployment commands, and a handful of ad-hoc forecasting and maintenance jobs.

## Why nothing here is a Dagster asset

The Dagster asset graph (`src/nged_substation_forecast/defs/`) exists to orchestrate the production
forecasting pipeline — scheduled, materialised, with lineage and retries — so it is the right home
for work that repeats on a schedule and depends on other tracked outputs. The scripts in this
directory do not fit that shape: a pre-commit hook runs once per commit and exits, a deployment is a
person running a command by hand during a release, and a baseline export or a one-off Delta
maintenance job runs whenever someone needs it, not on a schedule. Each is run directly, not through
Dagster: `uv run python <script>` for the Python scripts, or the shell script itself for the two
under `deploy/`.

Each script's own module docstring (or, for the two shell scripts, the comment header above the
code) is the source of truth for how it works and why each choice in it was made. The lines below
say only what each script is for and who runs it.

## `lint/` — the quality gates, and the helpers that fix what they flag

Three of these six modules are automated gates, wired into `.pre-commit-config.yaml` so that they
run on every commit touching a file they cover. Two more are helpers a person runs by hand, to
rewrap the prose that `pymarkdown`'s `MD013` and ruff's `E501` report as over-long. The sixth is a
library with no command line of its own.

- `check_docs_links.py` — **gate.** Resolves every link to the published docs site against the
  markdown sources, so a docstring cannot go on pointing at a page a rename moved or an anchor a
  heading rewrite killed. `mkdocs build --strict` validates links *inside* `docs/` only, which is
  why this script exists. The script runs as a pre-commit hook and again in
  `.github/workflows/ci.yml`, and scans the whole repository rather than the staged files.
- `check_marimo_notebooks.py` — **gate.** Checks that every name a marimo notebook's cells
  reference is bound by some cell, which is the failure `ruff check --fix` and `marimo check --fix`
  each leave behind and which ruff, ty, and pytest all pass over in silence. The script runs as a
  pre-commit hook, and `tests/test_marimo_notebooks.py` runs the same check in the test suite.
- `lint_docstring_markdown.py` — **gate.** Extracts each docstring with `ast` and pipes it through
  `pymarkdown scan-stdin`, so the markdown inside a docstring is held to the same standard as the
  markdown in a `.md` file. `mkdocstrings` renders those docstrings onto the `docs/api/` pages,
  where a malformed list renders just as badly as it would in a page.
- `markdown_wrap.py` — **library, not a command.** Holds `WIDTH`, the single column this
  repository's prose is wrapped at, and `reflow_text`, the whole-file markdown reflow the two
  `reflow_*` scripts below are built on. The prose-review and literature-review skills import from
  it too.
- `reflow_docs.py` — **run by hand.** Rewraps every markdown file named on the command line to
  `WIDTH`, in place.
- `reflow_python_prose.py` — **run by hand.** Rewraps the docstrings and the prose comment blocks
  inside every Python file named on the command line, leaving the code between them untouched. The
  script refuses any block a rewrap would damage — a directive, a doctest, a banner rule, hand
  alignment — rather than rewrapping the block wrongly.

## `deploy/` — the two commands that ship a new champion model

Both scripts are run by hand by a person doing a release, in the order given, and both take no
arguments so that nothing can be mistyped or drift. The two together are the recurring half of the
runbook at [Setting up the live service on
AWS](https://openclimatefix.github.io/nged-substation-forecast/live_service/aws/): the one-time
infrastructure steps around them stay in the AWS console.

- `build_and_verify_image.sh` — builds the production image with the champion model baked in, then
  smoke-tests it with no network access and no credentials, failing hard if MLflow appears anywhere
  in the runtime log. Step 4 of the runbook.
- `push_and_deploy_image.sh` — pushes that image to the Elastic Container Registry and registers a
  new Elastic Container Service task-definition revision pointing at it. Step 6 of the runbook.

## `forecasting/` — ad-hoc experiment and maintenance runs

None of these three scripts is on a schedule. The first two are run in sequence when the
weather-and-calendar-only baseline is rebuilt; the third is a migration that is run once per Delta
table and then left alone.

- `run_baseline_experiment.py` — runs the weather-and-calendar-only baseline experiment end to end
  in one process: register the experiment, train the fold, forecast, then compute the leaderboard
  metrics. Takes no arguments.
- `export_baseline_forecasts.py` — exports one cross-validation experiment's forecasts to three
  self-contained parquet files (full ensemble, ensemble mean, and quantiles) for offline analysis,
  each carrying the observed power beside the forecast so residuals can be computed directly.
- `rewrite_nwp_row_groups.py` — rewrites every `nwp` Delta partition whose Parquet row groups span
  more than one ensemble member, so that a single-member read can skip the rest of the partition. A
  one-off migration: it measures before it writes, skips a partition that is already aligned, and
  therefore resumes cleanly after an interruption.
