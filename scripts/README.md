# Scripts

Nothing here is a Dagster asset. The Dagster asset graph (`src/nged_substation_forecast/defs/`)
exists to orchestrate the production forecasting pipeline — scheduled, materialised, with lineage
and retries — so it's the right home for work that repeats on a schedule and depends on other
tracked outputs. The scripts in this directory don't fit that shape: a pre-commit hook runs once per
commit and exits, a deployment is a person running a command by hand during a release, and a
baseline export or a one-off Delta maintenance job runs whenever someone needs it, not on a
schedule. Each is run directly, not through Dagster: `uv run python <script>` for the Python
scripts, or the shell script itself for the two under `deploy/`.

- **`lint/`** — markdown, docstring, and marimo-notebook quality tooling. `check_docs_links.py`,
  `check_marimo_notebooks.py`, and `lint_docstring_markdown.py` are automated gates wired into
  `.pre-commit-config.yaml` and (for `check_docs_links.py`) `.github/workflows/ci.yml`.
  `markdown_wrap.py`, `reflow_docs.py`, and `reflow_python_prose.py` are the manual helpers a person
  runs by hand to fix what those gates flag.
- **`deploy/`** — `build_and_verify_image.sh` and `push_and_deploy_image.sh` build and ship the
  production Docker image to AWS, per the runbook in
  [`docs/live_service/aws.md`](../docs/live_service/aws.md).
- **`forecasting/`** — `export_baseline_forecasts.py` and `run_baseline_experiment.py` drive ad-hoc
  cross-validation (CV) and backtesting runs; `rewrite_nwp_row_groups.py` is a one-off Delta Lake
  maintenance script.
