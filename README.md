# NGED Flexpectation

[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved)

**NGED Flexpectation** is an NIA-funded project by [Open Climate Fix](https://openclimatefix.org/) to deliver state-of-the-art, probabilistic power forecasts for [National Grid Electricity Distribution](https://www.nationalgrid.com/electricity-distribution) (NGED). The forecasts cover NGED's substations and customer meters, with a 14-day horizon at half-hourly resolution, updated every 6 hours. The goal is to help NGED optimise flexibility procurement and manage network congestion.

This repository is the research and production-harness codebase. The system is orchestrated with Dagster, uses XGBoost as the initial forecasting model, and stores all data and forecasts as Delta Lake tables on S3.

To external contributors: this repo is in early-stage development with frequent breaking changes, so it is not suitable for external contributions at the moment.

---

## Documentation

For detailed information about the project, including architecture, design philosophy, and user guides, please visit our [documentation site](https://openclimatefix.github.io/nged-substation-forecast/).

## Development

This repo is a `uv` [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces): A single repo which contains multiple Python packages.

### Setup

1. Ensure [`uv`](https://docs.astral.sh/uv/) is installed following their [official documentation](https://docs.astral.sh/uv/getting-started/installation/).
2. **Install dependencies**: `uv sync`
3. **Install pre-commit hooks**: `uv run pre-commit install`

To run Dagster:

1. `uv run dg dev`
2. Open `http://localhost:3000` in your browser to see the project.

Optional: To allow Dagster to remember its state after you shut it down:

1. `mkdir ~/dagster_home/`
2. Put the following into `~/dagster_home/dagster.yaml`:

    ```yaml
    storage:
      sqlite:
        base_dir: "dagster_history"

    concurrency:
      pools:
        default_limit: 1  # Used to limit concurrency of ecmwf_ens asset.

    run_monitoring:
      # Without this, a crashed/killed run can leak its concurrency-pool slot (e.g. the pool
      # above) forever, since nothing else frees a slot held by a run that never reached a
      # normal finally-block exit. This lets the daemon self-heal: any run finished (in any
      # terminal status) for longer than the threshold has its slots freed automatically.
      enabled: true
      free_slots_after_run_end_seconds: 300

    python_logs:
      managed_python_loggers:
        - nged_data
      python_log_level: DEBUG
    ```

3. Add `export DAGSTER_HOME=<dagster_home_path>` to your `.bashrc` file, and restart your terminal.

Note: `dagster.yaml` is only read at process startup, so restart `dg dev` (and the daemon) after
editing it for changes to take effect.

### Linting & Formatting

- **Check linting**: `uv run ruff check .`
- **Fix linting**: `uv run ruff check . --fix`
- **Format code**: `uv run ruff format .`
- **Type checking**: `uv run ty check`
- **Markdown linting**: `uv run pymarkdown scan -r docs README.md CLAUDE.md metadata/README.md packages/*/README.md`

Markdown (README.md files, docs/*.md, and Python docstrings) is linted automatically by the
pre-commit hook, but when developing code or docs it's a good idea to run the markdown lint
command above yourself before committing, for faster feedback than waiting on the commit-time
hook.

### Testing

- **Run all tests**: `uv run pytest`
- **Run tests with coverage**: `uv run pytest --cov`

### Development

- **Run Dagster UI**: `uv run dagster dev`
- Open `http://localhost:3000` in your browser to see the project.
- **Run Marimo notebooks**: `uv run marimo edit packages/notebooks/some_notebook.py`
- **View MLflow experiments**: `uv run mlflow ui --gunicorn-opts "--workers 1"`, then open
  `http://localhost:5000`. The `--gunicorn-opts` flag is required on Python 3.14 — MLflow's default
  uvicorn+FastAPI server fails to start there; see
  [the ML experimentation docs](https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/dagster-workflow/#viewing-results-in-the-mlflow-ui).

### Documentation

The docs are built with [MkDocs](https://www.mkdocs.org/) (Material theme). The tooling is part of the `dev` dependency group, so `uv sync` installs it.

- **Serve docs locally with live reload**: `uv run mkdocs serve`, then open `http://localhost:8000`. The site rebuilds automatically as you edit files in `docs/`.
- **Build the static site**: `uv run mkdocs build` — renders the docs into the `site/` directory.

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)
