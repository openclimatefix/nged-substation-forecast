# Getting started on your laptop

This is the single walkthrough for going from a fresh clone to a running Dagster instance that
downloads real data and trains a model — entirely on your laptop, no AWS involved. Follow it
top to bottom the first time; later pages go deeper on each part and are linked as you reach them.

## Prerequisites

- **Python 3.14+** and [`uv`](https://docs.astral.sh/uv/) — install `uv` following its
  [official instructions](https://docs.astral.sh/uv/getting-started/installation/). `uv` manages
  the Python version and the virtual environment for you, so you do not need to install Python
  separately.
- **Credentials for NGED's telemetry bucket** — the URL, access key, and secret for the S3 bucket
  where NGED publishes its telemetry. Ask another team member if you do not have them. Nothing in
  this project works without them, because the raw power data lives there.

## Step 1 — Install the project

From your clone of the repository:

```bash
uv sync                    # create the virtualenv and install all workspace packages
uv run pre-commit install  # install the git hooks (lint, format, markdown, type-check on commit)
```

`uv sync` reads the workspace's lockfile and installs every package under `packages/` plus the
root Dagster application in one step.

## Step 2 — Create your `.env`

Configuration is read from a `.env` file in the repository root (and from environment variables,
which win over `.env`). Copy the committed template and fill in the three required NGED credentials:

```bash
cp .env.example .env
```

Then edit `.env` and set the three `NGED_S3_BUCKET_*` values from the prerequisites:

```dotenv
NGED_S3_BUCKET_URL=<nged source bucket url>
NGED_S3_BUCKET_ACCESS_KEY=<key>
NGED_S3_BUCKET_SECRET=<secret>
```

Those three are the only values you must set. Every other setting has a working default, so with
nothing else in `.env` all data and artifacts live under `<repo>/data` as plain files on disk. The
[Configuration reference](live_service/setup.md) explains the full menu — the three storage roots,
the derive-from-root convention, and the credentials you would add to move the data tables to S3.

`.env` is git-ignored; never commit real credentials.

> **Using a git worktree?** `Settings` reads `.env` from the repository root, so a worktree needs
> its own. Symlink the main checkout's file rather than copying secrets around:
> `ln -s ../<main-checkout>/.env .env`.

## Step 3 — Give Dagster a persistent home

Dagster works without any of this, but by default it forgets its run history and schedule state
when you stop it. Point it at a persistent directory so that state survives a restart — this also
matters later, because the live 6-hourly schedule only keeps time while the daemon runs
continuously.

1. Create the directory and a `dagster.yaml` inside it:

    ```bash
    mkdir -p ~/dagster_home
    ```

2. Put the following into `~/dagster_home/dagster.yaml`:

    ```yaml
    storage:
      sqlite:
        base_dir: "dagster_history"

    concurrency:
      pools:
        default_limit: 1  # Used to limit concurrency of the ecmwf_ens asset.

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

3. Tell Dagster where it is by adding `export DAGSTER_HOME=~/dagster_home` to your `.bashrc` (or
   equivalent) and restarting your terminal.

`dagster.yaml` is read only at process startup, so restart `dg dev` after editing it.

## Step 4 — Start Dagster

```bash
uv run dg dev
```

Leave it running and open <http://localhost:3000>. Everything from here is driven from that UI.

## Step 5 — Download data and train a model

In the Dagster UI, materialise the data assets in order — this is what pulls the real data onto
your laptop:

1. **`power_time_series_and_metadata`** — pulls NGED telemetry from S3 into a local Delta table.
2. **`h3_grid_weights`** — computes the spatial weights the NWP download needs (one-off).
3. **`ecmwf_ens`** — downloads ECMWF ensemble weather; materialise the daily partitions across
   your training window.
4. **`eligible_time_series`** — determines which series have enough coverage for a fold.

Then register an experiment and train a model. The full recipe — the run config for each job, what
`smoke_test` versus `full_cv` does, and how the trained model is tracked in MLflow — is
[Running an ML experiment end-to-end](ml_experimentation/dagster-workflow.md). Start there with a
`smoke_test` run to check the whole pipeline is wired up before committing to a long training run.

## Where to go next

- [Running an ML experiment end-to-end](ml_experimentation/dagster-workflow.md) — the data-to-model
  recipe in full, plus [Model configuration](ml_experimentation/model-configuration.md) for
  choosing features and hyperparameters.
- [Configuration reference](live_service/setup.md) — every `.env` setting, the storage-root model,
  and how to point the data tables at real S3.
- [Running the whole stack locally](live_service/local.md) — keeping the daemon and the 6-hourly
  forecast schedule running continuously, and an optional MinIO rehearsal of the S3 code paths.
- The [dashboard README](https://github.com/openclimatefix/nged-substation-forecast/tree/main/packages/dashboard#readme)
  — Marimo apps for inspecting forecasts, and the `.env.s3` toggle for viewing production data.
- [MLflow experiment tracking](https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/dagster-workflow/#viewing-results-in-the-mlflow-ui)
  — viewing your training runs (`uv run mlflow ui --gunicorn-opts "--workers 1"`).
