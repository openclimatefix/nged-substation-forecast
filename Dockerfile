# syntax=docker/dockerfile:1

# Production image for the live-forecast service. Bakes the champion model into the image at
# build time and loads it via a plain save/load — no MLflow, run ID, or cache lookup at
# runtime. See docs/architecture/production-deployment.md for the promotion runbook and
# docs/roadmap/live-service.md#production-model-artifacts for why this design was chosen.
#
# Build (arm64 — ARM Fargate is ~20% cheaper and the candidate control-plane boxes are
# Graviton; add --platform linux/arm64 if building on an x86 host):
#   docker build --build-arg MODEL_RUN_ID=<id> --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     -t nged-forecast:<id-short> .
#
# The champion model must already be promoted to data/production_model/ (via the
# `promoted_model` Dagster asset) before running this build — the build itself never talks to
# MLflow, so it copies that directory from the build context hermetically.

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Full workspace: uv needs every member's pyproject.toml present to resolve uv.lock, even
# though --no-editable below only ends up installing the root project's actual dependencies.
COPY pyproject.toml uv.lock README.md ./
COPY packages/ packages/
COPY src/ src/

# --no-editable installs every workspace package as a regular wheel into .venv, so .venv is
# fully self-contained and portable into the runtime stage with no source tree alongside it —
# verified empirically: the installed nged_substation_forecast package imports fine with the
# repo checkout absent entirely.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS runtime

ARG MODEL_RUN_ID
ARG GIT_SHA

LABEL org.opencontainers.image.source="https://github.com/openclimatefix/nged-substation-forecast" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.openclimatefix.model-run-id="${MODEL_RUN_ID}"

ENV GIT_SHA="${GIT_SHA}" \
    PATH="/app/.venv/bin:${PATH}" \
    PRODUCTION_MODEL_PATH="/app/data/production_model" \
    CV_CONFIG_PATH="/app/conf/cv/default.yaml" \
    NWP_METADATA_CSV_PATH="/app/metadata/nwp_metadata.csv"
# CV_CONFIG_PATH/NWP_METADATA_CSV_PATH override contracts.settings.Settings' PROJECT_ROOT-
# relative defaults, which resolve to a path *inside* .venv under this --no-editable install
# (PROJECT_ROOT = Path(__file__).parents[4] assumes an editable-install directory depth) —
# empirically confirmed necessary: without them, importing nged_substation_forecast.definitions
# (which eagerly loads the CV config at module scope, in defs/cv_assets.py) fails with
# FileNotFoundError: /app/.venv/conf/cv/default.yaml. cv_assets.py now reads CV_CONFIG_PATH
# directly (not via a Settings() instance, to stay credential-free at import time), so this env
# var is honoured; nwp_metadata_csv_path was already read via an instantiated Settings and so
# needed no source change.

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY data/production_model/ data/production_model/
COPY conf/ conf/
COPY metadata/ metadata/

ENTRYPOINT ["dagster"]
# The default target: live_forecasts_job is the real, already-existing partitioned job
# (defs/schedules.py). Under Option B, EcsRunLauncher overrides this command per run, and the
# same image separately serves as the code-location server — this default only matters for
# `docker run` smoke tests. Partition selection is via --tags, not --select/--partition:
# `dagster job execute` has no --partition flag at all, and `--select <asset>` hits a pre-
# existing, unrelated antlr4-python3-runtime/Python-3.14 incompatibility in Dagster's own
# asset-selection-string parser (confirmed reproducing outside Docker too, on plain `dg dev`).
# Job-name selection (-j) skips that parser entirely, so this is the reliable invocation:
#   docker run --network=none <image> job execute -j live_forecasts_job \
#     --tags '{"dagster/partition": "<key>"}'
CMD ["job", "execute", "-m", "nged_substation_forecast.definitions", "-j", "live_forecasts_job"]
