# syntax=docker/dockerfile:1

# Production image for the live-forecast service. Bakes the champion model into the image at
# build time and loads it via a plain save/load — no MLflow, run ID, or cache lookup at
# runtime. See docs/live_service/aws.md for the promotion/deployment runbook and
# docs/architecture/production-deployment.md for why this design was chosen.
#
# Build (always linux/arm64 — ARM Fargate is ~20% cheaper and the control-plane box is
# Graviton, so an amd64 image cannot run anywhere in the deployment; on an x86 host this
# needs QEMU registered, which scripts/build_and_verify_image.sh checks for):
#   docker build --platform linux/arm64 \
#     --build-arg MODEL_RUN_ID=<id> --build-arg GIT_SHA=$(git rev-parse HEAD) \
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
    PRODUCTION_MODEL_PATH="/app/data/production_model"

WORKDIR /app

# contracts.settings.PROJECT_ROOT walks up to the nearest ancestor holding uv.lock, so copying
# uv.lock here anchors the repo-relative default (conf/) at /app — which is where the COPY below
# places it. conf/ also keeps conf/model/ available so training jobs (register_experiment_job)
# can run in-container.
COPY --from=builder /app/.venv /app/.venv
COPY uv.lock ./
COPY data/production_model/ data/production_model/
COPY conf/ conf/

ENTRYPOINT ["dagster"]
# This ENTRYPOINT exists for `docker run` smoke-test ergonomics. In the deployed service every
# command spells the full argv instead: EcsRunLauncher's generated run command itself starts
# with "dagster", so the ECS task definition neutralises this ENTRYPOINT with /usr/bin/env
# (docs/live_service/aws.md, Step 9), and the control-plane services override it outright.
#
# The default target: live_forecasts_job is the real, already-existing partitioned job
# (defs/schedules.py). EcsRunLauncher overrides this command per run, and the
# same image separately serves as the code-location server — this default only matters for
# `docker run` smoke tests. Partition selection is via --tags, not --partition: `dagster job
# execute` has no --partition flag at all, and selecting the job by name keeps the smoke test
# exercising exactly the entry point the EcsRunLauncher uses. Inference needs no credentials of
# any kind, so this is the reliable invocation:
#   docker run --network=none \
#     <image> job execute -j live_forecasts_job --tags '{"dagster/partition": "<key>"}'
CMD ["job", "execute", "-m", "nged_substation_forecast.definitions", "-j", "live_forecasts_job"]
