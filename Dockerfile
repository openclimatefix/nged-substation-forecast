# syntax=docker/dockerfile:1

# Production image: the Dagster code-location server and the image every Dagster run executes in.
# No MLflow at runtime. See docs/live_service/aws.md for the deployment runbook and
# docs/architecture/production-deployment.md for why this design was chosen.
#
# Build (always linux/arm64 — ARM Fargate is ~20% cheaper and the control-plane box is
# Graviton, so an amd64 image cannot run anywhere in the deployment; on an x86 host this
# needs QEMU registered, which scripts/deploy/build_and_verify_image.sh checks for):
#   docker build --platform linux/arm64 \
#     --build-arg GIT_SHA=$(git rev-parse HEAD) \
#     -t nged-forecast:<commit-short> .

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

ARG GIT_SHA

LABEL org.opencontainers.image.source="https://github.com/openclimatefix/nged-substation-forecast" \
      org.opencontainers.image.revision="${GIT_SHA}"

ENV GIT_SHA="${GIT_SHA}" \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# contracts.settings.PROJECT_ROOT walks up to the nearest ancestor holding uv.lock, so copying
# uv.lock here anchors the repo-relative default (conf/) at /app — which is where the COPY below
# places it. conf/ also keeps conf/model/ available so training jobs (register_experiment_job)
# can run in-container.
COPY --from=builder /app/.venv /app/.venv
COPY uv.lock ./
COPY conf/ conf/

ENTRYPOINT ["dagster"]
# This ENTRYPOINT exists for `docker run` smoke-test ergonomics. In the deployed service every
# command spells the full argv instead: EcsRunLauncher's generated run command itself starts
# with "dagster", so the ECS task definition neutralises this ENTRYPOINT with /usr/bin/env
# (docs/live_service/aws.md, Step 9), and the control-plane services override it outright.
