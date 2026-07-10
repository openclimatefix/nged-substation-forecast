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
    PRODUCTION_MODEL_PATH="/app/data/production_model"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY data/production_model/ data/production_model/

ENTRYPOINT ["dagster"]
# The default target: live_forecasts_job is the real, already-existing partitioned job
# (defs/schedules.py). Under Option B, EcsRunLauncher overrides this command per run, and the
# same image separately serves as the code-location server — this default only matters for
# `docker run` smoke tests. Requires --partition <key> at run time, e.g.:
#   docker run --network=none <image> job execute -j live_forecasts_job --partition <key>
CMD ["job", "execute", "-m", "nged_substation_forecast.definitions", "-j", "live_forecasts_job"]
