#!/usr/bin/env bash
#
# Build the production image and smoke-test it with zero network access.
#
# This is Step 4 of the AWS setup runbook (docs/live_service/aws.md) as one command:
# build the image with the champion model baked in, then prove it runs hermetically. This header
# is the source of truth for *why* each choice below is made.
#
# Usage:
#   scripts/build_and_verify_image.sh          # no arguments — everything is derived
#
# The image is built for linux/arm64 REGARDLESS of the host architecture: the ECS task
# definition declares ARM64 (ARM Fargate is ~20% cheaper) and the control-plane box is a
# Graviton instance, so an amd64 image cannot run anywhere in the deployment — a native x86
# build pushes fine but every task launch then dies with "image Manifest does not contain
# descriptor matching platform 'linux/arm64 v8'" (aws.md Steps 9-11). On an x86 host both the
# build and the smoke test therefore run under QEMU user-mode emulation — slower than native
# but correct; the script checks the emulator is registered and prints the fix if it isn't.
#
# The smoke test's partition key is HARD-CODED and arbitrary. It only has to parse as
# YYYY-MM-DD-HH:MM: the offline run dies at the NWP lookup long before the slot's validity could
# matter, so no real partition is needed and there is no decision worth pushing onto the user.
# (Real slots matter only for genuine runs against real data tables — see the partition-semantics
# note in docs/live_service/operations.md.)
#
# The build never contacts MLflow — it only COPYs data/production_model/ (populated by Step 3's
# `promoted_model` asset) into the image, so it stays hermetic. The MODEL_RUN_ID and GIT_SHA
# build args become OCI labels purely for traceability (inspect with `docker inspect`).
#
# The smoke test choices:
#   - No credentials at all: inference reads the model baked into the image, so the smoke test
#     passes with an empty environment. The deployed task still gets the NGED source-bucket
#     values as secrets (runbook Step 8), because the hourly ingest schedule needs them.
#   - --network=none: proves runtime inference needs no network at all — the whole point of
#     baking the model in.
#   - Job selected with `-j live_forecasts_job`, not `--partition`: `dagster job execute` has no
#     --partition flag. Selecting by job name and passing the partition via --tags exercises
#     exactly the entry point the EcsRunLauncher invokes in production.
#
# What it gates on, and what it does not:
#   - HARD FAIL (exit 1) if the runtime touches MLflow. Baking the model in exists precisely so
#     production inference has no MLflow dependency at runtime; a single "mlflow" mention in the
#     log means that guarantee has regressed. This is the one check worth automating, because a
#     reintroduced runtime MLflow call is easy to miss by eye.
#   - NOT gated: "the model loaded" and "the only failure was missing NWP data". The container is
#     EXPECTED to exit non-zero here — no NWP table is mounted for this isolated test, so it fails
#     at the NWP-availability lookup, which runs *after* the model has already loaded. That
#     ordering is the proof the model loaded; there is no reliable log string to grep for it. Read
#     the printed log and confirm by eye.
#
# One failure to read for by name: a traceback naming a feature the code cannot parse means the
# model baked into this image predates a rename of that feature. See docs/live_service/operations.md
# Step 2 for what to do about it.

set -euo pipefail

# Arbitrary but well-formed (YYYY-MM-DD-HH:MM) — see the header: the offline smoke test never
# reaches the point where the slot's validity could matter.
PARTITION_KEY="2026-01-01-00:00"

PROMOTION_JSON="data/production_model/promotion.json"
if [[ ! -f "$PROMOTION_JSON" ]]; then
  echo "error: $PROMOTION_JSON not found — materialise the promoted_model asset first" >&2
  echo "       (aws.md Step 3), so the model this build bakes in exists on disk." >&2
  exit 2
fi

# The bake COPYs the promoted directory wholesale and never re-runs promotion, so a directory
# promoted before live inference started reading this file would build and pass the smoke test,
# then fail every 6-hourly slot in production.
TRAINED_METADATA="data/production_model/time_series_metadata.parquet"
if [[ ! -f "$TRAINED_METADATA" ]]; then
  echo "error: $TRAINED_METADATA not found — this model predates the frozen metadata copy" >&2
  echo "       live inference reads. Re-train, then re-materialise promoted_model." >&2
  exit 2
fi

# The run id is read from the directory the build COPYs from, so the OCI label can never drift
# from the model actually baked in. The image tag is its first 12 hex chars — unique per promoted
# model and human-readable.
RUN_ID="$(jq -r .mlflow_run_id "$PROMOTION_JSON")"
IMAGE="nged-forecast:${RUN_ID:0:12}"

# See the header: the deployment is ARM end-to-end, so the build always targets linux/arm64.
# On a non-ARM host that needs a QEMU binfmt handler registered with the kernel; fail fast
# with the fix rather than letting the build die mid-way with "exec format error".
if [[ "$(uname -m)" != "aarch64" && ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
  echo "error: this host is $(uname -m) and no arm64 emulator is registered, so the required" >&2
  echo "       linux/arm64 build cannot run. Register QEMU (once per boot) with:" >&2
  echo "         docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
  exit 2
fi

echo "==> Building ${IMAGE} for linux/arm64  (MODEL_RUN_ID=${RUN_ID})"
docker build \
  --platform linux/arm64 \
  --build-arg MODEL_RUN_ID="$RUN_ID" \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t "$IMAGE" .

echo
echo "==> Smoke-testing ${IMAGE} with zero network access (partition ${PARTITION_KEY})"
# No credentials of any kind, and --network=none: together they prove runtime inference needs
# neither.
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT
set +e
docker run --network=none --platform linux/arm64 \
  "$IMAGE" \
  job execute -m nged_substation_forecast.definitions -j live_forecasts_job \
  --tags "{\"dagster/partition\": \"${PARTITION_KEY}\"}" \
  >"$LOG_FILE" 2>&1
RUN_EXIT=$?
set -e

echo "----- container log (exit ${RUN_EXIT}) ----------------------------------------------"
cat "$LOG_FILE"
echo "-------------------------------------------------------------------------------------"
echo

# The one automated gate: hermeticity. Any real runtime MLflow use imports the `mlflow` package,
# so a broken build surfaces the literal string somewhere in the log (import error, log line, or
# traceback). A case-insensitive match is therefore a sound, low-rot signal.
if grep -qi "mlflow" "$LOG_FILE"; then
  echo "  [FAIL] MLflow appears in the runtime log — the image is NOT hermetic." >&2
  echo "         Runtime inference must not depend on MLflow. Grep the log above for 'mlflow'." >&2
  exit 1
fi
echo "  [PASS] zero MLflow mentions — runtime is hermetic."
echo
echo "  Confirm by eye that the log above shows:"
echo "    - the model loading, then failing ONLY at the NWP-availability lookup, and"
echo "    - missing NWP data (no DATA_PATH_INTERNAL mounted) as the sole cause — nothing else."
echo "  A non-zero container exit (${RUN_EXIT}) is EXPECTED here; a zero exit would be suspicious."
echo
echo "==> ${IMAGE} passed the automated hermeticity check."
echo "    Push + deploy it with scripts/push_and_deploy_image.sh (aws.md Step 6)."
