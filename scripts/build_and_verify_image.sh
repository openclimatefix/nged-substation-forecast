#!/usr/bin/env bash
#
# Build the production image and smoke-test it with zero network access.
#
# This is Steps 2-3 of the deployment runbook rolled into one command. The runbook explains the
# *why* behind every choice made here — dummy-but-present NGED creds, --network=none, selecting
# the job with -j rather than a --partition flag — and is the source of truth:
#   docs/live_service/deployment.md
#
# Usage:
#   scripts/build_and_verify_image.sh <partition-key>
#   e.g. scripts/build_and_verify_image.sh 2026-07-04-00:00
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
#     the printed log and confirm by eye, per deployment.md Step 3.

set -euo pipefail

PARTITION_KEY="${1:-}"
if [[ -z "$PARTITION_KEY" ]]; then
  echo "usage: $0 <partition-key>   e.g. $0 2026-07-04-00:00" >&2
  exit 2
fi

PROMOTION_JSON="data/production_model/promotion.json"
if [[ ! -f "$PROMOTION_JSON" ]]; then
  echo "error: $PROMOTION_JSON not found — materialise the promoted_model asset first" >&2
  echo "       (deployment.md Step 1), so the model this build bakes in exists on disk." >&2
  exit 2
fi

# The run id is read from the directory the build COPYs from, so the OCI label can never drift
# from the model actually baked in. The image tag is its first 12 hex chars — unique per promoted
# model and human-readable.
RUN_ID="$(jq -r .mlflow_run_id "$PROMOTION_JSON")"
IMAGE="nged-forecast:${RUN_ID:0:12}"

echo "==> Building ${IMAGE}  (MODEL_RUN_ID=${RUN_ID})"
docker build \
  --build-arg MODEL_RUN_ID="$RUN_ID" \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t "$IMAGE" .

echo
echo "==> Smoke-testing ${IMAGE} with zero network access (partition ${PARTITION_KEY})"
# Dummy NGED creds: Settings requires them at import, but they need only be PRESENT — never real,
# never reachable. --network=none proves runtime inference needs no network at all.
LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT
set +e
docker run --network=none \
  -e NGED_S3_BUCKET_URL=https://example.com/outbound/ \
  -e NGED_S3_BUCKET_ACCESS_KEY=dummy \
  -e NGED_S3_BUCKET_SECRET=dummy \
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
echo "  Confirm by eye (deployment.md Step 3) that the log above shows:"
echo "    - the model loading, then failing ONLY at the NWP-availability lookup, and"
echo "    - missing NWP data (no DATA_PATH_INTERNAL mounted) as the sole cause — nothing else."
echo "  A non-zero container exit (${RUN_EXIT}) is EXPECTED here; a zero exit would be suspicious."
echo
echo "==> ${IMAGE} passed the automated hermeticity check. Push it with deployment.md Step 5."
