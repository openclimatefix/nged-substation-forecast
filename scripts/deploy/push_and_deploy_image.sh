#!/usr/bin/env bash
#
# Push the already-built production image to the Elastic Container Registry (ECR) and point the
# Elastic Container Service (ECS) task definition at the pushed image.
#
# The script is Step 6 of the AWS setup runbook as one command and, together with
# scripts/deploy/build_and_verify_image.sh, the whole recurring champion-redeploy loop (aws.md
# "Redeploying a new champion model"). This header is the source of truth for *why* each choice
# below is made. The runbook is at
# <https://openclimatefix.github.io/nged-substation-forecast/live_service/aws/>.
#
# Usage:
#   scripts/deploy/push_and_deploy_image.sh          # no arguments — everything is derived
#
# Zero arguments by design, so nothing can be mistyped or drift:
#   - The image tag is derived from data/production_model/promotion.json exactly as
#     build_and_verify_image.sh derives it (first 12 hex chars of the promoted run id), so this
#     script can only ever push the image that script built and verified.
#   - The AWS account id comes from `aws sts get-caller-identity` — never typed by hand.
#   - Region, repository, task-definition family, and container name are the fixed names the
#     runbook establishes (eu-west-2 / nged-forecast everywhere). Region is passed explicitly on
#     every AWS call because the AWS command-line interface falls back to the configured default
#     region when `--region` is omitted, which aws.md Step 10 warns about by name.
#
# What the script does:
#   1. Push: ECR login — the instance or user credentials come from your AWS config or role, and no
#      static keys are handled here — then tag, then push.
#   2. Deploy: if the `nged-forecast` task-definition family exists, register a NEW REVISION that
#      is a copy of the latest revision with only the container image URI changed. The
#      EcsRunLauncher on the control-plane box resolves the family's latest revision at launch
#      time, so the next scheduled run picks the new image up with no restart on the box.
#
# First-time setup: the task-definition family does not exist until aws.md Step 9 creates it in the
# console, so on a fresh account the deploy half reports the missing family and exits 0 — the push
# half is all Step 6 needs. Re-running after the family exists (or when the latest revision already
# points at this image) is safe and idempotent.
#
# The revision is built by ALLOWLISTING the fields `register-task-definition` accepts from the
# `describe-task-definition` output; describe returns extra read-only fields — revision, status,
# registeredAt, and more — that register rejects. The list is an allowlist rather than a denylist,
# so a read-only field added to the describe output later cannot break this script.
#
# Deliberately NOT here: creating any infrastructure (buckets, Identity and Access Management roles,
# the cluster, the first task definition). One-time infrastructure stays in the console, per the
# runbook, until Stage 2 of the access-phasing plan replaces those console steps with
# infrastructure-as-code — ad-hoc bash that mutates infrastructure would be unreviewable.

set -euo pipefail

REGION="${AWS_REGION:-eu-west-2}"
REPO="nged-forecast"
TASK_FAMILY="nged-forecast"
CONTAINER_NAME="nged-forecast"

PROMOTION_JSON="data/production_model/promotion.json"
if [[ ! -f "$PROMOTION_JSON" ]]; then
  echo "error: $PROMOTION_JSON not found — materialise the promoted_model asset first" >&2
  echo "       (aws.md Step 3), then build the image (aws.md Step 4)." >&2
  exit 2
fi

RUN_ID="$(jq -r .mlflow_run_id "$PROMOTION_JSON")"
LOCAL_IMAGE="${REPO}:${RUN_ID:0:12}"

if ! docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
  echo "error: local image ${LOCAL_IMAGE} not found — run" >&2
  echo "       scripts/deploy/build_and_verify_image.sh first (aws.md Step 4), so only a" >&2
  echo "       built-and-verified image can be pushed." >&2
  exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REMOTE_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${RUN_ID:0:12}"

echo "==> Pushing ${LOCAL_IMAGE} to ${REMOTE_IMAGE}"
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker tag "$LOCAL_IMAGE" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"

echo
echo "==> Pointing the ${TASK_FAMILY} task definition at the new image"
if ! CURRENT_TASK_DEF="$(aws ecs describe-task-definition --region "$REGION" \
  --task-definition "$TASK_FAMILY" --output json 2>/dev/null)"; then
  echo "  Task-definition family '${TASK_FAMILY}' does not exist yet — nothing to deploy to."
  echo "  This is expected on first-time setup: the push above is all aws.md Step 6 needs;"
  echo "  create the task definition in the console next (aws.md Step 9). Re-run this script"
  echo "  on future redeploys and it will register new revisions automatically."
  exit 0
fi

CURRENT_IMAGE="$(echo "$CURRENT_TASK_DEF" \
  | jq -r --arg name "$CONTAINER_NAME" \
      '.taskDefinition.containerDefinitions[] | select(.name == $name) | .image')"
if [[ "$CURRENT_IMAGE" == "$REMOTE_IMAGE" ]]; then
  echo "  Latest revision already points at ${REMOTE_IMAGE} — nothing to do."
  exit 0
fi

# Allowlist the register-accepted fields, drop nulls (register rejects explicit nulls for
# fields that describe reports as absent), and swap the image on the named container only.
NEW_TASK_DEF="$(echo "$CURRENT_TASK_DEF" | jq \
  --arg name "$CONTAINER_NAME" --arg image "$REMOTE_IMAGE" '
  .taskDefinition
  | {family, taskRoleArn, executionRoleArn, networkMode, containerDefinitions, volumes,
     placementConstraints, requiresCompatibilities, cpu, memory, pidMode, ipcMode,
     proxyConfiguration, ephemeralStorage, runtimePlatform}
  | with_entries(select(.value != null))
  | .containerDefinitions |= map(if .name == $name then .image = $image else . end)
')"

NEW_REVISION="$(aws ecs register-task-definition --region "$REGION" \
  --cli-input-json "$NEW_TASK_DEF" \
  --query 'taskDefinition.revision' --output text)"

echo "  Registered ${TASK_FAMILY} revision ${NEW_REVISION}:"
echo "    ${CURRENT_IMAGE}"
echo "    -> ${REMOTE_IMAGE}"
echo
echo "==> Done. EcsRunLauncher resolves the family's latest revision at launch time, so the"
echo "    next scheduled run uses this image — no restart needed on the control-plane box."
echo "    (Control-plane containers only need updating when CODE changed, not just the model:"
echo "    see aws.md 'Redeploying a new champion model'.)"
