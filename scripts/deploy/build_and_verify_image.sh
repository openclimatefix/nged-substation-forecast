#!/usr/bin/env bash
#
# Build the production image and smoke-test it with zero network access.
#
# The script is Step 4 of the AWS setup runbook as one command: build the image, then prove that
# the Dagster definitions load inside it. This header is the source of truth for *why* each choice
# below is made. The runbook is the file `docs/live_service/aws.md`, cited below as aws.md and
# published at <https://openclimatefix.github.io/nged-substation-forecast/live_service/aws/>.
#
# Usage:
#   scripts/deploy/build_and_verify_image.sh          # no arguments — everything is derived
#
# The image is built for linux/arm64 REGARDLESS of the host architecture. The Elastic Container
# Service (ECS) task definition declares ARM64, and the control-plane box is a Graviton instance.
# The control-plane box is the always-on EC2 instance that runs Dagster, set up in aws.md Step 11.
# So an amd64 image cannot run anywhere in the deployment. ARM64 is declared because ARM Fargate is
# ~20% cheaper. A native x86 build pushes fine, but every task launch then dies with "image Manifest
# does not contain descriptor matching platform 'linux/arm64 v8'" (aws.md Steps 9-11). On an x86
# host both the build and the smoke test therefore run under QEMU user-mode emulation, which is
# slower than native but correct. The script checks the emulator is registered and prints the fix if
# it is not.
#
# The image tag is the short git commit hash of the checked-out commit, so every software release
# gets a new tag and the previous image stays available for rollback. The GIT_SHA build arg becomes
# an Open Container Initiative (OCI) label purely for traceability (inspect with `docker inspect`).
#
# The smoke test runs `dagster definitions validate` in the image with --network=none and no
# credentials. Validation imports every asset, check, job, and schedule module and resolves the
# whole asset graph, so a dangling import, a missing package in the image, or a broken asset
# dependency fails the build here rather than on the deployed control-plane box. It needs no
# network and no credentials, because loading the definitions reads no data. The script gates on
# the command's exit code: 0 means every code location loaded.

set -euo pipefail

IMAGE="nged-forecast:$(git rev-parse --short=8 HEAD)"

# See the header: the deployment is ARM end-to-end, so the build always targets linux/arm64. On a
# non-ARM host the build needs a QEMU binfmt handler registered with the kernel. So fail fast with
# the fix, rather than letting the build die mid-way with "exec format error".
if [[ "$(uname -m)" != "aarch64" && ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
  echo "error: this host is $(uname -m) and no arm64 emulator is registered, so the required" >&2
  echo "       linux/arm64 build cannot run. Register QEMU (once per boot) with:" >&2
  echo "         docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
  exit 2
fi

echo "==> Building ${IMAGE} for linux/arm64"
docker build \
  --platform linux/arm64 \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t "$IMAGE" .

echo
echo "==> Smoke-testing ${IMAGE} with zero network access: dagster definitions validate"
if ! docker run --rm --network=none --platform linux/arm64 \
  "$IMAGE" \
  definitions validate -m nged_substation_forecast.definitions; then
  echo "  [FAIL] The Dagster definitions do not load in ${IMAGE}. Read the log above." >&2
  exit 1
fi
echo
echo "==> ${IMAGE} passed the smoke test."
echo "    Push + deploy it with scripts/deploy/push_and_deploy_image.sh (aws.md Step 6)."
