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
#   scripts/deploy/build_and_verify_image.sh <bundle-dir>
#
# <bundle-dir> is the exported step-5 neural-network bundle that the live_primary_substation_forecasts
# asset serves: the directory holding its manifest.json. The bundle reaches the image as a second
# named build context, `primary_substation_bundle`, so it never has to sit inside the repository
# checkout or pass through .dockerignore. Everything else is derived.
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
# The image tag is "<git commit>-r<model_release>-<research commit>", derived by _image_tag.sh,
# which push_and_deploy_image.sh sources too. Every software release and every bundle refit
# therefore gets a new tag, and the previous image stays available for rollback. The GIT_SHA build
# arg becomes an Open Container Initiative (OCI) label purely for traceability (inspect with
# `docker inspect`).
#
# The smoke test runs `dagster definitions validate` in the image with --network=none and no
# credentials. Validation imports every asset, check, job, and schedule module and resolves the
# whole asset graph, so a dangling import, a missing package in the image, or a broken asset
# dependency fails the build here rather than on the deployed control-plane box. It needs no
# network and no credentials, because loading the definitions reads no data. The script gates on
# the command's exit code: 0 means every code location loaded.
#
# The bundle check then runs scripts/deploy/verify_bundle.py inside the image, also with
# --network=none. It compares the SHA-256 of every bundle file the image holds against the bundle's
# manifest.json, and loads the bundle the way the asset does. The image cannot change after it is
# built, so this one check at build time stands in for a check on every run. The script finally
# prints the bundle's research commit and the image size, so both are on record for the release.

set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 <bundle-dir>" >&2
  echo "       <bundle-dir> is the step-5 bundle directory that holds manifest.json." >&2
  exit 2
fi
BUNDLE_DIR="$(cd "$1" && pwd)"

# shellcheck source=scripts/deploy/_image_tag.sh
source "$(dirname "$0")/_image_tag.sh"
IMAGE="nged-forecast:$(image_tag "$BUNDLE_DIR")"

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
  --build-context primary_substation_bundle="$BUNDLE_DIR" \
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
echo "==> Checking the bundle inside ${IMAGE} against its manifest, with zero network access"
if ! docker run --rm -i --network=none --platform linux/arm64 --entrypoint python \
  "$IMAGE" \
  -I - /app/primary_substation_bundle <"$(dirname "$0")/verify_bundle.py"; then
  echo "  [FAIL] The bundle in ${IMAGE} does not match its manifest. Read the log above." >&2
  exit 1
fi

SIZE_BYTES="$(docker image inspect --format '{{.Size}}' "$IMAGE")"
echo
echo "==> ${IMAGE} passed the smoke test and the bundle check."
echo "    Image size: $((SIZE_BYTES / 1024 / 1024)) MiB"
echo "    Push + deploy it with scripts/deploy/push_and_deploy_image.sh ${BUNDLE_DIR} (aws.md Step 6)."
