#!/usr/bin/env bash
#
# Shared by build_and_verify_image.sh and push_and_deploy_image.sh, which source this file, so the
# two scripts derive the same image tag and the push can only ever push the image the build made.
#
# The tag is "<git commit>-r<model_release>-<research commit>": the short hash of the checked-out
# commit, the bundle's `model_release` from its manifest.json, and the first 8 characters of the
# research commit that trained the bundle. A monthly refit with no code change therefore gets a new
# tag, and the previous image stays available for rollback.

# Print the tag for the bundle directory $1, or exit 2 if the bundle has no readable manifest.
image_tag() {
  local bundle_dir="$1"
  local manifest="${bundle_dir}/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "error: ${manifest} not found; pass the bundle directory that holds manifest.json" >&2
    exit 2
  fi
  local release research
  release="$(jq -er '.model_release' "$manifest")"
  research="$(jq -er '.research_git.commit[0:8]' "$manifest")"
  echo "$(git rev-parse --short=8 HEAD)-r${release}-${research}"
}
