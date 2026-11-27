#!/usr/bin/env bash
#
# Shared by build_and_verify_image.sh and push_and_deploy_image.sh, which source this file, so the
# two scripts derive the same image tag and the push can only ever push the image the build made.
#
# The tag is "<git commit>-r<model_release>-<research commit>-<refit month>": the short hash of the
# checked-out commit, the bundle's `model_release` from its manifest.json, the first 8 characters of
# the research commit that trained the bundle, and the bundle's `refit_month` as YYYYMM. Two monthly
# refits from one research commit therefore get different tags, and the previous image stays
# available for rollback.

# Print the tag for the bundle directory $1, or exit 2 if the bundle has no readable manifest.
image_tag() {
  local bundle_dir="$1"
  local manifest="${bundle_dir}/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "error: ${manifest} not found; pass the bundle directory that holds manifest.json" >&2
    exit 2
  fi
  # The callers run this function inside a command substitution, which does not inherit set -e,
  # so each command that can fail exits explicitly rather than printing a malformed tag.
  local release research refit commit
  release="$(jq -er '.model_release' "$manifest")" || exit 2
  research="$(jq -er '.research_git.commit[0:8]' "$manifest")" || exit 2
  refit="$(jq -er '.refit_month | select(test("^[0-9]{4}-[0-9]{2}$")) | sub("-"; "")' "$manifest")" || exit 2
  commit="$(git rev-parse --short=8 HEAD)" || exit 2
  echo "${commit}-r${release}-${research}-${refit}"
}
