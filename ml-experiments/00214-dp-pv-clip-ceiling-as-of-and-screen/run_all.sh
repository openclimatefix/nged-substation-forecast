#!/usr/bin/env bash
# Runs every (arm, site) job of experiment 214, at most four at once, inside one capped scope.
# Usage: systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue ml-experiments/00214-*/run_all.sh
set -euo pipefail
D=ml-experiments/00214-dp-pv-clip-ceiling-as-of-and-screen
printf '%s\n' "k60 D5" "k60 D11" "k60 D14" "k60 D9" "k50 D9" "k70 D9" "k80 D9" "k90 D9" "k50 D5" "k70 D5" "k90 D5" \
  | xargs -P 4 -L 1 bash -c '{ '"$D"'/run_arm.sh "$0" "$1" > "/tmp/e214_$0_$1.log" 2>&1 && echo "done $0 $1"; } || echo "FAILED $0 $1"'
