#!/usr/bin/env bash
# Usage: run_chain.sh SOURCE SITE... : fit, forecast, and build the stage-2 arms for the sites.
# SOURCE is `mva` (the eight new sites) or `control` (MW sites from experiment 111's stores).
set -euo pipefail
source=$1; shift
D=ml-experiments/00252-gam-stack-at-eight-more-primaries
run() { E00252_SOURCE=$source E00252_ARM=a0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue uv run --no-sync python "$@"; }
for site in "$@"; do
  run $D/e00252_fit.py --cadences monthly --sites "$site" --workers 1
  run $D/e00252_forecast.py --arms a2 --sites "$site" --workers 1
  run $D/e00252s2_run.py --sites "$site" --workers 1
done
