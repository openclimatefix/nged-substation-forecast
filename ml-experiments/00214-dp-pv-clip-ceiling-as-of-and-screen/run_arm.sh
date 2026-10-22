#!/usr/bin/env bash
# Usage: run_arm.sh ARM SITE : fit, forecast (A2 chain), and arm D (stage 2) for one arm and one site.
# One job uses one core; run at most four at once (see run_all.sh).
set -euo pipefail
arm=$1; shift
D=ml-experiments/00214-dp-pv-clip-ceiling-as-of-and-screen
run() { E00214_ARM=$arm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python "$@"; }
run $D/e00214_fit.py --cadences monthly --sites "$@" --workers 1
run $D/e00214_forecast.py --arms a2 --sites "$@" --workers 1
run $D/e00214s2_run.py --sites "$@" --workers 1
