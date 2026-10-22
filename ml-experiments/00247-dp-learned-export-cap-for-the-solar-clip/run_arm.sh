#!/usr/bin/env bash
# Usage: run_arm.sh ARM SITE : fit, forecast (A2 chain), and arm D (stage 2) for one arm and one site.
# One job uses one core; run at most four at once (see run_all.sh).
set -euo pipefail
arm=$1; shift
D=ml-experiments/00247-dp-learned-export-cap-for-the-solar-clip
run() { E00247_ARM=$arm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python "$@"; }
run $D/e00247_fit.py --cadences monthly --sites "$@" --workers 1
run $D/e00247_forecast.py --arms a2 --sites "$@" --workers 1
run $D/e00247s2_run.py --sites "$@" --workers 1
