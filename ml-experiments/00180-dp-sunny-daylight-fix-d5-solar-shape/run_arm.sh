#!/usr/bin/env bash
# Usage: run_arm.sh ARM SITE... : fit, forecast (A2 chain), and arm D (stage 2) for one arm.
set -euo pipefail
arm=$1; shift
D=ml-experiments/00180-dp-sunny-daylight-fix-d5-solar-shape
run() { E00180_ARM=$arm systemd-run --user --scope -q -p MemoryMax=24G -p OOMPolicy=continue uv run --no-sync python "$@"; }
run $D/e00180_fit.py --cadences monthly --sites "$@" --workers 6
run $D/e00180_forecast.py --arms a2 --sites "$@" --workers 6
run $D/e00180s2_run.py --sites "$@" --workers 6
