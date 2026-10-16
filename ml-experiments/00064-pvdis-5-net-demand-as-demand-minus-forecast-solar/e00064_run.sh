#!/usr/bin/env bash
# Run the fits of experiment 00064 four jobs at a time (the slot is 8 cores and 12 GB of GPU).
#
# Stages, in order; each job skips work whose output file exists, so a stage can be re-run:
#   weights   20 cross-fits (5 series x 4 variants), then the collected table work/weights.parquet
#   predict   the predict-only arms sf, sf_member, sf_twin, sf_temp (needs weights.parquet)
#   refits    a, am, b, a_twin, am_twin: 5 arms x 3 seeds x 5 series = 75 fits (needs weights.parquet)
#   all       the three stages in order
#
# Check `uptime` and `nvidia-smi` first: each job uses one XGBoost thread and a few hundred MB of
# GPU memory, so four fit side by side.
#
#     bash ml-experiments/00064-*/e00064_run.sh [weights|predict|refits|all]
set -euo pipefail

STAGE="${1:-all}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITES=(D9 D10 D11 D14 D16)
SEEDS=(0 1 2)
VARIANTS=(all no_lags no_cloud_proxies neither)
ARMS=(a am b a_twin am_twin)
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=1

uptime
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

run_parallel() {
    # Reads one argument line per job on stdin; the script path is the first argument.
    xargs -P 4 -L 1 uv run python "$1"
}

stage_weights() {
    for site in "${SITES[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            echo "--job ${site} ${variant}"
        done
    done | run_parallel "${DIR}/e00064_weights.py"
    uv run python "${DIR}/e00064_weights.py" --collect
}

stage_predict() {
    for seed in "${SEEDS[@]}"; do
        for site in "${SITES[@]}"; do
            echo "--arm d1l_shift --seed ${seed} --site ${site}"
        done
    done | run_parallel "${DIR}/e00064_fit.py"
}

stage_refits() {
    for seed in "${SEEDS[@]}"; do
        for arm in "${ARMS[@]}"; do
            for site in "${SITES[@]}"; do
                echo "--arm ${arm} --seed ${seed} --site ${site}"
            done
        done
    done | run_parallel "${DIR}/e00064_fit.py"
}

case "${STAGE}" in
    weights) stage_weights ;;
    predict) stage_predict ;;
    refits) stage_refits ;;
    all)
        stage_weights
        stage_predict
        stage_refits
        ;;
    *)
        echo "usage: $0 [weights|predict|refits|all]" >&2
        exit 2
        ;;
esac
