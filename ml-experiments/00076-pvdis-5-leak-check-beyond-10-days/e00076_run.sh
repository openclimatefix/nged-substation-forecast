#!/usr/bin/env bash
# Run the fits of experiment 00076 four jobs at a time (slot D: 4 cores, a few hundred MB of GPU
# memory per job). Each job skips work whose output file exists, so the script can be re-run.
#
#     bash ml-experiments/00076-*/e00076_run.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1

uptime
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

for arm in am_train_only am0; do
    for seed in 0 1 2; do
        for site in D9 D10 D11 D14 D16; do
            echo "--arm ${arm} --seed ${seed} --site ${site}"
        done
    done
done | xargs -P 4 -L 1 uv run python "${DIR}/e00076_fit.py"
