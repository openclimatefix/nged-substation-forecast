#!/usr/bin/env bash
# Run the fits of experiment 00074 seven jobs at a time (the slot is 8 cores and 12 GB of GPU).
#
# Each job is one series, one seed, and one monthly refit (all nine arms), and writes
# work/arms_raw/<site>_s<seed>_<refit>.parquet; a job whose file exists is skipped, so the script
# can be re-run after a crash. Run `e00074_fit.py --prepare SITE` for each series first.
#
#     bash ml-experiments/00074-*/e00074_run.sh 0        # seed 0, which decides the verdict
#     bash ml-experiments/00074-*/e00074_run.sh 1 2      # extra seeds (post hoc)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITES=(D9 D10 D11 D14 D16)
REFITS=(2025-07-01 2025-08-01 2025-09-01 2025-10-01 2025-11-01 2025-12-01
        2026-01-01 2026-02-01 2026-03-01 2026-04-01 2026-05-01 2026-06-01)
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1

uptime
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

for seed in "$@"; do
    for refit in "${REFITS[@]}"; do
        for site in "${SITES[@]}"; do
            echo "--fit ${site} ${seed} ${refit}"
        done
    done
done | xargs -P 7 -L 1 uv run python "${DIR}/e00074_fit.py"
