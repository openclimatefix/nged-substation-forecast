#!/usr/bin/env bash
# Run the jobs of experiment 00078 four at a time (slot D: 4 cores, a few hundred MB of GPU memory
# per job). Each job skips work whose output file exists, so the script can be re-run.
#
#     uv run python ml-experiments/00078-*/e00078_capacity.py   # first: the capacity vintages
#     bash ml-experiments/00078-*/e00078_run.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1

uptime
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

for arm in am_sy am_sy_twin am_wk d1l_r d0_r d0_m0; do
    for seed in 0 1 2; do
        for site in D9 D10 D11 D14 D16; do
            echo "--arm ${arm} --seed ${seed} --site ${site}"
        done
    done
done | xargs -P 4 -L 1 uv run python "${DIR}/e00078_fit.py"
