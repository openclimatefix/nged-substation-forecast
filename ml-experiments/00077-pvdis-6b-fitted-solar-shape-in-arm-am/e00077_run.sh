#!/usr/bin/env bash
# Run the weights and fits of experiment 00077 for one solar shape, four jobs at a time
# (slot E: 6 cores and 8 GB of GPU; each job uses one XGBoost thread and a few hundred MB).
#
# Stages, in order; each job skips work whose output file exists, so a stage can be re-run:
#   weights   5 cross-fits (one per series), then the collected table
#   fits      arm am for 5 series x 3 seeds, and arm am_twin when the shape is `blend`
#
# Run `e00077_orient.py`, `e00077_shares.py`, `e00077_solar_inputs.py`, `e00077_capacity.py`, and
# `e00077_inputs.py` first (see `e00077_common.py`), then `e00077_assemble.py` and
# `e00077_stats.py` afterwards.
#
#     bash ml-experiments/00077-*/e00077_run.sh blend|s25|shuffled
set -euo pipefail

SHAPE="${1:?give the solar shape: blend, s25, or shuffled}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITES=(D9 D10 D11 D14 D16)
SEEDS=(0 1 2)
ARMS=(am)
if [[ "${SHAPE}" == "blend" ]]; then
    ARMS=(am am_twin)
fi
export E77_ARM="${SHAPE}" OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1
export MLFLOW_DISABLE_AGENT_HINT=1

uptime
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

run_parallel() {
    # Reads one argument line per job on stdin; the script path is the first argument.
    xargs -P 4 -L 1 uv run python "$1"
}

for site in "${SITES[@]}"; do
    echo "--job ${site} all"
done | run_parallel "${DIR}/e00077_weights.py"
uv run python "${DIR}/e00077_weights.py" --collect

for seed in "${SEEDS[@]}"; do
    for arm in "${ARMS[@]}"; do
        for site in "${SITES[@]}"; do
            echo "--arm ${arm} --seed ${seed} --site ${site}"
        done
    done
done | run_parallel "${DIR}/e00077_fit.py"
