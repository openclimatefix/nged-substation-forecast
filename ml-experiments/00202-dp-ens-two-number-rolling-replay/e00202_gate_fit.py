"""Gate G5: the copied fit reproduces experiment 193's fitted Q2 factors on its tuning months.

Experiment 193 fitted on the odd months of 2024-07 to 2025-06 (its `tuning` frame). Running
`fit_on` with float64 columns on the same rows must return the same factors.

    uv run --no-sync python ml-experiments/00202-*/e00202_gate_fit.py
"""

import json

import e00202_arms as arms
import e00202_common as common
import e00202_replay as replay
import numpy as np


def main() -> None:
    """Compare the refitted tuning factors with experiment 193's stored ones."""
    frame, delta, _ = replay.load_all()
    stats = arms.summaries(delta=delta)
    rows = np.flatnonzero((frame["period"] == "tuning").to_numpy())
    refit = replay.fit_on(frame=frame, stats=stats, rows=rows, spread=True)
    stored = json.loads(common.E193_FACTORS.read_text())["q2"]
    same = [list(map(float, a)) == list(map(float, b)) for a, b in zip(refit, stored, strict=True)]
    result = {"refit": refit, "stored_193": stored, "identical_per_band": same}
    common.write_json_atomic(path=common.WORK_DIR / "gate_g5_fit.json", content=result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
