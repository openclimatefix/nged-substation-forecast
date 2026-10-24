"""Experiment 00343, second step: spread factor, offset table, and scored quantiles for each arm.

Runs experiment 00326's `main` (experiment 33's spread factor, then experiment 259's offset table
fitted on the usable hindcast cases) on this experiment's out-of-fold hindcast quantiles, with
`MIX_DIR` and `FINAL_DIR` pointed at this experiment's folders. Then copies experiment 340's stored
default into `work/final/` (the default arm, refitted in experiment 336, read only) and writes the
controls to `controls.json`.

    uv run --no-sync python ml-experiments/00343-*/e00343_final.py   (PYTHONPATH as for the fit)
"""

import json
import shutil

import e00326_final as final326  # isort: skip  # ty: ignore[unresolved-import]
import e00343_common as common
import numpy as np


def main() -> None:
    """Write the arms' final quantiles and the controls."""
    c326 = common.c326
    c326.MIX_DIR = common.MIX_DIR
    c326.FINAL_DIR = common.FINAL_DIR
    final326.ARMS = common.FIT_ARMS
    final326.main()
    for name in ("default", "default_notable", "default_rep"):
        shutil.copy(common.DEFAULT_FINAL_DIR / f"{name}.npy", common.FINAL_DIR / f"{name}.npy")
    shutil.copy(
        common.DEFAULT_FINAL_DIR / "replicate_cases.parquet",
        common.FINAL_DIR / "replicate_cases.parquet",
    )
    # M0 is M without the offset table: the same file the spread factor alone produces.
    np.save(common.FINAL_DIR / "M0.npy", np.load(common.FINAL_DIR / "M_notable.npy"))
    (common.FINAL_DIR / "controls.json").write_text(
        json.dumps({"known_answer_test": common.known_answer_test()}, indent=1)
    )


if __name__ == "__main__":
    main()
