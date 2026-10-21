"""Copy experiment 82's thresholds and fitted logistic scores into this experiment's score folders.

`e00132_run.py score` reads `thresholds.json` and `models.pkl` from its `--out` folder. The fitted
scores are re-pickled against this folder's copy of the modules, and nothing is refitted.

    uv run --no-sync python ml-experiments/00132-*/e00132_prepare.py
"""

from __future__ import annotations

import pickle
import shutil
from typing import Final

from e00132_common import WORK_DIR
from e00132_real import EXPERIMENT_82_DIR, load_models

FOLDERS: Final[tuple[str, ...]] = ("score_old", "score_new")


def main() -> None:
    """Write the two input files into each score folder."""
    models = load_models()
    for folder in FOLDERS:
        out = WORK_DIR / folder
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(EXPERIMENT_82_DIR / "thresholds.json", out / "thresholds.json")
        (out / "models.pkl").write_bytes(pickle.dumps(models))
        print(f"{folder}: thresholds.json and models.pkl written")


if __name__ == "__main__":
    main()
