"""Tail calibration of experiment 324's arms on the kept rows: shares beyond the tail quantiles.

Writes `tails.json`: for each arm and row set, the share of observed power strictly below the 0.01
and 0.05 quantiles and strictly above the 0.95 and 0.99 quantiles (nominal 1%, 5%, 5%, 1%), and the
pinball loss at those four levels in points of capacity.

    PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting \
        uv run --no-sync python ml-experiments/00324-*/e00324_tails.py
"""

import json
from typing import Final

import e00303_score as s303  # ty: ignore[unresolved-import]
import e00324_common as common
import numpy as np

ARMS: Final[tuple[str, ...]] = (
    "W0",
    "P1",
    "P3",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
    "W7",
    "W8_W1",
    "W8_W4",
)
LEVEL_INDEX: Final[dict[str, int]] = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}


def main() -> None:
    """Write the tail table."""
    cases = s303.ordered_cases()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    kept = ~cases["masked"].to_numpy()
    sets = {
        "kept": kept,
        "kept_days_1_2": kept & cases["lead_day"].is_between(1, 2).to_numpy(),
        "kept_days_3_10": kept & cases["lead_day"].is_between(3, 10).to_numpy(),
    }
    out: dict = {}
    for arm in ARMS:
        q = np.load(common.FINAL_DIR / f"{arm}.npy").astype(np.float64)
        out[arm] = {}
        for name, rows in sets.items():
            record = {}
            for label, i in LEVEL_INDEX.items():
                level = float(s303.common.LEVELS[i])
                below = actual[None, rows] < q[:, rows, i]
                above = actual[None, rows] > q[:, rows, i]
                record[f"share_{'below' if level < 0.5 else 'above'}_{label}"] = float(
                    (below if level < 0.5 else above).mean()
                )
                error = actual[None, rows] - q[:, rows, i]
                record[f"pinball_{label}"] = float(
                    np.maximum(level * error, (level - 1.0) * error).mean() * 100.0
                )
            out[arm][name] = record
    (common.DATA_DIR / "tails.json").write_text(json.dumps(out, indent=1))
    for arm, sets_ in out.items():
        r = sets_["kept"]
        print(
            arm,
            *(f"{k.split('_', 1)[1]}={v:.3f}" for k, v in r.items()),
        )


if __name__ == "__main__":
    main()
