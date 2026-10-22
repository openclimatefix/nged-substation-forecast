"""Print every number the write-up of experiment 180 quotes that `e00180_score.py` does not.

Tail calibration (coverage and pinball loss at q01, q05, q95, q99), the median minus the mean
forecast, and the by-condition slices of D5 and of the placebos, for the control and arms P and K.

    uv run --no-sync python ml-experiments/00180-*/e00180_report.py
"""

import json
from typing import Final

import e00180_common as common
import e00180_score as score
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("c", "p", "k60")
TAIL: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def main() -> None:
    """Print and store the tables."""
    cache = common.load_cache()
    scores = json.loads((common.WORK_DIR / "scores_scored.json").read_text())
    out: dict = {"tail": {}, "median_minus_mean": {}}
    for site, arms in (("D5", ARMS), ("D10", ARMS), ("D16", ARMS)):
        for arm in arms:
            frame = score.add_slices(
                frame=score.arm_frame(cache=cache, arm=arm, site=site, period="scored", bg2=None)
            ).filter(pl.col("sunny"))
            actual = frame["actual"].to_numpy().astype(np.float64)
            row = {}
            for q in TAIL:
                predicted = frame[q].to_numpy().astype(np.float64)
                level = TAIL_LEVELS[q]
                error = actual - predicted
                row[q] = {
                    "coverage": float(np.mean(actual <= predicted)),
                    "pinball": float(np.mean(np.maximum(level * error, (level - 1.0) * error))),
                }
            out["tail"][f"{arm}|{site}"] = row
    for site in common.SCORED_SITES:
        raw = common.WORK_DIR / "arms" / "c" / "dp_preds_s2" / f"d_scored_{site}.parquet"
        frame = pl.read_parquet(raw)
        out["median_minus_mean"][site] = float((frame["q50"] - frame["mean"]).to_numpy().mean())
    common.write_json_atomic(path=common.WORK_DIR / "report.json", content=out)
    print("tail (sunny rows):")
    for key, row in out["tail"].items():
        print(
            " ", key, {q: (round(v["coverage"], 3), round(v["pinball"], 4)) for q, v in row.items()}
        )
    print(
        "median minus mean (points, all rows):",
        {k: round(v, 4) for k, v in out["median_minus_mean"].items()},
    )
    print("slices, D5, MAE (MW) and arm minus control:")
    for name in score.SLICES:
        cells = {arm: scores.get(f"{arm}|D5|{name}") for arm in ARMS}
        if cells["c"] is None:
            continue
        print(
            " ",
            name,
            cells["c"]["rows"],
            {a: round(c["mae"], 3) for a, c in cells.items() if c},
            "k60 minus c:",
            [round(x, 3) for x in cells["k60"]["mae_minus_control"]],
        )


if __name__ == "__main__":
    main()
