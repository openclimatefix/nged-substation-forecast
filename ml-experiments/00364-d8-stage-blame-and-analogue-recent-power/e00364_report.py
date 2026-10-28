# ruff: noqa: E501
"""Print every number the experiment 364 page quotes, from `work/scores.json`."""

import json
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00364-d8-stage-blame-and-analogue-recent-power"
SLICES: Final[tuple[str, ...]] = ("all", "0-24h", "24-72h", "72-168h", "step_days", "no_step_days")


def fmt(*, value: list[float], digits: int = 3) -> str:
    """Return `point (low to high)`."""
    return f"{value[0]:.{digits}f} ({value[1]:.{digits}f} to {value[2]:.{digits}f})"


def main() -> None:
    """Print arms, contrasts, monthly scores, and gates."""
    scores = json.loads((ML_EXPERIMENTS_DATA_DIR / SLUG / "work" / "scores.json").read_text())
    for key, block in scores.items():
        print(
            f"\n=== {key}: rows {block['rows']}, capacity {block['capacity_mw']:.3f} MW, gate {block['gate']}"
        )
        for sl in SLICES:
            if sl not in block["arms"]:
                continue
            print(f"\n-- arms, slice {sl}: score | MAE (MW) | bias (MW) | correlation")
            for arm, v in block["arms"][sl].items():
                print(
                    f"{arm:12s} {fmt(value=v['score'])} | {fmt(value=v['mae_mw'])} | {fmt(value=v['bias_mw'])} | {fmt(value=v['corr'], digits=4)}"
                )
            print(f"-- contrasts (first minus second), slice {sl}")
            for name, v in block["contrasts"][sl].items():
                print(
                    f"{name:24s} score {fmt(value=v['score'])} | MAE {fmt(value=v['mae_mw'])} | corr {fmt(value=v['corr'], digits=4)} | bias {fmt(value=v['bias_mw'])}"
                )
        print("\n-- monthly score and correlation, first arms")
        for month, arms in block["monthly"].items():
            print(
                month,
                {
                    a: (round(v["score"], 2), round(v["corr"], 3))
                    for a, v in arms.items()
                    if a in ("S1", "S4", "AN", "BG2_reblend")
                },
            )


if __name__ == "__main__":
    main()
