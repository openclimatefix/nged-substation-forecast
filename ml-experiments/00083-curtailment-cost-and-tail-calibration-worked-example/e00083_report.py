"""Apply the tail-calibration table to experiment 48's predictions.

Reads experiment 48's stored quantile forecasts of Primary, BSP, and GSP demand (the weekly
retrained XGBoost model, the best-guess stack, the manual heuristic, and the climatology), scores
them with `studies.tail_calibration`, and prints every number the write-up quotes into `report.md`.

    uv run python ml-experiments/00083-*/e00083_report.py
"""

from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import (
    align_arms,
    read_arms,
    tail_calibration_markdown,
    tail_calibration_table,
)

SOURCE_EXPERIMENT_ID: Final[int] = 48
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
ARMS: Final[tuple[str, ...]] = ("weekly", "frozen_cs", "manual_heuristic", "climatology")
EXPERIMENT_ID: Final[int] = 83
SLUG: Final[str] = "curtailment-cost-and-tail-calibration-worked-example"
REPORT_PATH = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "report.md"


def main() -> None:
    """Score the four arms and print the report."""
    raw = read_arms(experiment_id=SOURCE_EXPERIMENT_ID, names=ARMS, row_set_id=ROW_SET_ID)
    raw = {name: frame.drop("site", "group") for name, frame in raw.items()}
    arms = align_arms(arms=raw)
    lines = [
        "# Experiment 83: tail calibration on experiment 48",
        "",
        f"Rows per arm before alignment: { {n: f.height for n, f in raw.items()} }.",
        (
            "Rows per arm after alignment (shared keys, finite actual): "
            f"{ {n: f.height for n, f in arms.items()} }."
        ),
        "",
        "## Tail calibration, all 20 demand series pooled, every lead and run from 2025-07-02",
        "",
        tail_calibration_markdown(
            tables={name: tail_calibration_table(frame=frame) for name, frame in arms.items()},
            unit="fraction of capacity",
        ),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
