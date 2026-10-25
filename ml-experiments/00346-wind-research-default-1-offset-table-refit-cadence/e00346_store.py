"""Store the scored-year quantiles of arm x1, the monthly refit on an expanding window.

One row per farm, seed, run, and valid time, with the farm's anonymous label (no identifier), the
lead, the actual fraction of capacity, and the 11 quantiles. Written once with
`experiment_store.write_predictions`; a re-run raises because the set exists.

    uv run --no-sync python ml-experiments/00346-*/e00346_store.py   (PYTHONPATH: see e00346_common)
"""

import e00303_score as score303  # isort: skip  # ty: ignore[unresolved-import]
import e00346_common as common
import numpy as np
import polars as pl
from studies.experiment_store import FOLD_A_TUESDAYS_V1, write_predictions

NAME = "wind-research-default-1-monthly-expanding-offset-table"


def main() -> None:
    """Write the set."""
    cases = score303.ordered_cases().select(
        "site", "power_fcst_init_time", "valid_time", "lead_hours", "lead_day", "actual_norm"
    )
    quantiles = np.load(common.FINAL_DIR / "x1.npy")
    frames = [
        cases.hstack(
            pl.DataFrame(q.astype(np.float32), schema=list(common.c303.LEVEL_COLUMNS))
        ).with_columns(seed=pl.lit(seed, dtype=pl.Int32))
        for seed, q in zip(common.SEEDS, quantiles, strict=True)
    ]
    path = write_predictions(
        experiment_id=common.EXPERIMENT_ID,
        name=NAME,
        frame=pl.concat(frames),
        kind="quantiles",
        row_set_id=FOLD_A_TUESDAYS_V1,
    )
    print(path)


if __name__ == "__main__":
    main()
