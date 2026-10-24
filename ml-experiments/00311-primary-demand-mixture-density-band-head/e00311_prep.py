"""Write experiment 278's rows and arm-2 input matrix as memory-mapped arrays for the workers.

Duplicate (site, run, valid_time) rows across the early, tuning, replicate, and scored periods are
dropped, keeping the first. Output: `work/prep/*.npy` and `work/prep/rows.parquet`.

    PYTHONPATH=ml-experiments/00278-stack-retraining-cadence \
        uv run --no-sync python ml-experiments/00311-*/e00311_prep.py
"""

import e00278_h_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

import e00311_common as common  # isort: skip


def main() -> None:
    """Build the arrays."""
    frame = data.load_rows().with_row_index("orig")
    frame = frame.unique(subset=["site", "run", "valid_time"], keep="first", maintain_order=True)
    matrix = data.design_columns(frame=data.load_rows(), arm=common.INPUT_ARM)
    keep = frame["orig"].to_numpy()
    matrix = matrix[keep]
    width = ((frame["q95"] - frame["q05"]).to_numpy().astype(np.float64)) / 3.29
    scale = np.maximum(width, 0.15)
    common.PREP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(common.PREP_DIR / "x.npy", np.column_stack([matrix, np.log(scale)]).astype(np.float32))
    np.save(common.PREP_DIR / "mean.npy", frame["mean"].to_numpy().astype(np.float64))
    np.save(common.PREP_DIR / "scale.npy", scale)
    np.save(common.PREP_DIR / "actual.npy", frame["actual"].to_numpy().astype(np.float64))
    frame.select(
        "site", "run", "power_fcst_init_time", "valid_time", "lead_hours", "month_key", "period",
        pl.col("q50").alias("armd_q50"),
    ).write_parquet(common.PREP_DIR / "rows.parquet")  # fmt: skip
    print("rows", frame.height, "inputs", matrix.shape[1] + 1)


if __name__ == "__main__":
    main()
