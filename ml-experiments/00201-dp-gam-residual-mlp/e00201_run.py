"""Train every arm on the real power with three seeds, and store the scored-year predictions.

Writes `work/preds/arm<k>_seed<s>.parquet` (the scored-year rows' keys, actual, lead, site, era,
the corrected quantiles, and the network's median shift and log scale) and `work/fits.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00201-*/e00201_run.py
"""

import json
import os
from typing import Final

import e00201_common as common
import e00201_data as data
import e00201_nn as nn_
import numpy as np
import polars as pl
import torch

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
KEYS: Final[tuple[str, ...]] = (
    "site", "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "era",
    "actual",
)  # fmt: skip


def main() -> None:
    """Fit all arms and seeds."""
    torch.set_num_threads(1)
    loaded = data.load_data()
    frame, split = loaded.frame, loaded.split
    q = nn_.quantile_matrix(frame=frame)
    actual = frame["actual"].to_numpy().astype(np.float64)
    test = split == "test"
    out_dir = common.WORK_DIR / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = frame.select(list(KEYS)).filter(pl.Series(test))
    records = []
    for arm in (*data.ARMS, *data.ABLATION_ARMS):
        for seed in SEEDS:
            target = out_dir / f"arm{arm}_seed{seed}.parquet"
            if target.exists():
                continue
            fit = nn_.fit_predict(
                x=loaded.matrices[arm], q=q, actual=actual, split=split, seed=seed
            )
            columns = [
                pl.Series(name, fit.quantiles[test][:, i])
                for i, name in enumerate(common.QUANTILES)
            ]
            part = keys.with_columns(
                *columns,
                pl.Series("delta", fit.delta[test]),
                pl.Series("log_scale", fit.log_scale[test]),
            )
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            part.write_parquet(temporary)
            temporary.rename(target)
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "n_inputs": int(loaded.matrices[arm].shape[1]),
                    "epochs": fit.epochs,
                    "val_base": fit.base_val,
                    "val_best": fit.best_val,
                }
            )
            print(records[-1], flush=True)
    (common.WORK_DIR / "fits.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
