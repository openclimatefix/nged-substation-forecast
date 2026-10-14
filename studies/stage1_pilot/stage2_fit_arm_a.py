"""Fit arm (a): the ENS-trained single-stage quantile model, applied to every member.

Stage 1a's columns, the ENS control member's weather as the training weather, the training months
2024-04 to 2025-06 with the same cleaning as `stage1_fit_enstrained.py`, target power over effective
capacity, and the 9-level multi-quantile loss of `stage2_fit_stage1a_quantile.py`. One XGBoost
model per series and seed. Output:
`data/studies/per_study/stage2_pilot/arm_a/seed<s>/init=<date>.parquet` (keys and the 9 quantile
columns for every validation row and member), skipped if it exists.

    uv run python studies/stage1_pilot/stage2_fit_arm_a.py
"""

import time
from pathlib import Path
from typing import Final

import polars as pl
import stage1_inputs as inputs
from stage1_fit_enstrained import training_rows
from stage2_common import ENS_FEATURES_DIR, STAGE2_ROOT
from stage2_fit_stage1a_quantile import N_ROUNDS, fit, predict

OUT_DIR: Final[Path] = STAGE2_ROOT / "arm_a"


def main() -> None:
    """Fit every seed and write the per-member quantiles."""
    sites = inputs.site_table()
    technology = {int(r["time_series_id"]): r["technology"] for r in sites.iter_rows(named=True)}
    training = training_rows(sites=sites).rename({"power": "target"})
    val_files = sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet"))
    for seed in inputs.SEEDS:
        if all((OUT_DIR / f"seed{seed}" / p.name).exists() for p in val_files):
            continue
        started = time.monotonic()
        models = {
            series: fit(
                rows=training.filter(pl.col("time_series_id") == series),
                columns=list(inputs.FEATURES[technology[series]]),
                seed=seed,
                rounds=N_ROUNDS,
            )
            for series in technology
        }
        print(f"seed {seed}: fitted in {time.monotonic() - started:.0f} s", flush=True)
        for path in val_files:
            target = OUT_DIR / f"seed{seed}" / path.name
            if target.exists():
                continue
            pieces = []
            for key, part in (
                pl.read_parquet(path).partition_by("time_series_id", as_dict=True).items()
            ):
                series = int(key[0])
                pieces.append(
                    pl.concat(
                        [
                            part.select(*inputs.KEY_COLUMNS),
                            predict(
                                booster=models[series],
                                frame=part,
                                columns=list(inputs.FEATURES[technology[series]]),
                            ),
                        ],
                        how="horizontal",
                    )
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
        print(f"seed {seed}: done in {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
