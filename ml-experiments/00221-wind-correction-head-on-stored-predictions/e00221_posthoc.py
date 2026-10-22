# ruff: noqa: E501
"""Post hoc diagnostics of experiment 221: why the pooled heads under-cover out of sample.

For one seed, refit A1 and A2 with fewer epochs (2) and as linear heads (no hidden layers), write
those arms on the same rows, and write the in-sample score of the 12-epoch heads on their own
training rows beside the best guess's score on those rows. Labelled post hoc; no planned contrast
changes.

Output: `work/arms/<arm>_seed<seed>.parquet` for the variants and
`work/posthoc/insample_seed<seed>.parquet`.

    uv run --no-sync python ml-experiments/00221-wind-correction-head-on-stored-predictions/e00221_posthoc.py --seed 0
"""

import argparse
from datetime import UTC, datetime

import e00221_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00221_heads import (
    HEAD_ARMS,
    PURGE_HOURS,
    SCORED_MONTHS,
    arm_rows,
    base_quantiles,
    fit_both,
    fit_predict,
    prepare,
    write_arm,
)
from e00221_inputs import OUT_PATH

VARIANTS: dict[str, dict[str, int]] = {
    "short": {"epochs": 2, "hidden": 64},
    "linear": {"epochs": 12, "hidden": 0},
}


def main() -> None:
    """Fit the variants and the in-sample diagnostic for one seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    seed = parser.parse_args().seed
    torch.set_num_threads(2)
    frame = prepare(pl.read_parquet(OUT_PATH).filter((pl.col("seed") == seed) & ~pl.col("masked")))
    pieces: dict[str, list[pl.DataFrame]] = {}
    insample = []
    for month_index, month in enumerate(SCORED_MONTHS):
        year, mon = (int(part) for part in month.split("-"))
        start = datetime(year, mon, 1, tzinfo=UTC)
        train = frame.filter(
            (pl.col("power_fcst_init_time") < start)
            & (pl.col("valid_time") < start - np.timedelta64(PURGE_HOURS, "h"))
        )
        test = frame.filter(pl.col("month") == month)
        torch_seed = seed * 1000 + month_index
        for arm in ("A1", "A2"):
            for name, options in VARIANTS.items():
                quantiles = fit_predict(
                    spec=HEAD_ARMS[arm],
                    train=train,
                    test=test,
                    torch_seed=torch_seed,
                    epochs=options["epochs"],
                    hidden=options["hidden"],
                )
                pieces.setdefault(f"{arm}_{name}", []).append(
                    arm_rows(test=test, quantiles=quantiles, seed=seed)
                )
            _, trained = fit_both(
                spec=HEAD_ARMS[arm], train=train, test=test, torch_seed=torch_seed
            )
            insample.append(
                arm_rows(test=train, quantiles=trained, seed=seed).with_columns(
                    arm=pl.lit(arm), month=pl.lit(month)
                )
            )
            if arm == "A1":
                insample.append(
                    arm_rows(
                        test=train, quantiles=base_quantiles(frame=train, base="bg"), seed=seed
                    ).with_columns(arm=pl.lit("A0"), month=pl.lit(month))
                )
        print(f"seed {seed} {month} done", flush=True)
    for name, parts in pieces.items():
        write_arm(name=name, seed=seed, pieces=parts)
    target = common.WORK_DIR / "posthoc"
    target.mkdir(exist_ok=True)
    pl.concat(insample).write_parquet(target / f"insample_seed{seed}.parquet")


if __name__ == "__main__":
    main()
