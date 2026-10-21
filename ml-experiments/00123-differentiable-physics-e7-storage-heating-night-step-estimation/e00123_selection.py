"""The selection winter (2025-26) beside the headline winters, for experiment 123.

Prints, per fitted series, the step fitted to winter 2025-26 alone with its 90% interval, the
mean of the four headline winters' single-winter steps, and their difference; and the pooled
steps of the five sets for winter 2025-26 alone. Needs `numbers_all.json` (the `all` stage, run
after PR 1397 has merged).

    uv run python ml-experiments/00123-*/e00123_selection.py
"""

import json
from typing import Final

import e00123_common as common
import e00123_fit as fit
import e00123_pool as pool
import numpy as np
import polars as pl

HEADLINE: Final[tuple[int, ...]] = common.HEADLINE_WINTERS


def main() -> None:
    """Print the selection-winter table and pooled sets."""
    if not common.scored_winter_unlocked():
        msg = "winter 2025-26 is locked"
        raise PermissionError(msg)
    numbers = json.loads((common.WORK_DIR / "numbers_all.json").read_text())
    single = pl.DataFrame(numbers["per_winter"]).with_columns(
        winter=pl.col("winters").cast(pl.Int64)
    )
    rows = []
    for site in common.FITTED_SERIES:
        own = single.filter(pl.col("site") == site)
        selection = own.filter(pl.col("winter") == common.SELECTION_WINTER).row(0, named=True)
        headline = own.filter(pl.col("winter").is_in(list(HEADLINE)))
        mean = float(np.mean(headline["step"].to_list()))
        rows.append(
            {
                "site": site,
                "step_2025": selection["step"],
                "low": selection["step_low"],
                "high": selection["step_high"],
                "headline_winters_mean": mean,
                "difference": selection["step"] - mean,
                "positive_interval": selection["step_low"] > 0.0,
            }
        )
    frame = pl.DataFrame(rows)
    frame.write_parquet(common.WORK_DIR / "selection_winter.parquet")
    rng = np.random.default_rng(fit.SEED)
    pooled = []
    for name, sites in fit.POOLED_SETS.items():
        picked = single.filter(
            (pl.col("winter") == common.SELECTION_WINTER) & pl.col("site").is_in(list(sites))
        )
        result = pool.pool(
            values=picked["step"].to_numpy(), standard_errors=picked["step_sd"].to_numpy(), rng=rng
        )
        pooled.append({"set": name, **result.__dict__})
    (common.WORK_DIR / "selection_pooled.json").write_text(json.dumps(pooled, indent=1))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=2, tbl_width_chars=200):
        print("== winter 2025-26 alone, per series (points) ==")
        print(frame)
        print("== winter 2025-26 alone, pooled sets ==")
        print(
            pl.DataFrame(pooled).select(
                "set", "k", "mean", "low_hk", "high_hk", "low_boot", "high_boot", "label"
            )
        )


if __name__ == "__main__":
    main()
