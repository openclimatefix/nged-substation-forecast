"""Reproduction gate: scale 1.0 with the filter must equal experiment 349's stored arm A exactly.

PYTHONPATH=FOLDER_349:FOLDER_352
uv run --no-sync python ml-experiments/00352-*/e00352_gate.py
"""

import e00349_run as run349  # ty: ignore[unresolved-import]
import e00349_run_common as run_common  # ty: ignore[unresolved-import]
import e00352_common as common
import polars as pl

COLUMNS = ("q01", "q05", "q25", "q50", "q75", "q95", "q99", "mean", "actual")


def main() -> None:
    """Print the largest absolute gap per Primary and period, and fail on any nonzero gap."""
    worst = 0.0
    for site in run_common.SCORED_SITES:
        for period in common.PERIODS:
            stored = pl.read_parquet(run349.out_path(arm="pin", period=period, site=site))
            new = pl.read_parquet(
                common.preds_path(scale=1.0, filter_name="kf", period=period, site=site)
            )
            if stored.height != new.height:
                msg = f"{site} {period}: {stored.height} rows against {new.height}"
                raise ValueError(msg)
            gap = max(float(abs(stored[c] - new[c]).to_numpy().max()) for c in COLUMNS)
            worst = max(worst, gap)
            print(f"{site} {period}: {new.height:,} rows, largest gap {gap}")
    if worst != 0.0:
        msg = f"reproduction gate failed: largest gap {worst}"
        raise ValueError(msg)
    print("reproduction gate passed")


if __name__ == "__main__":
    main()
