"""Leave-one-farm-out change of arm c against P2 in score and 0.99 pinball loss (experiment 293).

Reads the per-row losses written by `e00293_score.py` and prints, for each dropped farm, the percent
change of arm c (tilt 20 with a soft cap) against P2 in the farm-equal mean (own history).

    uv run --no-sync python ml-experiments/00293-*/e00293_checks.py
"""

import e00293_common as common
import polars as pl


def main() -> None:
    """Print the leave-one-farm-out changes."""
    rows = pl.read_parquet(common.DATA_DIR / "row_losses.parquet").filter(
        (pl.col("setting") == "own_history") & pl.col("arm").is_in(["p2", "c_tilt20_soft_clip"])
    )
    per_farm = rows.group_by("arm", "time_series_id").agg(
        pl.col("score").mean(), pl.col("pin99").mean()
    )
    for metric in ("score", "pin99"):
        changes = {}
        for dropped in common.FARMS:
            kept = per_farm.filter(pl.col("time_series_id") != dropped)
            mean = kept.group_by("arm").agg(pl.col(metric).mean()).sort("arm")
            c, p2 = (
                mean["arm"].to_list().index("c_tilt20_soft_clip"),
                mean["arm"].to_list().index("p2"),
            )
            values = mean[metric].to_list()
            changes[dropped] = round(100.0 * (values[c] / values[p2] - 1.0), 2)
        print(metric, changes)


if __name__ == "__main__":
    main()
