"""Part 1: break experiment 349's sunniest-third contrast (arm B minus arm A) down.

Reads experiment 349's stored scored-year forecasts and writes the mean quantile-score difference
by Primary, lead band, and calendar month on the sunniest third of each Primary's daylight
half-hours, with the count of rows behind each cell.

    PYTHONPATH=FOLDER_349:FOLDER_352 \
    uv run --no-sync python ml-experiments/00352-*/e00352_part1.py
"""

import e00349_run_common as run_common  # ty: ignore[unresolved-import]
import e00349_score as score  # ty: ignore[unresolved-import]
import e00352_common as common
import numpy as np
import polars as pl

LEAD_BANDS = {"0 to 24 h": (0.0, 24.0), "24 to 72 h": (24.0, 72.0), "72 to 168 h": (72.0, 168.0)}


def main() -> None:
    """Print and store the breakdown."""
    sites = run_common.SCORED_SITES
    frames = {arm: score.load_arm(arm=arm, period="scored", sites=sites) for arm in ("pin", "b")}
    edges = score.tertile_edges(frame=frames["pin"])
    frames = {
        arm: score.add_slices(frame=score.with_losses(frame=f), edges=edges)
        for arm, f in frames.items()
    }
    keys = ["label", "run", "valid_time"]
    joined = (
        frames["pin"]
        .select(*keys, "third", "lead_hours", "calendar_month", pl.col("qs").alias("qs_a"))
        .join(
            frames["b"].select(*keys, pl.col("qs").alias("qs_b")),
            on=keys,
            how="inner",
            validate="1:1",
        )
        .filter(pl.col("third") == "sunny")
        .with_columns(difference=pl.col("qs_b") - pl.col("qs_a"))
    )
    band = (
        pl.when(pl.col("lead_hours") <= 24.0)
        .then(pl.lit("0 to 24 h"))
        .when(pl.col("lead_hours") <= 72.0)
        .then(pl.lit("24 to 72 h"))
        .otherwise(pl.lit("72 to 168 h"))
    )
    joined = joined.with_columns(lead_band=band)
    cells = (
        joined.group_by("label", "lead_band", "calendar_month")
        .agg(difference=pl.col("difference").mean(), n=pl.len())
        .sort("label", "lead_band", "calendar_month")
    )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(common.WORK_DIR / "part1_cells.parquet")
    with pl.Config(float_precision=3, tbl_rows=100):
        print("== by Primary (sunny rows, B minus A) ==")
        print(joined.group_by("label").agg(d=pl.col("difference").mean(), n=pl.len()).sort("label"))
        print("== by lead band, pooled ==")
        print(
            joined.group_by("lead_band")
            .agg(d=pl.col("difference").mean(), n=pl.len())
            .sort("lead_band")
        )
        print("== by Primary and lead band ==")
        print(
            joined.group_by("label", "lead_band")
            .agg(d=pl.col("difference").mean(), n=pl.len())
            .sort("label", "lead_band")
        )
        print("== by month, pooled ==")
        print(
            joined.group_by("calendar_month")
            .agg(d=pl.col("difference").mean(), n=pl.len())
            .sort("calendar_month")
        )
        print("== by Primary and month ==")
        print(
            joined.group_by("label", "calendar_month")
            .agg(d=pl.col("difference").mean(), n=pl.len())
            .sort("label", "calendar_month")
            .pivot(on="calendar_month", index="label", values="d")
        )
    print("pooled", float(np.mean(joined["difference"].to_numpy())), joined.height)


if __name__ == "__main__":
    main()
