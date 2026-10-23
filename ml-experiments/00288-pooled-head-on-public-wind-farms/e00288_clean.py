"""Drop curtailed half-hours, choose the kept farms, and write each farm's clean output.

Reads Elexon B1610 metered output, bid-offer acceptances (BOALF), and physical notifications (PN)
for the candidate BMUs. A half-hour is dropped when an acceptance overlaps it, or when the PN
exceeds the metered output by more than 5% of the farm's capacity (a half-hour with no PN is kept
and counted). Negative metered output (station load) is set to zero. Capacity is the 99.5th
percentile of the farm's clean output before 2024-04-01, and farm selection uses output before
2024-04-01 only (plus an availability check that a farm still reports after 2025-06-17).

Output, in `work/clean/`: `farms.parquet` (the kept farms with capacity, REPD turbine rating and
hub height, the share of half-hours dropped, `train_start`, the cell, and the spatial fold),
`actual_centred.parquet` (clean output over capacity at each half-hour instant, the mean of the two
settlement periods either side of it), and `selection.parquet` (every candidate and why it was kept
or dropped).

    uv run --no-sync python ml-experiments/00288-*/e00288_clean.py
"""

import json
from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
CLEAN_DIR: Final = WORK_DIR / "clean"
MARKET: Final = REPO_DATA_DIR / "studies/downloads/market_wind00288"
BASE_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
AVAILABLE_AFTER: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
WINDOW_SECONDS: Final[int] = 1800
PN_EXCESS_SHARE: Final[float] = 0.05
MIN_CLEAN_HALF_HOURS: Final[int] = 2 * 365 * 48
MAX_ZERO_SHARE: Final[float] = 0.3
CAPACITY_FACTOR_RANGE: Final[tuple[float, float]] = (0.10, 0.50)
MIN_LATE_HALF_HOURS: Final[int] = 1000
MAX_FARMS: Final[int] = 40
FOLDS: Final[int] = 5
LIVE_SHARE: Final[float] = 0.05
LIVE_RUN: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6


def acceptance_windows() -> pl.DataFrame:
    """Return `elexon_bmu_id` and `period_start` for every half-hour an acceptance overlaps."""
    boalf = pl.read_parquet(MARKET / "elexon_boalf" / "elexon_boalf.parquet").filter(
        pl.col("amendment_flag") != "DEL"
    )
    return (
        boalf.select(
            pl.col("bmu_id").alias("elexon_bmu_id"),
            start=pl.col("time_from").dt.epoch("s"),
            stop=pl.col("time_to").dt.epoch("s"),
        )
        .filter(pl.col("stop") > pl.col("start"))
        .with_columns(
            window=pl.int_ranges(
                pl.col("start") // WINDOW_SECONDS, (pl.col("stop") - 1) // WINDOW_SECONDS + 1
            )
        )
        .explode("window")
        .select(
            "elexon_bmu_id",
            period_start=(pl.col("window") * WINDOW_SECONDS * 1_000_000)
            .cast(pl.Datetime("us"))
            .dt.replace_time_zone("UTC"),
        )
        .unique()
        .with_columns(accepted=pl.lit(value=True))
    )


def half_hours() -> pl.DataFrame:
    """Return one row per BMU half-hour: metered MW, PN MW, and the acceptance flag."""
    b1610 = pl.read_parquet(MARKET / "elexon_b1610" / "elexon_b1610.parquet").select(
        "elexon_bmu_id",
        period_start=pl.col("half_hour_end_time") - pl.duration(minutes=30),
        mw=(pl.col("output_mwh") * 2).clip(lower_bound=0.0),
    )
    pn = pl.read_parquet(MARKET / "elexon_pn" / "elexon_pn_half_hourly.parquet").select(
        pl.col("bmu_id").alias("elexon_bmu_id"),
        pl.col("time").alias("period_start"),
        pn_mw=pl.col("mean_level_mw"),
    )
    return (
        b1610.join(pn, on=["elexon_bmu_id", "period_start"], how="left", validate="1:1")
        .join(acceptance_windows(), on=["elexon_bmu_id", "period_start"], how="left")
        .with_columns(accepted=pl.col("accepted").fill_null(value=False))
    )


def mark_clean(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add `capacity_mw` and `clean`, and return the frame and the per-farm capacities."""
    before = pl.col("period_start") < BASE_END
    provisional = (
        frame.filter(before, ~pl.col("accepted"))
        .group_by("elexon_bmu_id")
        .agg(provisional_capacity_mw=pl.col("mw").quantile(0.995))
    )
    frame = frame.join(provisional, on="elexon_bmu_id").with_columns(
        pn_excess=(pl.col("pn_mw") - pl.col("mw"))
        > PN_EXCESS_SHARE * pl.col("provisional_capacity_mw")
    )
    frame = frame.with_columns(
        clean=~pl.col("accepted") & ~pl.col("pn_excess").fill_null(value=False)
    )
    capacity = (
        frame.filter(before, pl.col("clean"))
        .group_by("elexon_bmu_id")
        .agg(capacity_mw=pl.col("mw").quantile(0.995))
    )
    return frame, capacity


def first_live_start(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `train_start`: six weeks after the first run of 48 half-hours above 5% of capacity."""
    live = frame.sort("elexon_bmu_id", "period_start").with_columns(
        live=(pl.col("mw") > LIVE_SHARE * pl.col("capacity_mw")).cast(pl.Int32)
    )
    runs = live.with_columns(
        run_end=pl.col("live").rolling_sum(window_size=LIVE_RUN).over("elexon_bmu_id") == LIVE_RUN
    )
    return (
        runs.filter(pl.col("run_end"))
        .group_by("elexon_bmu_id")
        .agg(first=pl.col("period_start").min().dt.offset_by(f"-{LIVE_RUN * 30 - 30}m"))
        .select("elexon_bmu_id", train_start=pl.col("first").dt.offset_by(f"{SETTLING_WEEKS}w"))
    )


def repd_static(*, candidates: pl.DataFrame) -> pl.DataFrame:
    """Return the candidates with hub height from REPD, or predicted from the turbine rating.

    The prediction is a straight line of hub height on turbine rating over every operational
    onshore REPD row that has both. It uses no farm's output.
    """
    body = json.loads(
        (
            REPO_DATA_DIR / "studies/per_study/solar_bmu_census/inputs/raw/repd_register.json"
        ).read_text()
    )["body"]
    repd = pl.read_csv(body.encode(), infer_schema_length=0).filter(
        pl.col("Technology Type") == "Wind Onshore",
        pl.col("Development Status (short)") == "Operational",
    )
    pairs = repd.select(
        turbine=pl.col("Turbine Capacity (MW)").cast(pl.Float64, strict=False),
        hub=pl.col("Height of Turbines (m)").cast(pl.Float64, strict=False),
    ).drop_nulls()
    slope, intercept = np.polyfit(pairs["turbine"].to_numpy(), pairs["hub"].to_numpy(), 1)
    median_turbine = float(candidates["turbine_mw"].drop_nulls().median())
    return candidates.with_columns(
        turbine_mw=pl.col("turbine_mw").fill_null(median_turbine),
        hub_imputed=pl.col("hub_height_m").is_null(),
    ).with_columns(
        hub_height_m=pl.coalesce(
            "hub_height_m", (slope * pl.col("turbine_mw").fill_null(median_turbine) + intercept)
        )
    )


def main() -> None:
    """Write the clean tables and print the selection."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pl.read_parquet(WORK_DIR / "farm_candidates.parquet").filter(pl.col("in_ens"))
    frame, capacity = mark_clean(frame=half_hours())
    frame = frame.join(capacity, on="elexon_bmu_id")
    before = pl.col("period_start") < BASE_END
    stats = (
        frame.group_by("elexon_bmu_id")
        .agg(
            half_hours=pl.len(),
            share_accepted=pl.col("accepted").mean(),
            share_pn_excess=pl.col("pn_excess").fill_null(value=False).mean(),
            share_dropped=1 - pl.col("clean").mean(),
            share_no_pn=pl.col("pn_mw").is_null().mean(),
            clean_before=(pl.col("clean") & before).sum(),
            clean_late=(pl.col("clean") & (pl.col("period_start") >= AVAILABLE_AFTER)).sum(),
            zero_share=pl.col("mw").filter(pl.col("clean") & before).eq(0).mean(),
            mean_clean_mw=pl.col("mw").filter(pl.col("clean") & before).mean(),
        )
        .join(capacity, on="elexon_bmu_id")
        .with_columns(capacity_factor=pl.col("mean_clean_mw") / pl.col("capacity_mw"))
        .join(first_live_start(frame=frame), on="elexon_bmu_id", how="left")
    )
    selection = (
        candidates.select(
            "elexon_bmu_id",
            "bmu_name",
            "match_ratio",
            "country",
            "region",
            "lat",
            "lon",
            "repd_name",
            "turbine_mw",
            "hub_height_m",
            "capacity_mw",
        )
        .rename({"capacity_mw": "repd_capacity_mw"})
        .join(stats, on="elexon_bmu_id", how="left")
    )
    selection = (
        repd_static(candidates=selection)
        .with_columns(
            enough=pl.col("clean_before") >= MIN_CLEAN_HALF_HOURS,
            zero_ok=pl.col("zero_share") < MAX_ZERO_SHARE,
            cf_ok=pl.col("capacity_factor").is_between(*CAPACITY_FACTOR_RANGE),
            late_ok=pl.col("clean_late") >= MIN_LATE_HALF_HOURS,
        )
        .with_columns(
            qualifies=pl.col("enough") & pl.col("zero_ok") & pl.col("cf_ok") & pl.col("late_ok")
        )
    )
    kept = (
        selection.filter(pl.col("qualifies")).sort("clean_before", descending=True).head(MAX_FARMS)
    )
    if kept.height < 15:
        raise SystemExit(f"stop rule: only {kept.height} farms qualify (the plan needs 15)")
    cells = pl.read_parquet(WORK_DIR / "farm_cells.parquet").select("elexon_bmu_id", "cell_id")
    kept = kept.join(cells, on="elexon_bmu_id")
    positions = np.column_stack([kept["lat"].to_numpy(), kept["lon"].to_numpy()])
    fold = KMeans(n_clusters=FOLDS, n_init=10, random_state=0).fit_predict(positions)
    kept = kept.with_columns(fold=pl.Series(fold, dtype=pl.Int32))
    selection.write_parquet(CLEAN_DIR / "selection.parquet")
    kept.write_parquet(CLEAN_DIR / "farms.parquet")
    names = kept["elexon_bmu_id"].to_list()
    clean = frame.filter(pl.col("elexon_bmu_id").is_in(names)).select(
        "elexon_bmu_id",
        "period_start",
        "clean",
        norm=pl.when(pl.col("clean")).then(pl.col("mw") / pl.col("capacity_mw")),
    )
    ends = clean.select(
        "elexon_bmu_id",
        valid_time=pl.col("period_start") + pl.duration(minutes=30),
        end_norm=pl.col("norm"),
    )
    starts = clean.select(
        "elexon_bmu_id", valid_time=pl.col("period_start"), start_norm=pl.col("norm")
    )
    centred = (
        ends.join(starts, on=["elexon_bmu_id", "valid_time"], how="inner")
        .select(
            "elexon_bmu_id",
            "valid_time",
            actual_norm=(pl.col("end_norm") + pl.col("start_norm")) / 2,
        )
        .drop_nulls("actual_norm")
        .sort("elexon_bmu_id", "valid_time")
    )
    centred.write_parquet(CLEAN_DIR / "actual_centred.parquet")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(12)
    qualify = selection.filter(pl.col("qualifies")).height
    print(f"{selection.height} candidates, {qualify} qualify, {kept.height} kept")
    print(selection.select("enough", "zero_ok", "cf_ok", "late_ok").sum())
    print(
        kept.select(
            "elexon_bmu_id",
            "bmu_name",
            "country",
            "capacity_mw",
            "turbine_mw",
            "hub_height_m",
            "hub_imputed",
            "share_accepted",
            "share_pn_excess",
            "share_no_pn",
            "fold",
        ).sort("fold", "elexon_bmu_id")
    )


if __name__ == "__main__":
    main()
