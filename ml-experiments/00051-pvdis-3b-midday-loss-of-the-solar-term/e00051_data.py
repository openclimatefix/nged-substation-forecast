"""Build experiment 51's summer-midday row table from experiment 00046's stored outputs.

Reads experiment 00046's `work/arms/*.parquet` (the stored quantiles of each arm), its CAMS
anomaly and unit output, its ensemble anomalies, and experiment 00040's capacities, and writes
`work/rows.parquet` (one row per series, forecast start, and valid time, with the context) and
`work/arms_long.parquet` (one row per arm, seed, and case, with the nine quantiles). It trains no
model. Rows are June to August (local calendar month), valid 09:00 to 15:00 UTC inclusive, 24 to 96
hours after the forecast origin, at the five verdict series.

The helpers are copied from experiment 00046's `e00046_common.py`; an experiment may not import
another experiment.

    uv run python ml-experiments/00051-*/e00051_data.py
"""

from datetime import time
from itertools import pairwise
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_46_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00046-embedded-solar-share-of-primary-forecast-error" / "work"
)
WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00051-pvdis-3b-midday-loss-of-the-solar-term" / "work"
)
INPUTS_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand" / "inputs"
CAPACITY_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00040-pv-capacity-error-bars" / "work" / "real_scores.parquet"
)
FRAME_00039_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
CAPACITY_SEASON_YEAR: Final[int] = 2023
VERDICT_SITES: Final[tuple[str, ...]] = ("D9", "D10", "D11", "D14", "D16")
SUMMER_MONTHS: Final[tuple[int, ...]] = (6, 7, 8)
MIDDAY_START: Final[time] = time(9, 0)
MIDDAY_END: Final[time] = time(15, 0)
LEAD_MIN_HOURS: Final[float] = 24.0
LEAD_MAX_HOURS: Final[float] = 96.0
ARMS: Final[tuple[str, ...]] = (
    "d0",
    "d1l",
    "d1l_ens_shift",
    "d1l_cams_shift",
    "d1l_ens_mean_shift",
    "d1l_ens_shift_fitted",
)
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
KEYS: Final[tuple[str, ...]] = ("site", "power_fcst_init_time", "valid_time")
COAST_KILOMETRES: Final[float] = 10.0
"""A series is coastal when its distance to the open coast is under this."""


def coast_distance_km(*, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Return the distance in km from each point to a coarse polyline of the Lincolnshire coast.

    The polyline runs from the north shore of the Wash to the Humber mouth. The distance is
    computed here and never written out, because the coordinates are not published.

    Args:
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        Distance in km, shape of the inputs.
    """
    coast = np.array(
        [
            (52.96, 0.07),
            (53.07, 0.30),
            (53.14, 0.35),
            (53.25, 0.35),
            (53.34, 0.25),
            (53.42, 0.19),
            (53.55, 0.08),
        ]
    )
    kilometres_per_degree = 111.2
    scale = np.array([kilometres_per_degree, kilometres_per_degree * np.cos(np.radians(53.1))])
    vertices = coast * scale
    points = np.column_stack([latitude, longitude]) * scale
    distances = np.full(points.shape[0], np.inf)
    for start, end in pairwise(vertices):
        segment = end - start
        fraction = np.clip(((points - start) @ segment) / (segment @ segment), 0.0, 1.0)
        nearest = start + fraction[:, None] * segment
        distances = np.minimum(distances, np.sqrt(((points - nearest) ** 2).sum(axis=1)))
    return distances


def main() -> None:
    """Write the row table and the long arm table."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    series = pl.read_parquet(INPUTS_DIR / "series.parquet").filter(
        pl.col("site").is_in(list(VERDICT_SITES))
    )
    capacity = (
        pl.read_parquet(CAPACITY_PATH)
        .filter(
            (pl.col("year") == CAPACITY_SEASON_YEAR) & pl.col("site").is_in(list(VERDICT_SITES))
        )
        .select("site", capacity_mw="C")
    )
    coast = coast_distance_km(
        latitude=series["latitude"].to_numpy().astype(np.float64),
        longitude=series["longitude"].to_numpy().astype(np.float64),
    )
    series = series.with_columns(coastal=pl.Series(coast < COAST_KILOMETRES))
    sites = series.select("time_series_id", "site", "effective_capacity_mw", "coastal").join(
        capacity, on="site", validate="1:1"
    )
    print("series that qualify as coastal:", int(sites["coastal"].sum()), "of", sites.height)

    def midday(frame: pl.LazyFrame) -> pl.LazyFrame:
        lead = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
        local_month = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month()
        return (
            frame.filter(pl.col("time_series_id").is_in(sites["time_series_id"].implode()))
            .with_columns(lead_hours=lead, local_month=local_month)
            .filter(
                pl.col("local_month").is_in(list(SUMMER_MONTHS))
                & pl.col("valid_time").dt.time().is_between(MIDDAY_START, MIDDAY_END)
                & (pl.col("lead_hours") >= LEAD_MIN_HOURS)
                & (pl.col("lead_hours") < LEAD_MAX_HOURS)
            )
        )

    long = []
    for arm in ARMS:
        path = EXPERIMENT_46_DIR / "arms" / f"{arm}.parquet"
        frame = (
            midday(pl.scan_parquet(path))
            .join(sites.lazy().select("time_series_id", "site"), on="time_series_id")
            .collect()
            .with_columns(arm=pl.lit(arm))
        )
        long.append(frame)
    arms_long = pl.concat(long)
    base = (
        arms_long.filter(pl.col("arm") == "d1l")
        .select(*KEYS, "time_series_id", "lead_hours", "local_month", "actual")
        .unique(subset=list(KEYS))
    )

    cams = (
        pl.read_parquet(FRAME_00039_PATH)
        .filter((pl.col("kind") == "substation") & pl.col("site").is_in(list(VERDICT_SITES)))
        .select(
            "site",
            pl.col("time").alias("valid_time"),
            clear_sky_index=pl.when(pl.col("ghi_u_clear") > 0)
            .then(pl.col("ghi_u") / pl.col("ghi_u_clear"))
            .otherwise(None),
        )
    )
    anomaly = pl.read_parquet(EXPERIMENT_46_DIR / "solar" / "cams_anomaly.parquet")
    u_cams = pl.read_parquet(EXPERIMENT_46_DIR / "solar" / "cams_u.parquet").rename(
        {"time": "valid_time", "u": "u_cams"}
    )
    ens = (
        pl.concat(
            [
                pl.read_parquet(path).filter(pl.col("site").is_in(list(VERDICT_SITES)))
                for path in sorted((EXPERIMENT_46_DIR / "solar" / "ens_val").glob("*.parquet"))
            ]
        )
        .group_by(*KEYS)
        .agg(ens_mean_calibrated=pl.col("anomaly_mean_calibrated").first())
    )
    era5 = pl.read_parquet(
        INPUTS_DIR / "era5_features.parquet",
        columns=["time_series_id", "valid_time", "temperature_2m"],
    ).filter(pl.col("time_series_id").is_in(sites["time_series_id"].implode()))
    candidates = base.height
    rows = (
        base.join(sites, on="time_series_id")
        .join(cams, on=["site", "valid_time"], how="left", validate="m:1")
        .join(anomaly, on=["site", "valid_time"], how="left", validate="m:1")
        .join(u_cams, on=["site", "valid_time"], how="left", validate="m:1")
        .join(ens, on=list(KEYS), how="left", validate="1:1")
        .join(era5, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            scale=pl.col("capacity_mw") / pl.col("effective_capacity_mw"),
        )
        .with_columns(
            oracle_shift=pl.col("scale") * pl.col("anomaly_cams"),
            ens_mean_shift=pl.col("scale") * pl.col("ens_mean_calibrated"),
        )
        .filter(
            pl.col("clear_sky_index").is_not_null()
            & pl.col("u_cams").is_not_null()
            & pl.col("anomaly_cams").is_not_null()
            & pl.col("actual").is_finite()
        )
    )
    print("candidate cases", candidates, "kept", rows.height, "dropped", candidates - rows.height)
    arms_long = arms_long.join(rows.select(*KEYS), on=list(KEYS), how="semi").select(
        "arm", "seed", *KEYS, *LEVEL_COLUMNS
    )
    # Self-checks: the stored shift arms equal the d1l quantiles minus the shift.
    d1l = arms_long.filter(pl.col("arm") == "d1l").select(*KEYS, "seed", d1l_q50="q50")
    for arm, column in (
        ("d1l_cams_shift", "oracle_shift"),
        ("d1l_ens_mean_shift", "ens_mean_shift"),
    ):
        check = (
            arms_long.filter(pl.col("arm") == arm)
            .select(*KEYS, "seed", "q50")
            .join(d1l, on=[*KEYS, "seed"])
            .join(rows.select(*KEYS, column), on=list(KEYS))
            .with_columns(gap=(pl.col("d1l_q50") - pl.col(column) - pl.col("q50")).abs())
        )
        print(arm, "max |d1l q50 - shift - arm q50|:", check["gap"].max(), "rows", check.height)
    print("rows", rows.height, "arms_long", arms_long.height)
    rows.write_parquet(WORK_DIR / "rows.parquet")
    arms_long.write_parquet(WORK_DIR / "arms_long.parquet")


if __name__ == "__main__":
    main()
