"""Extract the ENS members' weather at the experiment 00288 farms' cells, one parquet per run.

For each 00 UTC ensemble run in the hindcast span (daily, 2024-04-01 to 2025-06-16) and the scored
span (the 53 Tuesday runs from 2025-07-01 to 2026-06-30), reads the ENS Delta table's rows at the
H3 resolution-5 cells that contain the candidate farms and keeps the native 3-hourly and 6-hourly
valid times after the forecast's publication delay (9 hours) up to 351 hours after it, the lead
range experiment 259's rows cover. The columns match experiment 259's member rows: wind speed at
10 m and 100 m, the sine and cosine of the direction the wind blows from at both heights, and the
2 m temperature in degrees Celsius. `power_fcst_init_time` is the run time plus the publication
delay.

Output: `work/ens/<set>/init=<date>.parquet` (`set` is `hindcast` or `scored`) and
`work/farm_h3.parquet`, which maps each BMU to its cell.

    uv run --no-sync python ml-experiments/00288-*/e00288_ens.py
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import h3
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00288-pooled-head-on-public-wind-farms" / "work"
ENS_DIR: Final = WORK_DIR / "ens"
PUBLICATION_DELAY: Final = timedelta(hours=9)
MAX_LEAD: Final = timedelta(hours=351)
HINDCAST_FIRST: Final = datetime(2024, 4, 1, tzinfo=UTC)
HINDCAST_LAST: Final = datetime(2025, 6, 16, tzinfo=UTC)
SCORED_FIRST: Final = datetime(2025, 7, 1, tzinfo=UTC)
SCORED_LAST: Final = datetime(2026, 6, 30, tzinfo=UTC)
TRAIN_END: Final = datetime(2025, 6, 17, tzinfo=UTC)
VALID_END: Final = datetime(2026, 7, 1, tzinfo=UTC)
ENS_RESOLUTION: Final[int] = 5


def weather_columns() -> list[pl.Expr]:
    """Return the expressions turning ENS u and v components into the experiment's columns."""
    expressions = []
    for height in ("10", "100"):
        u, v = pl.col(f"wind_u_{height}m"), pl.col(f"wind_v_{height}m")
        bearing = ((270.0 - pl.arctan2(v, u).degrees()) % 360.0).radians()
        expressions += [
            (u**2 + v**2).sqrt().alias(f"wind_speed_{height}m"),
            bearing.sin().alias(f"wind_dir_sin_{height}m"),
            bearing.cos().alias(f"wind_dir_cos_{height}m"),
        ]
    return expressions


def runs() -> list[tuple[str, datetime]]:
    """Return (set name, run time) for every run to extract."""
    out = []
    day = HINDCAST_FIRST
    while day <= HINDCAST_LAST:
        out.append(("hindcast", day))
        day += timedelta(days=1)
    day = SCORED_FIRST
    while day <= SCORED_LAST:
        out.append(("scored", day))
        day += timedelta(days=7)
    return out


def main() -> None:
    """Write every missing run's member rows at the farms' cells."""
    farms = pl.read_parquet(WORK_DIR / "farm_candidates.parquet").filter(pl.col("in_ens"))
    farm_h3 = farms.select(
        "elexon_bmu_id",
        h3_index=pl.struct("lat", "lon").map_elements(
            lambda row: h3.str_to_int(h3.latlng_to_cell(row["lat"], row["lon"], ENS_RESOLUTION)),
            return_dtype=pl.Int64,
        ),
    )
    farm_h3.write_parquet(WORK_DIR / "farm_h3.parquet")
    cells = farm_h3["h3_index"].unique().to_list()
    table = str(REPO_DATA_DIR / "NWP")
    for set_name, init in runs():
        directory = ENS_DIR / set_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"init={init:%Y-%m-%d}.parquet"
        if path.exists():
            continue
        power_init = init + PUBLICATION_DELAY
        last = min(init + MAX_LEAD, (TRAIN_END if set_name == "hindcast" else VALID_END))
        frame = (
            pl.scan_delta(table)
            .filter(
                pl.col("init_time") == init,
                pl.col("h3_index").is_in(cells),
                pl.col("valid_time") > power_init,
                pl.col("valid_time") <= last,
            )
            .select(
                "h3_index",
                "ensemble_member",
                "valid_time",
                "wind_u_10m",
                "wind_v_10m",
                "wind_u_100m",
                "wind_v_100m",
                pl.col("temperature_2m").alias("temperature_c"),
            )
            .collect()
        )
        if frame.height:
            frame = (
                frame.with_columns(weather_columns())
                .with_columns(power_fcst_init_time=pl.lit(power_init))
                .select(
                    "h3_index",
                    "power_fcst_init_time",
                    "valid_time",
                    "ensemble_member",
                    "wind_speed_10m",
                    "wind_speed_100m",
                    "wind_dir_sin_10m",
                    "wind_dir_cos_10m",
                    "wind_dir_sin_100m",
                    "wind_dir_cos_100m",
                    "temperature_c",
                )
                .with_columns(
                    pl.exclude(
                        "h3_index", "power_fcst_init_time", "valid_time", "ensemble_member"
                    ).cast(pl.Float32)
                )
                .sort("h3_index", "valid_time", "ensemble_member")
            )
            partial = path.with_suffix(".partial")
            frame.write_parquet(partial)
            partial.rename(path)
        print(f"{set_name} {init:%Y-%m-%d}: {frame.height} rows", flush=True)


if __name__ == "__main__":
    main()
