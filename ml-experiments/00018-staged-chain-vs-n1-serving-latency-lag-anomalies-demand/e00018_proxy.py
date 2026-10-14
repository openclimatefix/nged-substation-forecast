"""Build the analysis proxy of experiment 00018: the ENS control member's earliest leads.

For each daily 00:00 UTC ENS run from 2024-04-01 to 2026-06-30, this reads the control member at
the 20 series' H3 cells, resamples it to the half-hourly grid with production's
`_upsample_nwp_to_half_hourly`, and keeps the half-hours from the run's start to 30 hours after it.
`weather_utils.analysis_proxy.select_analysis_proxy` then picks, for each series and valid time and
for each column, the value from the freshest run that holds one. A run is published 9 hours after
its start, so for any lag time before a forecast origin the proxy value comes from a run that had
been published by the origin.

Output: `work/proxy.parquet` with `time_series_id`, `valid_time`, and the stage 1 weather columns.

    uv run python \
        ml-experiments/00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/e00018_proxy.py
"""

import time
from datetime import UTC, datetime, timedelta
from typing import Final

import e00018_common as common
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from weather_utils.analysis_proxy import select_analysis_proxy

FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
KEEP_HOURS: Final[int] = 30
"""A run's rows kept: the first 30 hours, so a null first window falls back to the previous run."""
ENS_COLUMNS: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
}
WEATHER: Final[tuple[str, ...]] = (*ENS_COLUMNS, "wind_speed_10m")


def run_rows(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one run's control-member half-hours for every series, with the run's start time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    cells = series["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(common.ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select(
            "h3_index",
            "ensemble_member",
            "valid_time",
            *ENS_COLUMNS.values(),
            "wind_u_10m",
            "wind_v_10m",
        )
        .filter(
            pl.col("h3_index").is_in(cells)
            & (pl.col("ensemble_member") == 0)
            & (pl.col("valid_time") <= run + timedelta(hours=KEEP_HOURS + 6))
        )
        .join(
            series.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = series.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect()
    return (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt()
        )
        .rename({source: name for name, source in ENS_COLUMNS.items()})
        .filter(pl.col("valid_time") <= run + timedelta(hours=KEEP_HOURS))
        .select("time_series_id", "nwp_init_time", "valid_time", "ensemble_member", *WEATHER)
    )


def main() -> None:
    """Build and write the proxy, skipping when it exists."""
    if common.PROXY_PATH.exists():
        return
    series = common.series_table()
    pieces = []
    run = FIRST_RUN
    started = time.monotonic()
    while run <= LAST_RUN:
        pieces.append(run_rows(run=run, series=series))
        if len(pieces) % 50 == 0:
            print(f"{run:%Y-%m-%d}: {time.monotonic() - started:.0f} s", flush=True)
        run += timedelta(days=1)
    proxy = (
        select_analysis_proxy(
            pl.concat(pieces).lazy(), group_key="time_series_id", init_time_col="nwp_init_time"
        )
        .select("time_series_id", "valid_time", *WEATHER)
        .sort("time_series_id", "valid_time")
        .collect()
    )
    common.PROXY_PATH.parent.mkdir(parents=True, exist_ok=True)
    proxy.write_parquet(common.PROXY_PATH.with_suffix(".tmp"))
    common.PROXY_PATH.with_suffix(".tmp").rename(common.PROXY_PATH)
    print(proxy.height, "rows", flush=True)


if __name__ == "__main__":
    main()
