# ruff: noqa: E501
"""Forecast each Primary's demand with Chronos-2 and no training: Z0 (history only), Z1 (covariates).

For each scored-year origin (09:00 UTC) the context is the 2688 half-hours that end 3.5 hours before
the origin, taken from the readings stored in experiment 317's chain. The model forecasts 487
half-hours, and the first 7 (before the origin) are dropped. Z1 adds the ensemble-mean 24-hour
temperature (forecast over the horizon, from the nearest-lead run in the past), the sun's
elevation, the local half-hour of day, the day of week, and a bank-holiday flag, all known in
advance. Output: `work/forecasts/<arm>_<site>.parquet` with the 11 quantiles in points.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync --with chronos-forecasting python ml-experiments/00333-*/e00333_forecast.py [--limit-origins N]
"""

import argparse
import time
from datetime import datetime, timedelta

import e00333_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from chronos import Chronos2Pipeline  # ty: ignore[unresolved-import]
from studies.dp_calendar import bank_holiday_names
from studies.dp_features import default_input_paths
from studies.pv_physics import solar_position

COLUMNS = ["power_fcst_init_time", "valid_time", "lead_hours", "actual", "ens_mean_c", "period"]


def site_coordinates() -> dict[str, tuple[float, float]]:
    """Return each Primary's latitude and longitude (kept inside this process)."""
    table = pl.read_parquet(
        default_input_paths().series_table, columns=["site", "latitude", "longitude"]
    )
    return {r["site"]: (r["latitude"], r["longitude"]) for r in table.iter_rows(named=True)}


def covariate_frame(*, grid: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the deterministic covariates of the half-hours that end at each grid time."""
    zenith, _ = solar_position(
        stamps=grid.dt.offset_by("-15m"), latitude=latitude, longitude=longitude
    )
    local = grid.dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    half_hour = (
        local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30
    ).to_numpy()
    weekday = local.dt.weekday().to_numpy() - 1
    banks = bank_holiday_names()
    holiday = np.array([d in banks for d in local.dt.date().to_list()], dtype=np.float64)
    return pl.DataFrame(
        {
            "elevation_deg": 90.0 - zenith,
            "half_hour_sin": np.sin(2 * np.pi * half_hour / 48),
            "half_hour_cos": np.cos(2 * np.pi * half_hour / 48),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "bank_holiday": holiday,
        }
    )


def build_site(*, site: str, limit: int | None) -> tuple[list[dict], list[dict], list[dict]]:
    """Return the Z0 inputs, the Z1 inputs, and the keys of each forecast, for one Primary."""
    chain = pl.read_parquet(common.CHAIN_DIR / f"{site}.parquet", columns=COLUMNS)
    latitude, longitude = site_coordinates()[site]
    first, last = chain["valid_time"].min(), chain["valid_time"].max()
    assert isinstance(first, datetime)
    assert isinstance(last, datetime)
    grid = pl.datetime_range(first, last, interval="30m", time_zone="UTC", eager=True)
    base = pl.DataFrame({"valid_time": grid}).join(
        chain.unique("valid_time").select("valid_time", "actual"),
        on="valid_time",
        how="left",
        validate="1:1",
    )
    nearest = chain.sort("lead_hours").unique("valid_time", keep="first", maintain_order=True)
    base = base.join(
        nearest.select("valid_time", pl.col("ens_mean_c").alias("past_temp")),
        on="valid_time",
        how="left",
        validate="1:1",
    )
    fixed = covariate_frame(grid=grid, latitude=latitude, longitude=longitude)
    actual = base["actual"].to_numpy().astype(np.float32)
    past_temp = base["past_temp"].to_numpy().astype(np.float32)
    index_of = {t: i for i, t in enumerate(grid.to_list())}
    origins = (
        chain.filter(pl.col("period") == "scored")["power_fcst_init_time"].unique().sort().to_list()
    )
    if limit:
        origins = origins[:: max(len(origins) // limit, 1)][:limit]
    z0, z1, keys = [], [], []
    for origin in origins:
        end = index_of[origin + timedelta(minutes=30)] - common.TELEMETRY_DELAY_HALF_HOURS
        # `end` is the exclusive end of the context: the last reading ended at origin - 3.5 hours.
        start = max(end - common.CONTEXT_HALF_HOURS, 0)
        stop = end + common.PREDICT_HALF_HOURS
        if stop > len(grid):
            continue
        run = chain.filter(pl.col("power_fcst_init_time") == origin).select(
            "valid_time", "ens_mean_c"
        )
        future = pl.DataFrame({"valid_time": grid[end:stop]}).join(
            run, on="valid_time", how="left", validate="1:1"
        )
        future_temp = future["ens_mean_c"].to_numpy().astype(np.float32)
        # Half-hours up to the origin come from earlier runs. A run cut short ends before 240 hours,
        # and its last forecast temperature is held for the rest, because a later run is not known yet.
        before = future["valid_time"].to_numpy() <= np.datetime64(origin.replace(tzinfo=None))
        future_temp[before] = past_temp[end:stop][before]
        # A gap in the earlier runs at the start of the horizon is filled from the context's last value.
        joined = np.concatenate([past_temp[start:end], future_temp])
        filled = pl.Series(joined).fill_nan(None).forward_fill().to_numpy().astype(np.float32)
        future_temp = filled[end - start :]
        if not np.isfinite(future_temp).all():
            msg = f"{site} {origin}: a future covariate is not finite"
            raise ValueError(msg)
        target = actual[start:end].copy()
        z0.append({"target": target})
        past_cov = {"temp_c": past_temp[start:end]} | {
            c: fixed[c].to_numpy()[start:end].astype(np.float32) for c in fixed.columns
        }
        future_cov = {"temp_c": future_temp} | {
            c: fixed[c].to_numpy()[end:stop].astype(np.float32) for c in fixed.columns
        }
        z1.append({"target": target, "past_covariates": past_cov, "future_covariates": future_cov})
        keys.append(
            {"origin": origin, "valid_times": grid[end + common.TELEMETRY_DELAY_HALF_HOURS : stop]}
        )
    return z0, z1, keys


def forecast(*, pipeline: Chronos2Pipeline, inputs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return quantiles (n_inputs, 480, 11), sorted along the quantile axis, and means (n_inputs, 480)."""
    quantiles, means = pipeline.predict_quantiles(
        inputs,
        prediction_length=common.PREDICT_HALF_HOURS,
        quantile_levels=list(common.LEVELS),
        batch_size=24,
    )
    out = np.stack([q[0].numpy() for q in quantiles])  # (n, steps, levels)
    mean = np.stack([m[0].numpy() for m in means])
    return np.sort(out[:, common.TELEMETRY_DELAY_HALF_HOURS :, :], axis=2), mean[
        :, common.TELEMETRY_DELAY_HALF_HOURS :
    ]


def main() -> None:
    """Forecast every Primary and write one parquet per arm and Primary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-origins", type=int, default=None)
    parser.add_argument("--sites", nargs="*", default=list(common.SCORED_SITES))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    torch.set_num_threads(4)
    pipeline = Chronos2Pipeline.from_pretrained(common.MODEL_ID, device_map="cuda")
    common.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    for site in args.sites:
        z0, z1, keys = build_site(site=site, limit=args.limit_origins)
        for arm, inputs in (("z0", z0), ("z1", z1)):
            started = time.time()
            q, mean = forecast(pipeline=pipeline, inputs=inputs)
            frame = pl.concat(
                [
                    pl.DataFrame(
                        {
                            "site": site,
                            "power_fcst_init_time": k["origin"],
                            "valid_time": k["valid_times"],
                        }
                    )
                    .with_columns(
                        pl.DataFrame(
                            {f"{arm}_{n}": q[i, :, j] for j, n in enumerate(common.QUANTILES)}
                        )
                    )
                    .with_columns(pl.Series(f"{arm}_mean", mean[i]))
                    for i, k in enumerate(keys)
                ]
            )
            path = common.FORECAST_DIR / f"{arm}{args.tag}_{site}.parquet"
            frame.write_parquet(path.with_name(path.name + ".tmp"))
            path.with_name(path.name + ".tmp").rename(path)
            print(
                f"{site} {arm}: {len(inputs)} origins, {frame.height} rows, {time.time() - started:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
