# ruff: noqa: E501, PLR0915
r"""Build the dense per-run input tensors of experiment 334 for every Primary.

One run is one daily forecast origin (09:00 UTC). Each run gets 480 half-hour lead steps (0.5 to 240
hours). Rows up to 168 hours are exactly experiment 311's row set (the rows the earlier experiments
scored); steps beyond 168 hours are added for every run whose weather and reading exist. Weather is
the ensemble-mean temperature and shortwave irradiance of the run (experiment 111's store) in the
ensemble era, and the past-weather driver at the valid time before April 2024, exactly as in
experiment 278's input frame. Output: `work/prep/<site>_{x,y,keep,hist}.npy` and `<site>_runs.parquet`.

    export PYTHONPATH=$(ls -d ml-experiments/00{164,278,311,317,319,320,322,334}-* | tr '\\n' ':')
    uv run --no-sync python ml-experiments/00334-*/e00334_prep.py
"""

import e00164_common as c164  # ty: ignore[unresolved-import]
import e00278_h_data as h_data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names
from studies.dp_ens_weather import SunGeometry

import e00334_common as common  # isort: skip
import e00334_data as data  # isort: skip

HALF_HOUR = np.timedelta64(30, "m")
TIMELINE_START = np.datetime64("2019-11-01T00:00:00", "us")
TIMELINE_END = np.datetime64("2026-07-20T00:00:00", "us")
N_TIMELINE = int((TIMELINE_END - TIMELINE_START) // HALF_HOUR)
LONDON = "Europe/London"
DAY_TYPES = np.array(h_data.DAY_TYPES)
BASE_TEMPERATURE_C = h_data.BASE_TEMPERATURE_C


def timeline_index(*, times: np.ndarray) -> np.ndarray:
    """Return the index of each naive UTC time on the 30-minute timeline."""
    return ((times - TIMELINE_START) // HALF_HOUR).astype(np.int64)


def harmonics(*, x: np.ndarray, period: float, n: int) -> list[np.ndarray]:
    """Return sin and cos harmonics 1 to n of x with the given period."""
    return [f(2 * np.pi * k * x / period) for k in range(1, n + 1) for f in (np.sin, np.cos)]


def build_site(*, site: str, table, store_dir) -> None:  # noqa: ANN001
    """Build and save the arrays of one Primary."""
    rows = pl.read_parquet(common.E311_PREP / "rows.parquet").filter(pl.col("site") == site)
    frame = pl.read_parquet(
        common.E201_FEATURES / f"{site}.parquet",
        columns=[
            "run",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "actual",
            "switching_mask",
            "era",
            "drv_temp_c",
            "drv_sw_w_m2",
            "drv_temp_prev_c",
        ],
    )
    naive = lambda s: s.dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")  # noqa: E731
    # Timelines: reading, hand-filter flag, past-weather driver.
    valid_all = naive(frame["valid_time"])
    index_all = timeline_index(times=valid_all)
    reading = np.full(N_TIMELINE, np.nan, dtype=np.float32)
    reading[index_all] = frame["actual"].to_numpy()
    masked = np.zeros(N_TIMELINE, dtype=bool)
    masked[index_all] = frame["switching_mask"].to_numpy()
    pw = (frame["era"] == "past_weather").to_numpy()
    temp_pw = np.full(N_TIMELINE, np.nan, dtype=np.float32)
    sw_pw = np.full(N_TIMELINE, np.nan, dtype=np.float32)
    temp_pw[index_all[pw]] = frame["drv_temp_c"].to_numpy()[pw]
    sw_pw[index_all[pw]] = frame["drv_sw_w_m2"].to_numpy()[pw]

    # Runs: the origins of experiment 311's rows, with the period and month of their rows.
    runs = (
        rows.group_by("power_fcst_init_time")
        .agg(pl.col("period").first(), pl.col("month_key").first(), pl.col("run").first())
        .sort("power_fcst_init_time")
    )
    era = (
        frame.group_by("power_fcst_init_time")
        .agg(pl.col("era").first())
        .join(runs.select("power_fcst_init_time"), on="power_fcst_init_time")
    )
    runs = runs.join(era, on="power_fcst_init_time", validate="1:1").sort("power_fcst_init_time")
    n_runs = runs.height
    origin = naive(runs["power_fcst_init_time"])
    steps = np.arange(1, common.STEPS + 1)
    valid = origin[:, None] + steps[None, :] * HALF_HOUR
    v_index = timeline_index(times=valid)
    assert index_all.min() >= 0, "the timeline starts after the data"
    assert v_index.min() - 448 >= 0, "the timeline starts after the data"

    temp = np.full((n_runs, common.STEPS), np.nan, dtype=np.float32)
    sw = np.full_like(temp, np.nan)
    prev = np.full_like(temp, np.nan)
    keep = np.zeros((n_runs, common.STEPS), dtype=bool)
    # Steps up to 168 hours: scatter experiment 311's rows from the input frame.
    pos = {t: i for i, t in enumerate(origin)}
    joined = rows.select("power_fcst_init_time", "valid_time").join(
        frame.unique(subset=["power_fcst_init_time", "valid_time"]).select(
            "power_fcst_init_time", "valid_time", "drv_temp_c", "drv_sw_w_m2", "drv_temp_prev_c"
        ),
        on=["power_fcst_init_time", "valid_time"],
        how="left",
        validate="m:1",
    )
    r_idx = np.array([pos[t] for t in naive(joined["power_fcst_init_time"])])
    k_idx = ((naive(joined["valid_time"]) - origin[r_idx]) // HALF_HOUR).astype(np.int64) - 1
    temp[r_idx, k_idx] = joined["drv_temp_c"].to_numpy()
    sw[r_idx, k_idx] = joined["drv_sw_w_m2"].to_numpy()
    prev[r_idx, k_idx] = joined["drv_temp_prev_c"].to_numpy()
    keep[r_idx, k_idx] = True
    # Steps beyond 168 hours (and every step's ensemble values, which equal the frame's).
    with np.load(store_dir / f"{site}.npz") as stored:
        store_origins = stored["origins"].astype("datetime64[us]")
        t_mean = np.nanmean(
            stored["temperature"][:, :, : common.STEPS].astype(np.float64), axis=1
        ).astype(np.float32)
        s_mean = np.nanmean(
            stored["shortwave"][:, :, : common.STEPS].astype(np.float64), axis=1
        ).astype(np.float32)
    in_store = {t: i for i, t in enumerate(store_origins)}
    is_ens = (runs["era"] == "ens").to_numpy()
    for r in range(n_runs):
        if is_ens[r]:
            s = in_store[origin[r]]
            temp[r, common.SHORT_STEPS :] = t_mean[s, common.SHORT_STEPS :]
            sw[r, common.SHORT_STEPS :] = s_mean[s, common.SHORT_STEPS :]
            prev[r, 48:] = t_mean[s, :-48]
        else:
            tail = v_index[r, common.SHORT_STEPS :]
            temp[r, common.SHORT_STEPS :] = temp_pw[tail]
            sw[r, common.SHORT_STEPS :] = sw_pw[tail]
            prev[r, 48:] = temp_pw[v_index[r, 48:] - 48]
    actual = reading[v_index]
    extension_ok = (
        ~masked[v_index]
        & np.isfinite(actual)
        & np.isfinite(temp)
        & np.isfinite(sw)
        & np.isfinite(prev)
    )
    keep[:, common.SHORT_STEPS :] = extension_ok[:, common.SHORT_STEPS :]
    first_day = np.arange(common.STEPS) < 48
    prev[:, first_day] = temp[:, first_day]  # no reading of the day before the origin is stored
    short_ok = np.isfinite(temp) & np.isfinite(sw) & np.isfinite(prev)
    keep[:, : common.SHORT_STEPS] &= short_ok[:, : common.SHORT_STEPS]

    # Calendar and sun, on the flat valid-time array.
    flat = pl.Series(valid.ravel()).dt.replace_time_zone("UTC")
    local = flat.dt.convert_time_zone(LONDON)
    hh = (
        (local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30)
        .to_numpy()
        .astype(np.float64)
    )
    dow = (local.dt.weekday().cast(pl.Int32) - 1).to_numpy()
    doy = local.dt.ordinal_day().to_numpy().astype(np.float64)
    holiday_days = pl.Series(sorted(bank_holiday_names())).cast(pl.Date)
    bank = local.dt.date().is_in(holiday_days).to_numpy().astype(np.float64)
    month, day = local.dt.month().to_numpy(), local.dt.day().to_numpy()
    christmas = (((month == 12) & (day >= 24)) | ((month == 1) & (day <= 1))).astype(np.float64)
    e7 = ((hh >= 1) & (hh <= 14)).astype(np.float64)
    geometry = SunGeometry.build(
        end_times=valid.ravel(), latitude=table.latitude, longitude=table.longitude
    )
    elevation = 90.0 - geometry.zenith_deg
    shape, _ = geometry.pv_shape_and_ghi(
        ghi_w_m2=np.nan_to_num(sw.ravel().astype(np.float64))[None, :]
    )
    unit = np.nan_to_num(shape[0], nan=0.0)
    cell = temp.ravel() + 25.0 * unit
    unit_pv = unit * np.clip(1.0 - 0.004 * (cell - 25.0), 0.0, None)

    finite_readings = reading[np.isfinite(reading)][: common.STANDARDISE_READINGS]
    mu, sd = float(finite_readings.mean()), float(finite_readings.std())
    # Recent power per run: the newest reading at the origin (3 hours old), the 24 hours before it.
    newest = timeline_index(times=origin) - common.LATENCY_STEPS
    last = reading[newest]
    window = reading[newest[:, None] - np.arange(48)[None, :]]
    mean24 = np.nanmean(window, axis=1)
    recent_missing = (~np.isfinite(last) | ~np.isfinite(mean24)).astype(np.float64)
    hist_index = newest[:, None] - np.arange(common.HISTORY_STEPS - 1, -1, -1)[None, :]
    hist_raw = reading[hist_index]
    hist_missing = ~np.isfinite(hist_raw)
    hist_times = TIMELINE_START + hist_index * HALF_HOUR
    h_local = pl.Series(hist_times.ravel()).dt.replace_time_zone("UTC").dt.convert_time_zone(LONDON)
    h_hh = (
        (h_local.dt.hour().cast(pl.Int32) * 2 + h_local.dt.minute().cast(pl.Int32) // 30)
        .to_numpy()
        .astype(np.float64)
    )
    hist = np.stack(
        [
            np.where(hist_missing, 0.0, (hist_raw - mu) / sd).astype(np.float32),
            hist_missing.astype(np.float32),
            *[
                a.reshape(n_runs, -1).astype(np.float32)
                for a in harmonics(x=h_hh, period=48.0, n=1)
            ],
        ],
        axis=2,
    )

    lead_h = np.tile(steps * 0.5, n_runs)
    lag_hours = 24.0 * (np.floor((lead_h + 3.0) / 24.0) + 1.0)
    lag_index = v_index.ravel() - (lag_hours * 2).astype(np.int64)
    lag_raw = reading[lag_index]
    lag_missing = ~np.isfinite(lag_raw)
    t_flat = temp.ravel().astype(np.float64)
    sw_flat = sw.ravel().astype(np.float64)
    prev_flat = prev.ravel().astype(np.float64)
    repeat = lambda a: np.repeat(a, common.STEPS)  # noqa: E731
    columns = {
        "lead_h": lead_h,
        "temp": t_flat,
        "cold_hinge": np.maximum(BASE_TEMPERATURE_C - t_flat, 0.0),
        "warm_hinge": np.maximum(t_flat - BASE_TEMPERATURE_C, 0.0),
        "temp_prev": prev_flat,
        "sw": sw_flat,
        "solar_term": sw_flat * np.sin(np.deg2rad(np.maximum(elevation, 0.0))) / 1000.0,
        "dark": (elevation < 0.0).astype(np.float64),
        "elevation": elevation,
        "dow_s": harmonics(x=dow.astype(np.float64), period=7.0, n=1)[0],
        "dow_c": harmonics(x=dow.astype(np.float64), period=7.0, n=1)[1],
        "bank_holiday": bank,
        "christmas": christmas,
        "e7": e7,
        "lag_power": np.where(lag_missing, 0.0, (lag_raw - mu) / sd),
        "lag_missing": lag_missing.astype(np.float64),
        "lag_hours": lag_hours,
        "last_reading": repeat(np.where(np.isfinite(last), (last - mu) / sd, 0.0)),
        "mean24": repeat(np.where(np.isfinite(mean24), (mean24 - mu) / sd, 0.0)),
        "recent_missing": repeat(recent_missing),
        "unit_pv": unit_pv,
    }
    columns.update(
        zip(("hh_s1", "hh_c1", "hh_s2", "hh_c2"), harmonics(x=hh, period=48.0, n=2), strict=True)
    )
    columns.update(
        zip(
            ("doy_s1", "doy_c1", "doy_s2", "doy_c2"),
            harmonics(x=doy, period=365.25, n=2),
            strict=True,
        )
    )
    day_type = DAY_TYPES[dow]
    columns.update({f"dt{k}": (day_type == k).astype(np.float64) for k in range(5)})
    x = np.column_stack([columns[name] for name in common.FEATURES]).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0).reshape(n_runs, common.STEPS, len(common.FEATURES))
    y = np.where(keep, (np.nan_to_num(actual, nan=0.0)), 0.0).astype(np.float32)

    common.PREP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(common.PREP_DIR / f"{site}_x.npy", x)
    np.save(common.PREP_DIR / f"{site}_y.npy", y)
    np.save(common.PREP_DIR / f"{site}_keep.npy", keep)
    np.save(common.PREP_DIR / f"{site}_hist.npy", hist)
    np.save(common.PREP_DIR / f"{site}_valid_ens_temp24.npy", _window_mean(temp=temp))
    runs.with_columns(site=pl.lit(site), mu=pl.lit(mu), sd=pl.lit(sd)).write_parquet(
        common.PREP_DIR / f"{site}_runs.parquet"
    )
    print(
        site,
        "runs",
        n_runs,
        "kept steps",
        int(keep.sum()),
        "short",
        int(keep[:, : common.SHORT_STEPS].sum()),
        "of 311 rows",
        rows.height,
        flush=True,
    )


def _window_mean(*, temp: np.ndarray) -> np.ndarray:
    """Return the centred 49-half-hour mean of the temperature along the lead axis (edges clipped)."""
    filled = np.where(np.isfinite(temp), temp, np.nanmean(temp))
    csum = np.concatenate([np.zeros((len(temp), 1)), np.cumsum(filled, axis=1)], axis=1)
    n = temp.shape[1]
    low = np.clip(np.arange(n) - 24, 0, n - 1)
    high = np.clip(np.arange(n) + 24, 0, n - 1)
    return ((csum[:, high + 1] - csum[:, low]) / (high - low + 1)[None, :]).astype(np.float32)


def main() -> None:
    """Build every Primary."""
    cache = c164.load_cache()
    for site in common.SITES:
        build_site(site=site, table=cache.sites[site].table, store_dir=common.E111_STORE)
    sites = {
        s: {
            "time_series_id": cache.sites[s].table.time_series_id,
            "effective_capacity_mw": cache.sites[s].table.effective_capacity_mw,
        }
        for s in common.SITES
    }
    common.write_json_atomic(path=common.PREP_DIR / "sites.json", content=sites)
    data.combine()


if __name__ == "__main__":
    main()
