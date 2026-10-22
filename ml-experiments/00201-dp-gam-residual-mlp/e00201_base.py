"""Rebuild arm D of experiment 155 on past weather for the years before the ensemble forecasts.

Arm D is the convex model (T-all fits, monthly refit, 24-month window, pinned solar capacity)
with the Kalman filter on recent errors, the winter night offset, and recipe R. The ensemble
weather exists only from 2024-04, so for origins before 2024-07-01 the weather is the feature
cache's own ERA5 and CAMS values (past weather: no forecast error in the weather). Each row carries
`era = "past_weather"`. The ensemble era is experiment 155's stored arm D, copied unchanged with
`era = "ens"` by `e00201_join.py`.

The rebuild runs one job per substation: fit T-all at each monthly cut, run the filter on the
chained fits' residuals, forecast the daily 09:00 UTC origins, apply the offset and recipe R with
`e00201_corrections` (the same causal expanding-window functions as experiment 155), refit the
spread on the odd calendar months, and write one parquet per (substation, month of origin).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00201-*/e00201_base.py [--workers 6]
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00201_common as common
import e00201_corrections as corr
import e00201_filters as filters
import e00201_tall as tall
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    correction_at,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_pinned_capacity import read_vintages

BASE_DIR: Final[Path] = common.WORK_DIR / "base"
"""One file per (substation, month of origin): `<site>_<YYYY-MM>.parquet`, ascending months."""
FITS_DIR: Final[Path] = common.WORK_DIR / "tall_fits"
FIRST_CUT_FLOOR: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
LAST_CUT: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed", "arm",
    *common.QUANTILES, "actual", "mean", "level", "anomaly", "hour_band", "quarter",
    "switching_mask", "era",
)  # fmt: skip


def early_cuts(*, cache: FeatureCache, site: str) -> list[datetime]:
    """Return the monthly cuts from the site's first with 90 usable days to 2024-07-01."""
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    needed = common.MIN_TRAINING_DAYS * 48
    cuts = []
    cut = FIRST_CUT_FLOOR
    while cut <= LAST_CUT:
        if int((usable & (cache.end_times <= naive_utc(moment=cut))).sum()) >= needed:
            cuts.append(cut)
        cut = datetime(cut.year + (cut.month == 12), cut.month % 12 + 1, 1, tzinfo=UTC)
    return cuts


def filter_settings() -> filters.FilterSettings:
    """Return experiment 133's shared filter cell (POOL)."""
    hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


def fit_site(*, cache: FeatureCache, site: str) -> list[tall.TallFit]:
    """Return the site's T-all fits at every early cut, from disk if stored."""
    path = FITS_DIR / f"{site}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    vintages = read_vintages()
    fits = [
        tall.fit_tall(
            cache=cache, vintages=vintages, site=site, cut=cut, penalty_weight=common.T_ALL_LAMBDA
        )
        for cut in early_cuts(cache=cache, site=site)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(path)
    return fits


def residual_stream(*, cache: FeatureCache, fits: list[tall.TallFit], site: str) -> np.ndarray:
    """Return observed minus the chained fits' forecast on the cache's own weather.

    Fit `i` supplies the rows after its cut and up to the next cut, so no residual is in-sample.
    """
    features = cache.sites[site]
    stream = np.full(len(cache.end_times), np.nan)
    cuts = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=fit.cut), side="right")) - 1
        for fit in fits
    ]
    ends = [*cuts[1:], int(np.searchsorted(cache.end_times, naive_utc(moment=LAST_CUT)))]
    for fit, cut, end in zip(fits, cuts, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            stream[rows] = features.y[rows] - tall.predict_tall(cache=cache, fit=fit, index=rows)
    return stream


def origins_of_site(*, fits: list[tall.TallFit]) -> list[datetime]:
    """Return the daily 09:00 UTC origins from the first fit's cut to 2024-07-01."""
    first = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    day = datetime(first.year, first.month, first.day, 9, tzinfo=UTC)
    if day < first:
        day += timedelta(days=1)
    origins = []
    while day < common.EARLY_LAST_ORIGIN:
        origins.append(day)
        day += timedelta(days=1)
    return origins


def base_frame(
    *, cache: FeatureCache, site: str, fits: list[tall.TallFit], run: filters.StreamRun
) -> pl.DataFrame:
    """Return the Kalman-corrected past-weather forecast of every origin, with a zero spread."""
    features = cache.sites[site]
    frames = []
    run_zero = datetime(2024, 4, 1, tzinfo=UTC)
    for origin in origins_of_site(fits=fits):
        origin64 = np.datetime64(origin.replace(tzinfo=None), "us")
        position = int(np.searchsorted(cache.end_times, origin64, side="right")) - 1
        rows = np.arange(position + 1, position + 1 + common.HORIZON_HALF_HOURS)
        rows = rows[rows < len(cache.end_times)]
        if len(rows) < 48:
            continue
        fit = max((f for f in fits if f.cut <= origin - timedelta(hours=DEFAULT_DELAY_HOURS)),
                  key=lambda f: f.cut, default=None)  # fmt: skip
        if fit is None:
            continue
        predicted = np.full(len(rows), np.nan)
        ok = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
        )
        if not ok.any():
            continue
        predicted[ok] = tall.predict_tall(cache=cache, fit=fit, index=rows[ok])
        band = cache.calendar.band[rows].astype(np.int64)
        cutoff = usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=np.array([naive_utc(moment=origin)]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )
        correction = correction_at(
            trace=run.trace, cutoff=np.full(len(rows), cutoff[0]), target=rows, band=band
        )
        mean = predicted + correction.level + correction.anomaly
        keep = ok & features.present[rows] & np.isfinite(features.y[rows])
        if not keep.any():
            continue
        rows, mean = rows[keep], mean[keep]
        n = len(rows)
        valid = cache.end_times[rows]
        frames.append(
            pl.DataFrame(
                {
                    "time_series_id": np.full(n, features.table.time_series_id, dtype=np.int32),
                    "run": np.full(n, (origin - run_zero).days, dtype=np.int32),
                    "power_fcst_init_time": pl.Series(np.full(n, origin64)).dt.replace_time_zone(
                        "UTC"
                    ),
                    "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
                    "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(np.float64),
                    "seed": np.zeros(n, dtype=np.int8),
                    "arm": np.full(n, "d"),
                    "actual": features.y[rows].astype(np.float32),
                    "mean": mean.astype(np.float32),
                    "level": correction.level[keep].astype(np.float32),
                    "anomaly": correction.anomaly[keep].astype(np.float32),
                    "hour_band": cache.calendar.band[rows].astype(np.int8),
                    "quarter": cache.calendar.quarter[rows].astype(np.int8),
                    "switching_mask": features.switching_mask[rows],
                }
            )
        )
    return pl.concat(frames)


def build_site(*, site: str) -> str:
    """Build and write every month of past-weather arm D at one substation."""
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    fits = fit_site(cache=cache, site=site)
    fitted_at = time.monotonic() - started
    stream = residual_stream(cache=cache, fits=fits, site=site)
    run = filters.filter_stream(
        residual=stream, band=cache.calendar.band, settings=filter_settings()
    )
    frame = base_frame(cache=cache, site=site, fits=fits, run=run)
    frame = corr.add_calendar(frame=frame).with_columns(
        period=pl.when(pl.col("power_fcst_init_time").dt.month() % 2 == 1)
        .then(pl.lit("tuning"))
        .otherwise(pl.lit("replicate"))
    )
    mean_a = frame["mean"].to_numpy().astype(np.float64)
    offset, _ = corr.winter_offset_shift(frame=frame, variant="winter")
    mean_b = mean_a + offset
    shift = offset + corr.recipe_r_shift(frame=frame, forecast=mean_b)
    final = corr.apply_shift_and_refit_spread(frame=frame, shift=shift).with_columns(
        era=pl.lit("past_weather")
    )
    months = final.with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    ).partition_by("month", as_dict=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for (month,), part in sorted(months.items()):
        target = BASE_DIR / f"{site}_{month}.parquet"
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        part.select(list(OUTPUT_COLUMNS)).write_parquet(temporary)
        temporary.rename(target)
    return (
        f"{site}: {final.height:,} rows, {len(months)} months, {len(fits)} fits "
        f"({fitted_at:.0f} s), {time.monotonic() - started:.0f} s"
    )


def main() -> None:
    """Build every substation in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


def _job(site: str) -> str:
    return build_site(site=site)


if __name__ == "__main__":
    main()
