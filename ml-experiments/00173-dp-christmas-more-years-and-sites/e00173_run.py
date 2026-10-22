"""Forecast every Christmas season at every Primary substation and write the prediction frames.

For each substation and season (a year's 24 December), the script fits the convex model of
experiment 107's `g1_fc` arm on the season's cut, runs the Kalman filter of experiment 133's shared
noise setting over the fit's own residuals after the cut, forecasts daily 09:00 UTC runs of 168
hours with observed weather, applies recipe R, and fits the residual spread on the season's runs
with valid times from 16 November to 22 December. The first season (Christmas 2019-20) has no
spread, so its quantile columns are `NaN`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00173-*/e00173_run.py [--workers 6]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00173_common as common
import e00173_corrections as corr
import e00173_filters as filters
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, correction_at, usable_cutoff_positions
from studies.dp_ens import LeadHourSpread
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam

FIRST_QUARTER: int = 0
"""One spread table per season: every row is given this calendar quarter."""


def out_path(*, arm: str, site: str, season: int) -> Path:
    """Return the file that holds an arm's forecasts for one season at one substation."""
    return common.forecast_dir(arm=arm) / f"{site}_{season}.parquet"


def filter_settings() -> filters.FilterSettings:
    """Return experiment 133's shared filter noise setting (log q, logit phi, log sigmas)."""
    hyper = np.array(json.loads(common.POOL_PATH.read_text())["hyper"])
    return filters.FilterSettings(
        q=float(np.exp(hyper[0])),
        phi=float(1.0 / (1.0 + np.exp(-hyper[1]))),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        rho_a=0.0,
    )


def usable_rows(*, cache: FeatureCache, site: str, rows: np.ndarray) -> np.ndarray:
    """Return the rows whose weather inputs are finite, so the fit can predict them."""
    features = cache.sites[site]
    keep = (
        np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
        & np.isfinite(features.pv_unit_shape[rows])
        & np.isfinite(features.temperature_c[rows])
    )
    return rows[keep]


def forecast_season(*, cache: FeatureCache, site: str, season: int, arm: str) -> pl.DataFrame:
    """Return the season's runs at a substation: one row per scored half-hour with a reading."""
    features = cache.sites[site]
    cut = common.cut_of_season(season=season)
    tuned = json.loads(common.TUNED_PATH.read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        capacity="fitted",
    )
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=TrainingWindow(months=common.TRAINING_MONTHS if arm == "m12" else None),
        season_years=common.SEASON_YEARS,
    )
    end_times = cache.end_times
    last_run = common.last_run_of_season(season=season)
    n_days = (last_run - cut).days + 1
    cut_row = int(np.searchsorted(end_times, naive_utc(moment=cut), side="right")) - 1
    last_row = int(np.searchsorted(end_times, naive_utc(moment=last_run + timedelta(days=8))))
    last_row = min(last_row, len(end_times) - 1)
    stream = np.full(len(end_times), np.nan)
    rows = usable_rows(cache=cache, site=site, rows=np.arange(cut_row + 1, last_row + 1))
    rows = rows[np.isfinite(features.y[rows]) & features.present[rows]]
    stream[rows] = features.y[rows] - predict_gam(cache=cache, fit=fit, index=rows)
    run = filters.filter_stream(
        residual=stream, band=cache.calendar.band, settings=filter_settings()
    )
    frames = []
    for day in range(n_days):
        origin = cut + timedelta(days=day) + common.PUBLICATION_DELAY
        origin64 = naive_utc(moment=origin)
        position = int(np.searchsorted(end_times, origin64))
        if position + common.HORIZON_HALF_HOURS >= len(end_times):
            continue
        target = usable_rows(
            cache=cache,
            site=site,
            rows=np.arange(position + 1, position + 1 + common.HORIZON_HALF_HOURS),
        )
        target = target[np.isfinite(features.y[target]) & features.present[target]]
        if not len(target):
            continue
        mean = predict_gam(cache=cache, fit=fit, index=target)
        cutoff = usable_cutoff_positions(
            end_times=end_times,
            initialisation=np.array([origin64]),
            delay_hours=DEFAULT_DELAY_HOURS,
        )
        correction = correction_at(
            trace=run.trace,
            cutoff=np.full(len(target), cutoff[0]),
            target=target,
            band=cache.calendar.band[target].astype(np.int64),
        )
        valid = end_times[target]
        frames.append(
            pl.DataFrame(
                {
                    "run": np.full(len(target), day, dtype=np.int32),
                    "power_fcst_init_time": pl.Series(
                        np.full(len(target), origin64)
                    ).dt.replace_time_zone("UTC"),
                    "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
                    "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(np.float64),
                    "actual": features.y[target].astype(np.float64),
                    "mean_fit": mean,
                    "mean_filter": mean + correction.total,
                    "hour_band": cache.calendar.band[target].astype(np.int8),
                    "switching_mask": features.switching_mask[target],
                }
            )
        )
    frame = corr.add_calendar(frame=pl.concat(frames).sort(["power_fcst_init_time", "valid_time"]))
    shift = corr.recipe_r_shift(frame=frame, forecast=frame["mean_filter"].to_numpy())
    frame = frame.with_columns(
        mean=pl.Series(frame["mean_filter"].to_numpy() + shift),
        shift_r=pl.Series(shift),
        site=pl.lit(site),
        season=pl.lit(season),
        arm=pl.lit(arm),
        n_train_rows=pl.lit(fit.n_rows, dtype=pl.Int32),
        time_series_id=pl.lit(features.table.time_series_id, dtype=pl.Int32),
    )
    return attach_quantiles(frame=frame, season=season)


def attach_quantiles(*, frame: pl.DataFrame, season: int) -> pl.DataFrame:
    """Add `q01` to `q99`: the corrected mean plus a spread fitted before 23 December."""
    n = frame.height
    if season < common.FIRST_SEASON_WITH_SPREAD:
        return frame.with_columns(
            *[pl.lit(None, dtype=pl.Float32).alias(q) for q in common.QUANTILES]
        )
    first = datetime(season, *common.SPREAD_FIRST_VALID, tzinfo=UTC)
    last = datetime(season, *common.SPREAD_LAST_VALID, tzinfo=UTC) + timedelta(days=1)
    pool = (
        (pl.col("valid_time") >= first) & (pl.col("valid_time") < last) & ~pl.col("switching_mask")
    )
    fitted = frame.filter(pool)
    mean = frame["mean"].to_numpy()
    spread = LeadHourSpread.fit(
        residuals=(fitted["actual"] - fitted["mean"]).to_numpy(),
        band=fitted["hour_band"].to_numpy().astype(np.int64),
        quarter=np.full(fitted.height, FIRST_QUARTER, dtype=np.int64),
        lead_hours=fitted["lead_hours"].to_numpy(),
    )
    quantiles = spread.quantiles(
        mean=mean,
        band=frame["hour_band"].to_numpy().astype(np.int64),
        quarter=np.full(n, FIRST_QUARTER, dtype=np.int64),
        lead_hours=frame["lead_hours"].to_numpy(),
    )
    return frame.with_columns(
        *[
            pl.Series(name, quantiles[:, position].astype(np.float32))
            for position, name in enumerate(common.QUANTILES)
        ]
    )


_STATE: dict[str, FeatureCache] = {}


def _job(job: tuple[str, str, int]) -> str:
    torch.set_num_threads(1)
    arm, site, season = job
    target = out_path(arm=arm, site=site, season=season)
    if target.exists():
        return f"skip {target.name}"
    started = time.monotonic()
    if "cache" not in _STATE:
        _STATE["cache"] = common.load_cache()
    try:
        frame = forecast_season(cache=_STATE["cache"], site=site, season=season, arm=arm)
    except ValueError as error:
        return f"no fit {arm} {site} {season}: {error}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return f"{arm} {site} {season}: {frame.height:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Forecast every season at every substation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--arms", nargs="+", default=list(common.ARMS))
    parser.add_argument("--seasons", nargs="+", type=int, default=list(common.SEASONS))
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    jobs = [
        (arm, site, season) for arm in args.arms for season in args.seasons for site in args.sites
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
