"""Forecast GAM-K for 2024-04 to 2024-06, and assemble the K0 table of experiment 119.

K0 is GAM-K exactly as experiment 111 stored it. Experiment 111 forecast the tuning, replicate, and
scored months but not the three months before them, which ENC's first training year needs. This
script forecasts those 91 runs with experiment 111's stored monthly fits (which already start at
the 2024-04-01 cut), calibration, spread, and Kalman filter, then checks the method by forecasting
a sample of tuning-month runs and comparing them with the stored rows. It then writes one K0
parquet file per site holding the 456 runs of 2024-04 to 2025-06 and the 291 scored runs.

    uv run python ml-experiments/00119-*/e00119_gamk.py [--workers 3]
"""

import argparse
import json
import pickle
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Final

import e00119_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    run_kalman,
)
from studies.dp_ens import LeadHourSpread, forecast_run, proxy_residual_stream
from studies.dp_ens_weather import (
    AnalysisProxy,
    Calibration,
    EnsSiteRun,
    WeatherContext,
    ghi_climatology,
    proxy_cache,
)
from studies.dp_gam import QUANTILE_COLUMNS, GamFit, ResidualSpread

WORK_111: Final[Path] = common.EXP111_DIR / "work"
WARM_UP_FIRST: Final[datetime] = datetime(2024, 4, 1, tzinfo=common.FIRST_RUN.tzinfo)
WARM_UP_LAST: Final[datetime] = datetime(2024, 6, 30, tzinfo=common.FIRST_RUN.tzinfo)
PROXY_WARM_UP_RUNS: Final[int] = 9
"""Runs before this index lack a week of analysis proxy before their origin (experiment 111)."""
MIN_WINDOW_HALF_HOURS: Final[int] = 48
CLIMATOLOGY_BEFORE: Final[datetime] = datetime(2024, 7, 1, tzinfo=common.FIRST_RUN.tzinfo)
CHECK_RUNS: Final[tuple[int, ...]] = (100, 140, 200, 260)
"""Run indices (tuning and replicate months) re-forecast to check against the stored rows."""
K0_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "run",
    "power_fcst_init_time",
    "valid_time",
    "lead_hours",
    *QUANTILE_COLUMNS,
    "mean",
    "actual",
    "switching_mask",
)


def k0_path(*, site: str) -> Path:
    """Return the file that holds a site's K0 rows."""
    return common.WORK_DIR / "k0" / f"{site}.parquet"


def _spread(*, site: str, tuned: dict) -> LeadHourSpread:
    with np.load(WORK_111 / "spread" / f"gamk_{site}.npz") as stored:
        tables, factors = stored["tables"], stored["factors"]
    base = LeadHourSpread(
        by_lead_band=tuple(ResidualSpread(table=table) for table in tables), factors=factors
    )
    return base.with_factors(factors=tuned["spread_factors"]["gamk"])


def _calibration(*, site: str, shrink: Sequence[float]) -> Calibration:
    with np.load(WORK_111 / "calibration" / f"{site}.npz") as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )
    return calibration.with_solar_shrink(solar_shrink=shrink)


def forecast_runs(
    *,
    site: str,
    run_indices: list[int] | None,
    temperature_shift_k: float = 0.0,
    irradiance_scale: float = 1.0,
) -> pl.DataFrame:
    """Forecast GAM-K at one site with experiment 111's stored pieces.

    Args:
        site: The substation's label.
        run_indices: Indices into `common.run_dates()`; `None` for the 91 warm-up runs.
        temperature_shift_k: Added to every ensemble member's temperature in the forecast window
            only; the analysis proxy and the filter's residuals stay unperturbed.
        irradiance_scale: Multiplies every member's shortwave flux in the forecast window only.

    Returns:
        The rows of `K0_COLUMNS`, in points of effective capacity, plus `run`.
    """
    cache = common.load_cache()
    tuned = json.loads((WORK_111 / "tuned.json").read_text())
    dates = common.run_dates()
    origins = [d + common.PUBLICATION_DELAY for d in dates]
    with np.load(WORK_111 / "ens_store" / f"{site}.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    positions = np.searchsorted(cache.end_times, arrays["origins"])
    runs = [
        EnsSiteRun(
            origin_position=int(p),
            temperature_c=arrays["temperature"][i],
            shortwave_w_m2=arrays["shortwave"][i],
        )
        for i, p in enumerate(positions)
    ]
    context = WeatherContext(
        site=site,
        calibration=_calibration(site=site, shrink=tuned["solar_shrink"]),
        climatology=ghi_climatology(cache=cache, site=site, before=CLIMATOLOGY_BEFORE),
        proxy=AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times)),
    )
    fits: list[GamFit] = pickle.loads((WORK_111 / "fits" / "gamk" / f"{site}.pkl").read_bytes())
    hyper = tuned["filter_hyper"]["gamk"][site]
    stream = proxy_residual_stream(
        cache=proxy_cache(cache=cache, context=context), fits=fits, site=site
    )
    trace = partial(run_kalman, hyper=FilterHyperparameters.from_array(values=np.array(hyper)))(
        residual=stream, band=cache.calendar.band
    )
    spread = _spread(site=site, tuned=tuned)
    if run_indices is None:
        run_indices = [i for i, d in enumerate(dates) if WARM_UP_FIRST <= d <= WARM_UP_LAST]
    features = cache.sites[site]
    first_usable = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    frames = []
    for index in run_indices:
        run = runs[index]
        if index < PROXY_WARM_UP_RUNS or origins[index] < first_usable:
            continue
        if len(cache.end_times) - run.origin_position - 1 < MIN_WINDOW_HALF_HOURS:
            continue
        forecast = forecast_run(
            cache=cache,
            context=context,
            run=EnsSiteRun(
                origin_position=run.origin_position,
                temperature_c=run.temperature_c + temperature_shift_k,
                shortwave_w_m2=run.shortwave_w_m2 * irradiance_scale,
            ),
            origin=origins[index],
            fits=fits,
            spread=spread,
            trace=trace,
        )
        rows = np.arange(forecast.window.start, forecast.window.stop)
        keep = features.present[rows] & np.isfinite(features.y[rows])
        if not keep.any():
            continue
        rows = rows[keep]
        n = len(rows)
        valid = cache.end_times[rows]
        origin64 = np.datetime64(origins[index].replace(tzinfo=None), "us")
        frames.append(
            pl.DataFrame(
                {
                    "time_series_id": np.full(n, features.table.time_series_id, dtype=np.int32),
                    "run": np.full(n, index, dtype=np.int32),
                    "power_fcst_init_time": pl.Series(np.full(n, origin64)).dt.replace_time_zone(
                        "UTC"
                    ),
                    "valid_time": pl.Series(valid).dt.replace_time_zone("UTC"),
                    "lead_hours": ((valid - origin64) / np.timedelta64(1, "h")).astype(np.float64),
                    **{
                        c: forecast.quantiles[keep][:, p].astype(np.float32)
                        for p, c in enumerate(QUANTILE_COLUMNS)
                    },
                    "mean": forecast.mean[keep].astype(np.float32),
                    "actual": features.y[rows].astype(np.float32),
                    "switching_mask": features.switching_mask[rows],
                }
            ).select(K0_COLUMNS)
        )
    return pl.concat(frames)


def stored_rows(*, site: str) -> pl.DataFrame:
    """Return experiment 111's stored GAM-K rows at a site: tuning, replicate, and scored."""
    pieces = [
        pl.read_parquet(WORK_111 / "dp_preds" / f"gamk_{period}_{site}.parquet").select(K0_COLUMNS)
        for period in ("tuning", "replicate", "scored")
    ]
    return pl.concat(pieces)


def build_site(site: str) -> str:
    """Forecast the warm-up runs, check the method, and write the site's K0 file."""
    torch.set_num_threads(1)
    started = time.monotonic()
    stored = stored_rows(site=site)
    check = forecast_runs(site=site, run_indices=list(CHECK_RUNS))
    joined = check.join(stored, on=["run", "valid_time"], suffix="_stored", how="inner")
    worst = max(
        float((joined[c] - joined[f"{c}_stored"]).abs().to_numpy().max())
        for c in common.QUANTILE_COLUMNS
    )
    if joined.is_empty() or worst > 1e-4:
        msg = f"{site}: re-forecast differs from stored GAM-K by {worst} over {joined.height} rows"
        raise RuntimeError(msg)
    warm = forecast_runs(site=site, run_indices=None)
    k0 = pl.concat([warm, stored]).sort("run", "valid_time")
    if k0.select(pl.struct("run", "valid_time").is_duplicated().any()).item():
        msg = f"{site}: duplicate K0 keys"
        raise RuntimeError(msg)
    target = k0_path(site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    k0.write_parquet(target)
    return (
        f"{site}: check {joined.height:,} rows, worst difference {worst:.2e}; warm-up "
        f"{warm.height:,} rows; K0 {k0.height:,} rows, {k0['run'].n_unique()} runs, "
        f"{time.monotonic() - started:.0f} s"
    )


def main() -> None:
    """Build the K0 table of every scored site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(build_site, common.SCORED_SITES):
            print(line, flush=True)


if __name__ == "__main__":
    main()
