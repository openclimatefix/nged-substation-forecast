"""GAM-ens: the convex model trained on ENS control rows, as XGBoost is, for experiment 111.

Part 6's convex arms are fitted on ERA5 and CAMS and driven by calibrated ENS. GAM-ens departs from
that: it trains on the ENS control member's rows at leads of 0.5 to 168 hours, one row per run and
half-hour, each valid time weighted by one over the number of runs that cover it, so a valid time
counts once however many runs forecast it. It is fitted once, at 2025-07-01, on the rows from
2024-04-01 (15 months), with no drift and the capacity pinned, and is neither tuned nor replicated.

Each row's weather is the calibrated control member and the trailing temperature means are
stitched from the analysis proxy, exactly as at forecast time (`studies.dp_ens_weather.RunWindow`),
so the training weather is the weather the model is driven by. The forecast uses `gam0`'s spread
and no correction. The rows of the scored year are forecast with this single fit.

    uv run python ml-experiments/00111-*/e00111_gam_ens.py [--workers 4]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from datetime import UTC, datetime
from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_dp_run as dp_run
import e00111_ens as ens
import e00111_fits as fits_module
import numpy as np
import torch
from studies.dp_calendar import CalendarColumns
from studies.dp_ens import choose_fit  # noqa: F401
from studies.dp_ens_weather import HORIZON_HALF_HOURS, RunWindow
from studies.dp_features import FeatureCache, SiteFeatures
from studies.dp_gam import TrainingWindow, fit_gam, prepare_fit
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
ARM: Final[str] = "gamens"
FIRST_RUN_INDEX: Final[int] = 9
"""Runs before this index lack a week of analysis proxy before their origin."""


def expanded_cache(*, cache: FeatureCache, setup: ens.SiteSetup, cut: datetime) -> FeatureCache:
    """Stack every run's control-driven window into one cache, sorted by valid time.

    Args:
        cache: The full feature cache.
        setup: The substation's setup, with its calibration.
        cut: Only runs whose origin is before the cut contribute.

    Returns:
        A cache with one row per run and half-hour, whose `end_times` repeat and ascend.
    """
    site = setup.site
    windows: list[FeatureCache] = []
    for index, run in enumerate(setup.runs):
        if index < FIRST_RUN_INDEX or setup.origins[index] >= cut:
            continue
        if len(cache.end_times) - run.origin_position - 1 < HORIZON_HALF_HOURS:
            continue
        window = RunWindow.build(cache=cache, context=setup.context, run=run)
        windows.append(window.driven_caches(context=setup.context, run=run, members=[0])[0])
    ends = np.concatenate([w.end_times for w in windows])
    order = np.argsort(ends, kind="stable")
    calendar = CalendarColumns(
        **{
            f.name: np.concatenate([getattr(w.calendar, f.name) for w in windows])[order]
            for f in fields(CalendarColumns)
        }
    )
    table = windows[0].sites[site].table
    arrays = {
        f.name: np.concatenate([getattr(w.sites[site], f.name) for w in windows])[order]
        for f in fields(SiteFeatures)
        if f.name != "table"
    }
    return FeatureCache(
        key=cache.key,
        end_times=ends[order],
        calendar=calendar,
        sites={site: SiteFeatures(table=table, **arrays)},
        node_ids={site: (site,)},
    )


def coverage_weights(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return one over the number of training rows that share each row's valid time."""
    ends = cache.end_times[index]
    _, inverse, counts = np.unique(ends, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse])[None, :]


def fit_and_forecast(site: str) -> str:
    """Fit GAM-ens at one site and forecast the scored year, writing one parquet file."""
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    tuned = arms.read_tuned()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=arms.load_calibration(site=site).with_solar_shrink(
            solar_shrink=tuned["solar_shrink"]
        )
    )
    expanded = expanded_cache(cache=cache, setup=setup, cut=CUT)
    spec = fits_module.arm_spec(arm="gamk", settings=fits_module.tuned())
    pinned = pinned_capacity(vintages=read_vintages(), table=cache.sites[site].table, cut=CUT)
    _, index = prepare_fit(
        cache=expanded,
        site=site,
        spec=spec,
        cut=CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
    )
    fit = fit_gam(
        cache=expanded,
        site=site,
        specs=[spec],
        cut=CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
        weights=coverage_weights(cache=expanded, index=index),
        pinned=pinned,
    )
    spread = arms.spread_with_factors(arm="gam0", site=site, tuned=tuned)
    runs = setup.indices(periods={"scored"})
    frame = ens.forecast_frame(
        cache=cache, setup=setup, fits=[fit], spread=spread, trace=None, run_indices=runs, arm=ARM
    )
    target = dp_run.out_path(arm=ARM, period="scored", site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return (
        f"{ARM} {site}: {len(index):,} training rows, {frame.height:,} forecast rows, "
        f"{time.monotonic() - started:.0f} s"
    )


def main() -> None:
    """Fit GAM-ens at every substation with scored-year data."""
    dp_run.require_gate()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sites = [site for site in common.ALL_SITES if site != "D18"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(fit_and_forecast, sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
