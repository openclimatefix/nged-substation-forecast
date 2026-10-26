"""Fit and forecast every capacity scale of experiment 352, with and without the filter.

Each job is one (scale, Primary): it fits GAM-K at every monthly cut with the pinned capacity times
the scale, then forecasts every period twice, once with the filter's correction and once without
(the fitted mean plus the spread table). The spread is fitted on the tuning months of the same
arm. The fit, spread, and forecast code is experiment 349's.

    PYTHONPATH=FOLDER_349:FOLDER_352 \
    uv run --no-sync python ml-experiments/00352-*/e00352_run.py [--workers 6]
"""

import argparse
import functools
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import e00349_adaptor as adaptor  # ty: ignore[unresolved-import]
import e00349_ens as ens  # ty: ignore[unresolved-import]
import e00349_filters as filters  # ty: ignore[unresolved-import]
import e00349_fits as fits_module  # ty: ignore[unresolved-import]
import e00349_run as run349  # ty: ignore[unresolved-import]
import e00349_run_common as run_common  # ty: ignore[unresolved-import]
import e00352_common as common
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import proxy_cache
from studies.dp_gam import GamFit, fit_gam
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages


def scaled_pin(*, pin: PinnedCapacity, scale: float) -> PinnedCapacity:
    """Return the pin's weekly path and held value multiplied by `scale`."""
    return adaptor.with_path(
        pin=pin,
        points=adaptor.pin_on_weeks(pin=pin) * scale,
        held=pin.held_points * scale,
        source="pin",
    )


def fit_scaled(*, scale: float, site: str) -> list[GamFit]:
    """Return GAM-K's monthly fits at a Primary with the pin times `scale`."""
    cache = run_common.load_cache()
    vintages = read_vintages()
    cuts = run_common.monthly_cuts()
    spec = run_common.base_spec()
    first = fits_module.first_cut_index(cache=cache, site=site, cuts=cuts)
    return [
        fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=run_common.base_window(),
            season_years=run_common.SEASON_YEARS,
            pinned=scaled_pin(
                pin=pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut),
                scale=scale,
            ),
        )
        for cut in cuts[first:]
    ]


def _job(job: tuple[float, str]) -> str:
    torch.set_num_threads(1)
    scale, site = job
    started = time.monotonic()
    context = run349.load_site(site=site)
    fits = fit_scaled(scale=scale, site=site)
    proxy = proxy_cache(cache=context.cache, context=context.setup.context)
    stream = proxy_residual_stream(cache=proxy, fits=fits, site=site)
    run = filters.filter_stream(
        residual=stream,
        band=context.cache.calendar.band,
        settings=run349.settings_of_hyper(hyper=run349.noise_hyper(site=site)),
    )
    first = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    rows = 0
    for filter_name in common.FILTERS:
        trace = run.trace if filter_name == "kf" else None
        arm = common.arm_name(scale=scale, filter_name=filter_name)

        frame = functools.partial(
            ens.forecast_frame,
            cache=context.cache,
            setup=context.setup,
            fits=fits,
            trace=trace,
            arm=arm,
        )

        tuning = [
            i
            for i in context.setup.indices(periods={"tuning"})
            if context.setup.origins[i] >= first
        ]
        spread = ens.fit_spread_from_frame(frame=frame(run_indices=tuning, spread=ens.ZERO_SPREAD))
        for period in common.PERIODS:
            runs = [
                i
                for i in context.setup.indices(periods={period})
                if context.setup.origins[i] >= first
            ]
            if not runs:
                continue
            forecast = frame(run_indices=runs, spread=spread)
            target = common.preds_path(
                scale=scale, filter_name=filter_name, period=period, site=site
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            forecast.write_parquet(temporary)
            temporary.rename(target)
            rows += forecast.height
    return f"scale {scale} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Run every (scale, Primary) job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scales", nargs="+", type=float, default=list(common.SCALES))
    parser.add_argument("--sites", nargs="+", default=list(run_common.SCORED_SITES))
    args = parser.parse_args()
    jobs = [(scale, site) for scale in args.scales for site in args.sites]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
