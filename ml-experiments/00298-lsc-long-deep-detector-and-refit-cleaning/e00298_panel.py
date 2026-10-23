"""Build the clean daily residual panel: GAM_ONLY refitted monthly at all 16 Primaries.

For each site and each monthly cut from January 2020, the 24-month GAM_ONLY (unmasked) predicts the
following month. The daily mean of (reading minus forecast) in MW, the daily mean power, and the
daily mean forecast are stored. Months a site cannot train for are skipped.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        uv run --no-sync python e00298_panel.py --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import e00298_common as common
import polars as pl
import torch  # ty: ignore[unresolved-import]

_CACHE = None


def site_panel(site: str) -> pl.DataFrame:
    """Return one site's daily residual rows over every monthly cut."""
    torch.set_num_threads(1)
    cache = _CACHE
    assert cache is not None
    capacity = cache.sites[site].table.effective_capacity_mw / 100.0
    cuts = common.month_starts(first=common.FIRST_CUT, last=common.LAST_CUT)
    parts = []
    for cut in cuts:
        stop = common.add_months(moment=cut, months=1)
        index = common.target_rows(cache=cache, site=site, cut=cut, stop=stop)
        if len(index) < common.MIN_ROWS_PER_DAY:
            continue
        predicted = common.forecast_month(cache=cache, site=site, cut=cut, index=index)
        if predicted is None:
            continue
        y = cache.sites[site].y[index]
        days = common.utc_day_of(cache=cache, index=index)
        day, resid = common.daily_means(days=days, values=(y - predicted) * capacity)
        _, power = common.daily_means(days=days, values=y * capacity)
        _, expected = common.daily_means(days=days, values=predicted * capacity)
        parts.append(
            pl.DataFrame(
                {"site": site, "day": day, "residual_mw": resid, "power_mw": power,
                 "expected_mw": expected, "cut": cut.date()}
            )  # fmt: skip
        )
    return pl.concat(parts) if parts else pl.DataFrame()


def main() -> None:
    """Build the panel in parallel and write it."""
    global _CACHE  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    common.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE = common.load_cache()
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        frames = list(pool.map(site_panel, common.SITES))
    panel = pl.concat([f for f in frames if f.height])
    panel.write_parquet(common.PANEL_PATH)
    print(panel.group_by("site").agg(pl.len(), first=pl.col("day").min(), last=pl.col("day").max()).sort("site"))


if __name__ == "__main__":
    main()
