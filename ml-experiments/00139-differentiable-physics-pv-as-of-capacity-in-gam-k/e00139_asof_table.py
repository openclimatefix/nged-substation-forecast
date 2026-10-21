"""Build the as-of solar capacity table of experiment 139, one block per origin and Primary.

At an origin `O`, a fit of experiment 129's arm 2 uses only rows before `O`. Its raw capacity per
season-year is divided by a divisor, the median recovery of the five real farms' output injected at
the Primary's pinned capacity and fitted with the same cut `O`. So neither the capacity nor the
divisor sees data after `O`. Experiment 129 took its divisor from fits to 2026-07-31.

    uv run --no-sync python ml-experiments/00139-*/e00139_asof_table.py [--workers 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import e00139_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam, training_rows
from studies.dp_pinned_capacity import pinned_capacity, read_vintages


def table_path(*, site: str, origin: datetime) -> Path:
    """Return the file that holds one Primary's block for one origin."""
    return common.WORK_DIR / "asof" / f"asof_{site}_{origin:%Y%m%d}.parquet"


def capacity_by_year(*, cache: FeatureCache, site: str, origin: datetime, device: str) -> dict:
    """Return an arm 2 fit's capacity per season-year, in points, with rows before `origin`."""
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[common.ARM2_SPEC],
        cut=origin,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
        device=device,
        dtype=torch.float64,
    )
    return fit.capacity_points()


def year_row_counts(*, cache: FeatureCache, site: str, origin: datetime) -> dict[int, int]:
    """Return the number of training half-hours of each season-year before `origin`."""
    index = training_rows(
        features=cache.sites[site], cache=cache, cut=origin, window=common.TRAINING_WINDOW
    )
    years, counts = np.unique(np.asarray(cache.calendar.season_year)[index], return_counts=True)
    return {int(y): int(c) for y, c in zip(years, counts, strict=True)}


def build_block(*, site: str, origin: datetime, device: str) -> str:
    """Write one Primary's block: raw capacity, recoveries of five farms, divisors, estimates."""
    started = time.monotonic()
    torch.set_num_threads(1)
    cache = common.load_cache()
    table = cache.sites[site].table
    pinned = pinned_capacity(vintages=read_vintages(), table=table, cut=origin).held_points
    base = capacity_by_year(cache=cache, site=site, origin=origin, device=device)
    counts = year_row_counts(cache=cache, site=site, origin=origin)
    recoveries: dict[int, list[float]] = {year: [] for year in common.SEASON_YEARS}
    for farm in common.FARMS:
        injected = pinned * common.farm_norm(cache=cache, farm=farm)
        shifted = replace(
            cache,
            sites={
                **cache.sites,
                site: replace(cache.sites[site], y=cache.sites[site].y - injected),
            },
        )
        capacity = capacity_by_year(cache=shifted, site=site, origin=origin, device=device)
        for year in common.SEASON_YEARS:
            recoveries[year].append((capacity[year] - base[year]) / pinned)
    scale = table.effective_capacity_mw / 100.0
    rows = []
    for year in common.SEASON_YEARS:
        divisor = float(np.median(recoveries[year]))
        rows.append(
            {
                "site": site,
                "origin": origin,
                "season_year": year,
                "training_rows": counts.get(year, 0),
                "raw_points": base[year],
                "divisor": divisor,
                "estimate_points": base[year] / divisor if divisor > 0 else float("nan"),
                "estimate_mw": base[year] / divisor * scale if divisor > 0 else float("nan"),
                "pinned_points": pinned,
                "effective_capacity_mw": table.effective_capacity_mw,
            }
        )
    common.write_parquet_atomic(frame=pl.DataFrame(rows), path=table_path(site=site, origin=origin))
    return f"{site} {origin:%Y-%m-%d}: {time.monotonic() - started:.0f} s"


def _job(job: tuple[str, datetime, str]) -> str:
    site, origin, device = job
    return build_block(site=site, origin=origin, device=device)


def main() -> None:
    """Build every block that does not exist yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-site", default=None)
    arguments = parser.parse_args()
    jobs = [
        (site, origin, arguments.device)
        for origin in common.ORIGINS
        for site in common.DECISION_SITES
        if (arguments.only_site in (None, site))
        and not table_path(site=site, origin=origin).exists()
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
