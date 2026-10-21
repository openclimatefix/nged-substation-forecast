"""Fix experiment 119's three hard-day subsets before any ENC prediction is scored.

A day is the run date of a 00:00 UTC ENS run, and a day's valid times run from 09:00 that day to
09:00 the next (run leads 9 to 33 hours, experiment 111's analysis-proxy window). Thresholds pool
the runs of 2024-07-01 to 2026-06-30, and each subset holds 5% of those days (or of the
site-and-day pairs).

1. **Coldest 5%.** The mean over the five decision sites of the centre-cell control member's 2 m
   temperature, averaged over the day's 3-hourly steps. No power data.
2. **Most showery 5%, per site.** The standard deviation across the site's present patch cells of
   the control member's precipitation rate at each step, averaged over the day's steps. Only days
   whose patch-mean precipitation total is at least the median wet day's total can enter (a wet
   day has at least 0.1 mm). No power data.
3. **Largest GAM-K errors 5%.** The (site, day) pairs with the largest mean absolute error of K0's
   `q50`, over rows 15.5 to 39 hours after the origin, outside switching events, for days with at
   least 20 such rows. This uses GAM-K's outcomes only; no ENC prediction.

    uv run python ml-experiments/00119-*/e00119_hard_days.py
"""

import json
import math
from datetime import UTC, datetime
from typing import Final

import e00119_common as common
import e00119_gamk as gamk
import numpy as np
import polars as pl

POOL_START: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
SHARE: Final[float] = 0.05
DAY_STEPS: Final[range] = range(9, 34, 3)
"""Run-relative lead hours of the steps in a day's window."""
WET_DAY_MM: Final[float] = 0.1
STEP_SECONDS: Final[float] = 3 * 3600.0
MIN_ROWS: Final[int] = 20
D_PLUS_1: Final[tuple[float, float]] = (15.5, 39.0)


def hard_days() -> dict:
    """Return the three subsets as run-date strings."""
    frames = np.load(common.frames_path())
    layout = json.loads(common.patch_path().read_text())["patches"]
    union = [int(c) for c in frames["cells"]]
    position = {cell: i for i, cell in enumerate(union)}
    steps = [common.STEP_HOURS.index(h) for h in DAY_STEPS]
    temperature = common.FIELDS.index("temperature_2m")
    precipitation = common.FIELDS.index("precipitation_surface")
    dates = [d for d in common.run_dates() if d < common.HELD_OUT_START]
    keep = [i for i, d in enumerate(dates) if d >= POOL_START]
    control = frames["control"]
    centre_temperature = np.stack(
        [
            control[:, position[layout[site]["cells"][0]], :, temperature][:, steps].mean(axis=1)
            for site in common.DECISION_SITES
        ]
    ).mean(axis=0)
    n_days = len(keep)
    n_pick = math.ceil(SHARE * n_days)
    coldest = sorted(keep, key=lambda i: centre_temperature[i])[:n_pick]
    showery: dict[str, list[int]] = {}
    for site in common.SCORED_SITES:
        cells = [
            position[c]
            for c, ok in zip(layout[site]["cells"], layout[site]["present"], strict=True)
            if ok
        ]
        rate = control[:, cells][:, :, steps, precipitation]
        index = rate.std(axis=1).mean(axis=1)
        total_mm = rate.mean(axis=1).sum(axis=1) * STEP_SECONDS
        wet = [i for i in keep if total_mm[i] >= WET_DAY_MM]
        threshold = float(np.median(total_mm[wet]))
        eligible = [i for i in wet if total_mm[i] >= threshold]
        showery[site] = sorted(eligible, key=lambda i: -index[i])[:n_pick]
    errors = []
    for site in common.SCORED_SITES:
        k0 = pl.read_parquet(gamk.k0_path(site=site))
        daily = (
            k0.filter(
                ~pl.col("switching_mask"),
                pl.col("lead_hours").is_between(*D_PLUS_1),
                pl.col("run") >= keep[0],
            )
            .with_columns(error=(pl.col("actual") - pl.col("q50")).abs())
            .group_by("run")
            .agg(pl.col("error").mean(), pl.len().alias("n"))
            .filter(pl.col("n") >= MIN_ROWS)
            .with_columns(site=pl.lit(site))
        )
        errors.append(daily)
    pairs = pl.concat(errors)
    n_error = math.ceil(SHARE * pairs.height)
    worst = pairs.sort("error", descending=True).head(n_error)
    largest: dict[str, list[int]] = {
        site: sorted(worst.filter(pl.col("site") == site)["run"].to_list())
        for site in common.SCORED_SITES
    }

    def iso(indices: list[int]) -> list[str]:
        return [f"{dates[i]:%Y-%m-%d}" for i in sorted(indices)]

    return {
        "definition": "see e00119_hard_days.py; days are run dates (00:00 UTC runs)",
        "n_pool_days": n_days,
        "coldest": iso(coldest),
        "showery": {site: iso(v) for site, v in showery.items()},
        "largest_error": {site: iso(v) for site, v in largest.items()},
        "n_largest_error_pairs": n_error,
    }


def main() -> None:
    """Write `hard_days.json` and print the subsets' sizes."""
    result = hard_days()
    target = common.WORK_DIR / "hard_days.json"
    common.write_json_atomic(path=target, content=result)
    print(f"pool days {result['n_pool_days']}; coldest {len(result['coldest'])}")
    for site in common.SCORED_SITES:
        print(
            f"{site}: showery {len(result['showery'][site])}, "
            f"largest error {len(result['largest_error'][site])}"
        )


if __name__ == "__main__":
    main()
