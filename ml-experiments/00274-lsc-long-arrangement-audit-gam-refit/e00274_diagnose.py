"""Post hoc: where does a learned 30-day transfer show up in the forecast, by temperature and hour?

Fits GAM_ONLY at the end date of a 25% share transfer on the clean and the injected series at each
substation of a winter onset, and prints the mean forecast difference by temperature band and by
hour of day for the 48 hours after the end and for the following month.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= uv run --no-sync python e00274_diagnose.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Final

import e00274_fit as fit
import numpy as np
import polars as pl
from studies.dp_ens_weather import replace_site
from studies.dp_features import default_input_paths, load_or_build_feature_cache, naive_utc

SITES_SHOWN: Final[tuple[str, ...]] = ("D3", "D11")
ONSET: Final[datetime] = datetime(2023, 12, 1, tzinfo=UTC)
DURATION: Final[timedelta] = timedelta(days=30)


def main() -> None:
    """Print the forecast difference by temperature band and by hour."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=fit.CACHE_DIR)
    end = ONSET + DURATION
    stamps = cache.end_times
    active = (stamps > naive_utc(moment=ONSET)) & (stamps <= naive_utc(moment=end))
    for site in SITES_SHOWN:
        features = cache.sites[site]
        delta = np.where(active, 0.25 * features.y, 0.0)
        injected = replace_site(cache=cache, site=site, y=features.y + delta)
        stop = fit.add_months(moment=end.replace(day=1), months=2)
        index = fit.target_rows(cache=cache, site=site, cut=end, stop=stop)
        window = fit.WINDOWS["24m"]
        clean = fit.forecast(cache=cache, site=site, cut=end, window=window, index=index)
        moved = fit.forecast(cache=injected, site=site, cut=end, window=window, index=index)
        frame = pl.DataFrame(
            {
                "difference": moved - clean,
                "temperature_c": features.temperature_c[index],
                "hour": (
                    (stamps[index] - stamps[index].astype("datetime64[D]"))
                    // np.timedelta64(1, "h")
                ).astype(int),
                "days_after_end": (stamps[index] - naive_utc(moment=end)) / np.timedelta64(1, "D"),
            }
        )
        print(f"## {site}: mean transfer {float(np.mean(delta[active])):.2f} points")
        print(
            frame.with_columns(band=pl.col("temperature_c").cut([0, 5, 10, 15, 20]))
            .group_by("band")
            .agg(n=pl.len(), mean_difference=pl.col("difference").mean())
            .sort("band")
        )
        print(
            frame.with_columns(block=(pl.col("hour") // 6) * 6)
            .group_by("block")
            .agg(mean_difference=pl.col("difference").mean())
            .sort("block")
        )
        print(
            frame.with_columns(week=(pl.col("days_after_end") // 7).cast(pl.Int32))
            .group_by("week")
            .agg(mean_difference=pl.col("difference").mean())
            .sort("week")
        )


if __name__ == "__main__":
    main()
